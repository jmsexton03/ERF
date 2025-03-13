#include "ERF_Morrison.H"
#include "ERF_Constants.H"
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

/**
 * Updates conserved and microphysics variables in the provided MultiFabs from
 * the internal MultiFabs that store Microphysics module data.
 *
 * @param[out] cons Conserved variables
 * @param[out] qmoist: qv, qc, qi, qr, qs, qg
 */
void
Morrison::Copy_Micro_to_State (MultiFab& cons)
{
    // Get the temperature, density, theta, qt and qp from input
    for ( MFIter mfi(cons,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const auto& box3d = mfi.tilebox();

        auto states_arr = cons.array(mfi);

        auto rho_arr    = mic_fab_vars[MicVar_Morr::rho]->array(mfi);
        auto theta_arr  = mic_fab_vars[MicVar_Morr::theta]->array(mfi);

        auto qv_arr     = mic_fab_vars[MicVar_Morr::qv]->array(mfi);
        auto qc_arr     = mic_fab_vars[MicVar_Morr::qcl]->array(mfi);
        auto qi_arr     = mic_fab_vars[MicVar_Morr::qci]->array(mfi);

        auto qpr_arr     = mic_fab_vars[MicVar_Morr::qpr]->array(mfi);
        auto qps_arr     = mic_fab_vars[MicVar_Morr::qps]->array(mfi);
        auto qpg_arr     = mic_fab_vars[MicVar_Morr::qpg]->array(mfi);

        // get potential total density, temperature, qt, qp
        ParallelFor( box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            states_arr(i,j,k,RhoTheta_comp) = rho_arr(i,j,k)*theta_arr(i,j,k);

            states_arr(i,j,k,RhoQ1_comp)    = rho_arr(i,j,k)*std::max(0.0,qv_arr(i,j,k));
            states_arr(i,j,k,RhoQ2_comp)    = rho_arr(i,j,k)*std::max(0.0,qc_arr(i,j,k));
            states_arr(i,j,k,RhoQ3_comp)    = rho_arr(i,j,k)*std::max(0.0,qi_arr(i,j,k));

            states_arr(i,j,k,RhoQ4_comp)    = rho_arr(i,j,k)*std::max(0.0,qpr_arr(i,j,k));
            states_arr(i,j,k,RhoQ5_comp)    = rho_arr(i,j,k)*std::max(0.0,qps_arr(i,j,k));
            states_arr(i,j,k,RhoQ6_comp)    = rho_arr(i,j,k)*std::max(0.0,qpg_arr(i,j,k));
        });
    }

    // Fill interior ghost cells and periodic boundaries
    cons.FillBoundary(m_geom.periodicity());
}

/**
 * Updates microphysics variables and computes tendencies for one timestep.
 * This function applies all microphysical processes sequentially.
 *
 * @param[in] dt_advance Timestep for the advance
 * @param[in] sc Solver choices and configuration
 */
void
Morrison::Advance(const amrex::Real& dt_advance,
                 const SolverChoice& sc)
{
    BL_PROFILE("Morrison::Advance()");
    
    // Store timestep
    dt = dt_advance;
    
    // 5. CLOUD PROCESSES
    // Cloud phase thermodynamics and saturation adjustment
    Cloud(sc);

    // 3. Sublimation/Deposition of Ice Species
    SublimationDeposition(sc);

    // 6. ICE SEDIMENTATION
    // Cloud ice sedimentation
    if (m_iliq == 0) {  // Skip if liquid-only mode is active
        IceFall(sc);
    }
    
    // 9. PRECIPITATION PROCESSES
    // Autoconversion, collection, evaporation
    Precip(sc);
    
    // 10. PRECIPITATION SEDIMENTATION
    // Precipitation sedimentation
    PrecipFall(sc);
    
    // Compute reflectivity if requested
    if (m_do_radar_ref) {
        ComputeRadarReflectivity();
    }
}

void Morrison::rayleigh_soak_wetgraupel(const amrex::Real x,
                                      const amrex::Real xocm,
                                      const amrex::Real xobm,
                                      const amrex::Real fmelt,
                                      const bool melt_outside,
                                      const std::complex<amrex::Real>& m_w_0,
                                      const std::complex<amrex::Real>& m_i_0,
                                      const amrex::Real lambda_radar,
                                      amrex::Real& cback,
                                      const std::string& mixingrulestring,
                                      const std::string& matrixstring,
                                      const std::string& inclusionstring,
                                      const std::string& hoststring,
                                      const std::string& hostmatrixstring,
                                      const std::string& hostinclusionstring) const
{
    // Calculate particle diameter from mass
    amrex::Real rho_particle;
    if (hoststring == "snow") {
        rho_particle = m_rhosn;
    } else if (hoststring == "graupel") {
        rho_particle = m_rhog;
    } else {
        rho_particle = m_rhosu;
    }

    amrex::Real diameter = std::pow((6.0 * x) / (m_pi * rho_particle), 1.0/3.0);

    // Calculate effective dielectric constant based on mixing rule
    std::complex<amrex::Real> m_eff;

    if (mixingrulestring == "maxwell") {
        // Maxwell-Garnett mixing rule
        std::complex<amrex::Real> m_matrix, m_inclusion;

        if (melt_outside) {
            // Liquid coating (water outside, ice inside)
            m_matrix = m_w_0;
            m_inclusion = m_i_0;
        } else {
            // Ice coating (ice outside, water inside)
            m_matrix = m_i_0;
            m_inclusion = m_w_0;
        }

        // Volume fractions
        amrex::Real vol_inclusion = melt_outside ? (1.0 - fmelt) : fmelt;

        // Maxwell-Garnett formula
        std::complex<amrex::Real> beta = (m_inclusion - m_matrix) /
                                        (m_inclusion + 2.0 * m_matrix);
        m_eff = m_matrix * (1.0 + 3.0 * vol_inclusion * beta) /
                          (1.0 - vol_inclusion * beta);
    } else {
        // Simple volume-weighted average as fallback
        m_eff = fmelt * m_w_0 + (1.0 - fmelt) * m_i_0;
    }

    // Calculate Rayleigh backscattering cross-section
    amrex::Real k = 2.0 * m_pi / lambda_radar; // Wavenumber
    std::complex<amrex::Real> K = (m_eff * m_eff - 1.0) / (m_eff * m_eff + 2.0);
    amrex::Real K_squared = std::norm(K); // |K|^2

    // Backscattering cross-section
    cback = m_pi * m_pi * std::pow(diameter, 6) * K_squared / (lambda_radar * lambda_radar);
}

/**
 * Computes radar reflectivity from hydrometeor properties.
 * 
 */
void
Morrison::ComputeRadarReflectivity()
{
    BL_PROFILE("Morrison::ComputeRadarReflectivity()");

    if (!m_do_radar_ref || !m_radar) return;

    // Loop through grids
    for (MFIter mfi(*mic_fab_vars[MicVar_Morr::tabs]); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();

        // Get array data
        auto const& thermo_tabs = mic_fab_vars[MicVar_Morr::tabs]->array(mfi);
        auto const& thermo_pres = mic_fab_vars[MicVar_Morr::pres]->array(mfi);
        auto const& hydro_qr = mic_fab_vars[MicVar_Morr::qpr]->array(mfi);
        auto const& hydro_qs = mic_fab_vars[MicVar_Morr::qps]->array(mfi);
        auto const& hydro_qg = mic_fab_vars[MicVar_Morr::qpg]->array(mfi);
        auto const& hydro_nr = mic_fab_vars[MicVar_Morr::nr]->array(mfi);
        auto const& hydro_ns = mic_fab_vars[MicVar_Morr::ns]->array(mfi);
        auto const& hydro_ng = mic_fab_vars[MicVar_Morr::ng]->array(mfi);
        auto const& radar = m_radar->array(mfi);

        // Component indices
        const int t_comp = 0;   // Temperature
        const int p_comp = 1;   // Pressure
        const int qr_comp = 1;  // Rain
        const int qs_comp = 3;  // Snow
        const int qg_comp = 4;  // Graupel
        const int nr_comp = 6;  // Rain number
        const int ns_comp = 8;  // Snow number
        const int ng_comp = 9;  // Graupel number

        // ParallelFor loop over grid
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Default minimum reflectivity
            radar(i,j,k,0) = -35.0;

            // Calculate reflectivity for rain, snow, and graupel
            amrex::Real ze_rain = 1.0e-22;    // Minimum to avoid log issues
            amrex::Real ze_snow = 1.0e-22;
            amrex::Real ze_graupel = 1.0e-22;

            // Calculate rain reflectivity (if present)
            if (hydro_qr(i,j,k) >= m_qsmall) {
                const amrex::Real lamr = std::pow(m_pi * m_rhow * hydro_nr(i,j,k) / hydro_qr(i,j,k), 1.0/3.0);
                const amrex::Real n0r = hydro_nr(i,j,k) * lamr;
                //Corrected rain reflectivity
                ze_rain = n0r * gamma_function(4.0 + m_br) / std::pow(lamr, 4.0 + m_br);
            }

            // Calculate snow reflectivity (if present)
            if (hydro_qs(i,j,k) >= m_qsmall) {
                const amrex::Real lams = std::pow(m_cons1 * hydro_ns(i,j,k) / hydro_qs(i,j,k), 1.0/m_ds);
                const amrex::Real n0s = hydro_ns(i,j,k) * lams;

                // Check if melting is occurring
                if (thermo_tabs(i,j,k) >= 273.15) {
                    // Use rayleigh_soak_wetgraupel for melting snow
                    amrex::Real cback_s = 0.0; // Initialize
                    //Need to calculate fmelt_s, the melt fraction
                    //This requires finding the level where melting begins, k_0
                    //This is done in the outer loop, so pass 0 for now
                    rayleigh_soak_wetgraupel(hydro_qs(i,j,k), m_ds, m_bs, 0.0, m_melt_outside_s,
                                             m_m_w_0, m_m_i_0, m_lambda_radar, cback_s,
                                             m_mixingrulestring_s, m_matrixstring_s,
                                             m_inclusionstring_s, m_hoststring_s,
                                             m_hostmatrixstring_s, m_hostinclusionstring_s);
                    ze_snow = cback_s * n0s * gamma_function(1.0 + m_bs) / std::pow(lams, 1.0 + m_bs); // Simplified
                } else {
                    // Dielectric factor for dry snow
                    ze_snow = (0.176/0.93) * std::pow(6.0/m_pi, 2.0) *
                              std::pow(m_rhosn/900.0, 2.0) *
                              n0s * gamma_function(4.0 + m_bs) / std::pow(lams, 4.0 + m_bs);
                }
            }

            // Calculate graupel reflectivity (if present)
            if (hydro_qg(i,j,k) >= m_qsmall) {
                const amrex::Real lamg = std::pow(m_cons2 * hydro_ng(i,j,k) / hydro_qg(i,j,k), 1.0/m_dg);
                const amrex::Real n0g = hydro_ng(i,j,k) * lamg;

                // Check if melting is occurring
                if (thermo_tabs(i,j,k) >= 273.15) {
                    // Use rayleigh_soak_wetgraupel for melting graupel
                    amrex::Real cback_g = 0.0;
                    //Need to calculate fmelt_g, the melt fraction
                    //This requires finding the level where melting begins, k_0
                    //This is done in the outer loop, so pass 0 for now
                    rayleigh_soak_wetgraupel(hydro_qg(i,j,k), m_dg, m_bg, 0.0, m_melt_outside_g,
                                             m_m_w_0, m_m_i_0, m_lambda_radar, cback_g,
                                             m_mixingrulestring_g, m_matrixstring_g,
                                             m_inclusionstring_g, m_hoststring_g,
                                             m_hostmatrixstring_g, m_hostinclusionstring_g);
                    ze_graupel = cback_g * n0g * gamma_function(1.0 + m_bg) / std::pow(lamg, 1.0 + m_bg); // Simplified
                } else {
                    // Dielectric factor for graupel
                    ze_graupel = (0.176/0.93) * std::pow(6.0/m_pi, 2.0) *
                                 std::pow(m_rhog/900.0, 2.0) *
                                 n0g * gamma_function(4.0 + m_bg) / std::pow(lamg, 4.0 + m_bg);
                }
            }

            // Calculate total reflectivity and convert to dBZ
            radar(i,j,k,0) = 10.0 * std::log10(ze_rain + ze_snow + ze_graupel);

            // Ensure reflectivity is above minimum threshold
            radar(i,j,k,0) = amrex::max(radar(i,j,k,0), -35.0);
        });
    }
    // Find the melting level (k_0)
    int k_0 = -1;
    for (MFIter mfi(*mic_fab_vars[MicVar_Morr::tabs]); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        auto const& thermo_tabs = mic_fab_vars[MicVar_Morr::tabs]->array(mfi);
        auto const& hydro_qr = mic_fab_vars[MicVar_Morr::qpr]->array(mfi);
        auto const& hydro_qs = mic_fab_vars[MicVar_Morr::qps]->array(mfi);
        auto const& hydro_qg = mic_fab_vars[MicVar_Morr::qpg]->array(mfi);
        auto const& radar = m_radar->array(mfi);
        // Iterate from top down to find the first level where T > 273.15
        // and there's both rain and snow/graupel present.
        for (int k = box.bigEnd(2); k >= box.smallEnd(2); --k) {
            for (int j = box.bigEnd(1); j >= box.smallEnd(1); --j) {
                for (int i = box.bigEnd(0); i >= box.smallEnd(0); --i) {
                    if (thermo_tabs(i,j,k) > 273.15 && hydro_qr(i,j,k) > m_qsmall &&
                        (hydro_qs(i,j,k) > m_qsmall || hydro_qg(i,j,k) > m_qsmall))
                    {
                        k_0 = k;
                        break; // Found the level, exit the loop
                    }
                }
            if (k_0 != -1) break;
            }
        if (k_0 != -1) break; // Exit MFIter loop once k_0 is found.
        }
    if (k_0 != -1) break; // Exit MFIter loop once k_0 is found.
    }

    // Now, re-calculate reflectivity for melting snow/graupel using the correct fmelt.
    if (k_0 != -1) {
        for (MFIter mfi(*mic_fab_vars[MicVar_Morr::tabs]); mfi.isValid(); ++mfi) {
            const Box& box = mfi.validbox();
            auto const& thermo_tabs = mic_fab_vars[MicVar_Morr::tabs]->array(mfi);
            auto const& hydro_qr = mic_fab_vars[MicVar_Morr::qpr]->array(mfi);
            auto const& hydro_qs = mic_fab_vars[MicVar_Morr::qps]->array(mfi);
            auto const& hydro_qg = mic_fab_vars[MicVar_Morr::qpg]->array(mfi);
            auto const& hydro_nr = mic_fab_vars[MicVar_Morr::nr]->array(mfi);
            auto const& hydro_ns = mic_fab_vars[MicVar_Morr::ns]->array(mfi);
            auto const& hydro_ng = mic_fab_vars[MicVar_Morr::ng]->array(mfi);
            auto const& radar = m_radar->array(mfi);

            // Component indices
            const int t_comp = 0;   // Temperature
            const int p_comp = 1;   // Pressure
            const int qr_comp = 1;  // Rain
            const int qs_comp = 3;  // Snow
            const int qg_comp = 4;  // Graupel
            const int nr_comp = 6;  // Rain number
            const int ns_comp = 8;  // Snow number
            const int ng_comp = 9;  // Graupel number

            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // Only process levels below the melting level
                if (k < k_0) {
                    // Snow
                    if (hydro_qs(i,j,k) >= m_qsmall) {
                        const amrex::Real lams = std::pow(m_cons1 * hydro_ns(i,j,k) / hydro_qs(i,j,k), 1.0/m_ds);
                        const amrex::Real n0s = hydro_ns(i,j,k) * lams;
                        //Find total snow at k_0
                        amrex::Real qs_k0 = 0.0;
                        if (hydro_qs(i,j,k_0) >= m_qsmall){
                            qs_k0 = hydro_qs(i,j,k_0);
                        }
                        amrex::Real fmelt_s = 1.0 - hydro_qs(i,j,k) / qs_k0;
                        fmelt_s = std::max(0.0, std::min(fmelt_s, 1.0)); // Ensure 0 <= fmelt <= 1

                        amrex::Real cback_s = 0.0;
                        rayleigh_soak_wetgraupel(hydro_qs(i,j,k), m_ds, m_bs, fmelt_s, m_melt_outside_s,
                                                 m_m_w_0, m_m_i_0, m_lambda_radar, cback_s,
                                                 m_mixingrulestring_s, m_matrixstring_s,
                                                 m_inclusionstring_s, m_hoststring_s,
                                                 m_hostmatrixstring_s, m_hostinclusionstring_s);
                        amrex::Real ze_snow = cback_s * n0s * gamma_function(1.0 + m_bs) / std::pow(lams, 1.0 + m_bs);
                        // Update the snow reflectivity component
                        radar(i,j,k,0) = 10.0 * std::log10(std::pow(10.0, radar(i,j,k,0) / 10.0) + ze_snow - 1.0e-22);
                    }

                    // Graupel
                    if (hydro_qg(i,j,k) >= m_qsmall) {
                        const amrex::Real lamg = std::pow(m_cons2 * hydro_ng(i,j,k) / hydro_qg(i,j,k), 1.0/m_dg);
                        const amrex::Real n0g = hydro_ng(i,j,k) * lamg;
                        //Find total graupel at k_0
                        amrex::Real qg_k0 = 0.0;
                        if (hydro_qg(i,j,k_0) >= m_qsmall){
                            qg_k0 = hydro_qg(i,j,k_0);
                        }
                        amrex::Real fmelt_g = 1.0 - hydro_qg(i,j,k) / qg_k0;
                        fmelt_g = std::max(0.0, std::min(fmelt_g, 1.0)); // Ensure 0 <= fmelt <= 1

                        amrex::Real cback_g = 0.0;
                        rayleigh_soak_wetgraupel(hydro_qg(i,j,k), m_dg, m_bg, fmelt_g, m_melt_outside_g,
                                                 m_m_w_0, m_m_i_0, m_lambda_radar, cback_g,
                                                 m_mixingrulestring_g, m_matrixstring_g,
                                                 m_inclusionstring_g, m_hoststring_g,
                                                 m_hostmatrixstring_g, m_hostinclusionstring_g);

                        amrex::Real ze_graupel = cback_g * n0g * gamma_function(1.0 + m_bg) / std::pow(lamg, 1.0 + m_bg);
                        // Update the graupel reflectivity component
                        radar(i,j,k,0) = 10.0 * std::log10(std::pow(10.0, radar(i,j,k,0) / 10.0) + ze_graupel - 1.0e-22);
                    }
                }
            });
        }
    }
}

    /**
     * Calculates sublimation/deposition rates for ice, snow, and graupel.
     *
     * @param[in] sc SolverChoice containing solver configuration
     */
    void Morrison::SublimationDeposition(const SolverChoice& sc)
    {
        BL_PROFILE("Morrison::SublimationDeposition()");

        // Loop through grids
        for (amrex::MFIter mfi(*mic_fab_vars[MicVar_Morr::tabs]); mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.validbox();

            // Get array data
            auto const& thermo_tabs = mic_fab_vars[MicVar_Morr::tabs]->array(mfi);
            auto const& thermo_pres = mic_fab_vars[MicVar_Morr::pres]->array(mfi);
            auto const& hydro_qv = mic_fab_vars[MicVar_Morr::qv]->array(mfi);
            auto const& thermo_rho = mic_fab_vars[MicVar_Morr::rho]->array(mfi);
            auto const& hydro_qi = mic_fab_vars[MicVar_Morr::qci]->array(mfi);
            auto const& hydro_qs = mic_fab_vars[MicVar_Morr::qps]->array(mfi);
            auto const& hydro_qg = mic_fab_vars[MicVar_Morr::qpg]->array(mfi);
            auto const& hydro_ni = mic_fab_vars[MicVar_Morr::ni]->array(mfi);
            auto const& hydro_nr = mic_fab_vars[MicVar_Morr::nr]->array(mfi);
            auto const& hydro_ns = mic_fab_vars[MicVar_Morr::ns]->array(mfi);
            auto const& hydro_ng = mic_fab_vars[MicVar_Morr::ng]->array(mfi);
            auto const& tend = m_tend->array(mfi);

            // Component indices for thermodynamic variables
            const int t_comp = 0;   // Temperature
            const int p_comp = 1;   // Pressure
            const int qv_comp = 2;  // Water vapor mixing ratio
            const int rho_comp = 3; // Density

            // Component indices for hydrometeors
            const int qc_comp = 0;  // Cloud water
            const int qr_comp = 1;  // Rain
            const int qi_comp = 2;  // Cloud ice
            const int qs_comp = 3;  // Snow
            const int qg_comp = 4;  // Graupel
            const int nc_comp = 5;  // Cloud droplet number
            const int nr_comp = 6;  // Rain number
            const int ni_comp = 7;  // Cloud ice number
            const int ns_comp = 8;  // Snow number
            const int ng_comp = 9;  // Graupel number

            // Get table data for coefficients
            auto const& evaps1_t = evaps1.table();
            auto const& evaps2_t = evaps2.table();
            auto const& evapg1_t = evapg1.table();
            auto const& evapg2_t = evapg2.table();
            auto const& tabs1d_t = tabs1d.table();

            // Parallel execution over the box
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // Local variables
                const amrex::Real temp = thermo_tabs(i,j,k);
                const amrex::Real pres = thermo_pres(i,j,k);
                const amrex::Real qv = hydro_qv(i,j,k);
                const amrex::Real rho = thermo_rho(i,j,k);
                const amrex::Real qi = hydro_qi(i,j,k);
                const amrex::Real qs = hydro_qs(i,j,k);
                const amrex::Real qg = hydro_qg(i,j,k);
                const amrex::Real ni = hydro_ni(i,j,k);
                const amrex::Real ns = hydro_ns(i,j,k);
                const amrex::Real ng = hydro_ng(i,j,k);

                // Vertical index for 1D profile data
                const int k_1d = k - m_geom.ProbLo(2) / m_geom.CellSize(2);

                // Initialize process rates
                amrex::Real prd = 0.0;   // Deposition/sublimation of cloud ice
                amrex::Real prds = 0.0;  // Deposition/sublimation of snow
                amrex::Real prdg = 0.0;  // Deposition/sublimation of graupel
                amrex::Real eprd = 0.0;
                amrex::Real eprds = 0.0;
                amrex::Real eprdg = 0.0;

                // Calculate saturation vapor pressures
                amrex::Real evs = std::min(0.99*pres, calc_saturation_vapor_pressure(temp, 0));
                amrex::Real eis = std::min(0.99*pres, calc_saturation_vapor_pressure(temp, 1));
                if (eis > evs) eis = evs;

                // Calculate saturation mixing ratios
                amrex::Real qvs = m_ep_2 * evs / (pres - evs);
                amrex::Real qvi = m_ep_2 * eis / (pres - eis);

                // Calculate saturation ratios
                amrex::Real qvqvs = qv / qvs;
                amrex::Real qvqvsi = qv / qvi;

                // 1. Cloud Ice Sublimation/Deposition
                if (qi >= m_qsmall) {
                    // Calculate size distribution parameter for cloud ice
                    amrex::Real lami = std::pow(m_cons12 * ni / qi, 1.0/m_di);

                    // Apply limits to lambda
                    lami = amrex::max(lami, m_lammini);
                    lami = amrex::min(lami, m_lammaxi);

                    // Calculate deposition/sublimation rate (mass)
                    // Only include region of ice size dist < DCS
                    amrex::Real dum = (1.0 - std::exp(-lami * m_dcs) * (1.0 + lami * m_dcs));
                    prd = 2.0 * M_PI * ni * rho * (qv - qvi) /
                           (rho * ((m_fac_sub / (tabs1d_t(k_1d) * m_Rv) - 1.0) * m_fac_sub /
			   (1.414E3 * 1.496E-6 * std::pow(tabs1d_t(k_1d),1.5) / (tabs1d_t(k_1d) + 120.0) * tabs1d_t(k_1d)) +
                           m_Rv * tabs1d_t(k_1d) / (8.794E-5 * std::pow(tabs1d_t(k_1d),1.81) / pres * eis))) *
                           dum / (1.0 + (3.15e6 - 2370.0 * temp + 0.3337e6) * (3.15e6 - 2370.0 * temp + 0.3337e6) * qvi /
                           (m_Rv * temp * temp) * m_fac_sub / (m_cp * (1.0 + 0.887 * qv)));
                    if (prd < 0.0) eprd = prd;
                }

                // 2. Snow Sublimation/Deposition
                if (qs >= m_qsmall) {
                    // Calculate size distribution parameter for snow
                    amrex::Real lams = std::pow(m_cons1 * ns / qs, 1.0/m_ds);

                    // Apply limits to lambda
                    lams = amrex::max(lams, m_lammins);
                    lams = amrex::min(lams, m_lammaxs);

                    // Calculate deposition/sublimation rate (mass)
                    prds = (qv - qvi) /
                           (rho * ((m_fac_sub / (tabs1d_t(k_1d) * m_Rv) - 1.0) * m_fac_sub /
                           (1.414E3 * 1.496E-6 * std::pow(tabs1d_t(k_1d),1.5) / (tabs1d_t(k_1d) + 120.0) * tabs1d_t(k_1d)) +
                                   m_Rv * tabs1d_t(k_1d) / (8.794E-5 * std::pow(tabs1d_t(k_1d),1.81) / pres * eis))) *
                           (evaps1_t(k_1d) + evaps2_t(k_1d) / std::pow(lams, m_bs / 2.0));
                    if (qi < m_qsmall) prds = prds + prd; //Add deposition from cloud ice if no cloud ice
                    if (prds < 0.0) eprds = prds;
                }

                // 3. Graupel Sublimation/Deposition
                if (qg >= m_qsmall) {
                    // Calculate size distribution parameter for graupel
                    amrex::Real lamg = std::pow(m_cons2 * ng / qg, 1.0/m_dg);

                    // Apply limits to lambda
                    lamg = amrex::max(lamg, m_lamming);
                    lamg = amrex::min(lamg, m_lammaxg);

                    // Calculate deposition/sublimation rate (mass)
                    prdg = (qv - qvi) /
                           (rho * ((m_fac_sub / (tabs1d_t(k_1d) * m_Rv) - 1.0) * m_fac_sub /
                           (1.414E3 * 1.496E-6 * std::pow(tabs1d_t(k_1d),1.5) / (tabs1d_t(k_1d) + 120.0) * tabs1d_t(k_1d)) +
                           m_Rv * tabs1d_t(k_1d) / (8.794E-5 * std::pow(tabs1d_t(k_1d),1.81) / pres * eis))) *
                           (evapg1_t(k_1d) + evapg2_t(k_1d) / std::pow(lamg, m_bg / 2.0));
                    if (prdg < 0.0) eprdg = prdg;
                }

                // Apply limits to prevent excessive sublimation
                eprd = std::max(eprd, -qi / dt);
                eprds = std::max(eprds, -qs / dt);
                eprdg = std::max(eprdg, -qg / dt);

                // Update tendencies
                tend(i,j,k,qv_comp) += prd + prds + prdg;
                tend(i,j,k,qi_comp) += prd;
                tend(i,j,k,qs_comp) += prds;
                tend(i,j,k,qg_comp) += prdg;

                // Latent heating/cooling
                const amrex::Real cpm = m_cp * (1.0 + 0.887 * qv);
                tend(i,j,k,t_comp) += -(prd + prds + prdg) * m_fac_sub / cpm;
            });

        }
    }
