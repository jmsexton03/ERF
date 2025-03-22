#include "ERF_Morrison.H"
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_MultiFabUtil.H>
#include <cmath>

// Specific heat at constant pressure for dry air (J/kg/K)
constexpr amrex::Real CP = 1005.0;

/**
 * Precipitation fluxes P_{r/s/g} (A19)
 *
 * This function handles the sedimentation of precipitation species (rain, snow, graupel).
 * It corresponds approximately to lines 3650-4150 in the original Fortran code.
 */
void
Morrison::PrecipFall(const SolverChoice& /*sc*/)
{
    BL_PROFILE("Morrison::PrecipFall()");

    // Local vertical indexing
    const int klo = zlo;
    const int khi = zhi;

    // Adaptive time stepping for sedimentation to maintain numerical stability
    constexpr int max_split_steps = 10;  // Maximum number of substeps allowed for sedimentation

    // Accumulated precipitation variables
    amrex::Real rain_accum = 0.0;
    amrex::Real snow_accum = 0.0;
    amrex::Real graup_accum = 0.0;

    // Initialize number concentrations
    initialize_size_distributions();

    // Loop through the grids
    for (amrex::MFIter mfi(*mic_fab_vars[MicVar_Morr::qpr]); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();

        // Get data arrays for hydrometeors and thermodynamic variables
        auto const& qpr = mic_fab_vars[MicVar_Morr::qpr]->array(mfi);
        auto const& qps = mic_fab_vars[MicVar_Morr::qps]->array(mfi);
        auto const& qpg = mic_fab_vars[MicVar_Morr::qpg]->array(mfi);
        auto const& qp = mic_fab_vars[MicVar_Morr::qp]->array(mfi);
        auto const& rho = mic_fab_vars[MicVar_Morr::rho]->array(mfi);
        auto const& tabs = mic_fab_vars[MicVar_Morr::tabs]->array(mfi);
        auto const& pres = mic_fab_vars[MicVar_Morr::pres]->array(mfi);
        
        // Get cloud hydrometeor arrays
        auto const& qcl = mic_fab_vars[MicVar_Morr::qcl]->array(mfi);  // Cloud water mixing ratio
        auto const& qci = mic_fab_vars[MicVar_Morr::qci]->array(mfi);  // Cloud ice mixing ratio
        
        // Get number concentration arrays
        auto const& nc = mic_fab_vars[MicVar_Morr::nc]->array(mfi);  // Cloud droplet number concentration
        auto const& ni = mic_fab_vars[MicVar_Morr::ni]->array(mfi);  // Ice number concentration
        auto const& nr = mic_fab_vars[MicVar_Morr::nr]->array(mfi);  // Rain number concentration
        auto const& ns = mic_fab_vars[MicVar_Morr::ns]->array(mfi);  // Snow number concentration
        auto const& ng = mic_fab_vars[MicVar_Morr::ng]->array(mfi);  // Graupel number concentration

        // Get arrays for accumulated precipitation
        auto const& rain_arr = mic_fab_vars[MicVar_Morr::rain_accum]->array(mfi);
        auto const& snow_arr = mic_fab_vars[MicVar_Morr::snow_accum]->array(mfi);
        auto const& graup_arr = mic_fab_vars[MicVar_Morr::graup_accum]->array(mfi);

        // Create containers for sedimentation tendencies
        amrex::FArrayBox qrsten_fab(box, 1);  // Rain mixing ratio tendency
        amrex::FArrayBox qisten_fab(box, 1);  // Cloud ice mixing ratio tendency
        amrex::FArrayBox qcsten_fab(box, 1);  // Cloud water mixing ratio tendency
        amrex::FArrayBox qnisten_fab(box, 1); // Snow mixing ratio tendency
        amrex::FArrayBox qgsten_fab(box, 1);  // Graupel mixing ratio tendency
        auto const& qrsten = qrsten_fab.array();
        auto const& qisten = qisten_fab.array();
        auto const& qcsten = qcsten_fab.array();
        auto const& qnisten = qnisten_fab.array();
        auto const& qgsten = qgsten_fab.array();

        // Thermodynamic arrays needed for subsaturation check
        auto const& qv3d = mic_fab_vars[MicVar_Morr::qv]->array(mfi);
        auto const& tabs3d = mic_fab_vars[MicVar_Morr::tabs]->array(mfi);

        // Create arrays for thermodynamic calculations
        amrex::FArrayBox xxlv_fab(box, 1);  // Latent heat of vaporization
        amrex::FArrayBox xxls_fab(box, 1);  // Latent heat of sublimation
        amrex::FArrayBox cpm_fab(box, 1);   // Specific heat of moist air
        auto const& xxlv = xxlv_fab.array();
        auto const& xxls = xxls_fab.array();
        auto const& cpm = cpm_fab.array();

        // Initialize tendencies to zero
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            qrsten(i,j,k) = 0.0;
            qisten(i,j,k) = 0.0;
            qcsten(i,j,k) = 0.0;
            qnisten(i,j,k) = 0.0;
            qgsten(i,j,k) = 0.0;
        });

        // Create flux arrays for all species
        amrex::FArrayBox flux_qr_fab(box, 1); // Rain mass flux
        amrex::FArrayBox flux_qs_fab(box, 1); // Snow mass flux
        amrex::FArrayBox flux_qg_fab(box, 1); // Graupel mass flux
        amrex::FArrayBox flux_qi_fab(box, 1); // Cloud ice mass flux
        amrex::FArrayBox flux_qc_fab(box, 1); // Cloud water mass flux
        amrex::FArrayBox flux_nr_fab(box, 1); // Rain number flux
        amrex::FArrayBox flux_ns_fab(box, 1); // Snow number flux
        amrex::FArrayBox flux_ng_fab(box, 1); // Graupel number flux
        amrex::FArrayBox flux_ni_fab(box, 1); // Cloud ice number flux
        amrex::FArrayBox flux_nc_fab(box, 1); // Cloud water number flux

        auto const& flux_qr = flux_qr_fab.array();
        auto const& flux_qs = flux_qs_fab.array();
        auto const& flux_qg = flux_qg_fab.array();
        auto const& flux_qi = flux_qi_fab.array();
        auto const& flux_qc = flux_qc_fab.array();
        auto const& flux_nr = flux_nr_fab.array();
        auto const& flux_ns = flux_ns_fab.array();
        auto const& flux_ng = flux_ng_fab.array();
        auto const& flux_ni = flux_ni_fab.array();
        auto const& flux_nc = flux_nc_fab.array();

//----------------------------------------------------------------------
// Calculate terminal fall velocities for each hydrometeor species using proper physics:
// - Cloud water: Stokes law with temperature-dependent viscosity
// - Cloud ice: Power law with Ikawa and Saito (1991) air-density correction (0.35 exponent)
// - Rain/snow/graupel: Power law with Heymsfield and Bansemer (2006) air-density correction (0.54 exponent)
//----------------------------------------------------------------------
    // Calculate fall speeds for each hydrometeor species and track maximum
    // for determining time step splitting, following the WRF approach
    //----------------------------------------------------------------------
    // Arrays to store terminal velocities for mass and number
    amrex::FArrayBox fr_fab(box, 1);  // Mass-weighted fall speed for rain
    amrex::FArrayBox fs_fab(box, 1);  // Mass-weighted fall speed for snow
    amrex::FArrayBox fg_fab(box, 1);  // Mass-weighted fall speed for graupel 
    amrex::FArrayBox fi_fab(box, 1);  // Mass-weighted fall speed for cloud ice
    amrex::FArrayBox fc_fab(box, 1);  // Mass-weighted fall speed for cloud water
    
    amrex::FArrayBox fnr_fab(box, 1); // Number-weighted fall speed for rain
    amrex::FArrayBox fns_fab(box, 1); // Number-weighted fall speed for snow
    amrex::FArrayBox fng_fab(box, 1); // Number-weighted fall speed for graupel
    amrex::FArrayBox fni_fab(box, 1); // Number-weighted fall speed for cloud ice
    amrex::FArrayBox fnc_fab(box, 1); // Number-weighted fall speed for cloud water
    
    auto const& fr = fr_fab.array();
    auto const& fs = fs_fab.array();
    auto const& fg = fg_fab.array();
    auto const& fi = fi_fab.array();
    auto const& fc = fc_fab.array();
    
    auto const& fnr = fnr_fab.array();
    auto const& fns = fns_fab.array();
    auto const& fng = fng_fab.array();
    auto const& fni = fni_fab.array();
    auto const& fnc = fnc_fab.array();
    
    // Initialize fall speeds to zero
    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
        fr(i,j,k) = 0.0;
        fs(i,j,k) = 0.0;
        fg(i,j,k) = 0.0;
        fi(i,j,k) = 0.0;
        fc(i,j,k) = 0.0;
        
        fnr(i,j,k) = 0.0;
        fns(i,j,k) = 0.0;
        fng(i,j,k) = 0.0;
        fni(i,j,k) = 0.0;
        fnc(i,j,k) = 0.0;
    });
    
    // Initialize thermodynamic arrays for subsaturation checking
    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
        // Latent heat of vaporization (J/kg)
        xxlv(i,j,k) = 3.1484E6 - 2370.0*tabs(i,j,k);
        
        // Latent heat of sublimation (J/kg)
        xxls(i,j,k) = 3.15E6 - 2370.0*tabs(i,j,k) + 0.3337E6;
        
        // Specific heat of moist air (J/kg/K)
        cpm(i,j,k) = CP*(1.0 + 0.887*qv3d(i,j,k));
    });
    
    // Calculate fall speeds for all hydrometeors at all levels
    for (int k = klo; k <= khi; ++k) {
        for (int j = box.loVect()[1]; j <= box.hiVect()[1]; ++j) {
            for (int i = box.loVect()[0]; i <= box.hiVect()[0]; ++i) {
                // Calculate size distribution parameters for all hydrometeors
                amrex::Real lamc = 0.0, lamr = 0.0, lami = 0.0, lams = 0.0, lamg = 0.0;
                amrex::Real n0c = 0.0, n0r = 0.0, n0i = 0.0, n0s = 0.0, n0g = 0.0;
                amrex::Real pgam = 0.0;
                
                // Calculate size distribution parameters
                size_distributions_params(
                    qcl(i,j,k), qci(i,j,k), qpr(i,j,k), qps(i,j,k), qpg(i,j,k),
                    nc(i,j,k), ni(i,j,k), nr(i,j,k), ns(i,j,k), ng(i,j,k),
                    rho(i,j,k), tabs(i,j,k), pres(i,j,k),
                    lamc, lamr, lami, lams, lamg, pgam,
                    n0c, n0r, n0i, n0s, n0g);
                
                // Cloud water fall speed - use Stokes law with temperature-dependent viscosity
                if (qcl(i,j,k) >= m_qsmall && lamc > 0.0) {
                    // Calculate temperature-dependent dynamic viscosity (MU) using the correct formula
                    // MU(K) = 1.496E-6*T3D(K)**1.5/(T3D(K)+120.) in Fortran
                    const amrex::Real mu = 1.496E-6 * std::pow(tabs(i,j,k), 1.5) / (tabs(i,j,k) + 120.0);
                    
                    // Stokes fall velocity for cloud droplets (temperature-dependent)
                    // ACN(K) = G*RHOW/(18.*MU(K)) in Fortran
                    amrex::Real umc = m_g * m_rhow / (18.0 * mu);
                    // Apply gamma function approach for consistency with WRF
                    umc *= std::tgamma(4.0 + m_bc + pgam) / (std::pow(lamc, m_bc) * std::tgamma(pgam + 4.0));
                    
                    amrex::Real unc = m_g * m_rhow / (18.0 * mu);
                    unc *= std::tgamma(1.0 + m_bc + pgam) / (std::pow(lamc, m_bc) * std::tgamma(pgam + 1.0));
                    
                    fc(i,j,k) = umc;
                    fnc(i,j,k) = unc;
                }
                
                // Cloud ice fall speed - using Ikawa and Saito (1991) air-density correction with exponent 0.35
                // Different from the exponent 0.54 used for rain, snow, and graupel
                if (qci(i,j,k) >= m_qsmall && lami > 0.0) {
                    // Ikawa and Saito 1991 air-density correction for cloud ice
                    const amrex::Real air_density_factor = std::pow(m_rhosu/rho(i,j,k), 0.35);
                    amrex::Real umi = air_density_factor * m_ai * m_cons28 / std::pow(lami, m_bi);
                    amrex::Real uni = air_density_factor * m_ai * m_cons27 / std::pow(lami, m_bi);
                    
                    // Apply fall speed limits
                    umi = std::min(umi, 1.2 * air_density_factor);
                    uni = std::min(uni, 1.2 * air_density_factor);
                    
                    fi(i,j,k) = umi;
                    fni(i,j,k) = uni;
                }
                
                // Rain fall speed - using standard air density correction with exponent 0.54
                // This follows Heymsfield and Bansemer (2006)
                if (qpr(i,j,k) >= m_qsmall && lamr > 0.0) {
                    const amrex::Real air_density_factor = std::pow(m_rhosu/rho(i,j,k), 0.54);
                    amrex::Real umr = air_density_factor * m_ar * m_cons4 / std::pow(lamr, m_br);
                    amrex::Real unr = air_density_factor * m_ar * m_cons6 / std::pow(lamr, m_br);
                    
                    // Apply fall speed limits
                    umr = std::min(umr, 9.1 * air_density_factor);
                    unr = std::min(unr, 9.1 * air_density_factor);
                    
                    fr(i,j,k) = umr;
                    fnr(i,j,k) = unr;
                }
                
                // Snow fall speed - using standard air density correction with exponent 0.54
                // This follows Heymsfield and Bansemer (2006)
                if (qps(i,j,k) >= m_qsmall && lams > 0.0) {
                    const amrex::Real air_density_factor = std::pow(m_rhosu/rho(i,j,k), 0.54);
                    amrex::Real ums = air_density_factor * m_as * m_cons3 / std::pow(lams, m_bs);
                    amrex::Real uns = air_density_factor * m_as * m_cons5 / std::pow(lams, m_bs);
                    
                    // Apply fall speed limits
                    ums = std::min(ums, 1.2 * air_density_factor);
                    uns = std::min(uns, 1.2 * air_density_factor);
                    
                    fs(i,j,k) = ums;
                    fns(i,j,k) = uns;
                }
                
                // Graupel fall speed - using standard air density correction with exponent 0.54
                // This follows Heymsfield and Bansemer (2006)
                if (qpg(i,j,k) >= m_qsmall && lamg > 0.0) {
                    const amrex::Real air_density_factor = std::pow(m_rhosu/rho(i,j,k), 0.54);
                    amrex::Real umg = air_density_factor * m_ag * m_cons7 / std::pow(lamg, m_bg);
                    amrex::Real ung = air_density_factor * m_ag * m_cons8 / std::pow(lamg, m_bg);
                    
                    // Apply fall speed limits
                    umg = std::min(umg, 20.0 * air_density_factor);
                    ung = std::min(ung, 20.0 * air_density_factor);
                    
                    fg(i,j,k) = umg;
                    fng(i,j,k) = ung;
                }
                
                // Fix velocities below precipitation regions by propagating values down
                // If a velocity is zero at level k but nonzero at k+1, use the k+1 value
                // This matches WRF V3.3+ to prevent spurious accumulation of precipitation
                if (k < khi) {
                    if (fr(i,j,k) < 1.0e-10 && k+1 <= khi) {
                        fr(i,j,k) = fr(i,j,k+1);
                        fnr(i,j,k) = fnr(i,j,k+1);
                    }
                    if (fs(i,j,k) < 1.0e-10 && k+1 <= khi) {
                        fs(i,j,k) = fs(i,j,k+1);
                        fns(i,j,k) = fns(i,j,k+1);
                    }
                    if (fg(i,j,k) < 1.0e-10 && k+1 <= khi) {
                        fg(i,j,k) = fg(i,j,k+1);
                        fng(i,j,k) = fng(i,j,k+1);
                    }
                    if (fi(i,j,k) < 1.0e-10 && k+1 <= khi) {
                        fi(i,j,k) = fi(i,j,k+1);
                        fni(i,j,k) = fni(i,j,k+1);
                    }
                    if (fc(i,j,k) < 1.0e-10 && k+1 <= khi) {
                        fc(i,j,k) = fc(i,j,k+1);
                        fnc(i,j,k) = fnc(i,j,k+1);
                    }
                }
            }
        }
    }
    
    // Find maximum fall speed across all hydrometeors
    // This matches the RGVM calculation in WRF to determine time splitting
    amrex::Real max_fall_speed = 0.0;
    for (int k = klo; k <= khi; ++k) {
        for (int j = box.loVect()[1]; j <= box.hiVect()[1]; ++j) {
            for (int i = box.loVect()[0]; i <= box.hiVect()[0]; ++i) {
                // Find maximum fall speed for any species at this grid point
                amrex::Real local_max = std::max(fr(i,j,k), std::max(fi(i,j,k), 
                                       std::max(fs(i,j,k), std::max(fc(i,j,k), fg(i,j,k)))));
                
                // Also check number-weighted fall speeds
                local_max = std::max(local_max, std::max(fnr(i,j,k), std::max(fni(i,j,k),
                                     std::max(fns(i,j,k), std::max(fnc(i,j,k), fng(i,j,k))))));
                
                // Update global maximum if needed
                max_fall_speed = std::max(max_fall_speed, local_max);
            }
        }
    }
    
    //----------------------------------------------------------------------
    // Calculate number of sub-timesteps needed for stability - matches WRF approach
    //----------------------------------------------------------------------
    int num_split_steps = 1;
    amrex::Real dz_min = m_geom.CellSize(m_axis);
    
    if (max_fall_speed > 0.0) {
        // Calculate Courant number based on maximum fallspeed across all species
        // This matches the NSTEP calculation in WRF
        amrex::Real courant = max_fall_speed * dt / dz_min;
        
        // Calculate number of substeps needed for stability
        // Use ceil to ensure Courant ≤ CFL_MAX
        num_split_steps = static_cast<int>(std::ceil(courant / CFL_MAX));
        
        // Limit to maximum allowed number of substeps
        num_split_steps = std::max(1, std::min(num_split_steps, max_split_steps));
    }

// Calculate duration of each substep
const amrex::Real dt_sub = dt / static_cast<amrex::Real>(num_split_steps);

// Initialize flux arrays
amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
    flux_qr(i,j,k) = 0.0;
    flux_qs(i,j,k) = 0.0;
    flux_qg(i,j,k) = 0.0;
    flux_qi(i,j,k) = 0.0;
    flux_qc(i,j,k) = 0.0;
    flux_nr(i,j,k) = 0.0;
    flux_ns(i,j,k) = 0.0;
    flux_ng(i,j,k) = 0.0;
    flux_ni(i,j,k) = 0.0;
    flux_nc(i,j,k) = 0.0;
});

// Apply subsaturation check before entering the substep loop
amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
    // Calculate saturation ratios
    amrex::Real evs = std::min(0.99*pres(i,j,k),
                               calc_saturation_vapor_pressure(tabs(i,j,k), 0)); // Water saturation
    amrex::Real eis = std::min(0.99*pres(i,j,k), calc_saturation_vapor_pressure(tabs(i,j,k), 1));

    // Make sure ice saturation doesn't exceed water saturation near freezing
    if (eis > evs) eis = evs;

    // Calculate saturation mixing ratios
    amrex::Real qvs = 0.622*evs/(pres(i,j,k)-evs);  // Water saturation mixing ratio
    amrex::Real qvi = 0.622*eis/(pres(i,j,k)-eis);  // Ice saturation mixing ratio

    // Calculate saturation ratios
    amrex::Real qvqvs = qv3d(i,j,k)/qvs;  // Saturation ratio for liquid
    amrex::Real qvqvsi = qv3d(i,j,k)/qvi; // Saturation ratio for ice

    // At subsaturation, remove small amounts of cloud/precip water
    if (qvqvs < 0.9) {
        // For liquid water (rain and cloud water)
        if (qpr(i,j,k) < 1.E-8) {
            qv3d(i,j,k) = qv3d(i,j,k) + qpr(i,j,k);
            tabs3d(i,j,k) = tabs3d(i,j,k) - qpr(i,j,k)*xxlv(i,j,k)/cpm(i,j,k);
            qpr(i,j,k) = 0.0;
            nr(i,j,k) = 0.0;
        }
        if (qcl(i,j,k) < 1.E-8) {
            qv3d(i,j,k) = qv3d(i,j,k) + qcl(i,j,k);
            tabs3d(i,j,k) = tabs3d(i,j,k) - qcl(i,j,k)*xxlv(i,j,k)/cpm(i,j,k);
            qcl(i,j,k) = 0.0;
            nc(i,j,k) = 0.0;
        }
    }

    if (qvqvsi < 0.9) {
        // For ice species
        if (qci(i,j,k) < 1.E-8) {
            qv3d(i,j,k) = qv3d(i,j,k) + qci(i,j,k);
            tabs3d(i,j,k) = tabs3d(i,j,k) - qci(i,j,k)*xxls(i,j,k)/cpm(i,j,k);
            qci(i,j,k) = 0.0;
            ni(i,j,k) = 0.0;
        }
        if (qps(i,j,k) < 1.E-8) {
            qv3d(i,j,k) = qv3d(i,j,k) + qps(i,j,k);
            tabs3d(i,j,k) = tabs3d(i,j,k) - qps(i,j,k)*xxls(i,j,k)/cpm(i,j,k);
            qps(i,j,k) = 0.0;
            ns(i,j,k) = 0.0;
        }
        if (qpg(i,j,k) < 1.E-8) {
            qv3d(i,j,k) = qv3d(i,j,k) + qpg(i,j,k);
            tabs3d(i,j,k) = tabs3d(i,j,k) - qpg(i,j,k)*xxls(i,j,k)/cpm(i,j,k);
            qpg(i,j,k) = 0.0;
            ng(i,j,k) = 0.0;
        }
    }
});

// Begin substep loop for sedimentation
for (int ss = 0; ss < num_split_steps; ++ss) {
    // Calculate mass and number fluxes at cell interfaces for this substep
    for (int k = klo; k < khi; ++k) {
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k_local) {
            const int k = k_local + klo;  // Adjust k to global index

            if (k < khi) {  // Skip the top boundary
                //--------------------------------------------------------------
                // Rain fallout - recalculate fluxes based on current state
                //--------------------------------------------------------------
                if (qpr(i,j,k) > m_qsmall || qpr(i,j,k) >= 0.0) {
                    flux_qr(i,j,k) = fr(i,j,k) * qpr(i,j,k) * rho(i,j,k);
                    flux_nr(i,j,k) = fnr(i,j,k) * nr(i,j,k) * rho(i,j,k);
                } else {
                    flux_qr(i,j,k) = 0.0;
                    flux_nr(i,j,k) = 0.0;
                }

                //--------------------------------------------------------------
                // Snow fallout - recalculate fluxes based on current state
                //--------------------------------------------------------------
                if (qps(i,j,k) > m_qsmall || qps(i,j,k) >= 0.0) {
                    flux_qs(i,j,k) = fs(i,j,k) * qps(i,j,k) * rho(i,j,k);
                    flux_ns(i,j,k) = fns(i,j,k) * ns(i,j,k) * rho(i,j,k);
                } else {
                    flux_qs(i,j,k) = 0.0;
                    flux_ns(i,j,k) = 0.0;
                }

                //--------------------------------------------------------------
                // Graupel fallout - recalculate fluxes based on current state
                //--------------------------------------------------------------
                if (qpg(i,j,k) > m_qsmall || qpg(i,j,k) >= 0.0) {
                    flux_qg(i,j,k) = fg(i,j,k) * qpg(i,j,k) * rho(i,j,k);
                    flux_ng(i,j,k) = fng(i,j,k) * ng(i,j,k) * rho(i,j,k);
                } else {
                    flux_qg(i,j,k) = 0.0;
                    flux_ng(i,j,k) = 0.0;
                }

                //--------------------------------------------------------------
                // Cloud ice fallout - recalculate fluxes based on current state
                //--------------------------------------------------------------
                if (qci(i,j,k) > m_qsmall || qci(i,j,k) >= 0.0) {
                    flux_qi(i,j,k) = fi(i,j,k) * qci(i,j,k) * rho(i,j,k);
                    flux_ni(i,j,k) = fni(i,j,k) * ni(i,j,k) * rho(i,j,k);
                } else {
                    flux_qi(i,j,k) = 0.0;
                    flux_ni(i,j,k) = 0.0;
                }

                //--------------------------------------------------------------
                // Cloud water fallout - recalculate fluxes based on current state
                //--------------------------------------------------------------
                if (qcl(i,j,k) > m_qsmall || qcl(i,j,k) >= 0.0) {
                    flux_qc(i,j,k) = fc(i,j,k) * qcl(i,j,k) * rho(i,j,k);
                    flux_nc(i,j,k) = fnc(i,j,k) * nc(i,j,k) * rho(i,j,k);
                } else {
                    flux_qc(i,j,k) = 0.0;
                    flux_nc(i,j,k) = 0.0;
                }
            }
        });
    }

    // Apply sedimentation tendencies to state variables for this substep
    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
        // Calculate the tendencies due to sedimentation
        amrex::Real tend_qr = 0.0, tend_qs = 0.0, tend_qg = 0.0;
        amrex::Real tend_nr = 0.0, tend_ns = 0.0, tend_ng = 0.0;
        amrex::Real tend_qi = 0.0, tend_ni = 0.0;
        amrex::Real tend_qc = 0.0, tend_nc = 0.0;

        // Flux divergence for interior cells
        if (k < khi) {
            tend_qr -= flux_qr(i,j,k) / (rho(i,j,k) * m_geom.CellSize(m_axis));
            tend_qs -= flux_qs(i,j,k) / (rho(i,j,k) * m_geom.CellSize(m_axis));
            tend_qg -= flux_qg(i,j,k) / (rho(i,j,k) * m_geom.CellSize(m_axis));
            tend_qi -= flux_qi(i,j,k) / (rho(i,j,k) * m_geom.CellSize(m_axis));
            tend_qc -= flux_qc(i,j,k) / (rho(i,j,k) * m_geom.CellSize(m_axis));

            tend_nr -= flux_nr(i,j,k) / (rho(i,j,k) * m_geom.CellSize(m_axis));
            tend_ns -= flux_ns(i,j,k) / (rho(i,j,k) * m_geom.CellSize(m_axis));
            tend_ng -= flux_ng(i,j,k) / (rho(i,j,k) * m_geom.CellSize(m_axis));
            tend_ni -= flux_ni(i,j,k) / (rho(i,j,k) * m_geom.CellSize(m_axis));
            tend_nc -= flux_nc(i,j,k) / (rho(i,j,k) * m_geom.CellSize(m_axis));
        }

        if (k > klo && k < khi) {  // Add flux from below only if not top layer
            tend_qr += flux_qr(i,j,k-1) / (rho(i,j,k) * m_geom.CellSize(m_axis));
            tend_qs += flux_qs(i,j,k-1) / (rho(i,j,k) * m_geom.CellSize(m_axis));
            tend_qg += flux_qg(i,j,k-1) / (rho(i,j,k) * m_geom.CellSize(m_axis));
            tend_qi += flux_qi(i,j,k-1) / (rho(i,j,k) * m_geom.CellSize(m_axis));
            tend_qc += flux_qc(i,j,k-1) / (rho(i,j,k) * m_geom.CellSize(m_axis));

            tend_nr += flux_nr(i,j,k-1) / (rho(i,j,k) * m_geom.CellSize(m_axis));
            tend_ns += flux_ns(i,j,k-1) / (rho(i,j,k) * m_geom.CellSize(m_axis));
            tend_ng += flux_ng(i,j,k-1) / (rho(i,j,k) * m_geom.CellSize(m_axis));
            tend_ni += flux_ni(i,j,k-1) / (rho(i,j,k) * m_geom.CellSize(m_axis));
            tend_nc += flux_nc(i,j,k-1) / (rho(i,j,k) * m_geom.CellSize(m_axis));
        }

        // Apply tendencies for this substep
        qpr(i,j,k) += tend_qr * dt_sub;
        qps(i,j,k) += tend_qs * dt_sub;
        qpg(i,j,k) += tend_qg * dt_sub;
        qci(i,j,k) += tend_qi * dt_sub;
        qcl(i,j,k) += tend_qc * dt_sub;

        // Update number concentrations
        nr(i,j,k) += tend_nr * dt_sub;
        ns(i,j,k) += tend_ns * dt_sub;
        ng(i,j,k) += tend_ng * dt_sub;
        ni(i,j,k) += tend_ni * dt_sub;
        nc(i,j,k) += tend_nc * dt_sub;

        // Floor values to prevent negative concentrations
        qpr(i,j,k) = std::max(qpr(i,j,k), 0.0);
        qps(i,j,k) = std::max(qps(i,j,k), 0.0);
        qpg(i,j,k) = std::max(qpg(i,j,k), 0.0);
        qci(i,j,k) = std::max(qci(i,j,k), 0.0);
        qcl(i,j,k) = std::max(qcl(i,j,k), 0.0);
        nr(i,j,k) = std::max(nr(i,j,k), 0.0);
        ns(i,j,k) = std::max(ns(i,j,k), 0.0);
        ng(i,j,k) = std::max(ng(i,j,k), 0.0);
        ni(i,j,k) = std::max(ni(i,j,k), 0.0);
        nc(i,j,k) = std::max(nc(i,j,k), 0.0);

        // Set very small values to zero
        if (qpr(i,j,k) < m_qsmall) {
            qpr(i,j,k) = 0.0;
            nr(i,j,k) = 0.0;
        }
        if (qps(i,j,k) < m_qsmall) {
            qps(i,j,k) = 0.0;
            ns(i,j,k) = 0.0;
        }
        if (qpg(i,j,k) < m_qsmall) {
            qpg(i,j,k) = 0.0;
            ng(i,j,k) = 0.0;
        }
        if (qci(i,j,k) < m_qsmall) {
            qci(i,j,k) = 0.0;
            ni(i,j,k) = 0.0;
        }
        if (qcl(i,j,k) < m_qsmall) {
            qcl(i,j,k) = 0.0;
            nc(i,j,k) = 0.0;
        }
    });

    // Update accumulated precipitation at surface for this substep
    if (klo == 0) {  // Only if domain includes the surface
        for (int j = box.loVect()[1]; j <= box.hiVect()[1]; ++j) {
            for (int i = box.loVect()[0]; i <= box.hiVect()[0]; ++i) {
                // Accumulate precipitation at the surface (bottom of domain)
                rain_arr(i,j,klo) += flux_qr(i,j,klo) * dt_sub;
                snow_arr(i,j,klo) += flux_qs(i,j,klo) * dt_sub;
                graup_arr(i,j,klo) += flux_qg(i,j,klo) * dt_sub;

                // Also accumulate cloud water and cloud ice sedimentation
                rain_arr(i,j,klo) += flux_qc(i,j,klo) * dt_sub;
                snow_arr(i,j,klo) += flux_qi(i,j,klo) * dt_sub;

                // Accumulate totals for output (includes all precipitation)
                rain_accum += (flux_qr(i,j,klo) + flux_qc(i,j,klo)) * dt_sub;
                snow_accum += (flux_qs(i,j,klo) + flux_qi(i,j,klo)) * dt_sub;
                graup_accum += flux_qg(i,j,klo) * dt_sub;
            }
        }
    }
 }
        //----------------------------------------------------------------------
        // Update total precipitation and snow fields (for diagnostic output)
        //----------------------------------------------------------------------

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Update total precipitation mixing ratio (include all hydrometeors)
            qp(i,j,k) = qpr(i,j,k) + qps(i,j,k) + qpg(i,j,k) + qcl(i,j,k) + qci(i,j,k);
        });
    }

    // Reduce accumulated precipitation across processes if using MPI
    amrex::ParallelDescriptor::ReduceRealSum(rain_accum);
    amrex::ParallelDescriptor::ReduceRealSum(snow_accum);
    amrex::ParallelDescriptor::ReduceRealSum(graup_accum);

    // Calculate domain average precipitation rates if desired
    // (would scale by domain area and output as mm/hr)

    // Note: Accumulated precipitation now includes contributions from:
    // - rain_accum: rain + cloud water sedimentation
    // - snow_accum: snow + cloud ice sedimentation 
    // - graup_accum: graupel sedimentation
    // This matches the WRF implementation which tracks all surface precipitation
}
