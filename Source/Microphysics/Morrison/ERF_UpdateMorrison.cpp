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
    for (MFIter mfi(*m_thermo); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        
        // Get array data
        auto const& thermo = m_thermo->array(mfi);
        auto const& hydro = m_hydro->array(mfi);
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
            Real ze_rain = 1.0e-22;    // Minimum to avoid log issues
            Real ze_snow = 1.0e-22;
            Real ze_graupel = 1.0e-22;
            
            // Calculate rain reflectivity (if present)
            if (hydro(i,j,k,qr_comp) >= m_qsmall) {
                const Real lamr = std::pow(M_PI * m_rhow * hydro(i,j,k,nr_comp) / hydro(i,j,k,qr_comp), 1.0/3.0);
                const Real n0r = hydro(i,j,k,nr_comp) * lamr;
                
                ze_rain = n0r * m_cons4 / std::pow(lamr, 4.0 + m_br);
            }
            
            // Calculate snow reflectivity (if present)
            if (hydro(i,j,k,qs_comp) >= m_qsmall) {
                const Real lams = std::pow(m_cons1 * hydro(i,j,k,ns_comp) / hydro(i,j,k,qs_comp), 1.0/m_ds);
                const Real n0s = hydro(i,j,k,ns_comp) * lams;
                
                // Dielectric factor for dry snow
                ze_snow = (0.176/0.93) * std::pow(6.0/M_PI, 2.0) * 
                          std::pow(m_rhosn/900.0, 2.0) * 
                          n0s * m_cons3 / std::pow(lams, 4.0 + m_bs);
            }
            
            // Calculate graupel reflectivity (if present)
            if (hydro(i,j,k,qg_comp) >= m_qsmall) {
                const Real lamg = std::pow(m_cons2 * hydro(i,j,k,ng_comp) / hydro(i,j,k,qg_comp), 1.0/m_dg);
                const Real n0g = hydro(i,j,k,ng_comp) * lamg;
                
                // Dielectric factor for graupel
                ze_graupel = (0.176/0.93) * std::pow(6.0/M_PI, 2.0) * 
                             std::pow(m_rhog/900.0, 2.0) * 
                             n0g * m_cons7 / std::pow(lamg, 4.0 + m_bg);
            }
            
            // Calculate total reflectivity
            radar(i,j,k,0) = 10.0 * std::log10((ze_rain + ze_snow + ze_graupel) * 1.0e18);
            
            // Ensure reflectivity is above minimum threshold
            radar(i,j,k,0) = amrex::max(radar(i,j,k,0), -35.0);
        });
    }
}
