#include "ERF_Morrison.H"
#include "ERF_Constants.H"
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

// Existing functions...

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
    
    // 1. AEROSOL ACTIVATION
    // Activate cloud droplets from CCN (if activated)
    if (m_activate_type > 0) {
        ActivateCloudDroplets(sc);
    }
    
    // 2. HOMOGENEOUS FREEZING
    // Apply homogeneous freezing (below -40°C)
    ApplyHomogeneousFreezing(dt);
    
    // 3. ICE NUCLEATION
    // Primary ice nucleation
    if (m_inuc_type >= 0) {  // Skip if nucleation disabled
        ApplyIceNucleation(dt);
    }
    
    // 4. HETEROGENEOUS FREEZING
    // Heterogeneous freezing of droplets and rain
    if (m_iliq == 0) {  // Skip if liquid-only mode is active
        ApplyHeterogeneousFreezing(dt);
    }
    
    // 5. CLOUD PROCESSES
    // Cloud phase thermodynamics and saturation adjustment
    Cloud(sc);
    
    // 6. ICE SEDIMENTATION
    // Cloud ice sedimentation
    if (m_iliq == 0) {  // Skip if liquid-only mode is active
        IceFall(sc);
    }
    
    // 7. RAIN-ICE COLLISIONS
    // Rain-ice collisions
    if (m_iliq == 0) {  // Skip if liquid-only mode is active
        ApplyRainIceCollisions(dt);
    }
    
    // 8. RIME SPLINTERING
    // Rime splintering (Hallet-Mossop process)
    if (m_iliq == 0) {  // Skip if liquid-only mode is active
        ApplyRimeSplintering(dt);
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
