#include "ERF_Morrison.H"
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_MultiFabUtil.H>
#include <cmath>

/**
 * Sedimentation of cloud ice (A32)
 * 
 * This function implements the vertical sedimentation of cloud ice particles.
 * It corresponds to the ice sedimentation section in the MORR_TWO_MOMENT_MICRO
 * subroutine in the original WRF Fortran code (around lines 3400-3650).
 */
void 
Morrison::IceFall(const SolverChoice& /*sc*/)
{
    BL_PROFILE("Morrison::IceFall()");
    amrex::Print()<<"Got to IceFall"<<std::endl;
    // Local vertical indexing
    const int klo = zlo;
    const int khi = zhi;

    // Adaptive time steps for sedimentation to maintain numerical stability
    constexpr int max_split_steps = 10; // Maximum number of substeps allowed
    
    // Loop through the grids
    for (amrex::MFIter mfi(*mic_fab_vars[MicVar_Morr::qci]); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        
        // Get data arrays
        auto const& qci_arr = mic_fab_vars[MicVar_Morr::qci]->array(mfi);
        auto const& rho_arr = mic_fab_vars[MicVar_Morr::rho]->array(mfi);
        auto const& tabs_arr = mic_fab_vars[MicVar_Morr::tabs]->array(mfi);

        // Create temporary arrays for sedimentation
        amrex::FArrayBox fab_ni(box, 1); // Ice number concentration
        amrex::FArrayBox fab_fluxqi(box, 1); // Mass flux
        amrex::FArrayBox fab_fluxni(box, 1); // Number flux
        
        auto const& ni_arr = fab_ni.array();
        auto const& fluxqi_arr = fab_fluxqi.array();
        auto const& fluxni_arr = fab_fluxni.array();
        
        // Get number concentration array from storage 
        // Note: In a complete implementation, this would be properly stored and accessed

        // Initialize cloud ice number concentration based on mixing ratio
        // using a typical relationship or known values
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Estimate ice number concentration from mixing ratio
            // This is a placeholder - in a real implementation this would come from storage
            if (qci_arr(i,j,k) > m_qsmall) {
                // Typical ice concentration based on mixing ratio
                // Assuming a typical ice diameter of ~50 microns
                const amrex::Real typical_mass = 4.0/3.0 * M_PI * m_rhoi * std::pow(25.0e-6, 3);
                ni_arr(i,j,k) = qci_arr(i,j,k) / typical_mass;
            } else {
                ni_arr(i,j,k) = 0.0;
            }
        });
        
        // Calculate maximum fall speed for determining time step splitting
        amrex::Real max_fall_speed = 0.0;
        
        // First calculate maximum fall speed anywhere in the domain
        for (int k = klo; k <= khi; ++k) {
            for (int j = box.loVect()[1]; j <= box.hiVect()[1]; ++j) {
                for (int i = box.loVect()[0]; i <= box.hiVect()[0]; ++i) {
                    if (qci_arr(i,j,k) > m_qsmall) {
                        // Calculate size distribution parameters
                        amrex::Real lami = std::pow(m_cons12 * ni_arr(i,j,k) / qci_arr(i,j,k), 1.0/m_di);
                        
                        // Apply limits to lambda
                        lami = amrex::max(lami, m_lammini);
                        lami = amrex::min(lami, m_lammaxi);
                        
                        // Calculate fall speed with density correction
                        // Ikawa and Saito 1991 air-density correction (line 2018)
                        const amrex::Real air_density_factor = std::pow(m_rhosu/rho_arr(i,j,k), 0.35);
                        const amrex::Real fall_speed = air_density_factor * m_ai / std::pow(lami, m_bi);
                        
                        max_fall_speed = amrex::max(max_fall_speed, fall_speed);
                    }
                }
            }
        }
        
        // Calculate number of sub-timesteps needed for stability
        int num_split_steps = 1;
        amrex::Real dz_min = m_geom.CellSize(m_axis);
        
        if (max_fall_speed > 0.0) {
            // Calculate Courant number
            amrex::Real courant = max_fall_speed * dt / dz_min;
            
            // Calculate number of substeps needed for stability
            num_split_steps = static_cast<int>(std::ceil(courant / CFL_MAX));
            num_split_steps = amrex::min(num_split_steps, max_split_steps);
        }
        
        // Duration of each substep
        const amrex::Real dt_sub = dt / static_cast<amrex::Real>(num_split_steps);
        
        // Perform sedimentation over multiple sub-timesteps if necessary
        for (int step = 0; step < num_split_steps; ++step) {
            // Initialize fluxes to zero
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                fluxqi_arr(i,j,k) = 0.0;
                fluxni_arr(i,j,k) = 0.0;
            });
            
            // Calculate fluxes at cell interfaces
            for (int k = klo; k < khi; ++k) {
                for (int j = box.loVect()[1]; j <= box.hiVect()[1]; ++j) {
                    for (int i = box.loVect()[0]; i <= box.hiVect()[0]; ++i) {
                        // Calculate mass and number fluxes at k+1/2 interface
                        if (qci_arr(i,j,k) > m_qsmall) {
                            // Calculate size distribution parameters
                            amrex::Real lami = std::pow(m_cons12 * ni_arr(i,j,k) / qci_arr(i,j,k), 1.0/m_di);
                            
                            // Apply limits to lambda
                            lami = amrex::max(lami, m_lammini);
                            lami = amrex::min(lami, m_lammaxi);
                            
                            // Calculate number-weighted terminal velocity
                            const amrex::Real air_density_factor = std::pow(m_rhosu/rho_arr(i,j,k), 0.35);
                            amrex::Real vt_ice = air_density_factor * m_ai / std::pow(lami, m_bi);
                            
                            // Apply reasonable fall speed limit
                            vt_ice = amrex::min(vt_ice, 1.2 * air_density_factor);
                            
                            // Calculate fluxes (mass and number)
                            fluxqi_arr(i,j,k) = vt_ice * qci_arr(i,j,k) * rho_arr(i,j,k);
                            fluxni_arr(i,j,k) = vt_ice * ni_arr(i,j,k) * rho_arr(i,j,k);
                        }
                    }
                }
            }
            
            // Apply sedimentation tendencies
            for (int k = klo; k <= khi; ++k) {
                for (int j = box.loVect()[1]; j <= box.hiVect()[1]; ++j) {
                    for (int i = box.loVect()[0]; i <= box.hiVect()[0]; ++i) {
                        // Calculate the tendencies due to sedimentation
                        amrex::Real tend_qi = 0.0;
                        amrex::Real tend_ni = 0.0;

                        // Flux divergence for cell k
                        if (k < khi) {
                            tend_qi -= fluxqi_arr(i,j,k) / (rho_arr(i,j,k) * m_geom.CellSize(m_axis));
                            tend_ni -= fluxni_arr(i,j,k) / (rho_arr(i,j,k) * m_geom.CellSize(m_axis));
                        }
                        
                        if (k > klo) {
                            tend_qi += fluxqi_arr(i,j,k-1) / (rho_arr(i,j,k) * m_geom.CellSize(m_axis));
                            tend_ni += fluxni_arr(i,j,k-1) / (rho_arr(i,j,k) * m_geom.CellSize(m_axis));
                        }
                        
                        // Apply tendencies
                        qci_arr(i,j,k) += tend_qi * dt_sub;
                        ni_arr(i,j,k) += tend_ni * dt_sub;
                        
                        // Floor values to prevent negative concentrations
                        qci_arr(i,j,k) = amrex::max(qci_arr(i,j,k), 0.0);
                        ni_arr(i,j,k) = amrex::max(ni_arr(i,j,k), 0.0);
                        
                        // Set very small values to zero
                        if (qci_arr(i,j,k) < m_qsmall) {
                            qci_arr(i,j,k) = 0.0;
                            ni_arr(i,j,k) = 0.0;
                        }
                    }
                }
            }
        }
        
        // Update the total cloud condensate (qn) and total cloud (qt) fields
        auto const& qcl_arr = mic_fab_vars[MicVar_Morr::qcl]->array(mfi);
        auto const& qn_arr = mic_fab_vars[MicVar_Morr::qn]->array(mfi);
        auto const& qv_arr = mic_fab_vars[MicVar_Morr::qv]->array(mfi);
        auto const& qt_arr = mic_fab_vars[MicVar_Morr::qt]->array(mfi);
        
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Update total condensate and total cloud
            qn_arr(i,j,k) = qcl_arr(i,j,k) + qci_arr(i,j,k);
            qt_arr(i,j,k) = qv_arr(i,j,k) + qn_arr(i,j,k);
        });
    }
}
