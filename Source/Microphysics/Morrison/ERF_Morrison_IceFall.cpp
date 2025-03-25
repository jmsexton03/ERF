#include "ERF_Morrison.H"
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_MultiFabUtil.H>
#include <cmath>

/**
 * Sedimentation of cloud ice (A32) with enhanced ice microphysics
 *
 * This function implements the vertical sedimentation of cloud ice particles
 * and ice process enhancements including rime splintering.
 * It corresponds to the ice sedimentation and several ice processes in the
 * original WRF Fortran code.
 */
void
Morrison::IceFall(const SolverChoice& /*sc*/)
{
    BL_PROFILE("Morrison::IceFall()");
#ifdef AMREX_USE_GPU
    amrex::Print()<<"Unclear whether IceFall implemented properly for GPUS"<<std::endl;
#else
#ifdef ERF_USE_MORRCOLD
    // Local vertical indexing
    const int klo = zlo;
    const int khi = zhi;

    // Adaptive time steps for sedimentation to maintain numerical stability
    constexpr int max_split_steps = 10; // Maximum number of substeps allowed

    // Loop through the grids
    for (amrex::MFIter mfi(*mic_fab_vars[MicVar_Morr::qci]); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();

        // Get data arrays
        auto const& thermo_tabs = mic_fab_vars[MicVar_Morr::tabs]->array(mfi);
        auto const& hydro_qv = mic_fab_vars[MicVar_Morr::qv]->array(mfi);
        auto const& thermo_rho = mic_fab_vars[MicVar_Morr::rho]->array(mfi);
        auto const& hydro_qc = mic_fab_vars[MicVar_Morr::qcl]->array(mfi);
        auto const& hydro_qr = mic_fab_vars[MicVar_Morr::qpr]->array(mfi);
        auto const& hydro_qi = mic_fab_vars[MicVar_Morr::qci]->array(mfi);
        auto const& hydro_qs = mic_fab_vars[MicVar_Morr::qps]->array(mfi);
        auto const& hydro_nc = mic_fab_vars[MicVar_Morr::nc]->array(mfi);
        auto const& hydro_ni = mic_fab_vars[MicVar_Morr::ni]->array(mfi);
        auto const& hydro_ns = mic_fab_vars[MicVar_Morr::ns]->array(mfi);

        auto const& rho_arr = mic_fab_vars[MicVar_Morr::rho]->array(mfi);
        auto const& qci_arr = mic_fab_vars[MicVar_Morr::qci]->array(mfi);
        // Create temporary arrays for sedimentation
        amrex::FArrayBox fab_ni(box, 1);      // Ice number concentration
        amrex::FArrayBox fab_fluxqi(box, 1);  // Mass flux
        amrex::FArrayBox fab_fluxni(box, 1);  // Number flux

        auto const& ni_arr = fab_ni.array();
        auto const& fluxqi_arr = fab_fluxqi.array();
        auto const& fluxni_arr = fab_fluxni.array();

        // Apply homogeneous freezing process at very cold temperatures
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Get temperature
            const amrex::Real temp = thermo_tabs(i,j,k);

            // Apply freezing only below the homogeneous freezing threshold (T < -40°C)
            const amrex::Real t_homog_freeze = 233.15;
            if (temp <= t_homog_freeze) {
                // Q Process: MNUCCC
                // N Process: NNUCCC
                // Process: Homogeneous freezing of drops
                // Description: Homogeneous freezing of cloud water to cloud ice
                // Fraction: Cloud
                if (hydro_qc(i,j,k) >= m_qsmall) {
                    // Calculate latent heats
                    const amrex::Real xxlv = 3.1484e6 - 2370.0 * temp;  // Latent heat of vaporization
                    const amrex::Real xxls = 3.15e6 - 2370.0 * temp + 0.3337e6;  // Latent heat of sublimation
                    const amrex::Real xlf = xxls - xxlv;  // Latent heat of fusion

                    // Heat capacity including water vapor effect
                    const amrex::Real cpm = m_cp * (1.0 + 0.887 * hydro_qv(i,j,k));

                    // Transfer mass from cloud water to cloud ice
                    hydro_qi(i,j,k) += hydro_qc(i,j,k);

                    // Update temperature due to latent heat release
                    thermo_tabs(i,j,k) += hydro_qc(i,j,k) * xlf / cpm;

                    // Transfer number concentration
                    hydro_ni(i,j,k) += hydro_nc(i,j,k);

                    // Set cloud water and number to zero
                    hydro_qc(i,j,k) = 0.0;
                    hydro_nc(i,j,k) = 0.0;
                }
            }

            // Initialize cloud ice number concentration based on mixing ratio
            // This is for sedimentation calculations
            if (hydro_qi(i,j,k) > m_qsmall) {
                // Estimate ice number concentration from mixing ratio
                // Assuming a typical ice diameter of ~50 microns
                const amrex::Real typical_mass = 4.0/3.0 * M_PI * m_rhoi * std::pow(25.0e-6, 3);
                ni_arr(i,j,k) = hydro_qi(i,j,k) / typical_mass;
            } else {
                ni_arr(i,j,k) = 0.0;
            }

            // Apply rime splintering (Hallet-Mossop) process
            const amrex::Real t_hm_max = 270.16;
            const amrex::Real t_hm_min = 265.16;

            // Check if temperature is in the H-M process range
            if (temp < t_hm_max && temp > t_hm_min) {
                // Q Process: PSACWS
                // N Process: NPSACWS
                // Process: Collection of drops by snow
                // Description: Collection of cloud water by snow
                // Fraction: Cloud
                // Calculate temperature-dependent multiplication factor
                amrex::Real fmult = 0.0;

                if (temp <= t_hm_max && temp > 268.16) {
                    fmult = (t_hm_max - temp) / 2.0;
                } else if (temp >= t_hm_min && temp <= 268.16) {
                    fmult = (temp - t_hm_min) / 3.0;
                }

                // Process splinters from snow riming if necessary conditions are met
                if (hydro_qs(i,j,k) >= 0.1e-3 &&
                    ((hydro_qc(i,j,k) >= 0.5e-3) || (hydro_qr(i,j,k) >= 0.1e-3))) {

                    // Simple approximation of collection rates
                    amrex::Real psacws = 0.0;  // Cloud water collected by snow
                    if (hydro_qc(i,j,k) >= 0.5e-3) {
                        // Simplified collection rate
                        const amrex::Real lams = std::pow(m_cons1 * hydro_ns(i,j,k) /
                                               hydro_qs(i,j,k), 1.0/m_ds);

                        psacws = m_cons13 * m_as * hydro_qc(i,j,k) * thermo_rho(i,j,k) *
                                 hydro_ns(i,j,k) / std::pow(lams, m_bs + 3.0);

                        // Calculate ice splinters produced
                        if (psacws > 0.0) {
                            // Mass of individual splinters (kg)
                            const amrex::Real mmult = 4.0/3.0 * M_PI * m_rhoi * std::pow(5.0e-6, 3);

                            // Number of new splinters per second
                            const amrex::Real nmults = 35.0e4 * psacws * fmult * 1000.0;

                            // Mass of new splinters
                            const amrex::Real qmults = nmults * mmult;

                            // Constrain so mass transfer doesn't exceed what's rimed
                            const amrex::Real qmults_limited = amrex::min(qmults, psacws);

                            // Add new ice splinters to cloud ice
                            hydro_qi(i,j,k) += qmults_limited * dt;
                            hydro_ni(i,j,k) += nmults * dt;

                            // Remove mass that went to splinters from the rimed amount
                            psacws -= qmults_limited;
                        }

                        // Add remaining rimed mass to snow
                        hydro_qs(i,j,k) += psacws * dt;
                        hydro_qc(i,j,k) -= psacws * dt;
                    }
                }
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
        amrex::Real dz_min = m_geom.CellSizeArray(m_axis);

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
                            tend_qi -= fluxqi_arr(i,j,k) / (rho_arr(i,j,k) * m_geom.CellSizeArray(m_axis));
                            tend_ni -= fluxni_arr(i,j,k) / (rho_arr(i,j,k) * m_geom.CellSizeArray(m_axis));
                        }

                        if (k > klo) {
                            tend_qi += fluxqi_arr(i,j,k-1) / (rho_arr(i,j,k) * m_geom.CellSizeArray(m_axis));
                            tend_ni += fluxni_arr(i,j,k-1) / (rho_arr(i,j,k) * m_geom.CellSizeArray(m_axis));
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
#endif
#endif
}
