#include "ERF_Morrison.H"
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_MultiFabUtil.H>
#include <cmath>

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

    // Loop through the grids
    for (amrex::MFIter mfi(*mic_fab_vars[MicVar_Morr::qpr]); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();

        // Get data arrays for hydrometeors
        auto const& qpr = mic_fab_vars[MicVar_Morr::qpr]->array(mfi);
        auto const& qps = mic_fab_vars[MicVar_Morr::qps]->array(mfi);
        auto const& qpg = mic_fab_vars[MicVar_Morr::qpg]->array(mfi);
        auto const& rho = mic_fab_vars[MicVar_Morr::rho]->array(mfi);
        auto const& tabs = mic_fab_vars[MicVar_Morr::tabs]->array(mfi);

        // Get arrays for accumulated precipitation
        auto const& rain_arr = mic_fab_vars[MicVar_Morr::rain_accum]->array(mfi);
        auto const& snow_arr = mic_fab_vars[MicVar_Morr::snow_accum]->array(mfi);
        auto const& graup_arr = mic_fab_vars[MicVar_Morr::graup_accum]->array(mfi);

        // Create temporary arrays for number concentrations and fluxes
        amrex::FArrayBox nr_fab(box, 1); // Rain number concentration
        amrex::FArrayBox ns_fab(box, 1); // Snow number concentration
        amrex::FArrayBox ng_fab(box, 1); // Graupel number concentration

        amrex::FArrayBox flux_qr_fab(box, 1); // Rain mass flux
        amrex::FArrayBox flux_qs_fab(box, 1); // Snow mass flux
        amrex::FArrayBox flux_qg_fab(box, 1); // Graupel mass flux
        amrex::FArrayBox flux_nr_fab(box, 1); // Rain number flux
        amrex::FArrayBox flux_ns_fab(box, 1); // Snow number flux
        amrex::FArrayBox flux_ng_fab(box, 1); // Graupel number flux

        auto const& nr = nr_fab.array();
        auto const& ns = ns_fab.array();
        auto const& ng = ng_fab.array();

        auto const& flux_qr = flux_qr_fab.array();
        auto const& flux_qs = flux_qs_fab.array();
        auto const& flux_qg = flux_qg_fab.array();
        auto const& flux_nr = flux_nr_fab.array();
        auto const& flux_ns = flux_ns_fab.array();
        auto const& flux_ng = flux_ng_fab.array();

        // Initialize number concentrations
        // In a full implementation, these would come from state variables
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Placeholder number concentrations based on mixing ratios
            // and typical mean sizes - these would be properly stored and
            // retrieved in a complete implementation

            // Rain number concentration
            if (qpr(i,j,k) > m_qsmall) {
                // Assuming typical raindrop diameter of ~1 mm
                const amrex::Real typical_mass_r = 4.0/3.0 *m_pi* m_rhow * std::pow(0.5e-3, 3);
                nr(i,j,k) = qpr(i,j,k) / typical_mass_r;
            } else {
                nr(i,j,k) = 0.0;
            }

            // Snow number concentration
            if (qps(i,j,k) > m_qsmall) {
                // Assuming typical snow diameter of ~2 mm
                const amrex::Real typical_mass_s = 4.0/3.0 *m_pi* m_rhosn * std::pow(1.0e-3, 3);
                ns(i,j,k) = qps(i,j,k) / typical_mass_s;
            } else {
                ns(i,j,k) = 0.0;
            }

            // Graupel number concentration
            if (qpg(i,j,k) > m_qsmall) {
                // Assuming typical graupel diameter of ~3 mm
                const amrex::Real typical_mass_g = 4.0/3.0 *m_pi* m_rhog * std::pow(1.5e-3, 3);
                ng(i,j,k) = qpg(i,j,k) / typical_mass_g;
            } else {
                ng(i,j,k) = 0.0;
            }
        });

        //----------------------------------------------------------------------
        // Calculate maximum fall speeds for all precipitation species
        //----------------------------------------------------------------------
        amrex::Real max_fall_speed = 0.0;

        for (int k = klo; k <= khi; ++k) {
            for (int j = box.loVect()[1]; j <= box.hiVect()[1]; ++j) {
                for (int i = box.loVect()[0]; i <= box.hiVect()[0]; ++i) {
                    // Rain fall speed
                    if (qpr(i,j,k) > m_qsmall) {
                        // Calculate lambda parameter for rain (line ~3692)
                        const amrex::Real lamr = std::pow(m_pi * m_rhow * nr(i,j,k) / qpr(i,j,k), 1.0/3.0);

                        // Limit lambda to physical range
                        const amrex::Real lamr_limited = std::max(std::min(lamr, m_lammaxr), m_lamminr);

                        // Calculate density-corrected fall speed (line ~3708-3709)
                        const amrex::Real air_density_factor = std::pow(m_rhosu/rho(i,j,k), 0.54);
                        const amrex::Real fall_speed_r = air_density_factor * m_ar * m_cons4 /
                                                      std::pow(lamr_limited, m_br);

                        max_fall_speed = std::max(max_fall_speed, fall_speed_r);
                    }

                    // Snow fall speed
                    if (qps(i,j,k) > m_qsmall) {
                        // Calculate lambda parameter for snow (line ~3782)
                        const amrex::Real lams = std::pow(m_cons1 * ns(i,j,k) / qps(i,j,k), 1.0/m_ds);

                        // Limit lambda to physical range
                        const amrex::Real lams_limited = std::max(std::min(lams, m_lammaxs), m_lammins);

                        // Calculate density-corrected fall speed
                        const amrex::Real air_density_factor = std::pow(m_rhosu/rho(i,j,k), 0.54);
                        const amrex::Real fall_speed_s = air_density_factor * m_as * m_cons3 /
                                                      std::pow(lams_limited, m_bs);

                        max_fall_speed = std::max(max_fall_speed, fall_speed_s);
                    }

                    // Graupel fall speed
                    if (qpg(i,j,k) > m_qsmall) {
                        // Calculate lambda parameter for graupel
                        const amrex::Real lamg = std::pow(m_cons2 * ng(i,j,k) / qpg(i,j,k), 1.0/m_dg);

                        // Limit lambda to physical range
                        const amrex::Real lamg_limited = std::max(std::min(lamg, m_lammaxg), m_lamming);

                        // Calculate density-corrected fall speed
                        const amrex::Real air_density_factor = std::pow(m_rhosu/rho(i,j,k), 0.54);
                        const amrex::Real fall_speed_g = air_density_factor * m_ag * m_cons7 /
                                                      std::pow(lamg_limited, m_bg);

                        max_fall_speed = std::max(max_fall_speed, fall_speed_g);
                    }
                }
            }
        }

        //----------------------------------------------------------------------
        // Calculate number of sub-timesteps needed for stability (lines ~3797-3803)
        //----------------------------------------------------------------------
        int num_split_steps = 1;
        amrex::Real dz_min = m_geom.CellSize(m_axis);

        if (max_fall_speed > 0.0) {
            // Calculate Courant number
            amrex::Real courant = max_fall_speed * dt / dz_min;

            // Calculate number of substeps needed for stability
            num_split_steps = static_cast<int>(std::ceil(courant / CFL_MAX));
            num_split_steps = std::min(num_split_steps, max_split_steps);
        }

        // Duration of each substep
        const amrex::Real dt_sub = dt / static_cast<amrex::Real>(num_split_steps);

        //----------------------------------------------------------------------
        // Perform sedimentation over multiple sub-timesteps if necessary
        //----------------------------------------------------------------------
        for (int step = 0; step < num_split_steps; ++step) {
            // Initialize fluxes to zero
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                flux_qr(i,j,k) = 0.0;
                flux_qs(i,j,k) = 0.0;
                flux_qg(i,j,k) = 0.0;
                flux_nr(i,j,k) = 0.0;
                flux_ns(i,j,k) = 0.0;
                flux_ng(i,j,k) = 0.0;
            });

            //------------------------------------------------------------------
            // Calculate mass and number fluxes at cell interfaces (lines ~3826-3900)
            //------------------------------------------------------------------
            for (int k = klo; k < khi; ++k) {
                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k_local) {
                    const int k = k_local + klo;  // Adjust k to global index

                    if (k < khi) {  // Skip the top boundary
                        //--------------------------------------------------------------
                        // Rain fallout
                        //--------------------------------------------------------------
                        if (qpr(i,j,k) > m_qsmall) {
                            // Calculate size distribution parameters
                            amrex::Real lamr = std::pow(m_pi * m_rhow * nr(i,j,k) / qpr(i,j,k), 1.0/3.0);

                            // Apply limits to lambda
                            lamr = std::max(lamr, m_lamminr);
                            lamr = std::min(lamr, m_lammaxr);

                            // Calculate mass-weighted and number-weighted fall speeds
                            const amrex::Real air_density_factor = std::pow(m_rhosu/rho(i,j,k), 0.54);
                            amrex::Real umr = air_density_factor * m_ar * m_cons4 / std::pow(lamr, m_br);
                            amrex::Real unr = air_density_factor * m_ar * m_cons6 / std::pow(lamr, m_br);

                            // Apply fall speed limits
                            umr = std::min(umr, 9.1 * air_density_factor);
                            unr = std::min(unr, 9.1 * air_density_factor);

                            // Calculate fluxes (mass and number)
                            flux_qr(i,j,k) = umr * qpr(i,j,k) * rho(i,j,k);
                            flux_nr(i,j,k) = unr * nr(i,j,k) * rho(i,j,k);
                        }

                        //--------------------------------------------------------------
                        // Snow fallout
                        //--------------------------------------------------------------
                        if (qps(i,j,k) > m_qsmall) {
                            // Calculate size distribution parameters
                            amrex::Real lams = std::pow(m_cons1 * ns(i,j,k) / qps(i,j,k), 1.0/m_ds);

                            // Apply limits to lambda
                            lams = std::max(lams, m_lammins);
                            lams = std::min(lams, m_lammaxs);

                            // Calculate mass-weighted and number-weighted fall speeds
                            const amrex::Real air_density_factor = std::pow(m_rhosu/rho(i,j,k), 0.54);
                            amrex::Real ums = air_density_factor * m_as * m_cons3 / std::pow(lams, m_bs);
                            amrex::Real uns = air_density_factor * m_as * m_cons5 / std::pow(lams, m_bs);

                            // Apply fall speed limits
                            ums = std::min(ums, 1.2 * air_density_factor);
                            uns = std::min(uns, 1.2 * air_density_factor);

                            // Calculate fluxes (mass and number)
                            flux_qs(i,j,k) = ums * qps(i,j,k) * rho(i,j,k);
                            flux_ns(i,j,k) = uns * ns(i,j,k) * rho(i,j,k);
                        }

                        //--------------------------------------------------------------
                        // Graupel fallout
                        //--------------------------------------------------------------
                        if (qpg(i,j,k) > m_qsmall) {
                            // Calculate size distribution parameters
                            amrex::Real lamg = std::pow(m_cons2 * ng(i,j,k) / qpg(i,j,k), 1.0/m_dg);

                            // Apply limits to lambda
                            lamg = std::max(lamg, m_lamming);
                            lamg = std::min(lamg, m_lammaxg);

                            // Calculate mass-weighted and number-weighted fall speeds
                            const amrex::Real air_density_factor = std::pow(m_rhosu/rho(i,j,k), 0.54);
                            amrex::Real umg = air_density_factor * m_ag * m_cons7 / std::pow(lamg, m_bg);
                            amrex::Real ung = air_density_factor * m_ag * m_cons8 / std::pow(lamg, m_bg);

                            // Apply fall speed limits
                            umg = std::min(umg, 20.0 * air_density_factor);
                            ung = std::min(ung, 20.0 * air_density_factor);

                            // Calculate fluxes (mass and number)
                            flux_qg(i,j,k) = umg * qpg(i,j,k) * rho(i,j,k);
                            flux_ng(i,j,k) = ung * ng(i,j,k) * rho(i,j,k);
                        }
                    }
                });
            }

            //------------------------------------------------------------------
            // Apply sedimentation tendencies to state variables (lines ~3902-4000)
            //------------------------------------------------------------------
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // Calculate the tendencies due to sedimentation
                amrex::Real tend_qr = 0.0, tend_qs = 0.0, tend_qg = 0.0;
                amrex::Real tend_nr = 0.0, tend_ns = 0.0, tend_ng = 0.0;

                // Flux divergence for interior cells
                if (k < khi) {
                    tend_qr -= flux_qr(i,j,k) / (rho(i,j,k) * m_geom.CellSize(m_axis));
                    tend_qs -= flux_qs(i,j,k) / (rho(i,j,k) * m_geom.CellSize(m_axis));
                    tend_qg -= flux_qg(i,j,k) / (rho(i,j,k) * m_geom.CellSize(m_axis));

                    tend_nr -= flux_nr(i,j,k) / (rho(i,j,k) * m_geom.CellSize(m_axis));
                    tend_ns -= flux_ns(i,j,k) / (rho(i,j,k) * m_geom.CellSize(m_axis));
                    tend_ng -= flux_ng(i,j,k) / (rho(i,j,k) * m_geom.CellSize(m_axis));
                }

                if (k > klo) {
                    tend_qr += flux_qr(i,j,k-1) / (rho(i,j,k) * m_geom.CellSize(m_axis));
                    tend_qs += flux_qs(i,j,k-1) / (rho(i,j,k) * m_geom.CellSize(m_axis));
                    tend_qg += flux_qg(i,j,k-1) / (rho(i,j,k) * m_geom.CellSize(m_axis));

                    tend_nr += flux_nr(i,j,k-1) / (rho(i,j,k) * m_geom.CellSize(m_axis));
                    tend_ns += flux_ns(i,j,k-1) / (rho(i,j,k) * m_geom.CellSize(m_axis));
                    tend_ng += flux_ng(i,j,k-1) / (rho(i,j,k) * m_geom.CellSize(m_axis));
                }

                // Apply tendencies
                qpr(i,j,k) += tend_qr * dt_sub;
                qps(i,j,k) += tend_qs * dt_sub;
                qpg(i,j,k) += tend_qg * dt_sub;

                // Update number concentrations
                nr(i,j,k) += tend_nr * dt_sub;
                ns(i,j,k) += tend_ns * dt_sub;
                ng(i,j,k) += tend_ng * dt_sub;

                // Floor values to prevent negative concentrations
                qpr(i,j,k) = std::max(qpr(i,j,k), 0.0);
                qps(i,j,k) = std::max(qps(i,j,k), 0.0);
                qpg(i,j,k) = std::max(qpg(i,j,k), 0.0);
                nr(i,j,k) = std::max(nr(i,j,k), 0.0);
                ns(i,j,k) = std::max(ns(i,j,k), 0.0);
                ng(i,j,k) = std::max(ng(i,j,k), 0.0);

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
            });

            //------------------------------------------------------------------
            // Update accumulated precipitation at surface (lines ~4002-4006)
            //------------------------------------------------------------------
            if (klo == 0) {  // Only if domain includes the surface
                for (int j = box.loVect()[1]; j <= box.hiVect()[1]; ++j) {
                    for (int i = box.loVect()[0]; i <= box.hiVect()[0]; ++i) {
                        // Accumulate precipitation at the surface (bottom of domain)
                        rain_arr(i,j,klo) += flux_qr(i,j,klo) * dt_sub;
                        snow_arr(i,j,klo) += flux_qs(i,j,klo) * dt_sub;
                        graup_arr(i,j,klo) += flux_qg(i,j,klo) * dt_sub;

                        // Accumulate totals for output
                        rain_accum += flux_qr(i,j,klo) * dt_sub;
                        snow_accum += flux_qs(i,j,klo) * dt_sub;
                        graup_accum += flux_qg(i,j,klo) * dt_sub;
                    }
                }
            }
        }

        //----------------------------------------------------------------------
        // Update total precipitation and snow fields (for diagnostic output)
        //----------------------------------------------------------------------
        auto const& qp = mic_fab_vars[MicVar_Morr::qp]->array(mfi);

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Update total precipitation mixing ratio
            qp(i,j,k) = qpr(i,j,k) + qps(i,j,k) + qpg(i,j,k);
        });
    }

    // Reduce accumulated precipitation across processes if using MPI
    amrex::ParallelDescriptor::ReduceRealSum(rain_accum);
    amrex::ParallelDescriptor::ReduceRealSum(snow_accum);
    amrex::ParallelDescriptor::ReduceRealSum(graup_accum);

    // Calculate domain average precipitation rates if desired
    // (would scale by domain area and output as mm/hr)
}
