#include "ERF_Morrison.H"
#include "ERF_Constants.H"
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_MultiFabUtil.H>
#include <cmath>

/**
 * Autoconversion (A30), Accretion (A28), Evaporation (A24)
 * This function implements the microphysical processes for precipitation formation
 * and evaporation. It corresponds to a subset of the processes in the WRF
 * MORR_TWO_MOMENT_MICRO subroutine.
 */
void
Morrison::Precip(const SolverChoice& /*sc*/)
{
    BL_PROFILE("Morrison::Precip()");

    // Constants for microphysical processes
    constexpr amrex::Real one = 1.0;
    constexpr amrex::Real zero = 0.0;
    constexpr amrex::Real t_freezing = 273.15; // Freezing point of water in K

    // Loop through grids
    for (amrex::MFIter mfi(*mic_fab_vars[MicVar_Morr::qcl]); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        
        // Get array data
        auto const& qv = mic_fab_vars[MicVar_Morr::qv]->array(mfi);
        auto const& qcl = mic_fab_vars[MicVar_Morr::qcl]->array(mfi);
        auto const& qci = mic_fab_vars[MicVar_Morr::qci]->array(mfi);
        auto const& qpr = mic_fab_vars[MicVar_Morr::qpr]->array(mfi);
        auto const& qps = mic_fab_vars[MicVar_Morr::qps]->array(mfi);
        auto const& qpg = mic_fab_vars[MicVar_Morr::qpg]->array(mfi);
        auto const& rho = mic_fab_vars[MicVar_Morr::rho]->array(mfi);
        auto const& tabs = mic_fab_vars[MicVar_Morr::tabs]->array(mfi);
        auto const& pres = mic_fab_vars[MicVar_Morr::pres]->array(mfi);
        
        // Create temporary arrays for process rates and number concentrations
        amrex::FArrayBox temp_fab(box, 20);
        auto nc_arr = temp_fab.array(0);     // Cloud droplet number 
        auto nr_arr = temp_fab.array(1);     // Rain number
        auto ni_arr = temp_fab.array(2);     // Cloud ice number
        auto ns_arr = temp_fab.array(3);     // Snow number
        auto ng_arr = temp_fab.array(4);     // Graupel number
        auto prc_arr = temp_fab.array(5);    // Autoconversion rate: cloud -> rain
        auto pra_arr = temp_fab.array(6);    // Accretion rate: cloud by rain
        auto pracs_arr = temp_fab.array(7);  // Accretion rate: rain by snow
        auto psacw_arr = temp_fab.array(8);  // Accretion rate: cloud by snow
        auto psacr_arr = temp_fab.array(9);  // Accretion rate: rain by snow -> graupel
        auto pre_arr = temp_fab.array(10);   // Evaporation rate: rain
        auto nprc_arr = temp_fab.array(11);  // Number change: cloud autoconversion
        auto nragg_arr = temp_fab.array(12); // Number change: rain self-collection
        auto npra_arr = temp_fab.array(13);  // Number change: accretion cloud by rain
        
        // Initialize temporary arrays to zero
        temp_fab.setVal(0.0);

        // Work with size distribution parameters on the fly
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Process rates (initialize to zero)
            amrex::Real prc = 0.0;    // Autoconversion rate: cloud -> rain
            amrex::Real pra = 0.0;    // Accretion rate: cloud by rain
            amrex::Real pracs = 0.0;  // Accretion rate: rain by snow
            amrex::Real psacws = 0.0; // Accretion rate: cloud by snow
            amrex::Real psacr = 0.0;  // Accretion rate: rain by snow -> graupel
            amrex::Real pre = 0.0;    // Evaporation rate: rain
            amrex::Real nprc = 0.0;   // Number change: cloud autoconversion
            amrex::Real nprc1 = 0.0;  // Number change: cloud to rain number
            amrex::Real nragg = 0.0;  // Number change: rain self-collection
            amrex::Real npra = 0.0;   // Number change: accretion cloud by rain
            
            // Get ambient conditions
            const amrex::Real air_temperature = tabs(i,j,k);
            const amrex::Real air_pressure = pres(i,j,k);
            const amrex::Real air_density = rho(i,j,k);
            
            // Initialize size distribution parameters and fallspeeds
            amrex::Real lamr = 0.0;   // Slope parameter for rain
            amrex::Real lams = 0.0;   // Slope parameter for snow
            amrex::Real n0r = 0.0;    // Intercept parameter for rain
            amrex::Real n0s = 0.0;    // Intercept parameter for snow

            // Local cloud field values
            const amrex::Real qc = qcl(i,j,k);
            const amrex::Real qr = qpr(i,j,k);
            const amrex::Real qs = qps(i,j,k);
            const amrex::Real qi = qci(i,j,k);
            const amrex::Real qg = qpg(i,j,k);
            
            // Local number concentrations (would come from state in full implementation)
            // For this example, we'll use typical values based on mixing ratios
            amrex::Real nc = 100.0e6;   // Cloud droplet number (m^-3)
            amrex::Real nr = 1.0e6;     // Rain drop number (m^-3)
            amrex::Real ns = 1.0e6;     // Snow particle number (m^-3)
            amrex::Real ni = 1.0e5;     // Cloud ice number (m^-3)
            amrex::Real ng = 1.0e5;     // Graupel number (m^-3)
            
            // Store in the arrays for later use
            nc_arr(i,j,k) = nc / air_density;  // Convert to kg^-1
            nr_arr(i,j,k) = nr / air_density;
            ns_arr(i,j,k) = ns / air_density;
            ni_arr(i,j,k) = ni / air_density;
            ng_arr(i,j,k) = ng / air_density;
            
            //------------------------------------------------------------------
            // 1. Autoconversion of cloud water to rain (line ~1800-1830 in Fortran)
            //------------------------------------------------------------------
            if (qc >= 1.0e-6) {
                // Khairoutdinov and Kogan 2000 autoconversion rate (line ~1805-1808)
                prc = 1350.0 * std::pow(qc, 2.47) * 
                      std::pow(nc/1.0e6*air_density, -1.79);
                
                // Calculate number conversion rates (lines ~1810-1812)
                nprc1 = prc / m_cons29;
                nprc = prc / (qc / (nc / air_density));
                
                // Limit the number change to available (line ~1815)
                nprc = std::min(nprc, nc / air_density / dt);
                nprc1 = std::min(nprc1, nprc);
            }
            
            //------------------------------------------------------------------
            // 2. Calculate rain size distribution parameters for other processes
            //------------------------------------------------------------------
            if (qr >= m_qsmall) {
                // Calculate lambda parameter for rain (line ~1692-1693)
                lamr = std::pow(amrex::Math::pi * m_rhow * nr_arr(i,j,k) / qr, 1.0/3.0);
                
                // Apply limits to lambda (lines ~1698-1708)
                if (lamr < m_lamminr) {
                    lamr = m_lamminr;
                    n0r = lamr * lamr * lamr * lamr * qr / (amrex::Math::pi * m_rhow);
                    nr_arr(i,j,k) = n0r / lamr;
                } else if (lamr > m_lammaxr) {
                    lamr = m_lammaxr;
                    n0r = lamr * lamr * lamr * lamr * qr / (amrex::Math::pi * m_rhow);
                    nr_arr(i,j,k) = n0r / lamr;
                } else {
                    n0r = nr_arr(i,j,k) * lamr;
                }
            }
            
            //------------------------------------------------------------------
            // 3. Calculate snow size distribution parameters
            //------------------------------------------------------------------
            if (qs >= m_qsmall) {
                // Calculate lambda parameter for snow (lines ~1782-1783)
                lams = std::pow(m_cons1 * ns_arr(i,j,k) / qs, 1.0/m_ds);
                
                // Apply limits to lambda (lines ~1788-1799)
                if (lams < m_lammins) {
                    lams = m_lammins;
                    n0s = lams * lams * lams * lams * qs / m_cons1;
                    ns_arr(i,j,k) = n0s / lams;
                } else if (lams > m_lammaxs) {
                    lams = m_lammaxs;
                    n0s = lams * lams * lams * lams * qs / m_cons1;
                    ns_arr(i,j,k) = n0s / lams;
                } else {
                    n0s = ns_arr(i,j,k) * lams;
                }
            }
            
            //------------------------------------------------------------------
            // 4. Accretion of cloud droplets by rain (lines ~1980-1990)
            //------------------------------------------------------------------
            if (qr >= 1.0e-8 && qc >= 1.0e-8) {
                // Khairoutdinov and Kogan 2000 accretion formula
                const amrex::Real dum = (qc * qr);
                pra = 67.0 * std::pow(dum, 1.15);
                npra = pra / (qc / (nc / air_density));
            }
            
            //------------------------------------------------------------------
            // 5. Self-collection of rain (lines ~1992-2002)
            //------------------------------------------------------------------
            if (qr >= 1.0e-8) {
                // Check for drop breakup regime (lines ~1994-1998)
                const amrex::Real dum1 = 300.0e-6;
                amrex::Real dum = 1.0;
                if (1.0/lamr < dum1) {
                    dum = 1.0;
                } else if (1.0/lamr >= dum1) {
                    dum = 2.0 - std::exp(2300.0 * (1.0/lamr - dum1));
                }
                
                // Calculate rain self-collection rate (line ~2001)
                nragg = -5.78 * dum * nr_arr(i,j,k) * qr * air_density;
            }
            
            //------------------------------------------------------------------
            // 6. Accretion of cloud water by snow (lines ~1900-1913)
            //------------------------------------------------------------------
            if (qs >= 1.0e-8 && qc >= m_qsmall) {
                // Calculate accretion rate (lines ~1903-1906)
                psacws = m_cons13 * m_as * qc * air_density * n0s / 
                         std::pow(lams, m_bs + 3.0);
            }
            
            //------------------------------------------------------------------
            // 7. Collection of rain by snow (lines ~1915-1947)
            //------------------------------------------------------------------
            if (qr >= 1.0e-8 && qs >= 1.0e-8) {
                // Calculate fall speeds (lines ~1917-1925)
                amrex::Real ums = m_as * m_cons3 / std::pow(lams, m_bs);
                amrex::Real umr = m_ar * m_cons4 / std::pow(lamr, m_br);
                amrex::Real uns = m_as * m_cons5 / std::pow(lams, m_bs);
                amrex::Real unr = m_ar * m_cons6 / std::pow(lamr, m_br);
                
                // Apply density correction to fall speeds (lines ~1928-1931)
                const amrex::Real dum = std::pow(m_rhosu/air_density, 0.54);
                ums = std::min(ums, 1.2 * dum);
                uns = std::min(uns, 1.2 * dum);
                umr = std::min(umr, 9.1 * dum);
                unr = std::min(unr, 9.1 * dum);
                
                // Calculate mass accretion rate (lines ~1935-1942)
                pracs = m_cons41 * (std::sqrt(std::pow(1.2*umr-0.95*ums, 2) + 
                       0.08*ums*umr) * air_density * n0r * n0s / 
                       std::pow(lamr, 3) * (5.0/(std::pow(lamr,3)*lams) + 
                       2.0/(std::pow(lamr,2)*std::pow(lams,2)) + 
                       0.5/(lamr*std::pow(lams,3))));
                
                // Limit to available rain mass (line ~1944)
                pracs = std::min(pracs, qr/dt);
                
                //--------------------------------------------------------------
                // 8. Conversion to graupel via rain-snow collection (lines ~1969-1979)
                //--------------------------------------------------------------
                // Only calculate if rain and snow have significant mass
                if (qs >= 0.1e-3 && qr >= 0.1e-3) {
                    // Calculate portion converted to graupel (lines ~1970-1976)
                    amrex::Real dum = m_cons18 * std::pow(4.0/lams, 3) * std::pow(4.0/lams, 3) / 
                               (m_cons18 * std::pow(4.0/lams, 3) * std::pow(4.0/lams, 3) + 
                                m_cons19 * std::pow(4.0/lamr, 3) * std::pow(4.0/lamr, 3));
                    dum = std::min(dum, 1.0);
                    dum = std::max(dum, 0.0);
                    psacr = (1.0-dum) * pracs;
                }
            }
            
            //------------------------------------------------------------------
            // 9. Calculate evaporation of rain (lines ~2168-2195)
            //------------------------------------------------------------------
            if (qr >= m_qsmall) {
                // Calculate saturation values for this cell
                const amrex::Real evs = std::min(0.99*air_pressure, 
                                               calc_saturation_vapor_pressure(air_temperature, 0));
                const amrex::Real qvs = m_ep_2 * evs / (air_pressure - evs);
                
                // Calculate ventilation coefficient (lines ~2170-2176)
                const amrex::Real mu = 1.496e-6 * std::pow(air_temperature, 1.5) / 
                                     (air_temperature + 120.0);
                const amrex::Real sc = mu / (air_density * 
                                         8.794e-5 * std::pow(air_temperature, 1.81) / air_pressure);
                const amrex::Real dv = 8.794e-5 * std::pow(air_temperature, 1.81) / air_pressure;
                
                // Calculate evaporation rate (lines ~2177-2184)
                const amrex::Real epsr = 2.0 * amrex::Math::pi * n0r * air_density * dv * 
                       (m_f1r/(lamr*lamr) + m_f2r * std::sqrt(m_ar*air_density/mu) * 
                        std::pow(sc, 1.0/3.0) * m_cons9 / std::pow(lamr, m_cons34));
                
                // Apply evaporation only in subsaturated conditions (lines ~2188-2192)
                if (qv(i,j,k) < qvs) {
                    // Latent heating factor 
                    // Note: Full implementation would use pre-calculated AB array
                    const amrex::Real ab = 1.0 + 
                               ((3.1484e6 - 2370.0*air_temperature)*qvs) / 
                               (461.5*air_temperature*air_temperature);
                    
                    pre = epsr * (qv(i,j,k) - qvs) / ab;
                    pre = std::min(pre, 0.0); // Only allow evaporation, not condensation
                }
            }
            
            //------------------------------------------------------------------
            // Store all process rates in the arrays
            //------------------------------------------------------------------
            prc_arr(i,j,k) = prc;
            pra_arr(i,j,k) = pra;
            pracs_arr(i,j,k) = pracs;
            psacw_arr(i,j,k) = psacws;
            psacr_arr(i,j,k) = psacr;
            pre_arr(i,j,k) = pre;
            nprc_arr(i,j,k) = nprc;
            nragg_arr(i,j,k) = nragg;
            npra_arr(i,j,k) = npra;
        });
        
        //----------------------------------------------------------------------
        // Apply the microphysical process tendencies to the state variables
        //----------------------------------------------------------------------
        auto const& qn = mic_fab_vars[MicVar_Morr::qn]->array(mfi);
        auto const& qt = mic_fab_vars[MicVar_Morr::qt]->array(mfi);
        auto const& omega = mic_fab_vars[MicVar_Morr::omega]->array(mfi);
        auto const& theta = mic_fab_vars[MicVar_Morr::theta]->array(mfi);
        
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            //------------------------------------------------------------------
            // 1. Conservation checks to avoid depleting more than available water
            //------------------------------------------------------------------
            
            // Available cloud water
            const amrex::Real qc = qcl(i,j,k);
            
            // Calculate total sink for cloud water
            const amrex::Real sink_qc = (prc_arr(i,j,k) + pra_arr(i,j,k) + psacw_arr(i,j,k)) * dt;
            
            // Apply limiters if sinks exceed available cloud water
            amrex::Real ratio_c = 1.0;
            if (sink_qc > qc && qc >= m_qsmall) {
                ratio_c = qc / sink_qc;
                prc_arr(i,j,k) *= ratio_c;
                pra_arr(i,j,k) *= ratio_c;
                psacw_arr(i,j,k) *= ratio_c;
                nprc_arr(i,j,k) *= ratio_c;
                npra_arr(i,j,k) *= ratio_c;
            }
            
            // Available rain water
            const amrex::Real qr = qpr(i,j,k);
            
            // Calculate total sink for rain water
            const amrex::Real sink_qr = (pracs_arr(i,j,k) - pre_arr(i,j,k)) * dt;
            
            // Apply limiters if sinks exceed available rain water
            amrex::Real ratio_r = 1.0;
            if (sink_qr > qr && qr >= m_qsmall) {
                ratio_r = qr / sink_qr;
                pracs_arr(i,j,k) *= ratio_r;
                pre_arr(i,j,k) *= ratio_r;
            }
            
            //------------------------------------------------------------------
            // 2. Apply tendencies to state variables
            //------------------------------------------------------------------
            
            // Water vapor (evaporation)
            qv(i,j,k) += (-pre_arr(i,j,k)) * dt;
            
            // Cloud water (autoconversion, accretion)
            qcl(i,j,k) += (-prc_arr(i,j,k) - pra_arr(i,j,k) - psacw_arr(i,j,k)) * dt;
            
            // Rain water (autoconversion, accretion, evaporation)
            qpr(i,j,k) += (pre_arr(i,j,k) + pra_arr(i,j,k) + prc_arr(i,j,k) - 
                          pracs_arr(i,j,k)) * dt;
            
            // Snow (accretion of rain by snow)
            qps(i,j,k) += (psacw_arr(i,j,k) + pracs_arr(i,j,k) - psacr_arr(i,j,k)) * dt;
            
            // Graupel (conversion from snow)
            qpg(i,j,k) += psacr_arr(i,j,k) * dt;
            
            // Update total cloud condensate and total cloud water
            qn(i,j,k) = qcl(i,j,k) + qci(i,j,k);
            qt(i,j,k) = qv(i,j,k) + qn(i,j,k);
            
            //------------------------------------------------------------------
            // 3. Apply latent heating from microphysical processes
            //------------------------------------------------------------------
            
            // Calculate latent heats at this temperature
            const amrex::Real xxlv = 3.1484e6 - 2370.0 * tabs(i,j,k); // Latent heat of vaporization
            const amrex::Real xxls = 3.15e6 - 2370.0 * tabs(i,j,k) + 0.3337e6; // Latent heat of sublimation
            const amrex::Real xlf = xxls - xxlv; // Latent heat of fusion
            
            // Heat capacity including water vapor
            const amrex::Real cpm = m_cp * (1.0 + 0.887 * qv(i,j,k));
            
            // Apply temperature tendencies from latent heating
            tabs(i,j,k) += (pre_arr(i,j,k) * xxlv) / cpm * dt;
            
            // Update potential temperature
            // Note: Full implementation would use proper conversion
            const amrex::Real exner = std::pow(pres(i,j,k)/100000.0, m_rdOcp);
            theta(i,j,k) = tabs(i,j,k) / exner;
            
            // Apply limiter to remove tiny amounts of cloud water
            // and make sure values don't go negative
            if (qcl(i,j,k) < m_qsmall) qcl(i,j,k) = 0.0;
            if (qpr(i,j,k) < m_qsmall) qpr(i,j,k) = 0.0;
            if (qps(i,j,k) < m_qsmall) qps(i,j,k) = 0.0;
            if (qpg(i,j,k) < m_qsmall) qpg(i,j,k) = 0.0;
            
            // Update cloud diagnostics (for WRF-Chem compatibility)
            if (pre_arr(i,j,k) < 0.0 && qr >= m_qsmall) {
                // Calculate rain number evaporation rate (line ~3075-3079)
                const amrex::Real dum = pre_arr(i,j,k) * dt / qr;
                const amrex::Real nsubr = std::max(-1.0, dum) * nr_arr(i,j,k) / dt;
                
                // These would update the actual number concentrations in full implementation
            }
        });
    }
}

/**
 * Helper function to calculate saturation vapor pressure for water or ice.
 * This corresponds to the POLYSVP function in the Fortran code (line ~5580).
 * 
 * @param[in] T Temperature in Kelvin
 * @param[in] type 0 for liquid water, 1 for ice
 * @return Saturation vapor pressure in Pascals
 */
amrex::Real
Morrison::calc_saturation_vapor_pressure(const amrex::Real T, const int type) const
{
    amrex::Real polysvp = 0.0;
    amrex::Real dt = T - 273.15;  // Convert to Celsius

    if (type == 1) {  // Ice (lines ~5631-5644)
        if (T >= 195.8) {
            // Flatau et al. formula for ice
            const amrex::Real a0i = 6.11147274;
            const amrex::Real a1i = 0.503160820;
            const amrex::Real a2i = 0.188439774e-1;
            const amrex::Real a3i = 0.420895665e-3;
            const amrex::Real a4i = 0.615021634e-5;
            const amrex::Real a5i = 0.602588177e-7;
            const amrex::Real a6i = 0.385852041e-9;
            const amrex::Real a7i = 0.146898966e-11;
            const amrex::Real a8i = 0.252751365e-14;
            
            polysvp = a0i + dt*(a1i + dt*(a2i + dt*(a3i + dt*(a4i + dt*(a5i + dt*(a6i + dt*(a7i + a8i*dt)))))));
            polysvp *= 100.0;  // Convert from hPa to Pa
        } else {
            // Goff-Gratch formula for ice at cold temperatures
            polysvp = std::pow(10.0, (-9.09718*(273.16/T-1.0) - 3.56654*std::log10(273.16/T) + 
                      0.876793*(1.0-T/273.16) + std::log10(6.1071))) * 100.0;
        }
    } else {  // Water (lines ~5648-5665)
        if (T >= 202.0) {
           // Flatau et al. formula for liquid water
           const amrex::Real a0 = 6.11239921;
           const amrex::Real a1 = 0.443987641;
           const amrex::Real a2 = 0.142986287e-1;
           const amrex::Real a3 = 0.264847430e-3;
           const amrex::Real a4 = 0.302950461e-5;
           const amrex::Real a5 = 0.206739458e-7;
           const amrex::Real a6 = 0.640689451e-10;
           const amrex::Real a7 = -0.952447341e-13;
           const amrex::Real a8 = -0.976195544e-15;
           
           polysvp = a0 + dt*(a1 + dt*(a2 + dt*(a3 + dt*(a4 + dt*(a5 + dt*(a6 + dt*(a7 + a8*dt)))))));
           polysvp *= 100.0;  // Convert from hPa to Pa
       } else {
           // Goff-Gratch formula for water at cold temperatures
           polysvp = std::pow(10.0, (-7.90298*(373.16/T-1.0) + 5.02808*std::log10(373.16/T) -
                     1.3816e-7*(std::pow(10.0, (11.344*(1.0-T/373.16)))-1.0) +
                     8.1328e-3*(std::pow(10.0, (-3.49149*(373.16/T-1.0)))-1.0) +
                     std::log10(1013.246))) * 100.0;
       }
    }

    return polysvp;
}
