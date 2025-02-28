#include "ERF_Morrison.H"
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_MultiFabUtil.H>
#include <cmath>

/**
 * Split cloud components according to saturation pressures; 
 * source theta from latent heat.
 * 
 * This function handles cloud phase thermodynamics including saturation 
 * adjustment, phase partitioning between vapor/liquid/ice, and corresponding
 * latent heating/cooling.
 */
void 
Morrison::Cloud(const SolverChoice& /*sc*/)
{
    BL_PROFILE("Morrison::Cloud()");

    // Physical constants
    constexpr amrex::Real t_freeze = 273.15; // Freezing point in K
    constexpr amrex::Real t_homog_freeze = 233.15; // Homogeneous freezing threshold in K
    constexpr amrex::Real tbgmin = 253.15; // Min temp for mixed-phase (K)
    constexpr amrex::Real tbgmax = 273.15; // Max temp for mixed-phase (K)
    
    // Phase partitioning parameters (lines ~1760-1765)
    constexpr amrex::Real an = 1.0 / (tbgmax - tbgmin); // Slope for mixed-phase function
    constexpr amrex::Real bn = tbgmin / (tbgmax - tbgmin); // Intercept for mixed-phase function
    
    // Temperature threshold for homogeneous freezing
    constexpr amrex::Real tol = 1.0e-4; // Convergence tolerance for saturation adjustment

    // Loop through grids
    for (amrex::MFIter mfi(*mic_fab_vars[MicVar_Morr::qv]); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        
        // Get array data
        auto const& qv = mic_fab_vars[MicVar_Morr::qv]->array(mfi);
        auto const& qcl = mic_fab_vars[MicVar_Morr::qcl]->array(mfi);
        auto const& qci = mic_fab_vars[MicVar_Morr::qci]->array(mfi);
        auto const& qn = mic_fab_vars[MicVar_Morr::qn]->array(mfi);
        auto const& qt = mic_fab_vars[MicVar_Morr::qt]->array(mfi);
        auto const& rho = mic_fab_vars[MicVar_Morr::rho]->array(mfi);
        auto const& tabs = mic_fab_vars[MicVar_Morr::tabs]->array(mfi);
        auto const& pres = mic_fab_vars[MicVar_Morr::pres]->array(mfi);
        auto const& theta = mic_fab_vars[MicVar_Morr::theta]->array(mfi);
        
        //----------------------------------------------------------------------
        // Handle homogeneous freezing of cloud water (lines ~2352-2362)
        //----------------------------------------------------------------------
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Homogeneous freezing of cloud water (all liquid freezes below threshold)
            if (tabs(i,j,k) <= t_homog_freeze && qcl(i,j,k) >= m_qsmall) {
                // Convert all cloud water to cloud ice
                qci(i,j,k) += qcl(i,j,k);
                
                // Apply latent heating
                const amrex::Real xxlv = 3.1484e6 - 2370.0 * tabs(i,j,k); // Latent heat of vaporization
                const amrex::Real xxls = 3.15e6 - 2370.0 * tabs(i,j,k) + 0.3337e6; // Latent heat of sublimation
                const amrex::Real xlf = xxls - xxlv; // Latent heat of fusion
                
                // Heat capacity including water vapor
                const amrex::Real cpm = m_cp * (1.0 + 0.887 * qv(i,j,k));
                
                // Update temperature due to freezing
                tabs(i,j,k) += qcl(i,j,k) * xlf / cpm;
                
                // Clear cloud water
                qcl(i,j,k) = 0.0;
            }
        });
        
        //----------------------------------------------------------------------
        // Phase partitioning and saturation adjustment (lines ~2365-2389)
        //----------------------------------------------------------------------
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Check if there is any condensate to adjust
            if (qcl(i,j,k) < m_qsmall && qci(i,j,k) < m_qsmall && qv(i,j,k) < m_qsmall) {
                return; // Skip if no water to adjust
            }
            
            // Starting temperature
            amrex::Real T = tabs(i,j,k);
            
            // Calculate the saturation values at current temperature
            amrex::Real evs = std::min(0.99*pres(i,j,k), 
                                     calc_saturation_vapor_pressure(T, 0)); // Water saturation
            amrex::Real eis = std::min(0.99*pres(i,j,k), 
                                     calc_saturation_vapor_pressure(T, 1)); // Ice saturation
            
            // Ensure ice saturation doesn't exceed water saturation near freezing
            eis = std::min(eis, evs);
            
            // Calculate saturation mixing ratios
            amrex::Real qvs = m_ep_2 * evs / (pres(i,j,k) - evs);
            amrex::Real qvi = m_ep_2 * eis / (pres(i,j,k) - eis);
            
            // Calculate saturation ratios
            amrex::Real qvqvs = qv(i,j,k) / qvs;
            amrex::Real qvqvsi = qv(i,j,k) / qvi;
            
            //--------------------------------------------------------------
            // Newton iteration for saturation adjustment (similar to lines ~2389-2456)
            //--------------------------------------------------------------
            
            // Initialize values for Newton iteration
            int niter = 0;
            amrex::Real dtabs = 1.0;
            
            // Current water vapor and cloud water/ice mixing ratios
            amrex::Real qv_cur = qv(i,j,k);
            amrex::Real qc_cur = qcl(i,j,k);
            amrex::Real qi_cur = qci(i,j,k);
            
            // Iterate until convergence or max iterations
            while (std::abs(dtabs) > tol && niter < 20) {
                // Phase partitioning based on temperature
                amrex::Real omn, domn;
                
                // Cloud ice not permitted above max mixed-phase temperature
                if (T >= tbgmax) {
                    omn = 1.0;
                    domn = 0.0;
                }
                // Cloud water not permitted below min mixed-phase temperature
                else if (T <= tbgmin) {
                    omn = 0.0;
                    domn = 0.0;
                }
                // Mixed cloud phase between tbgmin and tbgmax
                else {
                    omn = an * T - bn;
                    domn = an;
                }
                
                // Recalculate saturation values at current temperature
                evs = std::min(0.99*pres(i,j,k), calc_saturation_vapor_pressure(T, 0));
                eis = std::min(0.99*pres(i,j,k), calc_saturation_vapor_pressure(T, 1));
                if (eis > evs) eis = evs;
                
                // Saturation mixing ratios
                qvs = m_ep_2 * evs / (pres(i,j,k) - evs);
                qvi = m_ep_2 * eis / (pres(i,j,k) - eis);
                
                // Calculate derivatives of saturation mixing ratios with respect to temperature
                const amrex::Real dum = m_Rv * T * T;
                const amrex::Real dqsdt = (3.1484e6 - 2370.0 * T) * qvs / dum;
                const amrex::Real dqsidt = (3.15e6 - 2370.0 * T + 0.3337e6) * qvi / dum;
                
                // Combined saturation values using phase partition function
                const amrex::Real qsat = omn * qvs + (1.0 - omn) * qvi;
                const amrex::Real dqsat = omn * dqsdt + (1.0 - omn) * dqsidt + 
                                       domn * qvs - domn * qvi;
                
                // Latent heat terms
                const amrex::Real lsterms = omn * m_fac_cond + (1.0 - omn) * m_fac_sub;
                const amrex::Real dlsterms = domn * m_fac_cond - domn * m_fac_sub;
                
                // Newton iteration function and derivative
                const amrex::Real f = -T + tabs(i,j,k) + lsterms * (qv_cur - qsat);
                const amrex::Real df = -1.0 + dlsterms * (qv_cur - qsat) - lsterms * dqsat;
                
                // Update temperature
                dtabs = -f / df;
                T += dtabs;
                
                niter++;
            }
            
            // Update saturation values at final temperature
            evs = std::min(0.99*pres(i,j,k), calc_saturation_vapor_pressure(T, 0));
            eis = std::min(0.99*pres(i,j,k), calc_saturation_vapor_pressure(T, 1));
            if (eis > evs) eis = evs;
            
            qvs = m_ep_2 * evs / (pres(i,j,k) - evs);
            qvi = m_ep_2 * eis / (pres(i,j,k) - eis);
            
            // Phase partitioning at final temperature
            amrex::Real omn;
            
            // Cloud ice not permitted above freezing
            if (T >= tbgmax) {
                omn = 1.0;
            }
            // Cloud water not permitted below min mixed-phase temperature
            else if (T <= tbgmin) {
                omn = 0.0;
            }
            // Mixed cloud phase between tbgmin and tbgmax
            else {
                omn = an * T - bn;
            }
            
            // Calculate combined saturation mixing ratio
            const amrex::Real qsat = omn * qvs + (1.0 - omn) * qvi;
            
            // Calculate excess or deficit of vapor
            const amrex::Real delta_qv = qv(i,j,k) - qsat;
            
            // Partition excess/deficit between liquid and ice based on temperature
            const amrex::Real delta_qc = std::max(-qcl(i,j,k), delta_qv * omn);
            const amrex::Real delta_qi = std::max(-qci(i,j,k), delta_qv * (1.0 - omn));
            
            // Update mixing ratios
            qv(i,j,k) = qsat;
            qcl(i,j,k) += delta_qc;
            qci(i,j,k) += delta_qi;
            
            // Update total condensate and total water
            qn(i,j,k) = qcl(i,j,k) + qci(i,j,k);
            qt(i,j,k) = qv(i,j,k) + qn(i,j,k);
            
            // Update temperature
            tabs(i,j,k) = T;
            
            // Update potential temperature
            const amrex::Real exner = std::pow(pres(i,j,k)/100000.0, m_rdOcp);
            theta(i,j,k) = T / exner;
            
            // Apply minimum thresholds
            if (qcl(i,j,k) < m_qsmall) qcl(i,j,k) = 0.0;
            if (qci(i,j,k) < m_qsmall) qci(i,j,k) = 0.0;
        });
        
        //----------------------------------------------------------------------
        // Transfer ice to snow if mean size exceeds threshold (lines ~2559-2570)
        //----------------------------------------------------------------------
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Only check if there is appreciable cloud ice
            if (qci(i,j,k) >= m_qsmall && tabs(i,j,k) < t_freeze) {
                // Get number concentration (placeholder - in full implementation this would be stored)
                amrex::Real ni = 1.0e5; // Placeholder number concentration
                
                // Calculate size distribution parameter
                amrex::Real lami = std::pow(m_cons12 * ni / qci(i,j,k), 1.0/m_di);
                
                // Check if mean size is valid
                if (lami >= 1.0e-10) {
                    // Check if mean size exceeds threshold (2*DCS)
                    if (1.0/lami >= 2.0*m_dcs) {
                        // Transfer all cloud ice to snow
                        // In a full implementation, would also transfer number concentration
                        auto& qps = mic_fab_vars[MicVar_Morr::qps]->array(mfi);
                        qps(i,j,k) += qci(i,j,k);
                        qci(i,j,k) = 0.0;
                    }
                }
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
