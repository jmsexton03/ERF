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
Morrison::Cloud(const SolverChoice& sc)
{
    BL_PROFILE("Morrison::Cloud()");

    // Physical constants
    constexpr amrex::Real t_freeze = 273.15; // Freezing point in K
    constexpr amrex::Real t_homog_freeze = 233.15; // Homogeneous freezing threshold in K
    constexpr amrex::Real tbgmin = 253.15; // Min temp for mixed-phase (K)
    constexpr amrex::Real tbgmax = 273.15; // Max temp for mixed-phase (K)
    
    // Phase partitioning parameters
    constexpr amrex::Real an = 1.0 / (tbgmax - tbgmin); // Slope for mixed-phase function
    constexpr amrex::Real bn = tbgmin / (tbgmax - tbgmin); // Intercept for mixed-phase function
    
    // Temperature threshold for homogeneous freezing
    constexpr amrex::Real tol = 1.0e-4; // Convergence tolerance for saturation adjustment

    // Loop through grids
    for (amrex::MFIter mfi(*mic_fab_vars[MicVar_Morr::tabs]); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        
        // Get array data
        auto const& thermo_tabs = mic_fab_vars[MicVar_Morr::tabs]->array(mfi);
        auto const& thermo_pres = mic_fab_vars[MicVar_Morr::pres]->array(mfi);
        auto const& thermo_rho = mic_fab_vars[MicVar_Morr::rho]->array(mfi);
        auto const& hydro_qv = mic_fab_vars[MicVar_Morr::qv]->array(mfi);
        auto const& hydro_qc = mic_fab_vars[MicVar_Morr::qcl]->array(mfi);
        auto const& hydro_qi = mic_fab_vars[MicVar_Morr::qci]->array(mfi);
        auto const& hydro_qr = mic_fab_vars[MicVar_Morr::qpr]->array(mfi);
        auto const& hydro_qs = mic_fab_vars[MicVar_Morr::qps]->array(mfi);
        auto const& hydro_qg = mic_fab_vars[MicVar_Morr::qpg]->array(mfi);
        auto const& hydro_nc = mic_fab_vars[MicVar_Morr::nc]->array(mfi);
        auto const& hydro_nr = mic_fab_vars[MicVar_Morr::nr]->array(mfi);
        auto const& hydro_ni = mic_fab_vars[MicVar_Morr::ni]->array(mfi);
        auto const& hydro_ns = mic_fab_vars[MicVar_Morr::ns]->array(mfi);
        auto const& hydro_ng = mic_fab_vars[MicVar_Morr::ng]->array(mfi);
        auto const& w = mic_fab_vars[MicVar_Morr::omega]->array(mfi);
        auto const& qn_arr = mic_fab_vars[MicVar_Morr::qn]->array(mfi);
        auto const& qt_arr = mic_fab_vars[MicVar_Morr::qt]->array(mfi);

        // This block implements cloud droplet activation (CCN activation)
        // This is a microphysical process that doesn't directly address 
        // cloud phase thermodynamics, but rather creates new cloud particles
        // Process: MNUCCC (Homogeneous freezing of drops, Cloud)
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Get local variables
            const amrex::Real temp = thermo_tabs(i,j,k);
            const amrex::Real pres = thermo_pres(i,j,k);
            const amrex::Real qv = hydro_qv(i,j,k);
            const amrex::Real rho = thermo_rho(i,j,k);
            const amrex::Real qc = hydro_qc(i,j,k);
            const amrex::Real nc = hydro_nc(i,j,k);
            const amrex::Real w_local = w(i,j,k);

            // Only activate if temperature is above freezing
            if (temp <= 273.15) return;

            // Only activate if there's little existing cloud water
            // This follows the approach in the original Morrison scheme
            if (qc > 0.05e-3) return;

            // Calculate effective vertical velocity (grid-scale + sub-grid)
            // Sub-grid velocity is set to 0.5 m/s following the original scheme
            amrex::Real w_eff = w_local;
            if (m_isub == 0) {
                w_eff = std::sqrt(w_local*w_local + 0.5*0.5);
            }

            // Only activate if upward motion
            if (w_eff <= 0.01) return;

            // Calculate supersaturation based on vertical velocity
            amrex::Real supersat = calculateSupersaturation(w_eff, temp, pres, qv);

            // Number of activated CCN (per kg)
            amrex::Real nact = 0.0;

            // Power-law CCN spectra (IACT = 1)
            if (m_iact == 1) {
                // NCCN = C*S^K, where S is supersaturation in %
                // Convert from cm^-3 to kg^-1
                nact = m_c1 * std::pow(supersat*100.0, m_k1) * 1.0e6 / rho;
            }
            // Lognormal aerosol size distribution (IACT = 2)
            else if (m_iact == 2) {
                // Calculate critical supersaturation for activation
                // First calculate parameters for activation
                amrex::Real alpha = std::pow(2.0*m_mw*0.0761/(m_rhow*m_r_v*temp), 1.5);
                amrex::Real gamma = m_r_v*temp*m_rhow/(m_mw*0.0761);
                amrex::Real psi = 2.0/3.0 * std::sqrt(alpha/gamma);

                // Calculate maximum supersaturation based on Ghan et al. (1993)
                amrex::Real eta1 = std::pow(supersat/(m_f11*m_nanew1), 1.0/m_f21);
                amrex::Real eta2 = std::pow(supersat/(m_f12*m_nanew2), 1.0/m_f22);
                amrex::Real smax = supersat;

                // Calculate number activated from each mode
                amrex::Real uu1 = 2.0*std::log(m_rm1/m_bact) / (3.0*std::sqrt(2.0)*std::log(m_sig1));
                amrex::Real uu2 = 2.0*std::log(m_rm2/m_bact) / (3.0*std::sqrt(2.0)*std::log(m_sig2));

                // Calculate number activated using error function
                amrex::Real n1 = 0.5*m_nanew1*(1.0 - ErrorFunction(uu1));
                amrex::Real n2 = 0.5*m_nanew2*(1.0 - ErrorFunction(uu2));

                // Total number activated (convert from m^-3 to kg^-1)
                nact = (n1 + n2) / rho;
            }

            // Limit activation to reasonable values
            nact = std::min(nact, 1.0e10);

            // Don't activate more than available CCN
            // For simplicity, assume total CCN concentration is 1000 cm^-3
            const amrex::Real nccn_max = 1000.0e6 / rho;  // Convert from cm^-3 to kg^-1
            nact = std::min(nact, nccn_max);

            // Don't activate if we already have more droplets than would be activated
            if (nc >= nact) return;

            // Calculate activation rate (number/kg/s)
            // Only activate the difference between current and activated number
            amrex::Real pccn = (nact - nc) / dt;

            // Update cloud droplet number concentration
            hydro_nc(i,j,k) += pccn * dt;
        });

        // This block implements homogeneous freezing of cloud water
        // It's part of the saturation adjustment process because it changes
        // water phase partitioning based on temperature thresholds
        // Process: MNUCCC (Homogeneous freezing of drops, Cloud)
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Homogeneous freezing of cloud water (all liquid freezes below threshold)
            if (thermo_tabs(i,j,k) <= t_homog_freeze && hydro_qc(i,j,k) >= m_qsmall) {
                // Convert all cloud water to cloud ice
                hydro_qi(i,j,k) += hydro_qc(i,j,k);
                
                // Apply latent heating
                const amrex::Real xxlv = 3.1484e6 - 2370.0 * thermo_tabs(i,j,k); // Latent heat of vaporization
                const amrex::Real xxls = 3.15e6 - 2370.0 * thermo_tabs(i,j,k) + 0.3337e6; // Latent heat of sublimation
                const amrex::Real xlf = xxls - xxlv; // Latent heat of fusion
                
                // Heat capacity including water vapor
                const amrex::Real cpm = m_cp * (1.0 + 0.887 * hydro_qv(i,j,k));
                
                // Update temperature due to freezing
                thermo_tabs(i,j,k) += hydro_qc(i,j,k) * xlf / cpm;
                
                // Transfer number concentration
                hydro_ni(i,j,k) += hydro_nc(i,j,k);
                
                // Clear cloud water and number
                hydro_qc(i,j,k) = 0.0;
                hydro_nc(i,j,k) = 0.0;
            }
        });

        // This block implements homogeneous freezing of rain
        // Also part of phase changes in the saturation adjustment process
        // Process: MNUCCR (Rain freezing, Precip)
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Homogeneous freezing of rain (all rain freezes below threshold)
            if (thermo_tabs(i,j,k) <= t_homog_freeze && hydro_qr(i,j,k) >= m_qsmall) {
                // Convert all rain water to graupel
                hydro_qg(i,j,k) += hydro_qr(i,j,k);

                // Apply latent heating
                const amrex::Real xxlv = 3.1484e6 - 2370.0 * thermo_tabs(i,j,k); // Latent heat of vaporization
                const amrex::Real xxls = 3.15e6 - 2370.0 * thermo_tabs(i,j,k) + 0.3337e6; // Latent heat of sublimation
                const amrex::Real xlf = xxls - xxlv; // Latent heat of fusion

                // Heat capacity including water vapor
                const amrex::Real cpm = m_cp * (1.0 + 0.887 * hydro_qv(i,j,k));

                // Update temperature due to freezing
                thermo_tabs(i,j,k) += hydro_qr(i,j,k) * xlf / cpm;

                // Transfer number concentration
                hydro_ng(i,j,k) += hydro_nr(i,j,k);

                // Clear rain water and number
                hydro_qr(i,j,k) = 0.0;
                hydro_nr(i,j,k) = 0.0;
            }
        });

        // This is the core saturation adjustment algorithm 
        // It redistributes water between vapor, liquid, and ice phases to
        // maintain thermodynamic equilibrium (saturation conditions)
        // Process: MNUCCC (Homogeneous freezing of drops, Cloud)
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Check if there is any condensate to adjust
            if (hydro_qc(i,j,k) < m_qsmall && hydro_qi(i,j,k) < m_qsmall && 
                hydro_qv(i,j,k) < m_qsmall) {
                return; // Skip if no water to adjust
            }
            
            // Starting temperature
            amrex::Real T = thermo_tabs(i,j,k);
            
            // Calculate the saturation values at current temperature
            amrex::Real evs = std::min(0.99*thermo_pres(i,j,k), 
                                     calc_saturation_vapor_pressure(T, 0)); // Water saturation
            amrex::Real eis = std::min(0.99*thermo_pres(i,j,k), 
                                     calc_saturation_vapor_pressure(T, 1)); // Ice saturation
            
            // Ensure ice saturation doesn't exceed water saturation near freezing
            eis = std::min(eis, evs);
            
            // Calculate saturation mixing ratios
            amrex::Real qvs = m_ep_2 * evs / (thermo_pres(i,j,k) - evs);
            amrex::Real qvi = m_ep_2 * eis / (thermo_pres(i,j,k) - eis);
            
            // Calculate saturation ratios
            amrex::Real qvqvs = hydro_qv(i,j,k) / qvs;
            amrex::Real qvqvsi = hydro_qv(i,j,k) / qvi;

            // Newton iteration for saturation adjustment
            // This is the core algorithm that determines the final equilibrium 
            // temperature and water phase distribution

            // Initialize values for Newton iteration
            int niter = 0;
            amrex::Real dtabs = 1.0;
            
            // Current water vapor and cloud water/ice mixing ratios
            amrex::Real qv_cur = hydro_qv(i,j,k);
            amrex::Real qc_cur = hydro_qc(i,j,k);
            amrex::Real qi_cur = hydro_qi(i,j,k);
            
            // Iterate until convergence or max iterations
            // This loop adjusts temperature to find equilibrium between
            // water phases considering latent heat effects
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
                evs = std::min(0.99*thermo_pres(i,j,k), calc_saturation_vapor_pressure(T, 0));
                eis = std::min(0.99*thermo_pres(i,j,k), calc_saturation_vapor_pressure(T, 1));
                if (eis > evs) eis = evs;
                
                // Saturation mixing ratios
                qvs = m_ep_2 * evs / (thermo_pres(i,j,k) - evs);
                qvi = m_ep_2 * eis / (thermo_pres(i,j,k) - eis);
                
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
                const amrex::Real f = -T + thermo_tabs(i,j,k) + lsterms * (qv_cur - qsat);
                const amrex::Real df = -1.0 + dlsterms * (qv_cur - qsat) - lsterms * dqsat;
                
                // Update temperature
                dtabs = -f / df;
                T += dtabs;
                
                niter++;
            }
            
            // Update saturation values at final temperature
            evs = std::min(0.99*thermo_pres(i,j,k), calc_saturation_vapor_pressure(T, 0));
            eis = std::min(0.99*thermo_pres(i,j,k), calc_saturation_vapor_pressure(T, 1));
            if (eis > evs) eis = evs;
            
            qvs = m_ep_2 * evs / (thermo_pres(i,j,k) - evs);
            qvi = m_ep_2 * eis / (thermo_pres(i,j,k) - eis);
            
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
            const amrex::Real delta_qv = hydro_qv(i,j,k) - qsat;
            
            // Partition excess/deficit between liquid and ice based on temperature
            const amrex::Real delta_qc = std::max(-hydro_qc(i,j,k), delta_qv * omn);
            const amrex::Real delta_qi = std::max(-hydro_qi(i,j,k), delta_qv * (1.0 - omn));

            // Update mixing ratios - this is where the actual redistribution occurs
            // between vapor, liquid, and ice to achieve saturation equilibrium
            hydro_qv(i,j,k) = qsat;
            hydro_qc(i,j,k) += delta_qc;
            hydro_qi(i,j,k) += delta_qi;
            
            // Update total condensate and total water will be done at the end of this kernel
            
            // Update temperature
            thermo_tabs(i,j,k) = T;
            
            // Note: We're storing absolute temperature in thermo_tabs, not potential temperature
            // The conversion to potential temperature happens elsewhere when needed
            
            // Apply minimum thresholds
            if (hydro_qc(i,j,k) < m_qsmall) hydro_qc(i,j,k) = 0.0;
            if (hydro_qi(i,j,k) < m_qsmall) hydro_qi(i,j,k) = 0.0;

            qn_arr(i,j,k) = hydro_qc(i,j,k) + hydro_qi(i,j,k);
            qt_arr(i,j,k) = hydro_qv(i,j,k) + qn_arr(i,j,k);
        });


        // Heterogeneous freezing of cloud droplets complements the saturation adjustment
        // by allowing phase transitions that occur at temperatures above homogeneous freezing
        // Process: MNUCCC (Homogeneous freezing of drops, Cloud)
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Only proceed if below 0°C but above homogeneous freezing temperature
            const amrex::Real temp = thermo_tabs(i,j,k);
            if (temp < 269.15 && temp > t_homog_freeze && hydro_qc(i,j,k) >= m_qsmall) {
                // Calculate number of contact nuclei per m^3 (Meyers et al. 1992)
                const amrex::Real n_contact = std::exp(-2.80 + 0.262 * (273.15 - temp)) * 1000.0;
                
                // Mean free path
                const amrex::Real dum = 7.37 * temp / (288.0 * 10.0 * thermo_pres(i,j,k)) / 100.0;
                
                // Effective diffusivity of contact nuclei (Brownian diffusion)
                const amrex::Real dap = m_cons37 * temp * (1.0 + dum / m_rin) / m_mu;
                
                // Get cloud droplet distribution parameters
                const amrex::Real rho = thermo_rho(i,j,k);
                const amrex::Real dum1 = thermo_pres(i,j,k) / (287.15 * temp);
                amrex::Real pgam = 0.0005714 * (hydro_nc(i,j,k) * rho / 1.0e6 * dum1) + 0.2714;
                pgam = 1.0 / (pgam * pgam) - 1.0;
                pgam = amrex::max(pgam, 2.0);
                pgam = amrex::min(pgam, 10.0);
                
                // Calculate distribution parameters
                const amrex::Real lamc = std::pow(m_cons26 * hydro_nc(i,j,k) * gamma_function(pgam + 4.0) /
                                         (hydro_qc(i,j,k) * gamma_function(pgam + 1.0)), 1.0/3.0);
                
                const amrex::Real cdist = hydro_nc(i,j,k) / gamma_function(pgam + 1.0);
                
                // Contact freezing rate
                const amrex::Real mnuccc = m_cons38 * dap * n_contact * 
                                          std::exp(std::log(cdist) + std::log(gamma_function(pgam + 5.0)) - 
                                          4.0 * std::log(lamc));
                
                const amrex::Real nnuccc = 2.0 * M_PI * dap * n_contact * cdist *
                                          gamma_function(pgam + 2.0) / lamc;
                
                // Immersion freezing (Bigg 1953)
                const amrex::Real imm_rate = std::exp(m_aimm * (273.15 - temp)) - 1.0;
                
                const amrex::Real mnuccc_imm = m_cons39 * 
                                            std::exp(std::log(cdist) + 
                                            std::log(gamma_function(7.0 + pgam)) - 
                                            6.0 * std::log(lamc)) * imm_rate;
                
                const amrex::Real nnuccc_imm = m_cons40 * 
                                            std::exp(std::log(cdist) + 
                                            std::log(gamma_function(pgam + 4.0)) - 
                                            3.0 * std::log(lamc)) * imm_rate;
                
                // Total heterogeneous freezing rate
                const amrex::Real mnuc_total = mnuccc + mnuccc_imm;
                const amrex::Real nnuc_total = nnuccc + nnuccc_imm;
                
                // Calculate maximum possible freezing based on available droplets
                const amrex::Real max_freeze_qc = hydro_qc(i,j,k) / dt;
                const amrex::Real max_freeze_nc = hydro_nc(i,j,k) / dt;
                
                // Apply freezing with limits
                const amrex::Real mnuc_limited = amrex::min(mnuc_total, max_freeze_qc);
                const amrex::Real nnuc_limited = amrex::min(nnuc_total, max_freeze_nc);
                
                // Transfer to ice
                hydro_qi(i,j,k) += mnuc_limited * dt;
                hydro_ni(i,j,k) += nnuc_limited * dt;
                
                // Remove from cloud water
                hydro_qc(i,j,k) -= mnuc_limited * dt;
                hydro_nc(i,j,k) -= nnuc_limited * dt;
                
                // Apply latent heating
                const amrex::Real xxlv = 3.1484e6 - 2370.0 * temp;
                const amrex::Real xxls = 3.15e6 - 2370.0 * temp + 0.3337e6;
                const amrex::Real xlf = xxls - xxlv;
                const amrex::Real cpm = m_cp * (1.0 + 0.887 * hydro_qv(i,j,k));
                
                thermo_tabs(i,j,k) += mnuc_limited * xlf / cpm * dt;
            }
        });
#if 0
        // This block implements primary ice nucleation (deposition/condensation freezing)
        // This process introduces new ice particles rather than changing phase of existing water,
        // so it's not directly part of the thermodynamic phase equilibrium adjustment
        //----------------------------------------------------------------------
        // Primary Ice Nucleation (NNUCCD, MNUCCD) - New Implementation
        //----------------------------------------------------------------------
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // Get local variables
            const amrex::Real temp = thermo_tabs(i,j,k);
            const amrex::Real rho = thermo_rho(i,j,k);
            const amrex::Real qi = hydro_qi(i,j,k);
            const amrex::Real qs = hydro_qs(i,j,k);
            const amrex::Real qg = hydro_qg(i,j,k);
            const amrex::Real ni = hydro_ni(i,j,k);
            const amrex::Real ns = hydro_ns(i,j,k);
            const amrex::Real ng = hydro_ng(i,j,k);

            // Initialize nucleation rates
            amrex::Real nnuccd = 0.0;
            amrex::Real mnuccd = 0.0;

            // Only proceed if below freezing
            if (temp < t_freeze) {
                // Calculate saturation vapor pressures (already done in saturation adjustment)
                amrex::Real evs = std::min(0.99*thermo_pres(i,j,k), calc_saturation_vapor_pressure(temp, 0));
                amrex::Real eis = std::min(0.99*thermo_pres(i,j,k), calc_saturation_vapor_pressure(temp, 1));
                if (eis > evs) eis = evs;

                // Calculate saturation mixing ratios
                amrex::Real qvs = m_ep_2 * evs / (thermo_pres(i,j,k) - evs);
                amrex::Real qvi = m_ep_2 * eis / (thermo_pres(i,j,k) - eis);

                // Calculate saturation ratios
                amrex::Real qvqvs = hydro_qv(i,j,k) / qvs;
                amrex::Real qvqvsi = hydro_qv(i,j,k) / qvi;

                // Ice nucleation parameterization
                // Cooper (1986) for mid-latitude conditions
                if (m_inuc_type == 0 || m_inuc == 0) { // Mid-latitude (Cooper)
                    // Only activate if supersaturated with respect to ice
                    // and below a certain temperature threshold
                    if (qvqvsi >= 1.0 && temp <= 265.15) {
                        // Calculate ice nucleation rate (per m^3)
                        amrex::Real kc2 = 0.005 * std::exp(0.304 * (t_freeze - temp)) * 1000.0; // Convert from L^-1 to m^-3

                        // Limit to a maximum value (e.g., 500 L^-1)
                        kc2 = std::min(kc2, 500.0e3);

                        // Convert to per kg
                        kc2 = kc2 / rho;

                        // Only nucleate if the nucleation rate exceeds existing ice number
                        if (kc2 > ni + ns + ng) {
                            nnuccd = (kc2 - (ni + ns + ng)) / dt;
                            mnuccd = nnuccd * m_mi0; // m_mi0 is the initial mass of nucleated ice
                        }
                    }
                }
                // MPACE (Morrison et al. 2007)
                else if (m_inuc_type == 1 || m_inuc == 1) { // Arctic (MPACE)
                    // Only activate if supersaturated with respect to ice
                    if (qvqvsi > 1.0) {
                        // Constant nucleation rate (per m^3)
                        amrex::Real kc2 = 0.16 * 1000.0; // Convert from L^-1 to m^-3

                        // Convert to per kg
                        kc2 = kc2 / rho;

                        // Only nucleate if the nucleation rate exceeds existing ice number
                        if (kc2 > ni + ns + ng) {
                            nnuccd = (kc2 - (ni + ns + ng)) / dt;
                            mnuccd = nnuccd * m_mi0; // m_mi0 is the initial mass of nucleated ice
                        }
                    }
                }

                // Apply nucleation (add to cloud ice) if there's any nucleation happening
                if (mnuccd > 0.0) {
                    hydro_qi(i,j,k) += mnuccd * dt;
                    hydro_ni(i,j,k) += nnuccd * dt;
                    
                    // Apply latent heating
                    const amrex::Real cpm = m_cp * (1.0 + 0.887 * hydro_qv(i,j,k));
                    thermo_tabs(i,j,k) += mnuccd * m_fac_sub * dt / cpm;
                }
            }
        });
        // This block handles ice-snow categorization based on particle size
        // It's a non-thermodynamic process that classifies ice particles
        // rather than adjusting phases based on thermodynamic equilibrium
        // Process: NSUBS (Snow evaporation, Precip)
        //----------------------------------------------------------------------
        // Transfer ice to snow if mean size exceeds threshold (existing code)
        //----------------------------------------------------------------------
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Only check if there is appreciable cloud ice
            if (hydro_qi(i,j,k) >= m_qsmall && thermo_tabs(i,j,k) < t_freeze) {
                // Calculate size distribution parameter
                const amrex::Real lami = (hydro_ni(i,j,k) > 0.0) ? 
                    std::pow(m_cons12 * hydro_ni(i,j,k) / hydro_qi(i,j,k), 1.0/m_di) : 0.0;
                
                // Check if mean size is valid
                if (lami >= 1.0e-10) {
                    // Check if mean size exceeds threshold (2*DCS)
                    if (1.0/lami >= 2.0*m_dcs) {
                        // Transfer all cloud ice to snow
                        hydro_qs(i,j,k) += hydro_qi(i,j,k);
                        hydro_ns(i,j,k) += hydro_ni(i,j,k);
                        hydro_qi(i,j,k) = 0.0;
                        hydro_ni(i,j,k) = 0.0;
                    }
                }
            }
        });
#endif
    }
}
