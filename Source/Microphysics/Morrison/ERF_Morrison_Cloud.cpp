// Add these includes if needed
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
    for (amrex::MFIter mfi(*m_thermo); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        
        // Get array data
        auto const& thermo = m_thermo->array(mfi);
        auto const& hydro = m_hydro->array(mfi);
        auto const& w = m_thermo->array(mfi);  // Vertical velocity

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
        const int nr_comp = 6;  // Rain drop number
        const int ni_comp = 7;  // Ice crystal number
        const int ns_comp = 8;  // Snow number
        const int ng_comp = 9;  // Graupel number

        // Parallel execution over the box
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Get local variables
            const amrex::Real temp = thermo(i,j,k,t_comp);
            const amrex::Real pres = thermo(i,j,k,p_comp);
            const amrex::Real qv = hydro(i,j,k,qv_comp);
            const amrex::Real rho = thermo(i,j,k,rho_comp);
            const amrex::Real qc = hydro(i,j,k,qc_comp);
            const amrex::Real nc = hydro(i,j,k,nc_comp);
            const amrex::Real w_local = w(i,j,k,3);  // Vertical velocity component

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
            hydro(i,j,k,nc_comp) += pccn * dt;
	});

        //----------------------------------------------------------------------
        // Handle homogeneous freezing of cloud water (replace existing code if any)
        //----------------------------------------------------------------------
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Homogeneous freezing of cloud water (all liquid freezes below threshold)
            if (thermo(i,j,k,t_comp) <= t_homog_freeze && hydro(i,j,k,qc_comp) >= m_qsmall) {
                // Convert all cloud water to cloud ice
                hydro(i,j,k,qi_comp) += hydro(i,j,k,qc_comp);
                
                // Apply latent heating
                const amrex::Real xxlv = 3.1484e6 - 2370.0 * thermo(i,j,k,t_comp); // Latent heat of vaporization
                const amrex::Real xxls = 3.15e6 - 2370.0 * thermo(i,j,k,t_comp) + 0.3337e6; // Latent heat of sublimation
                const amrex::Real xlf = xxls - xxlv; // Latent heat of fusion
                
                // Heat capacity including water vapor
                const amrex::Real cpm = m_cp * (1.0 + 0.887 * hydro(i,j,k,qv_comp));
                
                // Update temperature due to freezing
                thermo(i,j,k,t_comp) += hydro(i,j,k,qc_comp) * xlf / cpm;
                
                // Transfer number concentration
                hydro(i,j,k,ni_comp) += hydro(i,j,k,nc_comp);
                
                // Clear cloud water and number
                hydro(i,j,k,qc_comp) = 0.0;
                hydro(i,j,k,nc_comp) = 0.0;
            }
        });
        
        //----------------------------------------------------------------------
        // Phase partitioning and saturation adjustment (existing code)
        //----------------------------------------------------------------------
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Check if there is any condensate to adjust
            if (hydro(i,j,k,qc_comp) < m_qsmall && hydro(i,j,k,qi_comp) < m_qsmall && 
                hydro(i,j,k,qv_comp) < m_qsmall) {
                return; // Skip if no water to adjust
            }
            
            // Starting temperature
            amrex::Real T = thermo(i,j,k,t_comp);
            
            // Calculate the saturation values at current temperature
            amrex::Real evs = std::min(0.99*thermo(i,j,k,p_comp), 
                                     calc_saturation_vapor_pressure(T, 0)); // Water saturation
            amrex::Real eis = std::min(0.99*thermo(i,j,k,p_comp), 
                                     calc_saturation_vapor_pressure(T, 1)); // Ice saturation
            
            // Ensure ice saturation doesn't exceed water saturation near freezing
            eis = std::min(eis, evs);
            
            // Calculate saturation mixing ratios
            amrex::Real qvs = m_ep_2 * evs / (thermo(i,j,k,p_comp) - evs);
            amrex::Real qvi = m_ep_2 * eis / (thermo(i,j,k,p_comp) - eis);
            
            // Calculate saturation ratios
            amrex::Real qvqvs = hydro(i,j,k,qv_comp) / qvs;
            amrex::Real qvqvsi = hydro(i,j,k,qv_comp) / qvi;
            
            //--------------------------------------------------------------
            // Newton iteration for saturation adjustment (similar to existing code)
            //--------------------------------------------------------------
            
            // Initialize values for Newton iteration
            int niter = 0;
            amrex::Real dtabs = 1.0;
            
            // Current water vapor and cloud water/ice mixing ratios
            amrex::Real qv_cur = hydro(i,j,k,qv_comp);
            amrex::Real qc_cur = hydro(i,j,k,qc_comp);
            amrex::Real qi_cur = hydro(i,j,k,qi_comp);
            
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
                evs = std::min(0.99*thermo(i,j,k,p_comp), calc_saturation_vapor_pressure(T, 0));
                eis = std::min(0.99*thermo(i,j,k,p_comp), calc_saturation_vapor_pressure(T, 1));
                if (eis > evs) eis = evs;
                
                // Saturation mixing ratios
                qvs = m_ep_2 * evs / (thermo(i,j,k,p_comp) - evs);
                qvi = m_ep_2 * eis / (thermo(i,j,k,p_comp) - eis);
                
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
                const amrex::Real f = -T + thermo(i,j,k,t_comp) + lsterms * (qv_cur - qsat);
                const amrex::Real df = -1.0 + dlsterms * (qv_cur - qsat) - lsterms * dqsat;
                
                // Update temperature
                dtabs = -f / df;
                T += dtabs;
                
                niter++;
            }
            
            // Update saturation values at final temperature
            evs = std::min(0.99*thermo(i,j,k,p_comp), calc_saturation_vapor_pressure(T, 0));
            eis = std::min(0.99*thermo(i,j,k,p_comp), calc_saturation_vapor_pressure(T, 1));
            if (eis > evs) eis = evs;
            
            qvs = m_ep_2 * evs / (thermo(i,j,k,p_comp) - evs);
            qvi = m_ep_2 * eis / (thermo(i,j,k,p_comp) - eis);
            
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
            const amrex::Real delta_qv = hydro(i,j,k,qv_comp) - qsat;
            
            // Partition excess/deficit between liquid and ice based on temperature
            const amrex::Real delta_qc = std::max(-hydro(i,j,k,qc_comp), delta_qv * omn);
            const amrex::Real delta_qi = std::max(-hydro(i,j,k,qi_comp), delta_qv * (1.0 - omn));
            
            // Update mixing ratios
            hydro(i,j,k,qv_comp) = qsat;
            hydro(i,j,k,qc_comp) += delta_qc;
            hydro(i,j,k,qi_comp) += delta_qi;
            
            // Update total condensate and total water
            // (assuming these are tracked in separate variables)
            // hydro(i,j,k,qn_comp) = hydro(i,j,k,qc_comp) + hydro(i,j,k,qi_comp);
            // hydro(i,j,k,qt_comp) = hydro(i,j,k,qv_comp) + hydro(i,j,k,qn_comp);
            
            // Update temperature
            thermo(i,j,k,t_comp) = T;
            
            // Update potential temperature
            const amrex::Real exner = std::pow(thermo(i,j,k,p_comp)/100000.0, m_rdOcp);
            thermo(i,j,k,t_comp) = T / exner;
            
            // Apply minimum thresholds
            if (hydro(i,j,k,qc_comp) < m_qsmall) hydro(i,j,k,qc_comp) = 0.0;
            if (hydro(i,j,k,qi_comp) < m_qsmall) hydro(i,j,k,qi_comp) = 0.0;
        });


        //----------------------------------------------------------------------
        // Add heterogeneous freezing of cloud droplets (new code)
        //----------------------------------------------------------------------
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Only proceed if below 0°C but above homogeneous freezing temperature
            const amrex::Real temp = thermo(i,j,k,t_comp);
            if (temp < 269.15 && temp > t_homog_freeze && hydro(i,j,k,qc_comp) >= m_qsmall) {
                // Calculate number of contact nuclei per m^3 (Meyers et al. 1992)
                const amrex::Real n_contact = std::exp(-2.80 + 0.262 * (273.15 - temp)) * 1000.0;
                
                // Mean free path
                const amrex::Real dum = 7.37 * temp / (288.0 * 10.0 * thermo(i,j,k,p_comp)) / 100.0;
                
                // Effective diffusivity of contact nuclei (Brownian diffusion)
                const amrex::Real dap = m_cons37 * temp * (1.0 + dum / m_rin) / m_mu;
                
                // Get cloud droplet distribution parameters
                const amrex::Real rho = thermo(i,j,k,rho_comp);
                const amrex::Real dum1 = thermo(i,j,k,p_comp) / (287.15 * temp);
                amrex::Real pgam = 0.0005714 * (hydro(i,j,k,nc_comp) * rho / 1.0e6 * dum1) + 0.2714;
                pgam = 1.0 / (pgam * pgam) - 1.0;
                pgam = amrex::max(pgam, 2.0);
                pgam = amrex::min(pgam, 10.0);
                
                // Calculate distribution parameters
                const amrex::Real lamc = std::pow(m_cons26 * hydro(i,j,k,nc_comp) * gamma_function(pgam + 4.0) /
                                         (hydro(i,j,k,qc_comp) * gamma_function(pgam + 1.0)), 1.0/3.0);
                
                const amrex::Real cdist = hydro(i,j,k,nc_comp) / gamma_function(pgam + 1.0);
                
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
                const amrex::Real max_freeze_qc = hydro(i,j,k,qc_comp) / dt;
                const amrex::Real max_freeze_nc = hydro(i,j,k,nc_comp) / dt;
                
                // Apply freezing with limits
                const amrex::Real mnuc_limited = amrex::min(mnuc_total, max_freeze_qc);
                const amrex::Real nnuc_limited = amrex::min(nnuc_total, max_freeze_nc);
                
                // Transfer to ice
                hydro(i,j,k,qi_comp) += mnuc_limited * dt;
                hydro(i,j,k,ni_comp) += nnuc_limited * dt;
                
                // Remove from cloud water
                hydro(i,j,k,qc_comp) -= mnuc_limited * dt;
                hydro(i,j,k,nc_comp) -= nnuc_limited * dt;
                
                // Apply latent heating
                const amrex::Real xxlv = 3.1484e6 - 2370.0 * temp;
                const amrex::Real xxls = 3.15e6 - 2370.0 * temp + 0.3337e6;
                const amrex::Real xlf = xxls - xxlv;
                const amrex::Real cpm = m_cp * (1.0 + 0.887 * hydro(i,j,k,qv_comp));
                
                thermo(i,j,k,t_comp) += mnuc_limited * xlf / cpm * dt;
            }
        });

        //----------------------------------------------------------------------
        // Transfer ice to snow if mean size exceeds threshold (existing code)
        //----------------------------------------------------------------------
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Only check if there is appreciable cloud ice
            if (hydro(i,j,k,qi_comp) >= m_qsmall && thermo(i,j,k,t_comp) < t_freeze) {
                // Calculate size distribution parameter
                const amrex::Real lami = std::pow(m_cons12 * hydro(i,j,k,ni_comp) / 
                                        hydro(i,j,k,qi_comp), 1.0/m_di);
                
                // Check if mean size is valid
                if (lami >= 1.0e-10) {
                    // Check if mean size exceeds threshold (2*DCS)
                    if (1.0/lami >= 2.0*m_dcs) {
                        // Transfer all cloud ice to snow
                        hydro(i,j,k,qs_comp) += hydro(i,j,k,qi_comp);
                        hydro(i,j,k,ns_comp) += hydro(i,j,k,ni_comp);
                        hydro(i,j,k,qi_comp) = 0.0;
                        hydro(i,j,k,ni_comp) = 0.0;
                    }
                }
            }
        });
    }
}
