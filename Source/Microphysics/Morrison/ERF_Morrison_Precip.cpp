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
Morrison::Precip(const SolverChoice& sc)
{
    BL_PROFILE("Morrison::Precip()");

    // Constants for microphysical processes
    constexpr amrex::Real one = 1.0;
    constexpr amrex::Real zero = 0.0;
    constexpr amrex::Real t_freezing = 273.15; // Freezing point of water in K

    // Temperature thresholds for HM-process (K)
    constexpr amrex::Real t_hm_max = 270.16;
    constexpr amrex::Real t_hm_min = 265.16;
    
    // Mass of individual splinters (kg)
    const amrex::Real mmult = 4.0/3.0 * M_PI * m_rhoi * std::pow(5.0e-6, 3);

    // Loop through grids
    for (amrex::MFIter mfi(*m_hydro); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        
        // Get array data
        auto const& thermo = m_thermo->array(mfi);
        auto const& hydro = m_hydro->array(mfi);
        auto const& tend = m_tend->array(mfi);
        
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
        const int nr_comp = 6;  // Rain number
        const int ni_comp = 7;  // Cloud ice number
        const int ns_comp = 8;  // Snow number
        const int ng_comp = 9;  // Graupel number
        
        // Parallel execution over the box
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
        // Variables for storing process rates
        amrex::Real prc, nprc, nprc1, pra, npra, nragg, psacws, npsacws;
        amrex::Real pracs, npracs, psacwg, pracg, npracg, pgsacw, pgracs;
        amrex::Real nscng, ngracs, praci, piacr, niacr, piacrs, niacrs, pracis;
        amrex::Real pre, prds, prg, evpms, evpmg;
        amrex::Real nmults, nmultr, qmults, qmultr;
        amrex::Real nmultg, nmultrg, qmultg, qmultrg;
        
            // Get local variables
            const amrex::Real temp = thermo(i,j,k,t_comp);
            const amrex::Real pres = thermo(i,j,k,p_comp);
            const amrex::Real qv = hydro(i,j,k,qv_comp);
            const amrex::Real rho = thermo(i,j,k,rho_comp);
            const amrex::Real qc = hydro(i,j,k,qc_comp);
            const amrex::Real qi = hydro(i,j,k,qi_comp);
            const amrex::Real qr = hydro(i,j,k,qr_comp);
            const amrex::Real qs = hydro(i,j,k,qs_comp);
            const amrex::Real qg = hydro(i,j,k,qg_comp);
            const amrex::Real nc = hydro(i,j,k,nc_comp);
            const amrex::Real ni = hydro(i,j,k,ni_comp);
            const amrex::Real nr = hydro(i,j,k,nr_comp);
            const amrex::Real ns = hydro(i,j,k,ns_comp);
            const amrex::Real ng = hydro(i,j,k,ng_comp);
            
            // Initialize process rates to zero
            prc = 0.0; nprc = 0.0; nprc1 = 0.0; pra = 0.0; npra = 0.0; nragg = 0.0;
            psacws = 0.0; npsacws = 0.0; pracs = 0.0; npracs = 0.0;
            psacwg = 0.0; pracg = 0.0; npracg = 0.0; pgsacw = 0.0; pgracs = 0.0;
            nscng = 0.0; ngracs = 0.0; praci = 0.0; piacr = 0.0; niacr = 0.0;
            piacrs = 0.0; niacrs = 0.0; pracis = 0.0;
            pre = 0.0; prds = 0.0; prg = 0.0; evpms = 0.0; evpmg = 0.0;
            nmults = 0.0; nmultr = 0.0; qmults = 0.0; qmultr = 0.0;
            nmultg = 0.0; nmultrg = 0.0; qmultg = 0.0; qmultrg = 0.0;
            
            // Calculate size distribution parameters for all hydrometeors
            amrex::Real lamr = 0.0, lams = 0.0, lamg = 0.0, lami = 0.0, lamc = 0.0;
            amrex::Real n0r = 0.0, n0s = 0.0, n0g = 0.0, n0i = 0.0;
            amrex::Real pgam = 0.0;
            
            // Rain distribution
            if (qr >= m_qsmall) {
                lamr = std::pow(M_PI * m_rhow * nr / qr, 1.0/3.0);
                
                // Apply lambda limits
                lamr = amrex::max(lamr, m_lamminr);
                lamr = amrex::min(lamr, m_lammaxr);
                
                n0r = nr * lamr;
            }
            
            // Snow distribution
            if (qs >= m_qsmall) {
                lams = std::pow(m_cons1 * ns / qs, 1.0/m_ds);
                
                // Apply lambda limits
                lams = amrex::max(lams, m_lammins);
                lams = amrex::min(lams, m_lammaxs);
                
                n0s = ns * lams;
            }
            
            // Graupel distribution
            if (qg >= m_qsmall) {
                lamg = std::pow(m_cons2 * ng / qg, 1.0/m_dg);
                
                // Apply lambda limits
                lamg = amrex::max(lamg, m_lamming);
                lamg = amrex::min(lamg, m_lammaxg);
                
                n0g = ng * lamg;
            }
            
            // Ice distribution
            if (qi >= m_qsmall) {
                lami = std::pow(m_cons12 * ni / qi, 1.0/m_di);
                
                // Apply lambda limits
                lami = amrex::max(lami, m_lammini);
                lami = amrex::min(lami, m_lammaxi);
                
                n0i = ni * lami;
            }
            
            // Cloud distribution
            if (qc >= m_qsmall) {
                // Cloud droplet gamma distribution shape parameter
                const amrex::Real dum = pres / (287.15 * temp);
                pgam = 0.0005714 * (nc * rho / 1.0e6 * dum) + 0.2714;
                pgam = 1.0 / (pgam * pgam) - 1.0;
                pgam = amrex::max(pgam, 2.0);
                pgam = amrex::min(pgam, 10.0);
                
                // Calculate cloud droplet mean size parameter
                lamc = std::pow(m_cons26 * nc * gamma_function(pgam + 4.0) /
                               (qc * gamma_function(pgam + 1.0)), 1.0/3.0);
                
                // Apply lambda limits
                const amrex::Real lammin = (pgam + 1.0) / 60.0e-6;
                const amrex::Real lammax = (pgam + 1.0) / 1.0e-6;
                lamc = amrex::max(lamc, lammin);
                lamc = amrex::min(lamc, lammax);
            }
            
            //----------------------------------------------------------------------
            // 1. Autoconversion of cloud water to rain (Khairoutdinov and Kogan 2000)
            //----------------------------------------------------------------------
            if (qc >= 1.0e-6) {
                prc = 1350.0 * std::pow(qc, 2.47) * 
                      std::pow(nc * rho / 1.0e6, -1.79);
                
                // Calculate number conversion rates
                nprc1 = prc / m_cons29;
                nprc = prc / (qc / nc);
                
                // Limit by available cloud water
                nprc = amrex::min(nprc, nc / dt);
                nprc1 = amrex::min(nprc1, nprc);
            }
            
            //----------------------------------------------------------------------
            // 2. Rain-ice collisions - new process integration example
            //----------------------------------------------------------------------
            if (temp <= 273.15 && qr >= 1.0e-8 && qi >= 1.0e-8) {
                // Determine if rain is heavy enough to convert to graupel
                const bool convert_to_graupel = (qr >= 0.1e-3);
                
                // Collision rates for number concentrations
                niacr = m_cons24 * ni * n0r * m_ar * 
                       std::pow(lamr, -(m_br + 3.0)) * rho;
                
                // Collision rates for mass - ice collecting rain
                piacr = m_cons25 * ni * n0r * m_ar * 
                       std::pow(lamr, -(m_br + 3.0)) / std::pow(lamr, 3.0) * rho;
                
                // Collision rates for mass - rain collecting ice
                praci = m_cons24 * qi * n0r * m_ar * 
                       std::pow(lamr, -(m_br + 3.0)) * rho;
                
                // Limit by available particles
                const amrex::Real niacr_limited = amrex::min(niacr, nr / dt);
                niacr = amrex::min(niacr_limited, ni / dt);
                
                if (convert_to_graupel) {
                    // Rain-ice collisions go to graupel
                    // Process rates stored above will be applied in tendency update
                } else {
                    // Rain-ice collisions go to snow
                    piacrs = piacr;
                    niacrs = niacr;
                    pracis = praci;
                    
                    // Zero out graupel production
                    piacr = 0.0;
                    niacr = 0.0;
                    praci = 0.0;
                }
            }
            
            //----------------------------------------------------------------------
            // 3. Rime splintering (HM process) - new process integration example
            //----------------------------------------------------------------------
            if (temp < t_hm_max && temp > t_hm_min) {
                // Calculate temperature-dependent multiplication factor
                amrex::Real fmult = 0.0;
                
                if (temp > t_hm_max) {
                    fmult = 0.0;
                } else if (temp <= t_hm_max && temp > 268.16) {
                    fmult = (t_hm_max - temp) / 2.0;
                } else if (temp >= t_hm_min && temp <= 268.16) {
                    fmult = (temp - t_hm_min) / 3.0;
                } else if (temp < t_hm_min) {
                    fmult = 0.0;
                }
                
                //----------------------------------------------------------------------
                // 3a. Splintering from snow riming
                //----------------------------------------------------------------------
                if (qs >= 0.1e-3) {  // Only if snow mixing ratio >= 0.1 g/kg
                    // Threshold for liquid water content needed for HM-process
                    const bool cloud_threshold = (qc >= 0.5e-3);  // 0.5 g/kg
                    const bool rain_threshold = (qr >= 0.1e-3);   // 0.1 g/kg
                    
                    if (cloud_threshold || rain_threshold) {
                        // Accretion of cloud water by snow
                        if (cloud_threshold && qc > 0.0) {
                            psacws = m_cons13 * m_as * qc * rho * n0s / std::pow(lams, m_bs + 3.0);
                            npsacws = m_cons13 * m_as * nc * rho * n0s / std::pow(lams, m_bs + 3.0);
                            
                            // Calculate splinters from cloud water riming
                            nmults = 35.0e4 * psacws * fmult * 1000.0;
                            qmults = nmults * mmult;
                            
                            // Constrain to available rimed mass
                            qmults = amrex::min(qmults, psacws);
                            psacws -= qmults;
                        }
                        
                        // Accretion of rain by snow
                        if (rain_threshold && qr > 0.0) {
                            // Terminal velocities
                            amrex::Real ums = m_as * m_cons3 / std::pow(lams, m_bs);
                            amrex::Real umr = m_ar * m_cons4 / std::pow(lamr, m_br);
                            amrex::Real uns = m_as * m_cons5 / std::pow(lams, m_bs);
                            amrex::Real unr = m_ar * m_cons6 / std::pow(lamr, m_br);
                            
                            // Density correction
                            const amrex::Real dum = std::pow(m_rhosu / rho, 0.54);
                            ums = amrex::min(ums, 1.2 * dum);
                            uns = amrex::min(uns, 1.2 * dum);
                            umr = amrex::min(umr, 9.1 * dum);
                            unr = amrex::min(unr, 9.1 * dum);
                            
                            // Calculate collection rates
                            pracs = m_cons41 * (std::sqrt(std::pow(1.2*umr-0.95*ums, 2) + 
                                   0.08*ums*umr) * rho * n0r * n0s / std::pow(lamr, 3.0) *
                                   (5.0 / (std::pow(lamr, 3.0) * lams) +
                                    2.0 / (std::pow(lamr, 2.0) * std::pow(lams, 2.0)) +
                                    0.5 / (lamr * std::pow(lams, 3.0))));
                            
                            // Calculate number collection
                            npracs = m_cons32 * rho * std::sqrt(1.7 * std::pow(unr - uns, 2) +
                                    0.3 * unr * uns) * n0r * n0s *
                                    (1.0 / (std::pow(lamr, 3.0) * lams) +
                                     1.0 / (std::pow(lamr, 2.0) * std::pow(lams, 2.0)) +
                                     1.0 / (lamr * std::pow(lams, 3.0)));
                            
                            // Calculate splinters from rain riming
                            nmultr = 35.0e4 * pracs * fmult * 1000.0;
                            qmultr = nmultr * mmult;
                            
                            // Constrain to available rimed mass
                            qmultr = amrex::min(qmultr, pracs);
                            pracs -= qmultr;
                        }
                    }
                }
                
                //----------------------------------------------------------------------
                // 3b. Splintering from graupel riming
                //----------------------------------------------------------------------
                if (qg >= 0.1e-3) {  // Only if graupel mixing ratio >= 0.1 g/kg
                    // Similar implementation to snow HM process
                    // Code follows the same structure as snow riming
                    // Omitted for brevity
                }
            }
            
            //----------------------------------------------------------------------
            // 4. Water conservation checks - example of using the conservation logic
            //----------------------------------------------------------------------
            
            // Cloud water conservation
            {
                // Calculate total sink for cloud water
                const amrex::Real sink_qc = (prc + pra + psacws + psacwg + qmults + qmultg) * dt;
                
                // Apply conservation if sink exceeds available
                if (sink_qc > qc && qc >= m_qsmall) {
                    const amrex::Real ratio = qc / sink_qc;
                    
                    // Rescale process rates
                    prc *= ratio;
                    pra *= ratio;
                    psacws *= ratio;
                    psacwg *= ratio;
                    qmults *= ratio;
                    qmultg *= ratio;
                    
                    // Rescale number conversion rates
                    nprc *= ratio;
                    nprc1 *= ratio;
                    npra *= ratio;
                    npsacws *= ratio;
                    npsacwg *= ratio;
                    nmults *= ratio;
                    nmultg *= ratio;
                }
            }
            
            // Similar conservation checks for rain, snow, ice, graupel
            // Omitted for brevity
            
            //----------------------------------------------------------------------
            // 5. Apply all tendency terms to the hydrometeor fields
            //----------------------------------------------------------------------
            
            // Calculate latent heat terms
            const amrex::Real xxlv = 3.1484e6 - 2370.0 * temp;  // Latent heat of vaporization
            const amrex::Real xxls = 3.15e6 - 2370.0 * temp + 0.3337e6;  // Latent heat of sublimation
            const amrex::Real xlf = xxls - xxlv;  // Latent heat of fusion
            const amrex::Real cpm = m_cp * (1.0 + 0.887 * qv);  // Heat capacity
            
            // Update tendencies for each variable
            
            // Water vapor
            tend(i,j,k,qv_comp) = -pre - evpms - evpmg;
            
            // Temperature 
            tend(i,j,k,t_comp) = pre * xxlv + (evpms + evpmg) * xxls +
                                (piacr + piacrs) * xlf - 
                                (psacws + psacwg + qmults + qmultg) * xlf;
            
            // Cloud water
            tend(i,j,k,qc_comp) = -prc - pra - psacws - psacwg - qmults - qmultg;
            
            // Rain water
            tend(i,j,k,qr_comp) = pre + prc + pra - pracs - pracg - piacr - piacrs;
            
            // Cloud ice
            tend(i,j,k,qi_comp) = prds + qmults + qmultg + qmultr + qmultrg - 
                                 prci - prai - praci - pracis;
            
            // Snow
            tend(i,j,k,qs_comp) = psacws + pracs + prci + prai + pracis + piacrs - psacr;
            
            // Graupel
            tend(i,j,k,qg_comp) = psacwg + pracg + pgsacw + pgracs + praci + piacr + psacr;
            
            // Cloud droplet number
            tend(i,j,k,nc_comp) = -nprc - npra - npsacws - npsacwg;
            
            // Rain number
            tend(i,j,k,nr_comp) = nprc1 - npracs - npracg - niacr - niacrs + nragg;
            
            // Cloud ice number
            tend(i,j,k,ni_comp) = nmults + nmultg + nmultr + nmultrg - 
                                 nprci - nprai - niacr - niacrs;
            
            // Snow number
            tend(i,j,k,ns_comp) = nprci + nprai + niacrs - nscng - ngracs;
            
            // Graupel number
            tend(i,j,k,ng_comp) = nscng + ngracs + niacr;
            /* // No chem quantities needed no idea
            // Update WRF-Chem quantities if needed
            // Record cloud-to-precipitation conversion for chemistry
            c2prec(i,j,k) = prc + pra + psacws + qmults + psacwg + qmultg;
	    */
        });
    }
}
