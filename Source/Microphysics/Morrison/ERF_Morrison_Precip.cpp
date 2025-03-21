#include "ERF_Morrison.H"
#include "ERF_Constants.H"
#include "ERF_EOS.H"
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_MultiFabUtil.H>
#include <cmath>

/**
 * Calculates supersaturation based on vertical velocity
 *
 * @param[in] w Vertical velocity (m/s)
 * @param[in] T Temperature (K)
 * @param[in] P Pressure (Pa)
 * @param[in] qv Water vapor mixing ratio (kg/kg)
 * @return Supersaturation ratio (0-1)
 */
amrex::Real
Morrison::calculateSupersaturation(const amrex::Real w,
                                const amrex::Real T,
                                const amrex::Real P,
                                const amrex::Real qv) const
{
    // Constants
    const amrex::Real g = 9.81;       // Gravity (m/s^2)
    const amrex::Real cp = 1005.0;    // Specific heat of air (J/kg/K)
    const amrex::Real Lv = 2.5e6;     // Latent heat of vaporization (J/kg)
    const amrex::Real Rv = 461.5;     // Gas constant for water vapor (J/kg/K)
    const amrex::Real Ra = 287.0;     // Gas constant for dry air (J/kg/K)

    // Calculate saturation vapor pressure
    amrex::Real es = calc_saturation_vapor_pressure(T, 0);  // Water saturation vapor pressure

    // Calculate saturation mixing ratio
    amrex::Real qs = 0.622 * es / (P - es);

    // Calculate derivative of saturation mixing ratio with respect to temperature
    amrex::Real dqsdt = (Lv * qs) / (Rv * T * T);

    // Calculate supersaturation using simplified formula from Abdul-Razzak et al. (1998)
    amrex::Real alpha = g * w / (Ra * T);
    amrex::Real gamma = (Ra * T) / (es * Lv * Lv) * (cp * Ra * T + Lv * Lv * qs);

    // Calculate supersaturation
    amrex::Real supersat = alpha / gamma;

    // Ensure reasonable values
    supersat = std::max(supersat, 0.0);
    supersat = std::min(supersat, 0.05);  // Cap at 5% supersaturation

    return supersat;
}

/**
 * Calculates the error function for CCN activation
 *
 * @param[in] x Input value for error function
 * @return Error function value
 */
amrex::Real
Morrison::ErrorFunction(const amrex::Real x) const
{
    // Constants for approximation
    const amrex::Real a1 = 0.254829592;
    const amrex::Real a2 = -0.284496736;
    const amrex::Real a3 = 1.421413741;
    const amrex::Real a4 = -1.453152027;
    const amrex::Real a5 = 1.061405429;
    const amrex::Real p = 0.3275911;

    // Save the sign of x
    int sign = (x < 0) ? -1 : 1;
    amrex::Real absx = std::abs(x);

    // Approximation formula
    amrex::Real t = 1.0 / (1.0 + p * absx);
    amrex::Real y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * std::exp(-absx * absx);

    return sign * y;
}

/**
 * Calculates the complementary error function (1-erf)
 *
 * @param[in] x Input value
 * @return Complementary error function value
 */
amrex::Real
Morrison::ComplementaryErrorFunction(const amrex::Real x) const
{
    return 1.0 - ErrorFunction(x);
}

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
    constexpr amrex::Real qsmall = 1.0e-6;

    // Temperature thresholds for HM-process (K)
    constexpr amrex::Real t_hm_max = 270.16;
    constexpr amrex::Real t_hm_min = 265.16;
    
    // Mass of individual splinters (kg)
    const amrex::Real mmult = 4.0/3.0 * M_PI * m_rhoi * std::pow(5.0e-6, 3);
    amrex::Real rdOcp    = m_rdOcp;

    // Loop through grids
    for (amrex::MFIter mfi(*mic_fab_vars[MicVar_Morr::tabs]); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        
        // Get array data
        auto const& thermo_tabs = mic_fab_vars[MicVar_Morr::tabs]->array(mfi);
        auto const& thermo_pres = mic_fab_vars[MicVar_Morr::pres]->array(mfi);
        auto const& hydro_qv = mic_fab_vars[MicVar_Morr::qv]->array(mfi);
        auto const& thermo_rho = mic_fab_vars[MicVar_Morr::rho]->array(mfi);
        auto const& hydro_qc = mic_fab_vars[MicVar_Morr::qcl]->array(mfi);
        auto const& hydro_qr = mic_fab_vars[MicVar_Morr::qpr]->array(mfi);
        auto const& hydro_qi = mic_fab_vars[MicVar_Morr::qci]->array(mfi);
        auto const& hydro_qs = mic_fab_vars[MicVar_Morr::qps]->array(mfi);
        auto const& hydro_qg = mic_fab_vars[MicVar_Morr::qpg]->array(mfi);
        auto const& hydro_nc = mic_fab_vars[MicVar_Morr::nc]->array(mfi);
        auto const& hydro_nr = mic_fab_vars[MicVar_Morr::nr]->array(mfi);
        auto const& hydro_ni = mic_fab_vars[MicVar_Morr::ni]->array(mfi);
        auto const& hydro_ns = mic_fab_vars[MicVar_Morr::ns]->array(mfi);
        auto const& hydro_ng = mic_fab_vars[MicVar_Morr::ng]->array(mfi);
        auto const& qn_array = mic_fab_vars[MicVar_Morr::qn]->array(mfi);
        auto const& qt_array = mic_fab_vars[MicVar_Morr::qt]->array(mfi);
        auto const& qp_array = mic_fab_vars[MicVar_Morr::qp]->array(mfi);
        auto const& theta_array = mic_fab_vars[MicVar_Morr::theta]->array(mfi);

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
            amrex::Real pccn; // CCN activation rate
            amrex::Real npsacwg;    // No idea
            amrex::Real prci;    // No idea
            amrex::Real prai;    // No idea
            amrex::Real psacr;    // No idea
            amrex::Real nprci;    // No idea
            amrex::Real nprai;    // No idea

            // Get local variables
            const amrex::Real temp = thermo_tabs(i,j,k);
            const amrex::Real pres = thermo_pres(i,j,k);
            const amrex::Real qv = hydro_qv(i,j,k);
            const amrex::Real rho = thermo_rho(i,j,k);
            const amrex::Real qc = hydro_qc(i,j,k);
            const amrex::Real qi = hydro_qi(i,j,k);
            const amrex::Real qr = hydro_qr(i,j,k);
            const amrex::Real qs = hydro_qs(i,j,k);
            const amrex::Real qg = hydro_qg(i,j,k);
            const amrex::Real nc = hydro_nc(i,j,k);
            const amrex::Real ni = hydro_ni(i,j,k);
            const amrex::Real nr = hydro_nr(i,j,k);
            const amrex::Real ns = hydro_ns(i,j,k);
            const amrex::Real ng = hydro_ng(i,j,k);

            // Initialize process rates to zero
            prc = 0.0; nprc = 0.0; nprc1 = 0.0; pra = 0.0; npra = 0.0; nragg = 0.0;
            psacws = 0.0; npsacws = 0.0; pracs = 0.0; npracs = 0.0;
            psacwg = 0.0; pracg = 0.0; npracg = 0.0; pgsacw = 0.0; pgracs = 0.0;
            nscng = 0.0; ngracs = 0.0; praci = 0.0; piacr = 0.0; niacr = 0.0;
            piacrs = 0.0; niacrs = 0.0; pracis = 0.0;
            pre = 0.0; prds = 0.0; prg = 0.0; evpms = 0.0; evpmg = 0.0;
            nmults = 0.0; nmultr = 0.0; qmults = 0.0; qmultr = 0.0;
            nmultg = 0.0; nmultrg = 0.0; qmultg = 0.0; qmultrg = 0.0;
            pccn = 0.0; npsacwg = 0.0; prci = 0.0; prai = 0.0; psacr = 0.0;
            nprci = 0.0; nprai = 0.0;

            // Calculate size distribution parameters for all hydrometeors
            amrex::Real lamc = 0.0, lamr = 0.0, lami = 0.0, lams = 0.0, lamg = 0.0;
            amrex::Real n0c = 0.0, n0r = 0.0, n0i = 0.0, n0s = 0.0, n0g = 0.0;

            // Calculate size distribution parameters
            size_distributions_params(
                                      qc, qi, qr, qs, qg,
                                      nc, ni, nr, ns, ng,
                                      rho, temp, pres,
                                      lamc, lamr, lami, lams, lamg,
                                      n0c, n0r, n0i, n0s, n0g);
// unclear where ccn should live
#if 0            
            //----------------------------------------------------------------------
            // Q Process: PCCN
            // N Process: 
            // Process: CCN Activation
            // Description: New process added for cloud droplet activation
            // Fraction: Cloud
            //----------------------------------------------------------------------
            if (temp > 273.15) {  // Only activate above freezing
                // Only activate if there's little existing cloud water
                if (qc < 0.05e-3) {
                    // Calculate effective vertical velocity (grid-scale + sub-grid)
                    // Sub-grid velocity is set to 0.5 m/s following the original scheme
                    amrex::Real w_local = thermo_rho(i,j,k);  // Vertical velocity component
                    amrex::Real w_eff = w_local;
                    if (m_isub == 0) {
                        w_eff = std::sqrt(w_local*w_local + 0.5*0.5);
                    }

                    // Only activate if upward motion
                    if (w_eff > 0.01) {
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
                        if (nc < nact) {
                            // Calculate activation rate (number/kg/s)
                            // Only activate the difference between current and activated number
                            pccn = (nact - nc) / dt;
                        }
                    }
                }
            }
#endif
            //----------------------------------------------------------------------
            // Q Process: 
            // N Process: 
            // Process: Self-collection/breakup of rain
            // Description: Self-collection/breakup of rain
            // Fraction: Cloud
            //----------------------------------------------------------------------
            // self_collection_rain
            // WRF Line 1819
            if (qr >= m_qsmall)
            {
#ifdef ERF_USE_CAM
                // Self-collection of rain drops from Beheng(1994)
                // Simple formulation without breakup consideration
                nragg = -8.0 * nr * qr * rho;
#else
                // Self-collection with breakup model from Verlinde and Cotton (1993)
                // Modified implementation based on third code block
                amrex::Real dum1 = 300.0e-6;  // 0.3 mm threshold
                amrex::Real dum;

                if (1.0/lamr < dum1) {
                    dum = 1.0;
                } else {
                    dum = 2.0 - std::exp(2300.0 * (1.0/lamr - dum1));
                }

                nragg = -5.78 * dum * nr * qr * rho;

                // Note: This replaces the originally generated formulation (possibly (Verlinde and Cotton 1993, modification))
                // amrex::Real Dcr = 1.2e-3; // 1.2 mm
                // nragg = -5.78 * (1.0 - std::exp(-2300.0 * (1.0 / lamr - Dcr))) * nr * qr * rho;
#endif
            }

            //----------------------------------------------------------------------
            // Q Process: PRC
            // N Process: NPRC
            // Process: Liquid autoconversion
            // Description: Autoconversion of cloud water to rain (Khairoutdinov and Kogan 2000)
            // Fraction: Cloud
            //----------------------------------------------------------------------
            // kk2000_liq_autoconversion
            // WRF Line 2412
            if (qc >= qsmall) {
#ifdef ERF_USE_CAM
                // CAM implementation
                // Calculate variance coefficient if using non-uniform microphysics
                Real prc_coef = 1.0;
                if (!microp_uniform) {
                    prc_coef = var_coef(relvar, 2.47);
                }

                // CAM version with different exponent (-1.1) and additional coefficients
                prc = prc_coef * 0.01 * 1350.0 * std::pow(qc, 2.47) * 
                      std::pow(nc * 1.0e-6 * rho, -1.1);
              
                // Calculate number conversion rates using CAM approach
                nprc = prc * (1.0/droplet_mass_25um);
                nprc1 = prc * nc / qc;
#else
                // Original implementation
                prc = 1350.0 * std::pow(qc, 2.47) * 
                      std::pow(nc * rho / 1.0e6, -1.79);
              
                // Calculate number conversion rates
                nprc1 = prc / m_cons29;
                nprc = prc / (qc / nc);
//printf("%d %d %d\t%24.24g, %24.24g,  %24.24g, %24.24g, %24.24g, %24.24g\n",i,j,k,         hydro_nc(i,j,k) , qc ,nc ,rho ,nprc1, nprc);
#endif
                
                // Limit by available cloud water
                nprc = amrex::min(nprc, nc / dt);
                nprc1 = amrex::min(nprc1, nprc);
            }

            //----------------------------------------------------------------------
            // Q Process: PRA
            // N Process: NPRA
            // Process: Accretion of liquid to rain
            // Description: Accretion of cloud water by rain (Khairoutdinov and Kogan 2000)
            // Fraction: Cloud
            //----------------------------------------------------------------------
            // accrete_cloud_water_rain
            // WRF Line 2801
            if (qc >= qsmall && qr >= qsmall) {
                pra = 67.0 * std::pow((qc * qr), 1.15);

                // Calculate number accretion rate
                npra = pra / (qc / nc);

                // Limit by available cloud water
                npra = amrex::min(npra, nc / dt);
            }

#if 0
            //----------------------------------------------------------------------
            // Q Process: PRACI
            // N Process: NPACR
            // Process: Ice autoconversion
            // Description: Rain-ice collisions - new process integration example
            // Fraction: Cloud
            //----------------------------------------------------------------------
            //F2855
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
            // Q Process: PSACWS
            // N Process: NSACWS
            // Process: Collection of drops by snow
            // Description: Accretion of cloud water by snow
            // Fraction: Cloud
            //----------------------------------------------------------------------
            //F2602
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
                // 4a. Splintering from snow riming
                //----------------------------------------------------------------------
                if (qs >= 0.1e-3) {  // Only if snow mixing ratio >= 0.1 g/kg
                    // Threshold for liquid water content needed for HM-process
                    const bool cloud_threshold = (qc >= 0.5e-3);  // 0.5 g/kg
                    const bool rain_threshold = (qr >= 0.1e-3);   // 0.1 g/kg
                    
                    //F2446
                    if (cloud_threshold || rain_threshold) {
                        // Accretion of cloud water by snow
                        if (cloud_threshold && qc > 0.0) {
                            psacws = m_cons13 * m_as * qc * rho * n0s / std::pow(lams, m_bs + 3.0);
                            npsacws = m_cons13 * m_as * nc * rho * n0s / std::pow(lams, m_bs + 3.0);
                            //F2620                            
                            // Calculate splinters from cloud water riming
                            nmults = 35.0e4 * psacws * fmult * 1000.0;
                            qmults = nmults * mmult;
                            
                            // Constrain to available rimed mass
                            qmults = amrex::min(qmults, psacws);
                            psacws -= qmults;
                        }
                        //F3434
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
                //F2654                
                //----------------------------------------------------------------------
                // 4b. Splintering from graupel riming
                //----------------------------------------------------------------------
                // Process splinters from graupel riming if necessary conditions are met
                if (qg >= 0.1e-3) {  // Only if graupel mixing ratio >= 0.1 g/kg
                  // Threshold for liquid water content needed for HM-process
                  const bool cloud_threshold = (qc >= 0.5e-3);  // 0.5 g/kg
                  const bool rain_threshold = (qr >= 0.1e-3);   // 0.1 g/kg

                  if (cloud_threshold || rain_threshold) {
                    // Accretion of cloud water by graupel
                    if (cloud_threshold && qc > 0.0) {
                      // Calculate splinters from cloud water riming
                      nmultg = 35.0e4 * psacwg * fmult * 1000.0;
                      qmultg = nmultg * mmult;

                      // Constrain to available rimed mass
                      qmultg = amrex::min(qmultg, psacwg);
                      psacwg -= qmultg;
                    }

                    // Accretion of rain by graupel
                    if (rain_threshold && qr > 0.0) {
                      // Calculate splinters from rain riming
                      nmultrg = 35.0e4 * pracg * fmult * 1000.0;
                      qmultrg = nmultrg * mmult;

                      // Constrain to available rimed mass
                      qmultrg = amrex::min(qmultrg, pracg);
                      pracg -= qmultrg;
                    }
                  }
                }
            }
#endif

#if 0
            //----------------------------------------------------------------------
            // Q Process: PRCI
            // N Process: NPRCI
            // Process: Collection of ice to snow
            // Description: Autoconversion of Cloud Ice to Snow
            // Fraction: Cloud
            //----------------------------------------------------------------------
            //F2352
            // Starting temperature
            amrex::Real T = thermo_tabs(i,j,k);
            
            // Calculate the saturation values at current temperature
            amrex::Real evs = std::min(0.99*thermo_pres(i,j,k), 
                                     calc_saturation_vapor_pressure(T, 0)); // Water saturation
            amrex::Real eis = std::min(0.99*thermo_pres(i,j,k), calc_saturation_vapor_pressure(temp, 1));
            if (eis > evs) eis = evs;

            // Calculate saturation mixing ratios
            amrex::Real qvs = m_ep_2 * evs / (thermo_pres(i,j,k) - evs);
            amrex::Real qvi = m_ep_2 * eis / (thermo_pres(i,j,k) - eis);
            const amrex::Real dum = m_Rv * T * T;
            const amrex::Real dqsdt = (3.1484e6 - 2370.0 * T) * qvs / dum;
            if (qi >= m_qsmall && temp < t_freezing) {
                // Calculate size distribution parameter for cloud ice
                if (ni >= m_qsmall)
                {
                    lami = std::pow(m_cons12 * ni / qi, 1.0 / m_di);

                    // Apply limits to lambda
                    lami = amrex::max(lami, m_lammini);
                    lami = amrex::min(lami, m_lammaxi);

                    // Check if mean size exceeds threshold (2*DCS)
                    if (1.0 / lami >= 2.0 * m_dcs) {
                        // Autoconversion rate (mass)
                        prci = m_cons21 * (qv - qvi) * rho * n0i *
                            std::exp(-lami * m_dcs) * m_fac_sub / (1.0 + m_fac_sub * dqsdt);

                        // Autoconversion rate (number)
                        nprci = prci / m_cons22;

                        // Limit by available cloud ice
                        nprci = amrex::min(nprci, ni / dt);
                    }
                }
            }

            //----------------------------------------------------------------------
            // Q Process: PRAI
            // N Process: NPRAI
            // Process: Collection of ice to snow
            // Description: Collection of cloud ice by snow
            // Fraction: Cloud
            //----------------------------------------------------------------------
            //F2377
            if (qi >= m_qsmall && qs >= m_qsmall) {
                // PRAI: Rate of change of snow mixing ratio due to collection of cloud ice
                prai = m_cons23 * m_as * qi * rho * n0s / std::pow(lams, m_bs + 3.0);

                // NPRAI: Rate of change of snow number concentration due to collection of cloud ice
                nprai = m_cons23 * m_as * ni * rho * n0s / std::pow(lams, m_bs + 3.0);

                // Limit NPRAI by the available cloud ice number concentration
                nprai = amrex::min(nprai, ni / dt);
            }

            //----------------------------------------------------------------------
            // Q Process: PGSACW
            // N Process: NSCNG
            // Process: Collection of drops by snow
            // Description: Conversion of Rimed Cloud Water to Graupel
            // Fraction: Cloud
            //----------------------------------------------------------------------
            //F2477
            if (qs >= m_qsmall && qc >= m_qsmall) {
                // Only convert if snow and cloud water mixing ratios are significant
                if (qs >= 0.1e-3 && qc >= 0.5e-3) {
                    // Calculate mass-weighted fall speeds (already done in PrecipFall)
                    amrex::Real ums = m_as * m_cons3 / std::pow(lams, m_bs);

                    // Density correction (already done in PrecipFall)
                    const amrex::Real dum = std::pow(m_rhosu / rho, 0.54);
                    ums = amrex::min(ums, 1.2 * dum);

                    // Calculate collection rate of cloud water by snow (already done in psacws)
                    psacws = m_cons13 * m_as * qc * rho * n0s / std::pow(lams, m_bs + 3.0);

                    // Calculate portion of riming converted to graupel (Reisner et al. 1998)
                    pgsacw = amrex::min(psacws, m_cons17 * dt * n0s * qc * qc *
                                       m_as * m_as / (rho * std::pow(lams, 2.0 * m_bs + 2.0)));

                    // Calculate mass of embryo graupel (already initialized)
                    // Calculate number concentration of embryo graupel from riming of snow
                    nscng = (m_rhosn / (m_rhog - m_rhosn) * pgsacw) / m_mg0 * rho;

                    // Limit by available snow number
                    nscng = amrex::min(nscng, ns / dt);

                    // Remaining riming goes to snow
                    psacws -= pgsacw;
                }
            }
#endif
            //----------------------------------------------------------------------
            // Q Process: 
            // N Process: 
            // Process: Water conservation
            // Description: Apply all tendency terms to the hydrometeor fields
            // Fraction: Cloud
            //----------------------------------------------------------------------
            //F1938 
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
            // Rain water conservation
            {
                // Calculate total sink for rain water
                const amrex::Real sink_qr = (pracs + pracg + piacr + piacrs) * dt;

                // Apply conservation if sink exceeds available
                if (sink_qr > qr && qr >= m_qsmall) {
                    const amrex::Real ratio = qr / sink_qr;

                    // Rescale process rates
                    pracs *= ratio;
                    pracg *= ratio;
                    piacr *= ratio;
                    piacrs *= ratio;

                    // Rescale number conversion rates
                    npracs *= ratio;
                    npracg *= ratio;
                    niacr *= ratio;
                    niacrs *= ratio;
                }
            }
#if 0
            // Cloud ice conservation
            {
                // Calculate total sink for cloud ice
                const amrex::Real sink_qi = (prci + prai + praci + pracis) * dt;

                // Apply conservation if sink exceeds available
                if (sink_qi > qi && qi >= m_qsmall) {
                    const amrex::Real ratio = qi / sink_qi;

                    // Rescale process rates
                    prci *= ratio;
                    prai *= ratio;
                    praci *= ratio;
                    pracis *= ratio;

                    // Rescale number conversion rates
                    nprci *= ratio;
                    nprai *= ratio;
                    niacr *= ratio; // Shared with rain
                    niacrs *= ratio; // Shared with rain
                }
            }

            // Snow conservation
            {
                // Calculate total sink for snow
                const amrex::Real sink_qs = (psacr) * dt;

                // Apply conservation if sink exceeds available
                if (sink_qs > qs && qs >= m_qsmall) {
                    const amrex::Real ratio = qs / sink_qs;

                    // Rescale process rates
                    psacr *= ratio;
                }
            }

            // Graupel conservation
            {
                // Calculate total sink for graupel
                const amrex::Real sink_qg = (pgsacw + pgracs) * dt;

                // Apply conservation if sink exceeds available
                if (sink_qg > qg && qg >= m_qsmall) {
                    const amrex::Real ratio = qg / sink_qg;

                    // Rescale process rates
                    pgsacw *= ratio;
                    pgracs *= ratio;

                    // Rescale number conversion rates (nscng and ngracs are tendencies, not rates)
                    // No direct scaling needed here, as they are derived from mass tendencies
                }
            }

            //----------------------------------------------------------------------
            // Q Process: PRE
            // N Process: NSUBR
            // Process: Rain evaporation
            // Description: Accelerated Melting of Snow and Graupel due to Rain Collisions
            // Fraction: Precip
            //----------------------------------------------------------------------
            //F3950
            // Only apply if above freezing
            if (temp >= t_freezing) {
                const amrex::Real xxlv = 3.1484e6 - 2370.0 * temp;  // Latent heat of vaporization
                const amrex::Real xxls = 3.15e6 - 2370.0 * temp + 0.3337e6;  // Latent heat of sublimation
                const amrex::Real xlf = xxls - xxlv;  // Latent heat of fusion
                const amrex::Real cpm = m_cp * (1.0 + 0.887 * qv);  // Heat capacity

                //------------------------------------------------------------------
                // 14a. Accelerated Melting of Snow
                //------------------------------------------------------------------
                if (qs >= m_qsmall && qr >= m_qsmall) {
                    // Calculate size distribution parameters (already done above)

                    // Calculate mass-weighted fall speeds (already done in PrecipFall)
                    amrex::Real ums = m_as * m_cons3 / std::pow(lams, m_bs);
                    amrex::Real umr = m_ar * m_cons4 / std::pow(lamr, m_br);

                    // Density correction (already done in PrecipFall)
                    const amrex::Real dum = std::pow(m_rhosu / rho, 0.54);
                    ums = amrex::min(ums, 1.2 * dum);
                    umr = amrex::min(umr, 9.1 * dum);

                    // Calculate collection rate of rain by snow (already done in pracs)
                    // We reuse the pracs calculation, but it represents
                    // the mass of *rain* collected by snow, not the other way around.
                    pracs = m_cons41 * (std::sqrt(std::pow(1.2*umr-0.95*ums, 2) +
                                   0.08*ums*umr) * rho * n0r * n0s / std::pow(lamr, 3.0) *
                                   (5.0 / (std::pow(lamr, 3.0) * lams) +
                                    2.0 / (std::pow(lamr, 2.0) * std::pow(lams, 2.0)) +
                                    0.5 / (lamr * std::pow(lams, 3.0))));

                    // Calculate accelerated melting rate due to rain collisions
                    // Use Celsius for temperature, as in the Fortran code
                    amrex::Real psmlt_accel = -m_cpw / xlf * (temp - t_freezing) * pracs;

                    // Limit melting by available snow
                    psmlt_accel = std::max(psmlt_accel, -qs / dt);

                    // Apply melting tendency
                    prds += psmlt_accel;  // Add to existing snow deposition/sublimation
                    pre += psmlt_accel; // Add to rain production

                    // Apply latent cooling directly to temperature
                    thermo_tabs(i,j,k) -= psmlt_accel * xlf / cpm * dt;

                }

                //------------------------------------------------------------------
                // 14b. Accelerated Melting of Graupel
                //------------------------------------------------------------------
                if (qg >= m_qsmall && qr >= m_qsmall) {
                    // Calculate size distribution parameters (already done above)

                    // Calculate mass-weighted fall speeds (already done in PrecipFall)
                    amrex::Real umg = m_ag * m_cons7 / std::pow(lamg, m_bg);
                    amrex::Real umr = m_ar * m_cons4 / std::pow(lamr, m_br);

                    // Density correction (already done in PrecipFall)
                    const amrex::Real dum = std::pow(m_rhosu / rho, 0.54);
                    umg = amrex::min(umg, 20.0 * dum);
                    umr = amrex::min(umr, 9.1 * dum);

                    // Calculate collection rate of rain by graupel (already done in pracg)
                    // We reuse the pracg calculation.
                    pracg = m_cons41 * (std::sqrt(std::pow(1.2*umr-0.95*umg, 2) +
                                   0.08*umg*umr) * rho * n0r * n0g / std::pow(lamr, 3.0) *
                                   (5.0 / (std::pow(lamr, 3.0) * lamg) +
                                    2.0 / (std::pow(lamr, 2.0) * std::pow(lamg, 2.0)) +
                                    0.5 / (lamr * std::pow(lamg, 3.0))));

                    // Calculate accelerated melting rate due to rain collisions
                    // Use Celsius for temperature, as in the Fortran code
                    amrex::Real pgmlt_accel = -m_cpw / xlf * (temp - t_freezing) * pracg;

                    // Limit melting by available graupel
                    pgmlt_accel = std::max(pgmlt_accel, -qg / dt);

                    // Apply melting tendency
                    prg += pgmlt_accel;   // Add to existing graupel deposition/sublimation
                    pre += pgmlt_accel; // Add to rain

                    // Apply latent cooling
                    thermo_tabs(i,j,k) -= pgmlt_accel * xlf / cpm;
                }
            }           
#endif
            //----------------------------------------------------------------------
            // Q Process: 
            // N Process: 
            // Process: Water conservation
            // Description: Apply all tendency terms to the hydrometeor fields
            // Fraction: Cloud
            //----------------------------------------------------------------------
            //F1296 
            // Calculate latent heat terms
            const amrex::Real xxlv = 3.1484e6 - 2370.0 * temp;  // Latent heat of vaporization
            const amrex::Real xxls = 3.15e6 - 2370.0 * temp + 0.3337e6;  // Latent heat of sublimation
            const amrex::Real xlf = xxls - xxlv;  // Latent heat of fusion
            const amrex::Real cpm = m_cp * (1.0 + 0.887 * qv);  // Heat capacity
            
            // Update state variables with computed tendencies
            // Water vapor
            hydro_qv(i,j,k) += ( -pre - evpms - evpmg ) * dt;
            hydro_qv(i,j,k) = amrex::max(hydro_qv(i,j,k), 0.0);

            // Temperature
            thermo_tabs(i,j,k) += ( pre * xxlv + (evpms + evpmg) * xxls +
                                   (piacr + piacrs) * xlf -
                                   (psacws + psacwg + qmults + qmultg) * xlf ) * dt;

            // Update potential temperature (theta)
            theta_array(i,j,k) = getThgivenPandT(
                    thermo_tabs(i,j,k),  // Updated T
                    100.0 * thermo_pres(i,j,k),  // Convert pressure to Pascals
                    rdOcp  // Ratio of R/cp
                );

            // Cloud water
            hydro_qc(i,j,k) += ( -prc - pra - psacws - psacwg - qmults - qmultg ) * dt;
            hydro_qc(i,j,k) = amrex::max(hydro_qc(i,j,k), 0.0);

            // Rain water
            hydro_qr(i,j,k) += ( pre + prc + pra - pracs - pracg - piacr - piacrs ) * dt;
            hydro_qr(i,j,k) = amrex::max(hydro_qr(i,j,k), 0.0);
#if 0
            // Cloud ice
            hydro_qi(i,j,k) += ( prds + qmults + qmultg + qmultr + qmultrg -
                                 prci - prai - praci - pracis ) * dt;
            hydro_qi(i,j,k) = amrex::max(hydro_qi(i,j,k), 0.0);

            // Snow
            hydro_qs(i,j,k) += ( psacws + pracs + prci + prai + pracis + piacrs - psacr ) * dt;
            hydro_qs(i,j,k) = amrex::max(hydro_qs(i,j,k), 0.0);

            // Graupel
            hydro_qg(i,j,k) += ( psacwg + pracg + pgsacw + pgracs + praci + piacr + psacr ) * dt;
            hydro_qg(i,j,k) = amrex::max(hydro_qg(i,j,k), 0.0);
#endif
            // Cloud droplet number

//            printf("%d %d %d\t%24.24g, %24.24g,  %24.24g, %24.24g, %24.24g, %24.24g\n",i,j,k,         hydro_nc(i,j,k) , -nprc ,- npra ,- npsacws ,- npsacwg , dt);
            hydro_nc(i,j,k) += ( -nprc - npra - npsacws - npsacwg ) * dt;
            hydro_nc(i,j,k) = amrex::max(hydro_nc(i,j,k), 0.0);

            // Rain number
            hydro_nr(i,j,k) += ( nprc1 - npracs - npracg - niacr - niacrs + nragg ) * dt;
            hydro_nr(i,j,k) = amrex::max(hydro_nr(i,j,k), 0.0);
#if 0
            // Cloud ice number
            hydro_ni(i,j,k) += ( nmults + nmultg + nmultr + nmultrg -
                                 nprci - nprai - niacr - niacrs ) * dt;
            hydro_ni(i,j,k) = amrex::max(hydro_ni(i,j,k), 0.0);

            // Snow number
            hydro_ns(i,j,k) += ( nprci + nprai + niacrs - nscng - ngracs ) * dt;
            hydro_ns(i,j,k) = amrex::max(hydro_ns(i,j,k), 0.0);

            // Graupel number
            hydro_ng(i,j,k) += ( nscng + ngracs + niacr ) * dt;
            hydro_ng(i,j,k) = amrex::max(hydro_ng(i,j,k), 0.0);
#endif
            // Update derived quantities
            qn_array(i,j,k) = hydro_qc(i,j,k) + hydro_qi(i,j,k);
            qt_array(i,j,k) = hydro_qv(i,j,k) + qn_array(i,j,k);
            qp_array(i,j,k) = hydro_qr(i,j,k) + hydro_qs(i,j,k) + hydro_qg(i,j,k);

            // Set very small values to zero to avoid numerical issues
            if (hydro_qc(i,j,k) < m_qsmall) {
                hydro_qc(i,j,k) = 0.0;
                hydro_nc(i,j,k) = 0.0;
            }

            if (hydro_qi(i,j,k) < m_qsmall) {
                hydro_qi(i,j,k) = 0.0;
                hydro_ni(i,j,k) = 0.0;
            }

            if (hydro_qr(i,j,k) < m_qsmall) {
                hydro_qr(i,j,k) = 0.0;
                hydro_nr(i,j,k) = 0.0;
            }

            if (hydro_qs(i,j,k) < m_qsmall) {
                hydro_qs(i,j,k) = 0.0;
                hydro_ns(i,j,k) = 0.0;
            }

            if (hydro_qg(i,j,k) < m_qsmall) {
                hydro_qg(i,j,k) = 0.0;
                hydro_ng(i,j,k) = 0.0;
            }
            /* // No chem quantities needed no idea
            // Update WRF-Chem quantities if needed
            // Record cloud-to-precipitation conversion for chemistry
            c2prec(i,j,k) = prc + pra + psacws + qmults + psacwg + qmultg;
            */
        });
    }
}
