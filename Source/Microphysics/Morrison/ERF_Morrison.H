/*
 * Implementation of a 2-moment microphysics model
 * This model is based on the Morrison model as implemented in WRF (in Fortran)
 */
#ifndef ERF_Morrison_H
#define ERF_Morrison_H

#include <string>
#include <vector>
#include <memory>
#include <complex>

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_TableData.H>
#include <AMReX_MultiFabUtil.H>

#include "ERF_Constants.H"
#include "ERF_MicrophysicsUtils.H"
#include "ERF_IndexDefines.H"
#include "ERF_DataStruct.H"
#include "ERF_NullMoist.H"

namespace MicVar_Morr {
   enum {
      // independent variables
      rho=0, // density
      theta, // liquid/ice water potential temperature
      tabs,  // temperature
      pres,  // pressure
      // non-precipitating vars
      qt,    // total cloud
      qn,    // cloud condesnate (liquid + ice)
      qv,    // cloud vapor
      qcl,   // cloud water
      qci,   // cloud ice
      // precipitating vars
      qp,    // total precip
      qpr,   // precip rain
      qps,   // precip ice/snow
      qpg,   // graupel
      // number concentrations
      nc,    // cloud droplet number
      nr,    // rain number
      ni,    // cloud ice number
      ns,    // snow number
      ng,    // graupel number
      // derived vars
      rain_accum,
      snow_accum,
      graup_accum,
      omega,
      NumVars
  };
}

class Morrison : public NullMoist {

    using FabPtr = std::shared_ptr<amrex::MultiFab>;

public:
    // constructor
    Morrison () {}

    // destructor
    virtual ~Morrison () = default;

    // cloud physics
    void Cloud (const SolverChoice& sc);

    void SublimationDeposition (const SolverChoice& sc);

    // ice physics
    void IceFall (const SolverChoice& sc);

    // precip
    void Precip (const SolverChoice& sc);

    // precip fall
    void PrecipFall (const SolverChoice& sc);

    // Set up for first time
    void
    Define (SolverChoice& sc) override
    {
        m_fac_cond = lcond / sc.c_p;
        m_fac_fus  = lfus / sc.c_p;
        m_fac_sub  = lsub / sc.c_p;
        m_gOcp     = CONST_GRAV / sc.c_p;
        m_axis     = sc.ave_plane;
        m_rdOcp    = sc.rdOcp;
    }

    // init
    void
    Init (const amrex::MultiFab& cons_in,
          const amrex::BoxArray& grids,
          const amrex::Geometry& geom,
          const amrex::Real& dt_advance,
          std::unique_ptr<amrex::MultiFab>& z_phys_nd,
          std::unique_ptr<amrex::MultiFab>& detJ_cc) override;

    // Copy state into micro vars
    void
    Copy_State_to_Micro (const amrex::MultiFab& cons_in) override;

    // Copy state into micro vars
    void
    Copy_Micro_to_State (amrex::MultiFab& cons_in) override;

    void
    Update_Micro_Vars (amrex::MultiFab& cons_in) override
    {
        this->Copy_State_to_Micro(cons_in);
        this->initialize_size_distributions();
        this->Compute_Coefficients();
    }

    void
    Update_State_Vars (amrex::MultiFab& cons_in) override
    {
        this->Copy_Micro_to_State(cons_in);
    }

  
    // wrapper to do all the updating
    void
    Advance (const amrex::Real& dt_advance,
             const SolverChoice& sc) override;
    /*
    {
        dt = dt_advance;

        this->Cloud(sc);
        this->IceFall(sc);
        this->Precip(sc);
        this->PrecipFall(sc);

    }
    */
    
    amrex::MultiFab*
    Qmoist_Ptr (const int& varIdx) override
    {
        AMREX_ALWAYS_ASSERT(varIdx < m_qmoist_size);
        return mic_fab_vars[MicVarMap[varIdx]].get();
    }

    void
    Compute_Coefficients ();

    /**
     * Applies accumulated tendencies to state variables.
     * This function updates the state variables based on the tendencies
     * calculated during the microphysics processes.
     */
    void ApplyTendencies ();

    int
    Qmoist_Size () override { return Morrison::m_qmoist_size; }

    int
    Qstate_Size () override { return Morrison::m_qstate_size; }

    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::Real
    NewtonIterSat (int& i, int& j, int& k,
                   const int& model_type,
                   const amrex::Real& fac_cond,
                   const amrex::Real& fac_fus,
                   const amrex::Real& fac_sub,
                   const amrex::Real& an,
                   const amrex::Real& bn,
                   const amrex::Array4<amrex::Real>& tabs_array,
                   const amrex::Array4<amrex::Real>& pres_array,
                   const amrex::Array4<amrex::Real>& qv_array,
                   const amrex::Array4<amrex::Real>& qc_array,
                   const amrex::Array4<amrex::Real>& qi_array,
                   const amrex::Array4<amrex::Real>& qn_array,
                   const amrex::Array4<amrex::Real>& qt_array)
    {
        // Solution tolerance
        amrex::Real tol = 1.0e-6;

        // Saturation moisture fractions
        amrex::Real omn, domn;
        amrex::Real qsat, dqsat;
        amrex::Real qsatw, dqsatw;
        amrex::Real qsati, dqsati;

        // Newton iteration vars
        int  niter;
        amrex::Real fff, dfff, dtabs;
        amrex::Real lstar, dlstar;
        amrex::Real lstarw, lstari;
        amrex::Real delta_qv, delta_qc, delta_qi;

        // Initial guess for temperature & pressure
        amrex::Real tabs = tabs_array(i,j,k);
        amrex::Real pres = pres_array(i,j,k);

        niter = 0;
        dtabs = 1;
        //==================================================
        // Newton iteration to qv=qsat (cloud phase only)
        //==================================================
        do {
            // Latent heats and their derivatives wrt to T
            lstarw  = fac_cond;
            lstari  = fac_sub;
            domn    = 0.0;

            // Saturation moisture fractions
            erf_qsatw(tabs, pres, qsatw);
            erf_qsati(tabs, pres, qsati);
            erf_dtqsatw(tabs, pres, dqsatw);
            erf_dtqsati(tabs, pres, dqsati);

            if (model_type ==1) {
                // Cloud ice not permitted (condensation & fusion)
                if(tabs >= tbgmax) {
                    omn   = 1.0;
                }
                // Cloud water not permitted (sublimation & fusion)
                else if(tabs <= tbgmin) {
                    omn    = 0.0;
                    lstarw = fac_sub;
                }
                // Mixed cloud phase (condensation & fusion)
                else {
                    omn   = an*tabs-bn;
                    domn  = an;
                }
            } else {
                omn  = 1.0;
                domn = 0.0;
            }

            // Linear combination of each component
            qsat   =  omn * qsatw  + (1.0-omn) * qsati;
            dqsat  =  omn * dqsatw + (1.0-omn) * dqsati
                   + domn *  qsatw -     domn  *  qsati;
            lstar  =  omn * lstarw + (1.0-omn) * lstari;
            dlstar = domn * lstarw -     domn  * lstari;

            // Function for root finding:
            // 0 = -T_new + T_old + L_eff/C_p * (qv - qsat)
            fff   = -tabs + tabs_array(i,j,k) +  lstar*(qv_array(i,j,k) - qsat);

            // Derivative of function (T_new iterated on)
            dfff  = -1.0 + dlstar*(qv_array(i,j,k) - qsat) - lstar*dqsat;

            // Update the temperature
            dtabs = -fff/dfff;
            tabs += dtabs;

            // Update iteration
            niter = niter+1;
        } while(std::abs(dtabs) > tol && niter < 20);

        // Update qsat from last iteration (dq = dq/dt * dt)
        qsat += dqsat*dtabs;

        // Changes in each component
        delta_qv = qv_array(i,j,k) - qsat;
        delta_qc = std::max(-qc_array(i,j,k), delta_qv * omn);
        delta_qi = std::max(-qi_array(i,j,k), delta_qv * (1.0-omn));

        // Partition the change in non-precipitating q
        qv_array(i,j,k)  = qsat;
        qc_array(i,j,k) += delta_qc;
        qi_array(i,j,k) += delta_qi;
        qn_array(i,j,k)  = qc_array(i,j,k) + qi_array(i,j,k);
        qt_array(i,j,k)  = qv_array(i,j,k) + qn_array(i,j,k);

        // Return to temperature
        return tabs;
    }

    void
    Qmoist_Restart_Vars (const SolverChoice& /*a_sc*/,
                         std::vector<int>& a_idx,
                         std::vector<std::string>& a_names) const override
    {
        a_idx.clear();
        a_names.clear();
        a_idx.push_back( 8); a_names.push_back("RainAccum");
        a_idx.push_back( 9); a_names.push_back("SnowAccum");
        a_idx.push_back(10); a_names.push_back("GraupAccum");
    }

    /**
     * Activates cloud droplets from CCN
     * Based on the specified activation scheme (power law or lognormal)
     *
     * @param[in] sc SolverChoice containing solver configuration
     */
    void ActivateCloudDroplets (const SolverChoice& sc);
    
    /**
     * Performs water conservation checks for microphysical processes
     * Ensures that process rates don't deplete more water than is available
     * 
     * @param[in,out] process_rates Array of process rates to be adjusted
     * @param[in] available_water Available water for depletion (kg/kg)
     * @param[in] dt_in Timestep (s)
     * @return Scaling factor applied to process rates
     */
    amrex::Real EnforceWaterConservation (amrex::Vector<amrex::Real>& process_rates,
                                          const amrex::Real available_water,
                                          const amrex::Real dt_in);
    
    /**
     * Computes radar reflectivity from hydrometeor properties.
     * 
     */
    void ComputeRadarReflectivity ();

   /**
     * Calculates supersaturation based on vertical velocity
     *
     * @param[in] w Vertical velocity (m/s)
     * @param[in] T Temperature (K)
     * @param[in] P Pressure (Pa)
     * @param[in] qv Water vapor mixing ratio (kg/kg)
     * @return Supersaturation ratio (0-1)
     */
    amrex::Real calculateSupersaturation (const amrex::Real w, 
                                          const amrex::Real T, 
                                          const amrex::Real P, 
                                          const amrex::Real qv) const;
    
    /**
     * Calculates the error function for CCN activation
     * 
     * @param[in] x Input value for error function
     * @return Error function value
     */
    amrex::Real ErrorFunction (const amrex::Real x) const;
    
    /**
     * Calculates the complementary error function (1-erf)
     * 
     * @param[in] x Input value
     * @return Complementary error function value
     */
    amrex::Real ComplementaryErrorFunction (const amrex::Real x) const;

private:
    // Number of qmoist variables (qt, qv, qcl, qci, qp, qpr, qps, qpg)
    int m_qmoist_size = 11;

    // Number of qstate variables
    int m_qstate_size = 6;

    // CFL MAX for vertical advection
    static constexpr amrex::Real CFL_MAX = 0.5;

    // MicVar map (Qmoist indices -> MicVar enum)
    amrex::Vector<int> MicVarMap;

    // geometry
    amrex::Geometry m_geom;

    // valid boxes on which to evolve the solution
    amrex::BoxArray m_gtoe;

    // timestep
    amrex::Real dt;

    // number of vertical levels
    int nlev, zlo, zhi;

    // plane average axis
    int m_axis;

    // constants
    amrex::Real m_fac_cond;
    amrex::Real m_fac_fus;
    amrex::Real m_fac_sub;
    amrex::Real m_gOcp;
    amrex::Real m_rdOcp;

    // Pointer to terrain data
    amrex::MultiFab* m_z_phys_nd;
    amrex::MultiFab* m_detJ_cc;

    // independent variables
    amrex::Array<FabPtr, MicVar_Morr::NumVars> mic_fab_vars;

    // microphysics parameters/coefficients
    amrex::TableData<amrex::Real, 1> accrrc;
    amrex::TableData<amrex::Real, 1> accrsi;
    amrex::TableData<amrex::Real, 1> accrsc;
    amrex::TableData<amrex::Real, 1> coefice;
    amrex::TableData<amrex::Real, 1> evaps1;
    amrex::TableData<amrex::Real, 1> evaps2;
    amrex::TableData<amrex::Real, 1> accrgi;
    amrex::TableData<amrex::Real, 1> accrgc;
    amrex::TableData<amrex::Real, 1> evapg1;
    amrex::TableData<amrex::Real, 1> evapg2;
    amrex::TableData<amrex::Real, 1> evapr1;
    amrex::TableData<amrex::Real, 1> evapr2;

    // vertical plane average data
    amrex::TableData<amrex::Real, 1> rho1d;
    amrex::TableData<amrex::Real, 1> pres1d;
    amrex::TableData<amrex::Real, 1> tabs1d;
    amrex::TableData<amrex::Real, 1> qt1d;
    amrex::TableData<amrex::Real, 1> qv1d;
    amrex::TableData<amrex::Real, 1> qn1d;

    amrex::TableData<amrex::Real, 1> gamaz;
    amrex::TableData<amrex::Real, 1> zmid; // mid value of vertical coordinate in physical domain

    // Microphysics options/switches
    int m_iact = 2;    // CCN activation option (1: power-law, 2: lognormal aerosol)
    int m_inum = 1;    // Droplet number option (0: predict, 1: constant)
    int m_iliq = 0;    // Liquid-only option (0: include ice, 1: liquid only) 
    int m_inuc = 0;    // Ice nucleation option (0: mid-latitude, 1: arctic)
    int m_ibase = 2;   // Cloud base activation option
    int m_isub = 0;    // Sub-grid vertical velocity option
    int m_igraup = 0;  // Graupel option (0: include graupel, 1: no graupel)
    int m_ihail = 0;   // Graupel/hail option (0: graupel, 1: hail)
    bool m_do_radar_ref = false;  // Radar reflectivity calculation flag
    
    // Physical constants
    amrex::Real m_pi;          // Pi constant
    amrex::Real m_R;           // Gas constant for dry air (J/kg/K)
    amrex::Real m_Rv;          // Gas constant for water vapor (J/kg/K)
    amrex::Real m_cp;          // Specific heat at constant pressure (J/kg/K)
    amrex::Real m_g;           // Gravitational acceleration (m/s^2)
    amrex::Real m_ep_2;        // Molecular weight ratio (Rd/Rv)
    
    // Reference density and species densities
    amrex::Real m_rhosu;       // Standard air density at 850 mb (kg/m^3)
    amrex::Real m_rhow;        // Density of liquid water (kg/m^3)
    amrex::Real m_rhoi;        // Bulk density of cloud ice (kg/m^3)
    amrex::Real m_rhosn;       // Bulk density of snow (kg/m^3)
    amrex::Real m_rhog;        // Bulk density of graupel/hail (kg/m^3)
    
    // Fall speed parameters (V=AD^B)
    amrex::Real m_ai, m_bi;    // Cloud ice fall speed parameters
    amrex::Real m_ac, m_bc;    // Cloud droplet fall speed parameters
    amrex::Real m_as, m_bs;    // Snow fall speed parameters
    amrex::Real m_ar, m_br;    // Rain fall speed parameters
    amrex::Real m_ag, m_bg;    // Graupel/hail fall speed parameters
    
    // Microphysical parameters
    amrex::Real m_aimm;        // Parameter in Bigg immersion freezing
    amrex::Real m_bimm;        // Parameter in Bigg immersion freezing
    amrex::Real m_ecr;         // Collection efficiency between droplets/rain and snow/rain
    amrex::Real m_dcs;         // Threshold size for cloud ice autoconversion (m)
    amrex::Real m_mi0;         // Initial mass of nucleated ice crystal (kg)
    amrex::Real m_mg0;         // Mass of embryo graupel (kg)
    amrex::Real m_f1s;         // Ventilation parameter for snow
    amrex::Real m_f2s;         // Ventilation parameter for snow
    amrex::Real m_f1r;         // Ventilation parameter for rain
    amrex::Real m_f2r;         // Ventilation parameter for rain
    amrex::Real m_qsmall;      // Smallest allowed hydrometeor mixing ratio
    amrex::Real m_eii;         // Collection efficiency, ice-ice collisions
    amrex::Real m_eci;         // Collection efficiency, ice-droplet collisions
    amrex::Real m_cpw;         // Specific heat of liquid water (J/kg/K)
    amrex::Real m_rin;         // Radius of contact nuclei (m)
    amrex::Real m_mmult;       // Mass of splintered ice particle (kg)
    
    // Size distribution parameters
    amrex::Real m_ci, m_di;    // Cloud ice size distribution parameters
    amrex::Real m_cs, m_ds;    // Snow size distribution parameters
    amrex::Real m_cg, m_dg;    // Graupel size distribution parameters
    
    // Lambda limits for size distributions
    amrex::Real m_lammaxi, m_lammini;    // Cloud ice lambda limits
    amrex::Real m_lammaxr, m_lamminr;    // Rain lambda limits
    amrex::Real m_lammaxs, m_lammins;    // Snow lambda limits
    amrex::Real m_lammaxg, m_lamming;    // Graupel lambda limits
    
    // Constant droplet concentration (if INUM = 1)
    amrex::Real m_ndcnst = 250.0;  // Droplet number concentration (cm^-3)
    
    // CCN spectra parameters (for IACT = 1)
    amrex::Real m_k1;          // Exponent in CCN activation formula
    amrex::Real m_c1;          // Coefficient in CCN activation formula (cm^-3)
    
    // Aerosol activation parameters (for IACT = 2)
    amrex::Real m_mw;          // Molecular weight water (kg/mol)
    amrex::Real m_osm;         // Osmotic coefficient
    amrex::Real m_vi;          // Number of ions dissociated in solution
    amrex::Real m_epsm;        // Aerosol soluble fraction
    amrex::Real m_rhoa;        // Aerosol bulk density (kg/m^3)
    amrex::Real m_map;         // Molecular weight aerosol (kg/mol)
    amrex::Real m_ma;          // Molecular weight of air (kg/mol)
    amrex::Real m_rr;          // Universal gas constant (J/mol/K)
    amrex::Real m_bact;        // Activation parameter
    amrex::Real m_rm1;         // Geometric mean radius, mode 1 (m)
    amrex::Real m_rm2;         // Geometric mean radius, mode 2 (m)
    amrex::Real m_nanew1;      // Total aerosol concentration, mode 1 (m^-3)
    amrex::Real m_nanew2;      // Total aerosol concentration, mode 2 (m^-3)
    amrex::Real m_sig1;        // Standard deviation of aerosol dist, mode 1
    amrex::Real m_sig2;        // Standard deviation of aerosol dist, mode 2
    amrex::Real m_f11;         // Correction factor for activation, mode 1
    amrex::Real m_f12;         // Correction factor for activation, mode 1
    amrex::Real m_f21;         // Correction factor for activation, mode 2
    amrex::Real m_f22;         // Correction factor for activation, mode 2
    
    // Precomputed constants for efficiency
    amrex::Real m_cons1, m_cons2, m_cons3, m_cons4, m_cons5;
    amrex::Real m_cons6, m_cons7, m_cons8, m_cons9, m_cons10;
    amrex::Real m_cons11, m_cons12, m_cons13, m_cons14, m_cons15;
    amrex::Real m_cons16, m_cons17, m_cons18, m_cons19, m_cons20;
    amrex::Real m_cons21, m_cons22, m_cons23, m_cons24, m_cons25;
    amrex::Real m_cons26, m_cons27, m_cons28, m_cons29, m_cons30;
    amrex::Real m_cons31, m_cons32, m_cons33, m_cons34, m_cons35;
    amrex::Real m_cons36, m_cons37, m_cons38, m_cons39, m_cons40;
    amrex::Real m_cons41;
    
    // Radar reflectivity parameters
    amrex::Real m_xam_r, m_xbm_r, m_xmu_r;  // Rain reflectivity parameters
    amrex::Real m_xam_s, m_xbm_s, m_xmu_s;  // Snow reflectivity parameters
    amrex::Real m_xam_g, m_xbm_g, m_xmu_g;  // Graupel reflectivity parameters
    amrex::Real m_lambda_radar;  // Radar wavelength (10 cm)
    amrex::Real m_k_w;           // K_w parameter for liquid water
    amrex::Real m_lamda4;        // Lambda^4 for radar calculation
    amrex::Real m_pi5;           // Pi^5 for radar calculation
    
    // Radar arrays for Simpson integration
    amrex::Vector<amrex::Real> m_xxds, m_xxdg;     // Diameter values
    amrex::Vector<amrex::Real> m_xdts, m_xdtg;     // Diameter increments
    amrex::Vector<amrex::Real> m_simpson;          // Simpson's rule weights
    
    // Parameters for radar melting calculations
    bool m_melt_outside_s;     // Liquid coating of melting snow
    bool m_melt_outside_g;     // Liquid coating of melting graupel
    std::complex<amrex::Real> m_m_w_0;  // Dielectric constant for water
    std::complex<amrex::Real> m_m_i_0;  // Dielectric constant for ice
    std::string m_mixingrulestring_s;
    std::string m_matrixstring_s;
    std::string m_inclusionstring_s;
    std::string m_hoststring_s;
    std::string m_hostmatrixstring_s;
    std::string m_hostinclusionstring_s;
    std::string m_mixingrulestring_g;
    std::string m_matrixstring_g;
    std::string m_inclusionstring_g;
    std::string m_hoststring_g;
    std::string m_hostmatrixstring_g;
    std::string m_hostinclusionstring_g;
    
    // Storage for internal state
    std::unique_ptr<amrex::MultiFab> m_cons;    // Conserved variables
    std::unique_ptr<amrex::MultiFab> m_thermo;  // Thermodynamic variables
    std::unique_ptr<amrex::MultiFab> m_hydro;   // Hydrometeor variables
    std::unique_ptr<amrex::MultiFab> m_tend;    // Tendency variables
    std::unique_ptr<amrex::MultiFab> m_radar;   // Radar reflectivity
    
    // Sedimentation parameters
    amrex::Real m_max_sediment_courant;  // Maximum allowed Courant for sedimentation
    
    // Utility methods
    
    /**
     * Initializes physical constants and parameters for the Morrison microphysics scheme.
     */
    void initialize_constants();
    
    /**
     * Allocates internal MultiFabs for microphysics variables.
     * 
     * @param[in] grids The boxes on which we will evolve the solution
     * @param[in] geom Geometry associated with these MultiFabs and grids
     */
    void allocate_arrays(const amrex::BoxArray& grids, const amrex::Geometry& geom);
    
    /**
     * Copies input data to internal storage and performs initial calculations.
     * 
     * @param[in] cons_in Conserved variables input
     */
    void copy_input_data(const amrex::MultiFab& cons_in);
    
    /**
     * Initializes thermodynamic variables needed for microphysics.
     * 
     * @param[in] geom Geometry associated with these MultiFabs and grids
     */
    void initialize_thermodynamics(const amrex::Geometry& geom);
    
    /**
     * Initializes hydrometeor size distributions based on mixing ratios and number concentrations.
     */
    void initialize_size_distributions();

/**
     * Calculates size distribution parameters (lambda and n0) for all hydrometeor categories.
     * Uses either CAM, WRF, or simplified approach based on configuration flags.
     * Lambda (slope parameter) determines the characteristic size of particles.
     * N0 (intercept parameter) determines the overall number concentration scaling.
     *
     * @param[in] qc Cloud water mixing ratio (kg/kg)
     * @param[in] qi Cloud ice mixing ratio (kg/kg)
     * @param[in] qr Rain mixing ratio (kg/kg)
     * @param[in] qs Snow mixing ratio (kg/kg)
     * @param[in] qg Graupel mixing ratio (kg/kg)
     * @param[in] nc Cloud droplet number concentration (kg^-1)
     * @param[in] ni Cloud ice number concentration (kg^-1)
     * @param[in] nr Rain drop number concentration (kg^-1)
     * @param[in] ns Snow particle number concentration (kg^-1)
     * @param[in] ng Graupel particle number concentration (kg^-1)
     * @param[in] rho Air density (kg/m^3)
     * @param[in] temp Temperature (K)
     * @param[in] pres Pressure (Pa)
     * @param[out] lambda_c Cloud droplet size distribution slope parameter (m^-1)
     * @param[out] lambda_r Rain drop size distribution slope parameter (m^-1)
     * @param[out] lambda_i Ice crystal size distribution slope parameter (m^-1)
     * @param[out] lambda_s Snow size distribution slope parameter (m^-1)
     * @param[out] lambda_g Graupel size distribution slope parameter (m^-1)
     * @param[out] n0_c Cloud droplet size distribution intercept parameter (m^-4)
     * @param[out] n0_r Rain drop size distribution intercept parameter (m^-4)
     * @param[out] n0_i Ice crystal size distribution intercept parameter (m^-4)
     * @param[out] n0_s Snow size distribution intercept parameter (m^-4)
     * @param[out] n0_g Graupel size distribution intercept parameter (m^-4)
     * @param[in] use_simple_lambdas Flag to use simplified lambda calculations (default=false)
     */
    //size_dist_param_*
    //WRF Line 1525-1639
    //WRF Line 2122-2268
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void size_distributions_params(
                                   // Hydrometeor mixing ratios
                                   const amrex::Real qc, const amrex::Real qi, const amrex::Real qr, const amrex::Real qs, const amrex::Real qg,
                                   // Number concentrations
                                   const amrex::Real nc, const amrex::Real ni, const amrex::Real nr, const amrex::Real ns, const amrex::Real ng,
                                   // Thermodynamic state
                                   const amrex::Real rho, const amrex::Real temp, const amrex::Real pres,
                                   // Output lambda parameters
                                   amrex::Real &lambda_c, amrex::Real &lambda_r, amrex::Real &lambda_i, amrex::Real &lambda_s, amrex::Real &lambda_g,
                                   // Output pgam parameter
				   amrex::Real& pgam,
                                   // Output intercept parameters
                                   amrex::Real &n0_c, amrex::Real &n0_r, amrex::Real &n0_i, amrex::Real &n0_s, amrex::Real &n0_g,
                                   // Flag for simple lambda calculation method
                                   const bool use_simple_lambdas = false)
    {
        // Initialize output parameters to zero
        lambda_c = 0.0;
        lambda_r = 0.0;
        lambda_i = 0.0;
        lambda_s = 0.0;
        lambda_g = 0.0;

        n0_c = 0.0;
        n0_r = 0.0;
        n0_i = 0.0;
        n0_s = 0.0;
        n0_g = 0.0;

        // Use local copies of number concentrations to avoid modifying original values
        amrex::Real nc_local = nc;

        // Set constant droplet number concentration if needed (for calculations only)
        if (m_inum == 1) {
          nc_local = m_ndcnst * 1.0e6 / rho; // Convert from cm^-3 to kg^-1
        }

        // Make sure number concentrations are positive for calculations
        nc_local = amrex::max(nc_local, 1.0e-10);
        amrex::Real nr_local = amrex::max(nr, 1.0e-10);
        amrex::Real ni_local = amrex::max(ni, 1.0e-10);
        amrex::Real ns_local = amrex::max(ns, 1.0e-10);
        amrex::Real ng_local = amrex::max(ng, 1.0e-10);

#ifdef ERF_USE_CAM
        // ========================================================================
        // CAM APPROACH FOR SIZE DISTRIBUTION PARAMETERS
        // ========================================================================

        // Cloud droplets
        if (qc > m_qsmall) {
          // Get pgam from fit to observations of Martin et al. 1994
          pgam = 1.0 - 0.7 * exp(-0.008 * 1.e-6 * nc_local * rho);
          pgam = 1.0/(pgam*pgam) - 1.0;
          pgam = amrex::max(pgam, 2.0);

          // Calculate shape coefficient
          amrex::Real shape_coef = m_pi/6.0 * m_rhow * (pgam+1.0) * (pgam+2.0) * (pgam+3.0);

          // Lambda bounds (limits to between 2 and 50 microns mean size)
          amrex::Real lambda_min = (pgam+1.0)/50.0e-6;
          amrex::Real lambda_max = (pgam+1.0)/2.0e-6;

          // Calculate lambda parameter (c*n/q)^(1/d)
          lambda_c = pow(shape_coef * nc_local/qc, 1.0/3.0);

          // Apply bounds to lambda
          lambda_c = amrex::max(lambda_min, amrex::min(lambda_c, lambda_max));

          // Calculate gamma function value
          amrex::Real gamma_pgam_plus_1 = tgamma(pgam + 1.0);

          // Calculate intercept parameter n0 = n * lambda^(shape+1) / Gamma(shape+1)
          n0_c = nc_local * pow(lambda_c, pgam+1) / gamma_pgam_plus_1;
        }

        // Cloud ice
        if (qi > m_qsmall) {
          // Lambda bounds from CAM
          amrex::Real lambda_min = 1.0/500.0e-6;  // CAM standard bound
          amrex::Real lambda_max = 1.0/10.0e-6;   // CAM standard bound

          amrex::Real shape_coef = m_pi/6.0 * m_rhoi;
          lambda_i = pow(shape_coef * ni_local/qi, 1.0/3.0);

          // Apply bounds to lambda
          lambda_i = amrex::max(lambda_min, amrex::min(lambda_i, lambda_max));

          // For exponential distribution, n0 = n * lambda
          n0_i = ni_local * lambda_i;
        }

        // Rain
        if (qr > m_qsmall) {
          // Lambda bounds from CAM
          amrex::Real lambda_min = 1.0/500.0e-6;  // CAM standard bound
          amrex::Real lambda_max = 1.0/20.0e-6;   // CAM standard bound

          amrex::Real shape_coef = m_pi/6.0 * m_rhow;
          lambda_r = pow(shape_coef * nr_local/qr, 1.0/3.0);

          // Apply bounds to lambda
          lambda_r = amrex::max(lambda_min, amrex::min(lambda_r, lambda_max));

          // For exponential distribution, n0 = n * lambda
          n0_r = nr_local * lambda_r;
        }

        // Snow
        if (qs > m_qsmall) {
          // Lambda bounds from CAM
          amrex::Real lambda_min = 1.0/1000.0e-6;  // CAM standard bound
          amrex::Real lambda_max = 1.0/10.0e-6;    // CAM standard bound

          amrex::Real shape_coef = m_pi/6.0 * m_rhosn;
          lambda_s = pow(shape_coef * ns_local/qs, 1.0/3.0);

          // Apply bounds to lambda
          lambda_s = amrex::max(lambda_min, amrex::min(lambda_s, lambda_max));

          // For exponential distribution, n0 = n * lambda
          n0_s = ns_local * lambda_s;
        }

        // Graupel
        if (qg > m_qsmall) {
          // Lambda bounds from CAM
          amrex::Real lambda_min = 1.0/1000.0e-6;  // CAM standard bound
          amrex::Real lambda_max = 1.0/10.0e-6;    // CAM standard bound

          amrex::Real shape_coef = m_pi/6.0 * m_rhog;
          lambda_g = pow(shape_coef * ng_local/qg, 1.0/3.0);

          // Apply bounds to lambda
          lambda_g = amrex::max(lambda_min, amrex::min(lambda_g, lambda_max));

          // For exponential distribution, n0 = n * lambda
          n0_g = ng_local * lambda_g;
        }

#else
        if (use_simple_lambdas) {
            // ========================================================================
            // SIMPLIFIED APPROACH FOR SIZE DISTRIBUTION PARAMETERS
            // ========================================================================

            // Cloud ice
            if (qi > m_qsmall) {
                lambda_i = pow(m_cons12 * ni_local / qi, 1.0/m_di);
                lambda_i = amrex::max(lambda_i, m_lammini);
                lambda_i = amrex::min(lambda_i, m_lammaxi);
                // We don't calculate n0_i in the simplified approach
            }

            // Rain
            if (qr > m_qsmall) {
                lambda_r = pow(m_pi * m_rhow * nr_local / qr, 1.0/3.0);
                lambda_r = amrex::max(lambda_r, m_lamminr);
                lambda_r = amrex::min(lambda_r, m_lammaxr);
                // We don't calculate n0_r in the simplified approach
            }

            // Cloud droplets
            if (qc > m_qsmall) {
                amrex::Real dum = pres/(287.15 * temp);
                pgam = 0.0005714 * (nc_local/1.e6 * dum) + 0.2714;
                pgam = 1.0/(pgam*pgam) - 1.0;
                pgam = amrex::max(pgam, 2.0);
                pgam = amrex::min(pgam, 10.0);

                lambda_c = pow(m_cons26 * nc_local * tgamma(pgam + 4.0) / (qc * tgamma(pgam + 1.0)), 1.0/3.0);
                amrex::Real lambda_min = (pgam + 1.0)/60.0e-6;
                amrex::Real lambda_max = (pgam + 1.0)/1.0e-6;
                lambda_c = amrex::max(lambda_c, lambda_min);
                lambda_c = amrex::min(lambda_c, lambda_max);
                // We don't calculate n0_c in the simplified approach
            }

            // Snow
            if (qs > m_qsmall) {
                lambda_s = pow(m_cons1 * ns_local / qs, 1.0/m_ds);
                lambda_s = amrex::max(lambda_s, m_lammins);
                lambda_s = amrex::min(lambda_s, m_lammaxs);
                // We don't calculate n0_s in the simplified approach
            }

            // Graupel
            if (qg > m_qsmall) {
                lambda_g = pow(m_cons2 * ng_local / qg, 1.0/m_dg);
                lambda_g = amrex::max(lambda_g, m_lamming);
                lambda_g = amrex::min(lambda_g, m_lammaxg);
                // We don't calculate n0_g in the simplified approach
            }
        }
        else {
            // ========================================================================
            // USING WRF APPROACH FOR SIZE DISTRIBUTION PARAMETERS
            // ========================================================================

            // Cloud droplets
            if (qc > m_qsmall) {
              // Calculate air density factor (moist air density)
              amrex::Real dum = pres/(287.15*temp);

              // MARTIN ET AL. (1994) FORMULA FOR PGAM (WRF implementation)
              pgam = 0.0005714*(nc_local/1.0e6*dum) + 0.2714;
              pgam = 1.0/(pgam*pgam) - 1.0;
              pgam = amrex::max(pgam, 2.0);
              pgam = amrex::min(pgam, 10.0);

              // Calculate gamma function values
              amrex::Real gamma_pgam_plus_1 = tgamma(pgam + 1.0);
              amrex::Real gamma_pgam_plus_4 = tgamma(pgam + 4.0);

              // Calculate lambda parameter
              lambda_c = pow((m_cons26 * nc_local * gamma_pgam_plus_4) / (qc * gamma_pgam_plus_1), 1.0/3.0);

              // Lambda bounds from WRF - 60 micron max diameter, 1 micron min diameter
              amrex::Real lambda_min = (pgam + 1.0)/60.0e-6;
              amrex::Real lambda_max = (pgam + 1.0)/1.0e-6;

              // Apply bounds to lambda
              lambda_c = amrex::max(lambda_min, amrex::min(lambda_c, lambda_max));

              // Calculate intercept parameter n0 = n * lambda^(shape+1) / Gamma(shape+1)
              n0_c = nc_local * pow(lambda_c, pgam+1) / gamma_pgam_plus_1;
            }

            // Cloud ice
            if (qi > m_qsmall) {
              // Calculate lambda parameter
              lambda_i = pow(m_cons12 * ni_local / qi, 1.0/3.0);

              // Calculate intercept parameter (initial calculation)
              n0_i = ni_local * lambda_i;

              // Check for slope (apply bounds)
              if (lambda_i < m_lammini) {
                lambda_i = m_lammini;
                // Recalculate n0_i when lambda is adjusted
                n0_i = pow(lambda_i, 4.0) * qi / m_cons12;
                // Update ni_local (this was missing in the original C++)
                ni_local = n0_i / lambda_i;
              } else if (lambda_i > m_lammaxi) {
                lambda_i = m_lammaxi;
                // Recalculate n0_i when lambda is adjusted
                n0_i = pow(lambda_i, 4.0) * qi / m_cons12;
                // Update ni_local (this was missing in the original C++)
                ni_local = n0_i / lambda_i;
              }
            }

            // Rain
            if (qr > m_qsmall) {
              // Calculate lambda parameter using cons26 (pi*rhow/6)
              lambda_r = pow(m_cons26 * nr_local / qr, 1.0/3.0);

              // Apply bounds to lambda
              lambda_r = amrex::max(m_lamminr, amrex::min(lambda_r, m_lammaxr));

              // Calculate intercept parameter using WRF formula
              n0_r = pow(lambda_r, 4.0) * qr / m_cons26;
            }

            // Snow
            if (qs > m_qsmall) {
              // Calculate lambda parameter
              lambda_s = pow(m_cons1 * ns_local / qs, 1.0/m_ds);

              // Apply bounds to lambda
              lambda_s = amrex::max(m_lammins, amrex::min(lambda_s, m_lammaxs));

              // Calculate intercept parameter using WRF formula
              n0_s = pow(lambda_s, 4.0) * qs / m_cons1;
            }

            // Graupel
            if (qg > m_qsmall) {
              // Calculate lambda parameter
              lambda_g = pow(m_cons2 * ng_local / qg, 1.0/m_dg);

              // Apply bounds to lambda
              lambda_g = amrex::max(m_lamming, amrex::min(lambda_g, m_lammaxg));

              // Calculate intercept parameter using WRF formula
              n0_g = pow(lambda_g, 4.0) * qg / m_cons2;
            }
        }
#endif
    }

   /**
     * Initializes vertical grid information needed for sedimentation calculations.
     * 
     * @param[in,out] z_phys_nd Physical heights at cell centers
     * @param[in] detJ_cc Jacobian determinant at cell centers
     */
    void initialize_vertical_grid(std::unique_ptr<amrex::MultiFab>& z_phys_nd,
                                  std::unique_ptr<amrex::MultiFab>& detJ_cc);
    
    /**
     * Initializes parameters needed for radar reflectivity calculations.
     */
    void initialize_radar_parameters();
    
    /**
     * Initializes the radar reflectivity calculation module.
     */
    void initialize_radar_reflectivity();
    
    /**
     * Helper function to calculate the gamma function.
     * 
     * @param[in] x Input value
     * @return The gamma function evaluated at x
     */
    amrex::Real gamma_function(const amrex::Real x) const;

   /**
     * Calculates the backscattering cross-section of a wet/melting hydrometeor.
     * This function is based on the rayleigh_soak_wetgraupel subroutine in the Fortran code.
     *
     * @param[in] x Mass of the hydrometeor (kg)
     * @param[in] xocm Shape parameter for size distribution
     * @param[in] xobm Shape parameter for size distribution
     * @param[in] fmelt Melt fraction (0-1)
     * @param[in] melt_outside Flag for liquid coating (true if liquid is on the outside)
     * @param[in] m_w_0 Dielectric constant of water
     * @param[in] m_i_0 Dielectric constant of ice
     * @param[in] lambda_radar Radar wavelength (m)
     * @param[out] cback Backscattering cross-section (m^2)
     * @param[in] mixingrulestring Mixing rule string (e.g., "maxwell")
     * @param[in] matrixstring Matrix material string (e.g., "water")
     * @param[in] inclusionstring Inclusion material string (e.g., "spheroidal")
     * @param[in] hoststring Host material string (e.g., "air")
     * @param[in] hostmatrixstring Host-matrix material string (e.g., "icewater")
     * @param[in] hostinclusionstring Host-inclusion material string (e.g., "spheroidal")
     */
    void rayleigh_soak_wetgraupel (const amrex::Real x,
                                   const amrex::Real xocm,
                                   const amrex::Real xobm,
                                   const amrex::Real fmelt,
                                   const bool melt_outside,
                                   const std::complex<amrex::Real>& m_w_0,
                                   const std::complex<amrex::Real>& m_i_0,
                                   const amrex::Real lambda_radar,
                                   amrex::Real& cback,
                                   const std::string& mixingrulestring,
                                   const std::string& matrixstring,
                                   const std::string& inclusionstring,
                                   const std::string& hoststring,
                                   const std::string& hostmatrixstring,
                                   const std::string& hostinclusionstring) const;
  /**
   * Helper function to calculate saturation vapor pressure for water or ice.
   * This corresponds to the POLYSVP function in the Fortran code (line ~5580).
   * 
   * @param[in] T Temperature in Kelvin
   * @param[in] type 0 for liquid water, 1 for ice
   * @return Saturation vapor pressure in Pascals
   */
  amrex::Real
  calc_saturation_vapor_pressure (const amrex::Real T, const int type) const
  {
    amrex::Real polysvp = 0.0;
    amrex::Real del_T = T - 273.15;  // Convert to Celsius

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
            
            polysvp = a0i + del_T*(a1i + del_T*(a2i + del_T*(a3i + del_T*(a4i + del_T*(a5i + del_T*(a6i + del_T*(a7i + a8i*del_T)))))));
            polysvp *= 100.0;  // Convert from hPa to Pa
        } else {
            // Goff-Gratch formula for ice at cold temperatures
            polysvp = std::pow(10.0, (-9.09718*(273.16/T-1.0) - 3.56654*std::log10(273.16/T) + 
                                      0.876793*(1.0-T/273.16) + std::log10(6.1071))) * 100.0;
            polysvp = 0.0; 
        } // T
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
           
        polysvp = a0 + del_T*(a1 + del_T*(a2 + del_T*(a3 + del_T*(a4 + del_T*(a5 + del_T*(a6 + del_T*(a7 + a8*del_T)))))));
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
  
// Add to Morrison class private section:
private:
    // Activation and nucleation parameters
    int m_activate_type;  // 0: None, 1: Power law, 2: Lognormal aerosol, 3: WRF-Chem coupled
    int m_inuc_type;      // 0: Mid-latitude (Cooper), 1: Arctic (MPACE)

    amrex::Real m_a_w;    // Coefficient A for activation (m^3/2 kg^1/2)
    amrex::Real m_r_v;    // No idea
    amrex::Real m_mu;    // No idea

};

/**
 * Physical properties and constants namespace
 * This namespace contains common physical constants used throughout the ERF code
 */
namespace PhysProp {
    // Gas constants
    constexpr amrex::Real R_d = 287.15;       // Gas constant for dry air (J/kg/K)
    constexpr amrex::Real R_v = 461.5;        // Gas constant for water vapor (J/kg/K)
    
    // Specific heats
    constexpr amrex::Real c_p = 1005.0;       // Specific heat at constant pressure for dry air (J/kg/K)
    
    // Molecular weight ratios
    constexpr amrex::Real ep_2 = R_d / R_v;   // Molecular weight ratio (Rd/Rv)
    
    // Gravitational acceleration
    constexpr amrex::Real g = 9.806;          // Gravitational acceleration (m/s^2)
}

/**
 * Thermodynamic constants namespace
 * Contains constants used for thermodynamic calculations
 */
namespace ThermoProp {
    // Freezing point of water
    constexpr amrex::Real t_freeze = 273.15;  // Freezing point of water (K)
    
    // Mixed-phase bounds
    constexpr amrex::Real tbgmin = 253.15;    // Minimum temperature for mixed-phase (K)
    constexpr amrex::Real tbgmax = 273.15;    // Maximum temperature for mixed-phase (K)
    
    // Homogeneous freezing threshold
    constexpr amrex::Real t_homog_freeze = 233.15;  // Homogeneous freezing threshold (K)
    
    // Latent heats
    constexpr amrex::Real lcond = 2.5e6;      // Latent heat of condensation (J/kg)
    constexpr amrex::Real lfus = 3.336e5;     // Latent heat of fusion (J/kg)
    constexpr amrex::Real lsub = lcond + lfus; // Latent heat of sublimation (J/kg)
}
#endif
