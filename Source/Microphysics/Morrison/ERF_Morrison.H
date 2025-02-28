/*
 * Implementation of a 2-moment microphysics model
 * This model is based on the Morrison model as implemented in WRF (in Fortran)
 * 
 * References to the original WRF Fortran code (module_mp_morr_two_moment.F):
 * - The Morrison scheme is a bulk double-moment scheme described in 
 *   Morrison et al. (2005, JAS) and Morrison et al. (2009, MWR)
 * - It predicts mixing ratios and number concentrations of five hydrometeor species:
 *   cloud droplets, cloud ice, rain, snow, and graupel/hail
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
      qps,   // precip ice
      qpg,   // graupel
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

    /**
     * Cloud physics processes
     * Corresponds to condensation/evaporation and cloud phase partitioning
     * (Lines ~2040-2400 in Fortran code)
     */
    void Cloud (const SolverChoice& sc);

    /**
     * Ice sedimentation
     * Computes sedimentation of cloud ice 
     * (Lines ~3400-3650 in Fortran code)
     */
    void IceFall (const SolverChoice& sc);

    /**
     * Precipitation formation processes
     * Handles autoconversion, accretion, riming, etc.
     * (Lines ~2400-3400 in Fortran code)
     */
    void Precip (const SolverChoice& sc);

    /**
     * Precipitation sedimentation
     * Computes fallout/sedimentation of rain, snow, and graupel
     * (Lines ~3650-4050 in Fortran code)
     */
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

    /**
     * Initializes the Microphysics module.
     * 
     * This function corresponds to:
     * 1. MORR_TWO_MOMENT_INIT (lines 602-982) - Sets physical constants and parameters
     * 2. Beginning of MORR_TWO_MOMENT_MICRO (lines 1840-2025) - Initializes arrays and
     *    performs per-level calculations for thermodynamic variables
     *
     * @param[in] cons_in Conserved variables input
     * @param[in] grids The boxes on which we will evolve the solution
     * @param[in] geom Geometry associated with these MultiFabs and grids
     * @param[in] dt_advance Timestep for the advance
     * @param[in,out] z_phys_nd Physical heights at cell centers
     * @param[in] detJ_cc Jacobian determinant at cell centers
     */
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
             const SolverChoice& sc) override
    {
        dt = dt_advance;

        this->Cloud(sc);
        this->IceFall(sc);
        this->Precip(sc);
        this->PrecipFall(sc);
    }

    amrex::MultiFab*
    Qmoist_Ptr (const int& varIdx) override
    {
        AMREX_ALWAYS_ASSERT(varIdx < m_qmoist_size);
        return mic_fab_vars[MicVarMap[varIdx]].get();
    }

    /**
     * Computes derived coefficients used in microphysical rate calculations.
     * This corresponds to parts of MORR_TWO_MOMENT_INIT (lines ~740-820)
     */
    void
    Compute_Coefficients ();

    int
    Qmoist_Size () override { return Morrison::m_qmoist_size; }

    int
    Qstate_Size () override { return Morrison::m_qstate_size; }

    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::Real
    NewtonIterSat (int& i, int& j, int& k,
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
        amrex::Real tol = 1.0e-4;

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
            lstari  = fac_fus;
            domn    = 0.0;

            // Saturation moisture fractions
            erf_qsatw(tabs, pres, qsatw);
            erf_qsati(tabs, pres, qsati);
            erf_dtqsatw(tabs, pres, dqsatw);
            erf_dtqsati(tabs, pres, dqsati);

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

    /**
     * Initializes physical constants and parameters.
     * 
     * Corresponds to MORR_TWO_MOMENT_INIT (lines 602-850)
     * Sets up fall speed parameters, microphysical constants, and
     * precomputed constants for efficiency.
     */
    void initialize_constants();

    /**
     * Initializes thermodynamic variables needed for microphysics.
     * 
     * Corresponds to beginning of MORR_TWO_MOMENT_MICRO (lines 1840-2025).
     * Calculates latent heats, saturation vapor pressures, air density, etc.
     * at each vertical level.
     */
    void initialize_thermodynamics();

    /**
     * Initializes hydrometeor size distributions.
     * 
     * Relates to various parts of MORR_TWO_MOMENT_MICRO where size distribution
     * parameters are calculated based on mixing ratios and number concentrations.
     */
    void initialize_size_distributions();

    /**
     * Initializes vertical grid information for sedimentation.
     * 
     * Prepares the vertical coordinate and related information for 
     * the sedimentation calculations.
     * 
     * @param[in,out] z_phys_nd Physical heights at cell centers
     * @param[in] detJ_cc Jacobian determinant at cell centers
     */
    void initialize_vertical_grid(std::unique_ptr<amrex::MultiFab>& z_phys_nd,
                                 std::unique_ptr<amrex::MultiFab>& detJ_cc);

    /**
     * Calculates saturation vapor pressure for water or ice.
     * 
     * Corresponds to POLYSVP function (lines 4068-4160).
     * 
     * @param[in] T Temperature in Kelvin
     * @param[in] type 0 for liquid water, 1 for ice
     * @return Saturation vapor pressure in Pascals
     */
    amrex::Real calc_saturation_vapor_pressure(const amrex::Real T, const int type) const;

    /**
     * Helper function for Gamma function calculation.
     * 
     * Corresponds to GAMMA function (lines 4200-4341).
     * 
     * @param[in] x Input value
     * @return Value of the Gamma function at x
     */
    amrex::Real gamma_function(const amrex::Real x) const;

    /**
     * Initializes parameters for radar reflectivity calculations.
     * 
     * Corresponds to radar_init function reference in MORR_TWO_MOMENT_INIT
     * (line 951).
     */
    void initialize_radar_parameters();

    // Physical constants from Fortran (lines 650-675)
    amrex::Real m_pi;          // Pi constant 
    amrex::Real m_R;           // Gas constant for air
    amrex::Real m_Rv;          // Gas constant for water vapor
    amrex::Real m_rhosu;       // Standard air density at 850 mb
    amrex::Real m_rhow;        // Density of liquid water
    amrex::Real m_rhoi;        // Bulk density of cloud ice
    amrex::Real m_rhosn;       // Bulk density of snow
    amrex::Real m_rhog;        // Bulk density of graupel/hail

    // Fall speed parameters (lines 630-645)
    amrex::Real m_ai, m_bi;    // Cloud ice parameters
    amrex::Real m_ac, m_bc;    // Cloud droplet parameters
    amrex::Real m_as, m_bs;    // Snow parameters
    amrex::Real m_ar, m_br;    // Rain parameters
    amrex::Real m_ag, m_bg;    // Graupel/hail parameters

    // Microphysical parameters (lines 680-704)
    amrex::Real m_aimm;        // Parameter in Bigg immersion freezing
    amrex::Real m_bimm;        // Parameter in Bigg immersion freezing
    amrex::Real m_ecr;         // Collection efficiency
    amrex::Real m_dcs;         // Threshold size for cloud ice autoconversion
    amrex::Real m_mi0;         // Initial size of nucleated crystal
    amrex::Real m_mg0;         // Mass of embryo graupel
    amrex::Real m_f1s;         // Ventilation parameter for snow
    amrex::Real m_f2s;         // Ventilation parameter for snow
    amrex::Real m_f1r;         // Ventilation parameter for rain
    amrex::Real m_f2r;         // Ventilation parameter for rain
    amrex::Real m_qsmall;      // Smallest allowed hydrometeor mixing ratio
    amrex::Real m_eii;         // Collection efficiency
    amrex::Real m_eci;         // Collection efficiency
    amrex::Real m_rin;         // Radius of contact nuclei
    amrex::Real m_cpw;         // Specific heat of liquid water

    // Size distribution parameters (lines 707-719)
    amrex::Real m_ci, m_di;    // Cloud ice parameters
    amrex::Real m_cs, m_ds;    // Snow parameters
    amrex::Real m_cg, m_dg;    // Graupel parameters
    amrex::Real m_mmult;       // Mass of splintered ice particle

    // Lambda limits (lines 723-732)
    amrex::Real m_lammaxi, m_lammini;  // Cloud ice limits
    amrex::Real m_lammaxr, m_lamminr;  // Rain limits
    amrex::Real m_lammaxs, m_lammins;  // Snow limits
    amrex::Real m_lammaxg, m_lamming;  // Graupel limits

    // Precomputed constants for efficiency (lines 735-814)
    amrex::Real m_cons1, m_cons2, m_cons3, m_cons4, m_cons5;
    amrex::Real m_cons6, m_cons7, m_cons8, m_cons9, m_cons10;
    amrex::Real m_cons11, m_cons12, m_cons13, m_cons14, m_cons15;
    amrex::Real m_cons16, m_cons17, m_cons18, m_cons19, m_cons20;
    amrex::Real m_cons21, m_cons22, m_cons23, m_cons24, m_cons25;
    amrex::Real m_cons26, m_cons27, m_cons28, m_cons29, m_cons30;
    amrex::Real m_cons31, m_cons32, m_cons33, m_cons34, m_cons35;
    amrex::Real m_cons36, m_cons37, m_cons38, m_cons39, m_cons40;
    amrex::Real m_cons41;

    // Microphysics scheme options (lines 585-596)
    int m_ihail = 0;    // IHAIL: 0=graupel, 1=hail
    int m_inum = 1;     // INUM: 0=predict droplets, 1=constant droplets
    int m_iact = 2;     // IACT: CCN activation method
    int m_iliq = 0;     // ILIQ: 0=include ice, 1=liquid only
    int m_inuc = 0;     // INUC: ice nucleation method
    int m_ibase = 2;    // IBASE: droplet activation method
    int m_isub = 0;     // ISUB: sub-grid vertical velocity for activation
    int m_igraup = 0;   // IGRAUP: 0=include graupel, 1=no graupel
    
    // Constant droplet concentration (if INUM=1)
    amrex::Real m_ndcnst = 250.0; // line 616
    
    // CCN spectra parameters (IACT=1)
    amrex::Real m_c1;          // CCN concentration parameter
    amrex::Real m_k1;          // CCN spectra exponent
    
    // Aerosol parameters (IACT=2)
    amrex::Real m_mw;          // Molecular weight water (kg/mol)
    amrex::Real m_osm;         // Osmotic coefficient
    amrex::Real m_vi;          // Number of ion dissociated in solution
    amrex::Real m_epsm;        // Aerosol soluble fraction
    amrex::Real m_rhoa;        // Aerosol bulk density (kg/m^3)
    amrex::Real m_map;         // Molecular weight aerosol (kg/mol)
    amrex::Real m_ma;          // Molecular weight of air (kg/mol)
    amrex::Real m_rr;          // Universal gas constant
    amrex::Real m_bact;        // Activation parameter
    amrex::Real m_rm1, m_rm2;  // Geometric mean radius for modes 1 and 2
    amrex::Real m_nanew1, m_nanew2; // Total aerosol concentration
    amrex::Real m_sig1, m_sig2; // Standard deviation of aerosol S.D.
    amrex::Real m_f11, m_f12;  // Correction factors for activation
    amrex::Real m_f21, m_f22;  // Correction factors for activation
    
    // Radar parameters
    amrex::Real m_xam_r, m_xbm_r, m_xmu_r; // Rain radar parameters
    amrex::Real m_xam_s, m_xbm_s, m_xmu_s; // Snow radar parameters
    amrex::Real m_xam_g, m_xbm_g, m_xmu_g; // Graupel radar parameters
    amrex::Real m_lambda_radar; // Radar wavelength
    amrex::Real m_k_w;         // Dielectric factor for liquid water
    amrex::Real m_lamda4;      // λ⁴ precalculated
    amrex::Real m_pi5;         // π⁵ precalculated
    bool m_melt_outside_s;     // Melting snow configuration
    bool m_melt_outside_g;     // Melting graupel configuration
    std::complex<amrex::Real> m_m_w_0; // Dielectric constant for water
    std::complex<amrex::Real> m_m_i_0; // Dielectric constant for ice
    
    // Arrays for radar reflectivity calculation
    amrex::Vector<amrex::Real> m_xxds, m_xxdg; // Size bins
    amrex::Vector<amrex::Real> m_xdts, m_xdtg; // Size increments
    amrex::Vector<amrex::Real> m_simpson;      // Integration weights
    
    // Flag for radar reflectivity calculation
    bool m_do_radar_ref = false;
};
#endif
