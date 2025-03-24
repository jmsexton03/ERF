
#include <AMReX_GpuContainers.H>
#include "ERF_Morrison.H"
#include "ERF_IndexDefines.H"
#include "ERF_PlaneAverage.H"
#include "ERF_EOS.H"
#include "ERF_TileNoZ.H"

using namespace amrex;

/**
 * Initializes the Microphysics module.
 *
 * @param[in] cons_in Conserved variables input
 * @param[in] grids The boxes on which we will evolve the solution
 * @param[in] geom Geometry associated with these MultiFabs and grids
 * @param[in] dt_advance Timestep for the advance
 * @param[in,out] z_phys_nd Physical heights at cell centers
 * @param[in] detJ_cc Jacobian determinant at cell centers
 */
void
Morrison::Init (const MultiFab& cons_in,
                const BoxArray& grids,
                const Geometry& geom,
                const Real& dt_advance,
                std::unique_ptr<MultiFab>& z_phys_nd,
                std::unique_ptr<MultiFab>& detJ_cc)
{
    BL_PROFILE("Morrison::Init()");

    // Store timestep
    dt = dt_advance;
    m_geom = geom;
    m_gtoe = grids;

    m_z_phys_nd = z_phys_nd.get();
    m_detJ_cc   = detJ_cc.get();

    MicVarMap.resize(m_qmoist_size);
    MicVarMap = {MicVar_Morr::qt, MicVar_Morr::qv , MicVar_Morr::qcl, MicVar_Morr::qci,
                 MicVar_Morr::qp, MicVar_Morr::qpr, MicVar_Morr::qps, MicVar_Morr::qpg,
                 MicVar_Morr::rain_accum, MicVar_Morr::snow_accum, MicVar_Morr::graup_accum};

    // initialize microphysics variables
    for (auto ivar = 0; ivar < MicVar_Morr::NumVars; ++ivar) {
        mic_fab_vars[ivar] = std::make_shared<MultiFab>(cons_in.boxArray(), cons_in.DistributionMap(),
                                                        1, cons_in.nGrowVect());
        mic_fab_vars[ivar]->setVal(0.);
    }

    // Set class data members
    for ( MFIter mfi(cons_in, TileNoZ()); mfi.isValid(); ++mfi) {
        const auto& box3d = mfi.tilebox();

        const auto& lo = lbound(box3d);
        const auto& hi = ubound(box3d);

        nlev = box3d.length(2);
        zlo  = lo.z;
        zhi  = hi.z;

        // parameters
        accrrc.resize({zlo},  {zhi});
        accrsi.resize({zlo},  {zhi});
        accrsc.resize({zlo},  {zhi});
        coefice.resize({zlo}, {zhi});
        evaps1.resize({zlo},  {zhi});
        evaps2.resize({zlo},  {zhi});
        accrgi.resize({zlo},  {zhi});
        accrgc.resize({zlo},  {zhi});
        evapg1.resize({zlo},  {zhi});
        evapg2.resize({zlo},  {zhi});
        evapr1.resize({zlo},  {zhi});
        evapr2.resize({zlo},  {zhi});

        // data (input)
        rho1d.resize({zlo}, {zhi});
        pres1d.resize({zlo}, {zhi});
        tabs1d.resize({zlo}, {zhi});
        gamaz.resize({zlo}, {zhi});
        zmid.resize({zlo}, {zhi});
    }
    // Initialize physical constants
    // Set microphysics control parameters
    m_inum = 1;           // Use constant droplet number concentration
    m_ndcnst = 250.0;     // Droplet number concentration (cm^-3)
    initialize_constants();

    // Set microphysics control parameters
    m_activate_type = 2;  // Lognormal aerosol activation
    m_inuc_type = 0;      // Mid-latitude ice nucleation (Cooper)
    m_iliq = 0;           // Include ice processes
    m_igraup = 0;         // Include graupel processes
    m_ihail = 0;          // Use graupel (0) instead of hail (1)
    m_isub = 0;           // Sub-grid vertical velocity option
    m_do_radar_ref = false; // Disable radar reflectivity by default

    // Ensure consistency between m_iact and m_activate_type
    m_iact = m_activate_type;

    // Initialize water vapor gas constant for activation calculations
    m_r_v = m_Rv;

    // Allocate internal MultiFabs for microphysics variables
    allocate_arrays(grids, geom);

    // Initialize thermodynamic variables
    initialize_thermodynamics(geom);

    // Initialize hydrometeor size distributions
    initialize_size_distributions();

    // Initialize height and vertical coordinate information
    initialize_vertical_grid(z_phys_nd, detJ_cc);

    // Initialize radar diagnostics if enabled
    if (m_do_radar_ref) {
        initialize_radar_parameters();
        initialize_radar_reflectivity();
    }

    // Initialize viscosity parameter for contact nucleation
    m_mu = 1.496E-6 * std::pow(293.15, 1.5) / (293.15 + 120.0);

}

/**
 * Initializes physical constants and parameters for the Morrison microphysics scheme.
 * Includes all constants needed for the full range of microphysical processes.
 */
void
Morrison::initialize_constants ()
{
    // Mathematical constants
    m_pi = 3.1415926535897932384626434;

    // Basic physical constants from AMReX physics
    m_R = PhysProp::R_d;         // Gas constant for dry air (J/kg/K)
    m_Rv = PhysProp::R_v;        // Gas constant for water vapor (J/kg/K)
    m_cp = PhysProp::c_p;        // Specific heat at constant pressure (J/kg/K)
    m_g = PhysProp::g;           // Gravitational acceleration (m/s^2)
    m_ep_2 = PhysProp::ep_2;     // Molecular weight ratio (Rd/Rv)

    // Reference density
    m_rhosu = 85000.0/(m_R*273.15);  // Standard air density at 850 mb (kg/m^3)

    // Densities for different hydrometeor species
    m_rhow = 997.0;     // Density of liquid water (kg/m^3)
    m_rhoi = 500.0;     // Bulk density of cloud ice (kg/m^3)
    m_rhosn = 100.0;    // Bulk density of snow (kg/m^3)

    // Set density for graupel or hail based on configuration
    if (m_ihail == 0) {
        m_rhog = 400.0; // Bulk density of graupel (kg/m^3)
    } else {
        m_rhog = 900.0; // Bulk density of hail (kg/m^3)
    }

    // Fall speed parameters (V=AD^B) for different hydrometeors
    // Cloud ice
    m_ai = 700.0;
    m_bi = 1.0;

    // Cloud droplets
    m_ac = 3.0E7;
    m_bc = 2.0;

    // Snow
    m_as = 11.72;
    m_bs = 0.41;

    // Rain
    m_ar = 841.99667;
    m_br = 0.8;

    // Graupel/hail (dependent on configuration)
    if (m_ihail == 0) {
        // Graupel parameters
        m_ag = 19.3;
        m_bg = 0.37;
    } else {
        // Hail parameters (Matsun and Huggins 1980)
        m_ag = 114.5;
        m_bg = 0.5;
    }

    // Microphysical parameters
    m_aimm = 0.66;       // Parameter in Bigg immersion freezing
    m_bimm = 100.0;      // Parameter in Bigg immersion freezing
    m_ecr = 1.0;         // Collection efficiency between rain and snow/graupel
    m_dcs = 125.0E-6;    // Threshold size for cloud ice autoconversion (m)
    m_mi0 = 4.0/3.0*m_pi*m_rhoi*std::pow(10.0E-6, 3);  // Initial mass of nucleated ice crystal (kg)
    m_mg0 = 1.6E-10;     // Mass of embryo graupel (kg)

    // Ventilation parameters
    m_f1s = 0.86;        // Ventilation parameter for snow
    m_f2s = 0.28;        // Ventilation parameter for snow
    m_f1r = 0.78;        // Ventilation parameter for rain
    m_f2r = 0.308;       // Ventilation parameter for rain

    // Smallest allowed hydrometeor mixing ratio
    m_qsmall = 1.0E-14;

    // Collection efficiencies
    m_eii = 0.1;         // Ice-ice collision efficiency
    m_eci = 0.7;         // Ice-droplet collision efficiency

    // Specific heat of liquid water (J/kg/K)
    m_cpw = 4187.0;

    // Size distribution parameters
    m_ci = m_rhoi * m_pi / 6.0;
    m_di = 3.0;
    m_cs = m_rhosn * m_pi / 6.0;
    m_ds = 3.0;
    m_cg = m_rhog * m_pi / 6.0;
    m_dg = 3.0;

    // Radius of contact nuclei (m)
    m_rin = 0.1E-6;

    // Mass of splintered ice particle (kg)
    m_mmult = 4.0/3.0*m_pi*m_rhoi*std::pow(5.0E-6, 3);

    // Set lambda limits for size distributions
    // Maximum and minimum values for lambda parameter in size distributions
    m_lammaxi = 1.0/1.0E-6;
    m_lammini = 1.0/(2.0*m_dcs + 100.0E-6);
    m_lammaxr = 1.0/20.0E-6;
    m_lamminr = 1.0/2800.0E-6;
    m_lammaxs = 1.0/10.0E-6;
    m_lammins = 1.0/2000.0E-6;
    m_lammaxg = 1.0/20.0E-6;
    m_lamming = 1.0/2000.0E-6;

    // Set CCN parameters for different environments
    if (m_activate_type == 1) {
        // Maritime CCN spectrum parameters (modified from Rasmussen et al. 2002)
        // NCCN = C*S^K, where S is supersaturation in %
        m_k1 = 0.4;        // Exponent in CCN activation formula
        m_c1 = 120.0;      // Coefficient in CCN activation formula (cm^-3)
    }

    // Initialize aerosol activation parameters for lognormal distribution
    if (m_activate_type == 2) {
        // Parameters for ammonium sulfate
        m_mw = 0.018;      // Molecular weight of water (kg/mol)
        m_osm = 1.0;       // Osmotic coefficient
        m_vi = 3.0;        // Number of ions dissociated in solution
        m_epsm = 0.7;      // Aerosol soluble fraction
        m_rhoa = 1777.0;   // Aerosol bulk density (kg/m^3)
        m_map = 0.132;     // Molecular weight of aerosol (kg/mol)
        m_ma = 0.0284;     // Molecular weight of air (kg/mol)
        m_rr = 8.3145;     // Universal gas constant (J/mol/K)
        m_bact = m_vi * m_osm * m_epsm * m_mw * m_rhoa / (m_map * m_rhow);
        m_a_w = 2.0 * m_mw * 0.0761 / (m_rhow * m_r_v * 293.15);  // "A" parameter

        // Aerosol size distribution parameters for MPACE (Morrison et al. 2007, JGR)
        // Mode 1
        m_rm1 = 0.052E-6;  // Geometric mean radius, mode 1 (m)
        m_sig1 = 2.04;     // Standard deviation of aerosol size distribution, mode 1
        m_nanew1 = 72.2E6; // Total aerosol concentration, mode 1 (m^-3)
        m_f11 = 0.5 * std::exp(2.5 * std::pow(std::log(m_sig1), 2));
        m_f21 = 1.0 + 0.25 * std::log(m_sig1);

        // Mode 2
        m_rm2 = 1.3E-6;    // Geometric mean radius, mode 2 (m)
        m_sig2 = 2.5;      // Standard deviation of aerosol size distribution, mode 2
        m_nanew2 = 1.8E6;  // Total aerosol concentration, mode 2 (m^-3)
        m_f12 = 0.5 * std::exp(2.5 * std::pow(std::log(m_sig2), 2));
        m_f22 = 1.0 + 0.25 * std::log(m_sig2);
    }

    // Precompute constants for efficiency
    m_cons1 = gamma_function(1.0 + m_ds) * m_cs;
    m_cons2 = gamma_function(1.0 + m_dg) * m_cg;
    m_cons3 = gamma_function(4.0 + m_bs) / 6.0;
    m_cons4 = gamma_function(4.0 + m_br) / 6.0;
    m_cons5 = gamma_function(1.0 + m_bs);
    m_cons6 = gamma_function(1.0 + m_br);
    m_cons7 = gamma_function(4.0 + m_bg) / 6.0;
    m_cons8 = gamma_function(1.0 + m_bg);
    m_cons9 = gamma_function(5.0/2.0 + m_br/2.0);
    m_cons10 = gamma_function(5.0/2.0 + m_bs/2.0);
    m_cons11 = gamma_function(5.0/2.0 + m_bg/2.0);
    m_cons12 = gamma_function(1.0 + m_di) * m_ci;
    m_cons13 = gamma_function(m_bs + 3.0) * m_pi / 4.0 * m_eci;
    m_cons14 = gamma_function(m_bg + 3.0) * m_pi / 4.0 * m_eci;
    m_cons15 = -1108.0 * m_eii * std::pow(m_pi, (1.0-m_bs)/3.0) *
               std::pow(m_rhosn, (-2.0-m_bs)/3.0) / (4.0*720.0);
    m_cons16 = gamma_function(m_bi + 3.0) * m_pi / 4.0 * m_eci;
    m_cons17 = 4.0 * 2.0 * 3.0 * m_rhosu * m_pi * m_eci * m_eci *
               gamma_function(2.0*m_bs + 2.0) / (8.0*(m_rhog-m_rhosn));
    m_cons18 = m_rhosn * m_rhosn;
    m_cons19 = m_rhow * m_rhow;
    m_cons20 = 20.0 * m_pi * m_pi * m_rhow * m_bimm;
    m_cons21 = 4.0 / (m_dcs * m_rhoi);
    m_cons22 = m_pi * m_rhoi * std::pow(m_dcs, 3) / 6.0;
    m_cons23 = m_pi / 4.0 * m_eii * gamma_function(m_bs + 3.0);
    m_cons24 = m_pi / 4.0 * m_ecr * gamma_function(m_br + 3.0);
    m_cons25 = m_pi * m_pi / 24.0 * m_rhow * m_ecr * gamma_function(m_br + 6.0);
    m_cons26 = m_pi / 6.0 * m_rhow;
    m_cons27 = gamma_function(1.0 + m_bi);
    m_cons28 = gamma_function(4.0 + m_bi) / 6.0;
    m_cons29 = 4.0/3.0 * m_pi * m_rhow * std::pow(25.0E-6, 3);
    m_cons30 = 4.0/3.0 * m_pi * m_rhow;
    m_cons31 = m_pi * m_pi * m_ecr * m_rhosn;
    m_cons32 = m_pi / 2.0 * m_ecr;
    m_cons33 = m_pi * m_pi * m_ecr * m_rhog;
    m_cons34 = 5.0/2.0 + m_br/2.0;
    m_cons35 = 5.0/2.0 + m_bs/2.0;
    m_cons36 = 5.0/2.0 + m_bg/2.0;
    m_cons37 = 4.0 * m_pi * 1.38E-23 / (6.0 * m_pi * m_rin);
    m_cons38 = m_pi * m_pi / 3.0 * m_rhow;
    m_cons39 = m_pi * m_pi / 36.0 * m_rhow * m_bimm;
    m_cons40 = m_pi / 6.0 * m_bimm;
    m_cons41 = m_pi * m_pi * m_ecr * m_rhow;

    // Set CCN parameters for different environments
    if (m_iact == 1) {
        // Maritime CCN spectrum parameters (modified from Rasmussen et al. 2002)
        // NCCN = C*S^K, where S is supersaturation in %
        m_k1 = 0.4;        // Exponent in CCN activation formula
        m_c1 = 120.0;      // Coefficient in CCN activation formula (cm^-3)
    }

    // Initialize aerosol activation parameters for IACT=2
    if (m_iact == 2) {
        // Parameters for ammonium sulfate
        m_mw = 0.018;      // Molecular weight of water (kg/mol)
        m_osm = 1.0;       // Osmotic coefficient
        m_vi = 3.0;        // Number of ions dissociated in solution
        m_epsm = 0.7;      // Aerosol soluble fraction
        m_rhoa = 1777.0;   // Aerosol bulk density (kg/m^3)
        m_map = 0.132;     // Molecular weight of aerosol (kg/mol)
        m_ma = 0.0284;     // Molecular weight of air (kg/mol)
        m_rr = 8.3145;     // Universal gas constant (J/mol/K)
        m_bact = m_vi * m_osm * m_epsm * m_mw * m_rhoa / (m_map * m_rhow);

        // Aerosol size distribution parameters for MPACE (Morrison et al. 2007, JGR)
        // Mode 1
        m_rm1 = 0.052E-6;  // Geometric mean radius, mode 1 (m)
        m_sig1 = 2.04;     // Standard deviation of aerosol size distribution, mode 1
        m_nanew1 = 72.2E6; // Total aerosol concentration, mode 1 (m^-3)
        m_f11 = 0.5 * std::exp(2.5 * std::pow(std::log(m_sig1), 2));
        m_f21 = 1.0 + 0.25 * std::log(m_sig1);

        // Mode 2
        m_rm2 = 1.3E-6;    // Geometric mean radius, mode 2 (m)
        m_sig2 = 2.5;      // Standard deviation of aerosol size distribution, mode 2
        m_nanew2 = 1.8E6;  // Total aerosol concentration, mode 2 (m^-3)
        m_f12 = 0.5 * std::exp(2.5 * std::pow(std::log(m_sig2), 2));
        m_f22 = 1.0 + 0.25 * std::log(m_sig2);
    }

    // Initialize radar reflectivity parameters
    m_xam_r = m_pi * m_rhow / 6.0;
    m_xbm_r = 3.0;
    m_xmu_r = 0.0;
    m_xam_s = m_cs;
    m_xbm_s = m_ds;
    m_xmu_s = 0.0;
    m_xam_g = m_cg;
    m_xbm_g = m_dg;
    m_xmu_g = 0.0;
}

/**
 * Allocates internal MultiFabs for microphysics variables.
 *
 * @param[in] grids The boxes on which we will evolve the solution
 * @param[in] geom Geometry associated with these MultiFabs and grids
 */
void
Morrison::allocate_arrays(const BoxArray& grids, const Geometry& /*geom*/)
{
    // Get number of ghost cells for different variables
    const int n_ghost_micro = 1; // Number of ghost cells for microphysics variables

    // Define component indices for each variable
    const int n_comps_tend = 10;  // Number of tendency components

    // Create distribution mapping
    DistributionMapping dm(grids);

    // Allocate MultiFabs for tendency variables
    m_tend.reset(new MultiFab(grids, dm, n_comps_tend, n_ghost_micro));

    // Initialize all data to zero
    m_tend->setVal(0.0);

    // Initialize number concentration fields if they don't exist
    if (!mic_fab_vars[MicVar_Morr::nc]) {
        mic_fab_vars[MicVar_Morr::nc] = std::make_shared<MultiFab>(grids, dm, 1, n_ghost_micro);
        mic_fab_vars[MicVar_Morr::nc]->setVal(0.0);
    }

    if (!mic_fab_vars[MicVar_Morr::nr]) {
        mic_fab_vars[MicVar_Morr::nr] = std::make_shared<MultiFab>(grids, dm, 1, n_ghost_micro);
        mic_fab_vars[MicVar_Morr::nr]->setVal(0.0);
    }

    if (!mic_fab_vars[MicVar_Morr::ni]) {
        mic_fab_vars[MicVar_Morr::ni] = std::make_shared<MultiFab>(grids, dm, 1, n_ghost_micro);
        mic_fab_vars[MicVar_Morr::ni]->setVal(0.0);
    }

    if (!mic_fab_vars[MicVar_Morr::ns]) {
        mic_fab_vars[MicVar_Morr::ns] = std::make_shared<MultiFab>(grids, dm, 1, n_ghost_micro);
        mic_fab_vars[MicVar_Morr::ns]->setVal(0.0);
    }

    if (!mic_fab_vars[MicVar_Morr::ng]) {
        mic_fab_vars[MicVar_Morr::ng] = std::make_shared<MultiFab>(grids, dm, 1, n_ghost_micro);
        mic_fab_vars[MicVar_Morr::ng]->setVal(0.0);
    }

    // Initialize accumulation variables
    if (!mic_fab_vars[MicVar_Morr::rain_accum]) {
        mic_fab_vars[MicVar_Morr::rain_accum] = std::make_shared<MultiFab>(grids, dm, 1, n_ghost_micro);
        mic_fab_vars[MicVar_Morr::rain_accum]->setVal(0.0);
    }

    if (!mic_fab_vars[MicVar_Morr::snow_accum]) {
        mic_fab_vars[MicVar_Morr::snow_accum] = std::make_shared<MultiFab>(grids, dm, 1, n_ghost_micro);
        mic_fab_vars[MicVar_Morr::snow_accum]->setVal(0.0);
    }

    if (!mic_fab_vars[MicVar_Morr::graup_accum]) {
        mic_fab_vars[MicVar_Morr::graup_accum] = std::make_shared<MultiFab>(grids, dm, 1, n_ghost_micro);
        mic_fab_vars[MicVar_Morr::graup_accum]->setVal(0.0);
    }

    if (!mic_fab_vars[MicVar_Morr::omega]) {
        mic_fab_vars[MicVar_Morr::omega] = std::make_shared<MultiFab>(grids, dm, 1, n_ghost_micro);
        mic_fab_vars[MicVar_Morr::omega]->setVal(0.0);
    }

    // Allocate radar reflectivity MultiFab if needed
    if (m_do_radar_ref) {
        m_radar.reset(new MultiFab(grids, dm, 1, 0)); // No ghost cells needed for radar
        m_radar->setVal(-35.0); // Initialize with minimum reflectivity value
    }
}

void
Morrison::copy_input_data (const MultiFab& cons_in)
{
    // Copy data from conserved variables to microphysics variables
    // This will be handled by Copy_State_to_Micro
    Copy_State_to_Micro(cons_in);
}

void
Morrison::initialize_thermodynamics (const Geometry& /*geom*/)
{
    // Initialize thermodynamic variables
    // This is now handled by the Copy_State_to_Micro method
    // Additional thermodynamic calculations can be done here if needed
}

void
Morrison::initialize_size_distributions ()
{
    // Initialize size distributions for hydrometeors
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*mic_fab_vars[MicVar_Morr::qcl]); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();

        // Get array accessors
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

        auto const& thermo_rho = mic_fab_vars[MicVar_Morr::rho]->array(mfi);
        auto const& thermo_temp = mic_fab_vars[MicVar_Morr::tabs]->array(mfi);
        auto const& thermo_pres = mic_fab_vars[MicVar_Morr::pres]->array(mfi);

        // Initialize size distribution parameters
        amrex::ParallelFor( box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // Get density for this cell
            Real rho = thermo_rho(i,j,k);
            Real temp = thermo_temp(i,j,k);
            Real pres = thermo_pres(i,j,k);

            // Set constant droplet number concentration if specified
            if (m_inum == 1) {
              hydro_nc(i,j,k) = m_ndcnst * 1.0e6 / rho; // Convert from cm^-3 to kg^-1
            }

            // Make sure number concentrations are positive
            hydro_nc(i,j,k) = amrex::max(hydro_nc(i,j,k), 0.0);
            hydro_nr(i,j,k) = amrex::max(hydro_nr(i,j,k), 0.0);
            hydro_ni(i,j,k) = amrex::max(hydro_ni(i,j,k), 0.0);
            hydro_ns(i,j,k) = amrex::max(hydro_ns(i,j,k), 0.0);
            hydro_ng(i,j,k) = amrex::max(hydro_ng(i,j,k), 0.0);

#ifdef ERF_USE_CAM
            // ========================================================================
            // CAM APPROACH FOR ALL HYDROMETEORS
            // ========================================================================

            // Cloud droplets
            if (hydro_qc(i,j,k) > m_qsmall) {
              // Get pgam from fit to observations of Martin et al. 1994
              Real pgam = 1.0 - 0.7 * exp(-0.008 * 1.e-6 * hydro_nc(i,j,k) * rho);
              pgam = 1.0/(pgam*pgam) - 1.0;
              pgam = amrex::max(pgam, 2.0);

              // Calculate shape coefficient
              Real shape_coef = m_pi/6.0 * m_rhow * (pgam+1.0) * (pgam+2.0) * (pgam+3.0);

              // Lambda bounds (limits to between 2 and 50 microns mean size)
              Real lambda_min = (pgam+1.0)/50.0e-6;
              Real lambda_max = (pgam+1.0)/2.0e-6;

              // Calculate lambda parameter (c*n/q)^(1/d)
              Real lambda_c = pow(shape_coef * hydro_nc(i,j,k)/hydro_qc(i,j,k), 1.0/3.0);

              // Check for slope and adjust vars
              if (lambda_c < lambda_min) {
                lambda_c = lambda_min;
                hydro_nc(i,j,k) = pow(lambda_c, 3.0) * hydro_qc(i,j,k)/shape_coef;
              } else if (lambda_c > lambda_max) {
                lambda_c = lambda_max;
                hydro_nc(i,j,k) = pow(lambda_c, 3.0) * hydro_qc(i,j,k)/shape_coef;
              }
            }

            // Cloud ice
            if (hydro_qi(i,j,k) > m_qsmall) {
              // CAM approach - using standard lambda bounds
              Real lambda_min = 1.0/500.0e-6;  // CAM standard bound
              Real lambda_max = 1.0/10.0e-6;   // CAM standard bound

              Real shape_coef = m_pi/6.0 * m_rhoi;
              Real lambda_i = pow(shape_coef * hydro_ni(i,j,k)/hydro_qi(i,j,k), 1.0/3.0);

              if (lambda_i < lambda_min) {
                lambda_i = lambda_min;
                hydro_ni(i,j,k) = pow(lambda_i, 3.0) * hydro_qi(i,j,k)/shape_coef;
              } else if (lambda_i > lambda_max) {
                lambda_i = lambda_max;
                hydro_ni(i,j,k) = pow(lambda_i, 3.0) * hydro_qi(i,j,k)/shape_coef;
              }
            }

            // Rain
            if (hydro_qr(i,j,k) > m_qsmall) {
              // CAM approach - using standard lambda bounds
              Real lambda_min = 1.0/500.0e-6;  // CAM standard bound
              Real lambda_max = 1.0/20.0e-6;   // CAM standard bound

              Real shape_coef = m_pi/6.0 * m_rhow;
              Real lambda_r = pow(shape_coef * hydro_nr(i,j,k)/hydro_qr(i,j,k), 1.0/3.0);

              if (lambda_r < lambda_min) {
                lambda_r = lambda_min;
                hydro_nr(i,j,k) = pow(lambda_r, 3.0) * hydro_qr(i,j,k)/shape_coef;
              } else if (lambda_r > lambda_max) {
                lambda_r = lambda_max;
                hydro_nr(i,j,k) = pow(lambda_r, 3.0) * hydro_qr(i,j,k)/shape_coef;
              }
            }

            // Snow
            if (hydro_qs(i,j,k) > m_qsmall) {
              // CAM approach - using standard lambda bounds
              Real lambda_min = 1.0/1000.0e-6;  // CAM standard bound
              Real lambda_max = 1.0/10.0e-6;    // CAM standard bound

              Real shape_coef = m_pi/6.0 * m_rhosn;
              Real lambda_s = pow(shape_coef * hydro_ns(i,j,k)/hydro_qs(i,j,k), 1.0/3.0);

              if (lambda_s < lambda_min) {
                lambda_s = lambda_min;
                hydro_ns(i,j,k) = pow(lambda_s, 3.0) * hydro_qs(i,j,k)/shape_coef;
              } else if (lambda_s > lambda_max) {
                lambda_s = lambda_max;
                hydro_ns(i,j,k) = pow(lambda_s, 3.0) * hydro_qs(i,j,k)/shape_coef;
              }
            }

            // Graupel
            if (hydro_qg(i,j,k) > m_qsmall) {
              // CAM approach - using standard lambda bounds
              Real lambda_min = 1.0/1000.0e-6;  // CAM standard bound
              Real lambda_max = 1.0/10.0e-6;    // CAM standard bound

              Real shape_coef = m_pi/6.0 * m_rhog;
              Real lambda_g = pow(shape_coef * hydro_ng(i,j,k)/hydro_qg(i,j,k), 1.0/3.0);

              if (lambda_g < lambda_min) {
                lambda_g = lambda_min;
                hydro_ng(i,j,k) = pow(lambda_g, 3.0) * hydro_qg(i,j,k)/shape_coef;
              } else if (lambda_g > lambda_max) {
                lambda_g = lambda_max;
                hydro_ng(i,j,k) = pow(lambda_g, 3.0) * hydro_qg(i,j,k)/shape_coef;
              }
            }

#else
            // ========================================================================
            // USING WRF APPROACH FOR ALL HYDROMETEORS
            // ========================================================================

            // Cloud droplets
            if (hydro_qc(i,j,k) > m_qsmall && m_inum == 0) {
              // Calculate air density factor (moist air density)
              Real dum = pres/(287.15*temp);

              // MARTIN ET AL. (1994) FORMULA FOR PGAM (WRF implementation)
              Real pgam = 0.0005714*(hydro_nc(i,j,k)/1.0e6*dum) + 0.2714;
              pgam = 1.0/(pgam*pgam) - 1.0;
              pgam = amrex::max(pgam, 2.0);
              pgam = amrex::min(pgam, 10.0);

              // CONS26 equivalent (coefficient for distribution calculation)
              Real cons26 = m_pi * m_rhow / 6.0;

              // Calculate gamma function values using tgamma from cmath
              Real gamma_pgam_plus_1 = tgamma(pgam + 1.0);
              Real gamma_pgam_plus_4 = tgamma(pgam + 4.0);

              // Calculate lambda parameter
              Real lambda_c = pow((cons26 * hydro_nc(i,j,k) * gamma_pgam_plus_4) /
                                  (hydro_qc(i,j,k) * gamma_pgam_plus_1), 1.0/3.0);

              // Lambda bounds from WRF - 60 micron max diameter, 1 micron min diameter
              Real lambda_min = (pgam + 1.0)/60.0e-6;
              Real lambda_max = (pgam + 1.0)/1.0e-6;

              // Check for slope and adjust vars
              if (lambda_c < lambda_min) {
                lambda_c = lambda_min;
                // Adjust number concentration using WRF formulation
                hydro_nc(i,j,k) = exp(3.0*log(lambda_c) + log(hydro_qc(i,j,k)) +
                                      log(gamma_pgam_plus_1) - log(gamma_pgam_plus_4))/cons26;
              } else if (lambda_c > lambda_max) {
                lambda_c = lambda_max;
                // Adjust number concentration using WRF formulation
                hydro_nc(i,j,k) = exp(3.0*log(lambda_c) + log(hydro_qc(i,j,k)) +
                                      log(gamma_pgam_plus_1) - log(gamma_pgam_plus_4))/cons26;
              }
            }

            // Cloud ice - using bounds from WRF
            if (hydro_qi(i,j,k) > m_qsmall) {
              // Calculate lambda parameter
              Real cons12 = m_pi * m_rhoi / 6.0; // CI constant from WRF
              Real lambda_i = pow(cons12 * hydro_ni(i,j,k) / hydro_qi(i,j,k), 1.0/3.0);

              // Use lambda limits from WRF
              Real lammini = 1.0/(2.0*m_dcs + 100.0e-6);  // WRF bound
              Real lammaxi = 1.0/1.0e-6;                  // WRF bound

              // Check for slope and adjust vars
              if (lambda_i < lammini) {
                lambda_i = lammini;
                Real n0i = pow(lambda_i, 4.0) * hydro_qi(i,j,k) / cons12;
                hydro_ni(i,j,k) = n0i / lambda_i;
              } else if (lambda_i > lammaxi) {
                lambda_i = lammaxi;
                Real n0i = pow(lambda_i, 4.0) * hydro_qi(i,j,k) / cons12;
                hydro_ni(i,j,k) = n0i / lambda_i;
              }
            }

            // Rain - using bounds from WRF
            if (hydro_qr(i,j,k) > m_qsmall) {
              // Calculate lambda parameter
              Real lambda_r = pow(m_pi * m_rhow * hydro_nr(i,j,k) / hydro_qr(i,j,k), 1.0/3.0);

              // Use lambda limits from WRF
              Real lamminr = 1.0/2800.0e-6;  // WRF bound
              Real lammaxr = 1.0/20.0e-6;    // WRF bound

              // Check for slope and adjust vars
              if (lambda_r < lamminr) {
                lambda_r = lamminr;
                Real n0r = pow(lambda_r, 4.0) * hydro_qr(i,j,k) / (m_pi * m_rhow);
                hydro_nr(i,j,k) = n0r / lambda_r;
              } else if (lambda_r > lammaxr) {
                lambda_r = lammaxr;
                Real n0r = pow(lambda_r, 4.0) * hydro_qr(i,j,k) / (m_pi * m_rhow);
                hydro_nr(i,j,k) = n0r / lambda_r;
              }
            }

            // Snow - using bounds from WRF
            if (hydro_qs(i,j,k) > m_qsmall) {
              // Calculate lambda parameter
              Real lambda_s = pow(m_pi * m_rhosn * hydro_ns(i,j,k) / hydro_qs(i,j,k), 1.0/3.0);

              // Use lambda limits from WRF
              Real lammins = 1.0/2000.0e-6;  // WRF bound
              Real lammaxs = 1.0/10.0e-6;    // WRF bound

              // Check for slope and adjust vars
              if (lambda_s < lammins) {
                lambda_s = lammins;
                Real n0s = pow(lambda_s, 4.0) * hydro_qs(i,j,k) / (m_pi * m_rhosn);
                hydro_ns(i,j,k) = n0s / lambda_s;
              } else if (lambda_s > lammaxs) {
                lambda_s = lammaxs;
                Real n0s = pow(lambda_s, 4.0) * hydro_qs(i,j,k) / (m_pi * m_rhosn);
                hydro_ns(i,j,k) = n0s / lambda_s;
              }
            }

            // Graupel - using bounds from WRF
            if (hydro_qg(i,j,k) > m_qsmall) {
              // Calculate lambda parameter
              Real lambda_g = pow(m_pi * m_rhog * hydro_ng(i,j,k) / hydro_qg(i,j,k), 1.0/3.0);

              // Use lambda limits from WRF
              Real lamming = 1.0/2000.0e-6;  // WRF bound
              Real lammaxg = 1.0/20.0e-6;    // WRF bound

              // Check for slope and adjust vars
              if (lambda_g < lamming) {
                lambda_g = lamming;
                Real n0g = pow(lambda_g, 4.0) * hydro_qg(i,j,k) / (m_pi * m_rhog);
                hydro_ng(i,j,k) = n0g / lambda_g;
              } else if (lambda_g > lammaxg) {
                lambda_g = lammaxg;
                Real n0g = pow(lambda_g, 4.0) * hydro_qg(i,j,k) / (m_pi * m_rhog);
                hydro_ng(i,j,k) = n0g / lambda_g;
              }
            }
#endif

            // Make sure number concentrations are positive (final check)
            hydro_nc(i,j,k) = amrex::max(hydro_nc(i,j,k), 0.0);
            hydro_nr(i,j,k) = amrex::max(hydro_nr(i,j,k), 0.0);
            hydro_ni(i,j,k) = amrex::max(hydro_ni(i,j,k), 0.0);
            hydro_ns(i,j,k) = amrex::max(hydro_ns(i,j,k), 0.0);
            hydro_ng(i,j,k) = amrex::max(hydro_ng(i,j,k), 0.0);
        });
    } // mfi
}

/**
 * Initializes vertical grid information needed for sedimentation calculations.
 *
 * @param[in,out] z_phys_nd Physical heights at cell centers
 * @param[in] detJ_cc Jacobian determinant at cell centers
 */
void
Morrison::initialize_vertical_grid (std::unique_ptr<MultiFab>& z_phys_nd,
                                    std::unique_ptr<MultiFab>& detJ_cc)
{
       // Store pointers to vertical grid information
       m_z_phys_nd = z_phys_nd.get();
       m_detJ_cc = detJ_cc.get();

       // Set maximum allowed Courant number for sedimentation
       m_max_sediment_courant = 0.9;
#if 0
       // Initialize vertical grid for each cell
       for (MFIter mfi(*m_z_phys_nd); mfi.isValid(); ++mfi) {
           const Box& box = mfi.validbox();
           auto const& z_phys = m_z_phys_nd->array(mfi);
           auto const& detJ = m_detJ_cc->array(mfi);

           // Store vertical grid information for use in sedimentation calculations
           amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
               // Additional initialization if needed
           });
       }
#endif
}

/**
 * Initializes parameters needed for radar reflectivity calculations.
 */
void
Morrison::initialize_radar_parameters ()
{
    if (!m_do_radar_ref) return;

    // Set up parameters for radar reflectivity calculations
    m_lambda_radar = 0.10;   // 10 cm wavelength
    m_k_w = 0.93;            // K_w parameter for liquid water

    // Additional radar parameters from original code
    m_lamda4 = std::pow(m_lambda_radar, 4.0);
    m_pi5 = std::pow(m_pi, 5.0);

    // Set up coefficients for melting calculations if needed
}

/**
 * Helper function to calculate the gamma function.
 *
 * @param[in] x Input value
 * @return The gamma function evaluated at x
 */
Real
Morrison::gamma_function (const Real x) const
{
    // Implementation of gamma function using Lanczos approximation
    // or another suitable approximation method

    // For this example, we'll use a simplified implementation
    if (x <= 0.0) {
        // Handle negative values with reflection formula if needed
        // For now, just return a large value as an error indicator
        return 1.0e30;
    }

    // Stirling's approximation for large x
    if (x > 12.0) {
        Real y = x - 1.0;
        return std::sqrt(2.0*m_pi) * std::pow(y, y+0.5) * std::exp(-y) *
               (1.0 + 1.0/(12.0*y) + 1.0/(288.0*y*y) - 139.0/(51840.0*y*y*y));
    }

    // For smaller x, use recursion with known values
    // Gamma(n+1) = n*Gamma(n)
    // Base case: Gamma(1) = 1
    if (std::abs(x - std::floor(x)) < 1.0e-10) {
        // Integer case
        int n = static_cast<int>(x);
        Real result = 1.0;
        for (int i = 1; i < n; ++i) {
            result *= i;
        }
        return result;
    }

    // For non-integer values between 1 and 2, use a polynomial approximation
    // For a complete implementation, this would be more sophisticated

    // For simplicity in this example:
    return std::tgamma(x); // Use standard library gamma function
}


/**
* Initializes the radar reflectivity calculation module.
* This is analogous to the radar_init subroutine in the original code.
*/
void
Morrison::initialize_radar_reflectivity ()
{
   if (!m_do_radar_ref) return;

   // Set up arrays for Simpson integration of reflectivity
   const int nrbins = 300;  // Number of bins for integration

   // Allocate arrays for radar calculations
      m_xxds.resize(nrbins);
      m_xxdg.resize(nrbins);
      m_xdts.resize(nrbins);
      m_xdtg.resize(nrbins);
      m_simpson.resize(nrbins);

   // Initialize arrays for reflectivity calculation
   Real maxD = 2.0e-2;  // Maximum diameter for integration (m)
   Real dD = maxD / nrbins;  // Diameter increment

   // Set up integration arrays
      for (int n = 0; n < nrbins; ++n) {
          m_xxds[n] = (n+0.5) * dD;
          m_xxdg[n] = (n+0.5) * dD;
          m_xdts[n] = dD;
          m_xdtg[n] = dD;
   }

   // Simpson's rule integration weights
      m_simpson[0] = 1.0;
      for (int n = 1; n < nrbins-1; ++n) {
       if (n % 2 == 0) {
           m_simpson[n] = 4.0;
       } else {
           m_simpson[n] = 2.0;
       }
   }
      m_simpson[nrbins-1] = 1.0;

   // Parameters for wet/melting hydrometeors
   // These would be the same as in the original code for the
   // rayleigh_soak_wetgraupel calculation

   // Parameters for dielectric matrix calculation
   m_melt_outside_s = false;  // Liquid coating of melting snow
   m_melt_outside_g = true;   // Liquid coating of melting graupel

   // Dielectric constants
   m_m_w_0 = std::complex<Real>(8.8, 0.4);  // Dielectric constant for water
   m_m_i_0 = std::complex<Real>(3.2, 0.0);  // Dielectric constant for ice

   // Matrix and inclusion strings for different hydrometeors
   m_mixingrulestring_s = "maxwell";
   m_matrixstring_s = "water";
   m_inclusionstring_s = "spheroidal";
   m_hoststring_s = "snow";  // Changed to "snow" to properly identify snow particles
   m_hostmatrixstring_s = "icewater";
   m_hostinclusionstring_s = "spheroidal";

   m_mixingrulestring_g = "maxwell";
   m_matrixstring_g = "water";
   m_inclusionstring_g = "spheroidal";
   m_hoststring_g = "graupel";  // Changed to "graupel" to properly identify graupel particles
   m_hostmatrixstring_g = "icewater";
   m_hostinclusionstring_g = "spheroidal";

   // Set radar wavelength and parameters
   m_lambda_radar = 0.10;   // 10 cm wavelength
   m_k_w = 0.93;            // K_w parameter for liquid water
   m_lamda4 = std::pow(m_lambda_radar, 4.0);
   m_pi5 = std::pow(m_pi, 5.0);
}

/**
 * Initializes the Microphysics module.
 *
 * @param[in] cons_in Conserved variables input
 */
void
Morrison::Copy_State_to_Micro (const MultiFab& cons_in)
{
    // Get the temperature, density, theta, qt and qp from input
    for ( MFIter mfi(cons_in); mfi.isValid(); ++mfi) {
        const auto& box3d = mfi.growntilebox();

        auto states_array = cons_in.array(mfi);

        // Non-precipitating
        auto qv_array    = mic_fab_vars[MicVar_Morr::qv]->array(mfi);
        auto qc_array    = mic_fab_vars[MicVar_Morr::qcl]->array(mfi);
        auto qi_array    = mic_fab_vars[MicVar_Morr::qci]->array(mfi);
        auto qn_array    = mic_fab_vars[MicVar_Morr::qn]->array(mfi);
        auto qt_array    = mic_fab_vars[MicVar_Morr::qt]->array(mfi);

        // Precipitating
        auto qpr_array   = mic_fab_vars[MicVar_Morr::qpr]->array(mfi);
        auto qps_array   = mic_fab_vars[MicVar_Morr::qps]->array(mfi);
        auto qpg_array   = mic_fab_vars[MicVar_Morr::qpg]->array(mfi);
        auto qp_array    = mic_fab_vars[MicVar_Morr::qp]->array(mfi);

        auto rho_array   = mic_fab_vars[MicVar_Morr::rho]->array(mfi);
        auto theta_array = mic_fab_vars[MicVar_Morr::theta]->array(mfi);
        auto tabs_array  = mic_fab_vars[MicVar_Morr::tabs]->array(mfi);
        auto pres_array  = mic_fab_vars[MicVar_Morr::pres]->array(mfi);

        // Get pressure, theta, temperature, density, and qt, qp
        ParallelFor( box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            rho_array(i,j,k)   = states_array(i,j,k,Rho_comp);
            theta_array(i,j,k) = states_array(i,j,k,RhoTheta_comp)/states_array(i,j,k,Rho_comp);

            qv_array(i,j,k)    = std::max(0.0,states_array(i,j,k,RhoQ1_comp)/states_array(i,j,k,Rho_comp));
            qc_array(i,j,k)    = std::max(0.0,states_array(i,j,k,RhoQ2_comp)/states_array(i,j,k,Rho_comp));
            qi_array(i,j,k)    = std::max(0.0,states_array(i,j,k,RhoQ3_comp)/states_array(i,j,k,Rho_comp));
            qn_array(i,j,k)    = qc_array(i,j,k) + qi_array(i,j,k);
            qt_array(i,j,k)    = qv_array(i,j,k) + qn_array(i,j,k);

            qpr_array(i,j,k)   = std::max(0.0,states_array(i,j,k,RhoQ4_comp)/states_array(i,j,k,Rho_comp));
            qps_array(i,j,k)   = std::max(0.0,states_array(i,j,k,RhoQ5_comp)/states_array(i,j,k,Rho_comp));
            qpg_array(i,j,k)   = std::max(0.0,states_array(i,j,k,RhoQ6_comp)/states_array(i,j,k,Rho_comp));
             qp_array(i,j,k)   = qpr_array(i,j,k) + qps_array(i,j,k) + qpg_array(i,j,k);

            tabs_array(i,j,k)  = getTgivenRandRTh(states_array(i,j,k,Rho_comp),
                                                  states_array(i,j,k,RhoTheta_comp),
                                                  qv_array(i,j,k));
            //pres_array(i,j,k)  = getPgivenRTh(states_array(i,j,k,RhoTheta_comp), qv_array(i,j,k));
            pres_array(i,j,k)  = getPgivenRTh(states_array(i,j,k,RhoTheta_comp), qv_array(i,j,k)) * 0.01;
        });
    }
}


void Morrison::Compute_Coefficients ()
{
    auto dz   = m_geom.CellSize(2);
    auto lowz = m_geom.ProbLo(2);

    auto accrrc_t  = accrrc.table();
    auto accrsi_t  = accrsi.table();
    auto accrsc_t  = accrsc.table();
    auto coefice_t = coefice.table();
    auto evaps1_t  = evaps1.table();
    auto evaps2_t  = evaps2.table();
    auto accrgi_t  = accrgi.table();
    auto accrgc_t  = accrgc.table();
    auto evapg1_t  = evapg1.table();
    auto evapg2_t  = evapg2.table();
    auto evapr1_t  = evapr1.table();
    auto evapr2_t  = evapr2.table();

    auto rho1d_t  = rho1d.table();
    auto pres1d_t = pres1d.table();
    auto tabs1d_t = tabs1d.table();

    auto gamaz_t  = gamaz.table();
    auto zmid_t   = zmid.table();

    Real gam3  = erf_gammafff(3.0             );
    Real gamr1 = erf_gammafff(3.0+b_rain      );
    Real gamr2 = erf_gammafff((5.0+b_rain)/2.0);
    Real gams1 = erf_gammafff(3.0+b_snow      );
    Real gams2 = erf_gammafff((5.0+b_snow)/2.0);
    Real gamg1 = erf_gammafff(3.0+b_grau      );
    Real gamg2 = erf_gammafff((5.0+b_grau)/2.0);

    // calculate the plane average variables
    PlaneAverage rho_ave(mic_fab_vars[MicVar_Morr::rho].get(), m_geom, m_axis);
    PlaneAverage theta_ave(mic_fab_vars[MicVar_Morr::theta].get(), m_geom, m_axis);
    PlaneAverage qv_ave(mic_fab_vars[MicVar_Morr::qv].get(), m_geom, m_axis);
    rho_ave.compute_averages(ZDir(), rho_ave.field());
    theta_ave.compute_averages(ZDir(), theta_ave.field());
    qv_ave.compute_averages(ZDir(), qv_ave.field());

    // get host variable rho, and rhotheta
    int ncell = rho_ave.ncell_line();

    Gpu::HostVector<Real> rho_h(ncell), theta_h(ncell), qv_h(ncell);
    rho_ave.line_average(0, rho_h);
    theta_ave.line_average(0, theta_h);
    qv_ave.line_average(0, qv_h);

    // copy data to device
    Gpu::DeviceVector<Real> rho_d(ncell), theta_d(ncell), qv_d(ncell);
    Gpu::copyAsync(Gpu::hostToDevice, rho_h.begin(), rho_h.end(), rho_d.begin());
    Gpu::copyAsync(Gpu::hostToDevice, theta_h.begin(), theta_h.end(), theta_d.begin());
    Gpu::copyAsync(Gpu::hostToDevice, qv_h.begin(), qv_h.end(), qv_d.begin());
    Gpu::streamSynchronize();

    Real* rho_dptr   = rho_d.data();
    Real* theta_dptr = theta_d.data();
    Real* qv_dptr    = qv_d.data();

    Real gOcp = m_gOcp;

    ParallelFor(nlev, [=] AMREX_GPU_DEVICE (int k) noexcept
    {
        Real RhoTheta = rho_dptr[k]*theta_dptr[k];
        Real pressure = getPgivenRTh(RhoTheta, qv_dptr[k]);
        rho1d_t(k)    = rho_dptr[k];
        pres1d_t(k)   = pressure*0.01;
        // NOTE: Limit the temperature to the melting point of ice to avoid a divide by
        //       0 condition when computing the cold evaporation coefficients. This should
        //       not affect results since evporation requires snow/graupel to be present
        //       and thus T<273.16
        tabs1d_t(k)   = std::min(getTgivenRandRTh(rho_dptr[k], RhoTheta, qv_dptr[k]),273.16);
        zmid_t(k)     = lowz + (k+0.5)*dz;
        gamaz_t(k)    = gOcp*zmid_t(k);
    });

    if(round(gam3) != 2) {
        std::cout << "cannot compute gamma-function in Microphysics::Init" << std::endl;
        std::exit(-1);
    }

    // Populate all the coefficients
    ParallelFor(nlev, [=] AMREX_GPU_DEVICE (int k) noexcept
    {
        Real Prefactor;
        Real pratio = sqrt(1.29 / rho1d_t(k));
        //Real rrr1   = 393.0/(tabs1d_t(k)+120.0)*std::pow((tabs1d_t(k)/273.0),1.5);
        //Real rrr2   = std::pow((tabs1d_t(k)/273.0),1.94)*(1000.0/pres1d_t(k));
        Real estw   = 100.0*erf_esatw(tabs1d_t(k));
        Real esti   = 100.0*erf_esati(tabs1d_t(k));

        // accretion by snow:
        Real coef1   = 0.25 * PI * nzeros * a_snow * gams1 * pratio/pow((PI * rhos * nzeros/rho1d_t(k) ) , ((3.0+b_snow)/4.0));
        Real coef2   = exp(0.025*(tabs1d_t(k) - 273.15));
        accrsi_t(k)  =  coef1 * coef2 * esicoef;
        accrsc_t(k)  =  coef1 * esccoef;
        coefice_t(k) =  coef2;

        // evaporation of snow:
        coef1 = (lsub/(tabs1d_t(k)*R_v)-1.0)*lsub/(therco*tabs1d_t(k));
        coef2 = R_v * R_d / (diffelq * esti);
        Prefactor = 2.0 * PI * nzeros / (rho1d_t(k) * (coef1 + coef2));
        Prefactor *= (2.0/PI); // Shape factor snow
        evaps1_t(k) = Prefactor * 0.65 * sqrt(rho1d_t(k) / (PI * rhos * nzeros));
        evaps2_t(k) = Prefactor * 0.44 * sqrt(a_snow * rho1d_t(k) / muelq) * gams2
                    * sqrt(pratio) * pow(rho1d_t(k) / (PI * rhos * nzeros) , ((5.0+b_snow)/8.0));

        // accretion by graupel:
        coef1 = 0.25*PI*nzerog*a_grau*gamg1*pratio/pow((PI*rhog*nzerog/rho1d_t(k)) , ((3.0+b_grau)/4.0));
        coef2 = exp(0.025*(tabs1d_t(k) - 273.15));
        accrgi_t(k) = coef1 * coef2 * egicoef;
        accrgc_t(k) = coef1 * egccoef;

        // evaporation of graupel:
        coef1 = (lsub/(tabs1d_t(k)*R_v)-1.0)*lsub/(therco*tabs1d_t(k));
        coef2 = R_v * R_d / (diffelq * esti);
        Prefactor = 2.0 * PI * nzerog / (rho1d_t(k) * (coef1 + coef2)); // Shape factor for graupel is 1
        evapg1_t(k) = Prefactor * 0.78 * sqrt(rho1d_t(k) / (PI * rhog * nzerog));
        evapg2_t(k) = Prefactor * 0.31 * sqrt(a_grau * rho1d_t(k) / muelq) * gamg2
                    * sqrt(pratio) * pow(rho1d_t(k) / (PI * rhog * nzerog) , ((5.0+b_grau)/8.0));

        // accretion by rain:
        accrrc_t(k) = 0.25 * PI * nzeror * a_rain * gamr1 * pratio/pow((PI * rhor * nzeror / rho1d_t(k)) , ((3.0+b_rain)/4.))* erccoef;

        // evaporation of rain:
        coef1 = (lcond/(tabs1d_t(k)*R_v)-1.0)*lcond/(therco*tabs1d_t(k));
        coef2 = R_v * R_d / (diffelq * estw);
        Prefactor = 2.0 * PI * nzeror / (rho1d_t(k) * (coef1 + coef2)); // Shape factor for rain is 1
        evapr1_t(k) = Prefactor * 0.78 * sqrt(rho1d_t(k) / (PI * rhor * nzeror));
        evapr2_t(k) = Prefactor * 0.31 * sqrt(a_rain * rho1d_t(k) / muelq) * gamr2
                    * sqrt(pratio) * pow(rho1d_t(k) / (PI * rhor * nzeror) , ((5.0+b_rain)/8.0));
    });
}
