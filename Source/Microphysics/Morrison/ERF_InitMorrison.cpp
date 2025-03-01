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
Morrison::Init(const MultiFab& cons_in,
               const BoxArray& grids,
               const Geometry& geom,
               const Real& dt_advance,
               std::unique_ptr<MultiFab>& z_phys_nd,
               std::unique_ptr<MultiFab>& detJ_cc)
{
    BL_PROFILE("Morrison::Init()");

    // Store timestep
    dt = dt_advance;

    // Initialize physical constants
    initialize_constants();

    // Set microphysics control parameters
    m_activate_type = 2;  // Lognormal aerosol activation
    m_inuc_type = 0;      // Mid-latitude ice nucleation (Cooper)
    m_iliq = 0;           // Include ice processes
    m_igraup = 0;         // Include graupel processes
    m_ihail = 0;          // Use graupel (0) instead of hail (1)

    // Allocate internal MultiFabs for microphysics variables
    allocate_arrays(grids, geom);

    // Copy input data to internal storage
    copy_input_data(cons_in);

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
}

/**
 * Initializes physical constants and parameters for the Morrison microphysics scheme.
 * Includes all constants needed for the full range of microphysical processes.
 */
void
Morrison::initialize_constants()
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
Morrison::allocate_arrays(const BoxArray& grids, const Geometry& geom)
{
    // Get number of ghost cells for different variables
    const int n_ghost_cons = 1;  // Number of ghost cells for conserved variables
    const int n_ghost_micro = 1; // Number of ghost cells for microphysics variables

    // Define component indices for each variable
    const int n_comps_cons = 5;  // rho, rho*u, rho*v, rho*w, rho*e
    const int n_comps_thermo = 14; // T, p, qv, etc.
    const int n_comps_hydro = 10;  // qc, qr, qi, qs, qg, nc, nr, ni, ns, ng

    // Create distribution mapping
    DistributionMapping dm(grids);

    // Allocate MultiFabs for conserved variables
    m_cons.reset(new MultiFab(grids, dm, n_comps_cons, n_ghost_cons));

    // Allocate MultiFabs for thermodynamic variables
    m_thermo.reset(new MultiFab(grids, dm, n_comps_thermo, n_ghost_micro));

    // Allocate MultiFabs for hydrometeor variables
    m_hydro.reset(new MultiFab(grids, dm, n_comps_hydro, n_ghost_micro));

    // Allocate MultiFabs for tendency variables
    m_tend.reset(new MultiFab(grids, dm, n_comps_hydro, n_ghost_micro));

    // Initialize all data to zero
    m_cons->setVal(0.0);
    m_thermo->setVal(0.0);
    m_hydro->setVal(0.0);
    m_tend->setVal(0.0);

    // Allocate auxiliary arrays for microphysics calculations
    if (m_do_radar_ref) {
        m_radar.reset(new MultiFab(grids, dm, 1, 0)); // No ghost cells needed for radar
        m_radar->setVal(-35.0); // Initialize with minimum reflectivity value
    }
}

/**
 * Copies input data to internal storage and performs initial calculations.
 *
 * @param[in] cons_in Conserved variables input
 */
void
Morrison::copy_input_data(const MultiFab& cons_in)
{
    // Copy conserved variables from input to internal storage
    MultiFab::Copy(*m_cons, cons_in, 0, 0, m_cons->nComp(), m_cons->nGrow());

    // Extract density, momentum, and energy components for convenience
    // inside parallel region
    const int rho_comp = 0;
    const int momx_comp = 1;
    const int momy_comp = 2;
    const int momz_comp = 3;
    const int energy_comp = 4;

    // Loop over boxes and initialize derived variables
    #ifdef AMREX_USE_OMP
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif
    for (MFIter mfi(*m_cons); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();

        // Get array accessors
        const auto& cons = m_cons->array(mfi);

        // Initialize hydrometeors based on input data
        // This would typically be done by other init functions
        // called in initialize_thermodynamics and initialize_size_distributions
    }
}

/**
 * Initializes thermodynamic variables needed for microphysics.
 *
 * @param[in] geom Geometry associated with these MultiFabs and grids
 */
void
Morrison::initialize_thermodynamics(const Geometry& geom)
{
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

    // Calculate thermodynamic variables from conserved variables
    #ifdef AMREX_USE_OMP
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif
    for (MFIter mfi(*m_cons); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();

        // Get array accessors
        const auto& cons = m_cons->array(mfi);
        const auto& thermo = m_thermo->array(mfi);
        const auto& hydro = m_hydro->array(mfi);

        // Calculate temperature, pressure, etc. from conserved variables
        amrex::ParallelFor(box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // This would calculate temperature, pressure, and other
                // thermodynamic variables from density, momentum, and energy

                // Set minimum values for hydrometeor variables
                hydro(i,j,k,qc_comp) = amrex::max(hydro(i,j,k,qc_comp), 0.0);
                hydro(i,j,k,qr_comp) = amrex::max(hydro(i,j,k,qr_comp), 0.0);
                hydro(i,j,k,qi_comp) = amrex::max(hydro(i,j,k,qi_comp), 0.0);
                hydro(i,j,k,qs_comp) = amrex::max(hydro(i,j,k,qs_comp), 0.0);
                hydro(i,j,k,qg_comp) = amrex::max(hydro(i,j,k,qg_comp), 0.0);

                // Calculate saturation vapor pressures
                Real T = thermo(i,j,k,t_comp);
                Real p = thermo(i,j,k,p_comp);

                // Water saturation vapor pressure
                Real evs = amrex::min(0.99*p, calc_saturation_vapor_pressure(T, 0));

                // Ice saturation vapor pressure
                Real eis = amrex::min(0.99*p, calc_saturation_vapor_pressure(T, 1));

                // Make sure ice saturation doesn't exceed water saturation near freezing
                if (eis > evs) eis = evs;

                // Calculate saturation mixing ratios
                Real qvs = m_ep_2 * evs / (p - evs);
                Real qvi = m_ep_2 * eis / (p - eis);

                // Store these in thermodynamic variables array
                // Component indices would continue here...
            }
        );
    }
}

/**
 * Initializes hydrometeor size distributions based on mixing ratios and number concentrations.
 */
void
Morrison::initialize_size_distributions()
{
    // Component indices for hydrometeors
    const int qc_comp = 0;  // Cloud water mixing ratio
    const int qr_comp = 1;  // Rain mixing ratio
    const int qi_comp = 2;  // Cloud ice mixing ratio
    const int qs_comp = 3;  // Snow mixing ratio
    const int qg_comp = 4;  // Graupel mixing ratio
    const int nc_comp = 5;  // Cloud droplet number
    const int nr_comp = 6;  // Rain drop number
    const int ni_comp = 7;  // Ice crystal number
    const int ns_comp = 8;  // Snow number
    const int ng_comp = 9;  // Graupel number

    // Component indices for thermodynamic variables
    const int rho_comp = 3; // Density

    // Initialize size distributions for hydrometeors
    #ifdef AMREX_USE_OMP
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif
    for (MFIter mfi(*m_hydro); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();

        // Get array accessors
        const auto& hydro = m_hydro->array(mfi);
        const auto& thermo = m_thermo->array(mfi);

        // Initialize size distribution parameters
        amrex::ParallelFor(box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // Get density for this cell
                Real rho = thermo(i,j,k,rho_comp);

                // Set constant droplet number concentration if specified
                if (m_inum == 1) {
                    hydro(i,j,k,nc_comp) = m_ndcnst * 1.0e6 / rho; // Convert from cm^-3 to kg^-1
                }

                // Make sure number concentrations are positive
                hydro(i,j,k,nc_comp) = amrex::max(hydro(i,j,k,nc_comp), 0.0);
                hydro(i,j,k,nr_comp) = amrex::max(hydro(i,j,k,nr_comp), 0.0);
                hydro(i,j,k,ni_comp) = amrex::max(hydro(i,j,k,ni_comp), 0.0);
                hydro(i,j,k,ns_comp) = amrex::max(hydro(i,j,k,ns_comp), 0.0);
                hydro(i,j,k,ng_comp) = amrex::max(hydro(i,j,k,ng_comp), 0.0);

                // Calculate size distribution parameters for each hydrometeor species
                // (Lambda, N0, etc.) based on mixing ratio and number concentration
                // This would follow the same approach as the original FORTRAN code
            }
        );
    }
}

/**
 * Initializes vertical grid information needed for sedimentation calculations.
 *
 * @param[in,out] z_phys_nd Physical heights at cell centers
 * @param[in] detJ_cc Jacobian determinant at cell centers
 */
void
Morrison::initialize_vertical_grid(std::unique_ptr<MultiFab>& z_phys_nd,
                                  std::unique_ptr<MultiFab>& detJ_cc)
{
  /*
  // Store pointers to vertical grid information
    m_z_phys_nd = std::move(z_phys_nd);
    m_detJ_cc = std::move(detJ_cc);
  */
    // Initialize any sedimentation-specific parameters
    // For example: maximum allowed Courant number for sedimentation,
    // minimum allowed layer thickness, etc.
    m_max_sediment_courant = 1.0;
}

/**
 * Initializes parameters needed for radar reflectivity calculations.
 */
void
Morrison::initialize_radar_parameters()
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
Morrison::gamma_function(const Real x) const
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
Morrison::initialize_radar_reflectivity()
{
   if (!m_do_radar_ref) return;

   // Set up arrays for Simpson integration of reflectivity
   const int nrbins = 300;  // Number of bins for integration

   // Allocate arrays for radar calculations
   m_xxds.resize(nrbins+1);
   m_xxdg.resize(nrbins+1);
   m_xdts.resize(nrbins+1);
   m_xdtg.resize(nrbins+1);
   m_simpson.resize(nrbins+1);

   // Initialize arrays for reflectivity calculation
   Real maxD = 2.0e-2;  // Maximum diameter for integration (m)
   Real dD = maxD / nrbins;  // Diameter increment

   // Set up integration arrays
   for (int n = 1; n <= nrbins; ++n) {
       m_xxds[n] = (n-0.5) * dD;
       m_xxdg[n] = (n-0.5) * dD;
       m_xdts[n] = dD;
       m_xdtg[n] = dD;
   }

   // Simpson's rule integration weights
   m_simpson[1] = 1.0;
   for (int n = 2; n < nrbins; ++n) {
       if (n % 2 == 0) {
           m_simpson[n] = 4.0;
       } else {
           m_simpson[n] = 2.0;
       }
   }
   m_simpson[nrbins] = 1.0;

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
   m_hoststring_s = "air";
   m_hostmatrixstring_s = "icewater";
   m_hostinclusionstring_s = "spheroidal";

   m_mixingrulestring_g = "maxwell";
   m_matrixstring_g = "water";
   m_inclusionstring_g = "spheroidal";
   m_hoststring_g = "air";
   m_hostmatrixstring_g = "icewater";
   m_hostinclusionstring_g = "spheroidal";
}
