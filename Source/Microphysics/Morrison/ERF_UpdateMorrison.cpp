#include "ERF_Morrison.H"
#include "ERF_Constants.H"
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_MultiFabUtil.H>

/**
 * Copy state variables to microphysics variables.
 * This function transfers data from the model's conserved variables
 * to the microphysics internal variables.
 *
 * @param[in] cons_in Input MultiFab containing conserved variables
 */
void
Morrison::Copy_State_to_Micro(const amrex::MultiFab& cons_in)
{
    BL_PROFILE("Morrison::Copy_State_to_Micro()");

    // Component indices for conserved state variables
    const int rho_comp = 0;   // Density
    const int momx_comp = 1;  // X-momentum
    const int momy_comp = 2;  // Y-momentum
    const int momz_comp = 3;  // Z-momentum
    const int energy_comp = 4; // Total energy
    const int rhotheta_comp = 5; // Density * theta
    
    // Component indices for moisture variables (depend on model configuration)
    // These should match the configuration in the main model
    const int qv_comp = 6;    // Water vapor
    const int qc_comp = 7;    // Cloud water
    const int qi_comp = 8;    // Cloud ice
    const int qr_comp = 9;    // Rain
    const int qs_comp = 10;   // Snow
    const int qg_comp = 11;   // Graupel
    
    // Additional number concentration indices
    // These would be defined based on the model configuration
    const int nc_comp = 12;   // Cloud droplet number
    const int ni_comp = 13;   // Cloud ice number
    const int nr_comp = 14;   // Rain number
    const int ns_comp = 15;   // Snow number
    const int ng_comp = 16;   // Graupel number

    // Loop through grids
    for (amrex::MFIter mfi(cons_in); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        
        // Get array accessors for conserved variables
        auto const& cons = cons_in.array(mfi);
        
        // Get array accessors for microphysics variables
        auto& rho = mic_fab_vars[MicVar_Morr::rho]->array(mfi);
        auto& theta = mic_fab_vars[MicVar_Morr::theta]->array(mfi);
        auto& tabs = mic_fab_vars[MicVar_Morr::tabs]->array(mfi);
        auto& pres = mic_fab_vars[MicVar_Morr::pres]->array(mfi);
        
        auto& qv = mic_fab_vars[MicVar_Morr::qv]->array(mfi);
        auto& qcl = mic_fab_vars[MicVar_Morr::qcl]->array(mfi);
        auto& qci = mic_fab_vars[MicVar_Morr::qci]->array(mfi);
        auto& qpr = mic_fab_vars[MicVar_Morr::qpr]->array(mfi);
        auto& qps = mic_fab_vars[MicVar_Morr::qps]->array(mfi);
        auto& qpg = mic_fab_vars[MicVar_Morr::qpg]->array(mfi);
        
        auto& qn = mic_fab_vars[MicVar_Morr::qn]->array(mfi);
        auto& qt = mic_fab_vars[MicVar_Morr::qt]->array(mfi);
        auto& qp = mic_fab_vars[MicVar_Morr::qp]->array(mfi);
        auto& omega = mic_fab_vars[MicVar_Morr::omega]->array(mfi);
        
        // Copy conserved variables to microphysics variables
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Copy basic state variables
            rho(i,j,k) = cons(i,j,k,rho_comp);
            
            // Calculate potential temperature from conserved rhotheta
            theta(i,j,k) = cons(i,j,k,rhotheta_comp) / rho(i,j,k);
            
            // Copy moisture variables
            qv(i,j,k) = cons(i,j,k,qv_comp);
            qcl(i,j,k) = cons(i,j,k,qc_comp);
            qci(i,j,k) = cons(i,j,k,qi_comp);
            qpr(i,j,k) = cons(i,j,k,qr_comp);
            qps(i,j,k) = cons(i,j,k,qs_comp);
            qpg(i,j,k) = cons(i,j,k,qg_comp);
            
            // Ensure non-negative values for all hydrometeors
            qv(i,j,k) = amrex::max(qv(i,j,k), 0.0);
            qcl(i,j,k) = amrex::max(qcl(i,j,k), 0.0);
            qci(i,j,k) = amrex::max(qci(i,j,k), 0.0);
            qpr(i,j,k) = amrex::max(qpr(i,j,k), 0.0);
            qps(i,j,k) = amrex::max(qps(i,j,k), 0.0);
            qpg(i,j,k) = amrex::max(qpg(i,j,k), 0.0);
            
            // Calculate derived quantities
            qn(i,j,k) = qcl(i,j,k) + qci(i,j,k);           // Total non-precipitating condensate
            qt(i,j,k) = qv(i,j,k) + qn(i,j,k);             // Total non-precipitating water
            qp(i,j,k) = qpr(i,j,k) + qps(i,j,k) + qpg(i,j,k); // Total precipitation
            
            // Calculate temperature from potential temperature
            // This assumes an initial pressure field is already available
            const amrex::Real exner = std::pow(pres(i,j,k)/100000.0, m_rdOcp);
            tabs(i,j,k) = theta(i,j,k) * exner;
            
            // Calculate vertical velocity (needed for microphysics calculations)
            omega(i,j,k) = cons(i,j,k,momz_comp) / rho(i,j,k);
            
            // In a full implementation, we would also copy number concentrations
            // for each hydrometeor species from the conserved state
        });
    }
}

/**
 * Copy microphysics variables back to state variables.
 * This function transfers data from the microphysics internal variables
 * to the model's conserved variables.
 *
 * @param[in,out] cons_in Output MultiFab for conserved variables
 */
void
Morrison::Copy_Micro_to_State(amrex::MultiFab& cons_in)
{
    BL_PROFILE("Morrison::Copy_Micro_to_State()");

    // Component indices for conserved state variables
    const int rho_comp = 0;   // Density
    const int momx_comp = 1;  // X-momentum
    const int momy_comp = 2;  // Y-momentum
    const int momz_comp = 3;  // Z-momentum
    const int energy_comp = 4; // Total energy
    const int rhotheta_comp = 5; // Density * theta
    
    // Component indices for moisture variables
    const int qv_comp = 6;    // Water vapor
    const int qc_comp = 7;    // Cloud water
    const int qi_comp = 8;    // Cloud ice
    const int qr_comp = 9;    // Rain
    const int qs_comp = 10;   // Snow
    const int qg_comp = 11;   // Graupel
    
    // Additional number concentration indices
    const int nc_comp = 12;   // Cloud droplet number
    const int ni_comp = 13;   // Cloud ice number
    const int nr_comp = 14;   // Rain number
    const int ns_comp = 15;   // Snow number
    const int ng_comp = 16;   // Graupel number

    // Loop through grids
    for (amrex::MFIter mfi(cons_in); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        
        // Get array accessors for conserved variables
        auto& cons = cons_in.array(mfi);
        
        // Get array accessors for microphysics variables
        auto const& rho = mic_fab_vars[MicVar_Morr::rho]->array(mfi);
        auto const& theta = mic_fab_vars[MicVar_Morr::theta]->array(mfi);
        auto const& tabs = mic_fab_vars[MicVar_Morr::tabs]->array(mfi);
        
        auto const& qv = mic_fab_vars[MicVar_Morr::qv]->array(mfi);
        auto const& qcl = mic_fab_vars[MicVar_Morr::qcl]->array(mfi);
        auto const& qci = mic_fab_vars[MicVar_Morr::qci]->array(mfi);
        auto const& qpr = mic_fab_vars[MicVar_Morr::qpr]->array(mfi);
        auto const& qps = mic_fab_vars[MicVar_Morr::qps]->array(mfi);
        auto const& qpg = mic_fab_vars[MicVar_Morr::qpg]->array(mfi);
        
        auto const& omega = mic_fab_vars[MicVar_Morr::omega]->array(mfi);
        
        // Copy microphysics variables back to conserved variables
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Update conserved thermodynamic variables
            cons(i,j,k,rhotheta_comp) = rho(i,j,k) * theta(i,j,k);
            
            // Update moisture variables
            cons(i,j,k,qv_comp) = qv(i,j,k);
            cons(i,j,k,qc_comp) = qcl(i,j,k);
            cons(i,j,k,qi_comp) = qci(i,j,k);
            cons(i,j,k,qr_comp) = qpr(i,j,k);
            cons(i,j,k,qs_comp) = qps(i,j,k);
            cons(i,j,k,qg_comp) = qpg(i,j,k);
            
            // Update momentum (vertical component may have changed if including sedimentation drag)
            cons(i,j,k,momz_comp) = rho(i,j,k) * omega(i,j,k);
            
            // Update energy
            // In a full implementation, we would also need to recalculate the total energy
            // based on the updated temperature, moisture, and potential energy
            
            // In a full implementation, we would also copy number concentrations
            // for each hydrometeor species back to the conserved state
        });
    }
}

/**
 * Compute diagnostic coefficients needed for microphysics calculations.
 * These include size distribution parameters, fall speeds, and other
 * derived quantities.
 */
void
Morrison::Compute_Coefficients()
{
    BL_PROFILE("Morrison::Compute_Coefficients()");

    // Loop through grids
    for (amrex::MFIter mfi(*mic_fab_vars[MicVar_Morr::qcl]); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        
        // Get array accessors for microphysics variables
        auto const& qcl = mic_fab_vars[MicVar_Morr::qcl]->array(mfi);
        auto const& qci = mic_fab_vars[MicVar_Morr::qci]->array(mfi);
        auto const& qpr = mic_fab_vars[MicVar_Morr::qpr]->array(mfi);
        auto const& qps = mic_fab_vars[MicVar_Morr::qps]->array(mfi);
        auto const& qpg = mic_fab_vars[MicVar_Morr::qpg]->array(mfi);
        auto const& rho = mic_fab_vars[MicVar_Morr::rho]->array(mfi);
        auto const& tabs = mic_fab_vars[MicVar_Morr::tabs]->array(mfi);
        auto const& pres = mic_fab_vars[MicVar_Morr::pres]->array(mfi);
        
        // Create temporary arrays for the coefficients
        amrex::FArrayBox lam_fab(box, 5);  // Lambda parameters for size distributions
        auto lamr = lam_fab.array(0);      // Rain lambda
        auto lams = lam_fab.array(1);      // Snow lambda
        auto lami = lam_fab.array(2);      // Cloud ice lambda
        auto lamc = lam_fab.array(3);      // Cloud water lambda
        auto lamg = lam_fab.array(4);      // Graupel lambda
        
        amrex::FArrayBox n0_fab(box, 5);   // Intercept parameters for size distributions
        auto n0r = n0_fab.array(0);        // Rain N0
        auto n0s = n0_fab.array(1);        // Snow N0
        auto n0i = n0_fab.array(2);        // Cloud ice N0
        auto n0c = n0_fab.array(3);        // Cloud water N0
        auto n0g = n0_fab.array(4);        // Graupel N0
        
        amrex::FArrayBox pgam_fab(box, 1); // Spectral shape parameter for droplets
        auto pgam = pgam_fab.array(0);
        
        // Calculate distribution parameters
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Number concentrations (these would come from state in a full implementation)
            // For this example, we'll use typical values based on mixing ratios
            
            // Cloud droplet number (assume 100 cm^-3)
            amrex::Real nc = 100.0e6 / rho(i,j,k);  // convert to kg^-1
            
            // Rain number concentration
            amrex::Real nr = 0.0;
            if (qpr(i,j,k) > m_qsmall) {
                // Assume typical raindrop diameter of ~1 mm
                const amrex::Real typical_mass_r = 4.0/3.0 * amrex::Math::pi * m_rhow * std::pow(0.5e-3, 3);
                nr = qpr(i,j,k) / typical_mass_r;
            }
            
            // Snow number concentration
            amrex::Real ns = 0.0;
            if (qps(i,j,k) > m_qsmall) {
                // Assume typical snow diameter of ~2 mm
                const amrex::Real typical_mass_s = 4.0/3.0 * amrex::Math::pi * m_rhosn * std::pow(1.0e-3, 3);
                ns = qps(i,j,k) / typical_mass_s;
            }
            
            // Ice number concentration
            amrex::Real ni = 0.0;
            if (qci(i,j,k) > m_qsmall) {
                // Assume typical ice diameter of ~50 microns
                const amrex::Real typical_mass_i = 4.0/3.0 * amrex::Math::pi * m_rhoi * std::pow(25.0e-6, 3);
                ni = qci(i,j,k) / typical_mass_i;
            }
            
            // Graupel number concentration
            amrex::Real ng = 0.0;
            if (qpg(i,j,k) > m_qsmall) {
                // Assume typical graupel diameter of ~3 mm
                const amrex::Real typical_mass_g = 4.0/3.0 * amrex::Math::pi * m_rhog * std::pow(1.5e-3, 3);
                ng = qpg(i,j,k) / typical_mass_g;
            }
            
            // Calculate size distribution parameters for each species
            
            // Cloud droplets
            if (qcl(i,j,k) >= m_qsmall) {
                // Calculate spectral shape parameter using Martin et al. (1994) formula
                // Lines ~1750-1755
                const amrex::Real dum = pres(i,j,k) / (287.15 * tabs(i,j,k));
                pgam(i,j,k) = 0.0005714 * (nc * rho(i,j,k) / 1.0e6) + 0.2714;
                pgam(i,j,k) = 1.0 / (pgam(i,j,k) * pgam(i,j,k)) - 1.0;
                pgam(i,j,k) = amrex::max(pgam(i,j,k), 2.0);
                pgam(i,j,k) = amrex::min(pgam(i,j,k), 10.0);
                
                // Calculate lambda parameter - Line ~1758
                lamc(i,j,k) = std::pow((m_cons26 * nc * gamma_function(pgam(i,j,k) + 4.0) / 
                              (qcl(i,j,k) * gamma_function(pgam(i,j,k) + 1.0))), 1.0/3.0);
                
                // Apply limits to lambda - Lines ~1763-1772
                const amrex::Real lammin = (pgam(i,j,k) + 1.0) / 60.0e-6;
                const amrex::Real lammax = (pgam(i,j,k) + 1.0) / 1.0e-6;
                
                if (lamc(i,j,k) < lammin) {
                    lamc(i,j,k) = lammin;
                } else if (lamc(i,j,k) > lammax) {
                    lamc(i,j,k) = lammax;
                }
            }
            
            // Rain
            if (qpr(i,j,k) >= m_qsmall) {
                // Calculate lambda parameter - Line ~1692
                lamr(i,j,k) = std::pow(amrex::Math::pi * m_rhow * nr / qpr(i,j,k), 1.0/3.0);
                
                // Apply limits to lambda - Lines ~1698-1708
                if (lamr(i,j,k) < m_lamminr) {
                    lamr(i,j,k) = m_lamminr;
                    n0r(i,j,k) = std::pow(lamr(i,j,k), 4) * qpr(i,j,k) / (amrex::Math::pi * m_rhow);
                } else if (lamr(i,j,k) > m_lammaxr) {
                    lamr(i,j,k) = m_lammaxr;
                    n0r(i,j,k) = std::pow(lamr(i,j,k), 4) * qpr(i,j,k) / (amrex::Math::pi * m_rhow);
                } else {
                    n0r(i,j,k) = nr * lamr(i,j,k);
                }
            }
            
            // Snow
            if (qps(i,j,k) >= m_qsmall) {
                // Calculate lambda parameter - Line ~1782
                lams(i,j,k) = std::pow(m_cons1 * ns / qps(i,j,k), 1.0/m_ds);
                
                // Apply limits to lambda - Lines ~1788-1799
                if (lams(i,j,k) < m_lammins) {
                    lams(i,j,k) = m_lammins;
                    n0s(i,j,k) = std::pow(lams(i,j,k), 4) * qps(i,j,k) / m_cons1;
                } else if (lams(i,j,k) > m_lammaxs) {
                    lams(i,j,k) = m_lammaxs;
                    n0s(i,j,k) = std::pow(lams(i,j,k), 4) * qps(i,j,k) / m_cons1;
                } else {
                    n0s(i,j,k) = ns * lams(i,j,k);
                }
            }
            
            // Cloud ice
            if (qci(i,j,k) >= m_qsmall) {
                // Calculate lambda parameter - Lines ~1691-1692
                lami(i,j,k) = std::pow(m_cons12 * ni / qci(i,j,k), 1.0/m_di);
                
                // Apply limits to lambda - Lines ~1697-1708
                if (lami(i,j,k) < m_lammini) {
                    lami(i,j,k) = m_lammini;
                    n0i(i,j,k) = std::pow(lami(i,j,k), 4) * qci(i,j,k) / m_cons12;
                } else if (lami(i,j,k) > m_lammaxi) {
                    lami(i,j,k) = m_lammaxi;
                    n0i(i,j,k) = std::pow(lami(i,j,k), 4) * qci(i,j,k) / m_cons12;
                } else {
                    n0i(i,j,k) = ni * lami(i,j,k);
                }
            }
            
            // Graupel
            if (qpg(i,j,k) >= m_qsmall) {
                // Calculate lambda parameter
                lamg(i,j,k) = std::pow(m_cons2 * ng / qpg(i,j,k), 1.0/m_dg);
                
                // Apply limits to lambda
                if (lamg(i,j,k) < m_lamming) {
                    lamg(i,j,k) = m_lamming;
                    n0g(i,j,k) = std::pow(lamg(i,j,k), 4) * qpg(i,j,k) / m_cons2;
                } else if (lamg(i,j,k) > m_lammaxg) {
                    lamg(i,j,k) = m_lammaxg;
                    n0g(i,j,k) = std::pow(lamg(i,j,k), 4) * qpg(i,j,k) / m_cons2;
                } else {
                    n0g(i,j,k) = ng * lamg(i,j,k);
                }
            }
        });
    }
}
