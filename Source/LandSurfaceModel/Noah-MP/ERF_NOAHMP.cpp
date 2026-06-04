
#include<iostream>
#include<string>

#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_ParallelDescriptor.H>

#include <ERF_NOAHMP.H>
#include <ERF_Constants.H>
#include <ERF_EOS.H>

using namespace amrex;

#ifdef ERF_USE_NOAHMP_MPMD
namespace {
int s_app0_root_global = -1;
int s_app1_root_global = -1;
}

void
NOAHMP::SetMPMDRootRanks (int app0_root_global, int app1_root_global)
{
    s_app0_root_global = app0_root_global;
    s_app1_root_global = app1_root_global;
}
#endif

/* Initialize lsm data structures */
void
NOAHMP::Init (const int& lev,
              const MultiFab& cons_in,
              const Geometry& geom,
              const Real& dt)
{

    m_dt   = dt;
    m_geom = geom;

    Box domain = geom.Domain();
    khi_lsm    = domain.smallEnd(2) - 1;

    LsmDataMap.resize(m_lsm_data_size);
    LsmDataMap = {LsmData_NOAHMP::t_sfc             , LsmData_NOAHMP::sfc_emis          ,
                  LsmData_NOAHMP::sfc_alb_dir_vis   , LsmData_NOAHMP::sfc_alb_dir_nir   ,
                  LsmData_NOAHMP::sfc_alb_dif_vis   , LsmData_NOAHMP::sfc_alb_dif_nir   ,
                  LsmData_NOAHMP::cos_zenith_angle  , LsmData_NOAHMP::sw_flux_dn        ,
                  LsmData_NOAHMP::sw_flux_dn_dir_vis, LsmData_NOAHMP::sw_flux_dn_dir_nir,
                  LsmData_NOAHMP::sw_flux_dn_dif_vis, LsmData_NOAHMP::sw_flux_dn_dif_nir,
                  LsmData_NOAHMP::lw_flux_dn        };
    LsmDataName.resize(m_lsm_data_size);
    LsmDataName = {"t_sfc"             , "sfc_emis"          ,
                   "sfc_alb_dir_vis"   , "sfc_alb_dir_nir"   ,
                   "sfc_alb_dif_vis"   , "sfc_alb_dif_nir"   ,
                   "cos_zenith_angle"  , "sw_flux_dn"        ,
                   "sw_flux_dn_dir_vis", "sw_flux_dn_dir_nir",
                   "sw_flux_dn_dif_vis", "sw_flux_dn_dif_nir",
                   "lw_flux_dn"        };


    LsmFluxMap.resize(m_lsm_flux_size);
    LsmFluxMap = {LsmFlux_NOAHMP::t_flux         , LsmFlux_NOAHMP::q_flux         ,
                  LsmFlux_NOAHMP::tau13          , LsmFlux_NOAHMP::tau23          };
    LsmFluxName.resize(m_lsm_flux_size);
    LsmFluxName = {"t_flux"         , "q_flux"         ,
                   "tau13"          , "tau23"          };

    ParmParse pp("erf");
    pp.query("plot_int_1" , m_plot_int_1);

    amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                      << " rank " << amrex::ParallelDescriptor::MyProc()
                      << "] NOAHMP::Init start lev=" << lev
                      << " domain lo=(" << domain.smallEnd(0) << "," << domain.smallEnd(1) << "," << domain.smallEnd(2) << ")"
                      << " hi=(" << domain.bigEnd(0) << "," << domain.bigEnd(1) << "," << domain.bigEnd(2) << ")"
                      << " cons boxes=" << cons_in.boxArray().size()
                      << " local boxes=" << cons_in.local_size()
                      << " dt=" << dt
                      << std::endl;

    // NOTE: All boxes in ba extend from zlo to zhi, so this transform is valid.
    //       If that were to change, the dm and new ba are no longer valid and
    //       direct copying between lsm data/flux vars cannot be done in a parfor.

    // Set 2D box array for lsm data
    IntVect ng(1,1,0);
    BoxArray ba = cons_in.boxArray();
    DistributionMapping dm = cons_in.DistributionMap();
    BoxList bl_lsm = ba.boxList();
    for (auto& b : bl_lsm) { b.setRange(2,0); }
    BoxArray ba_lsm(std::move(bl_lsm));

    // Set up lsm geometry
    const RealBox& dom_rb = m_geom.ProbDomain();
    const Real*    dom_dx = m_geom.CellSize();
    RealBox lsm_rb = dom_rb;
    Real lsm_dx[AMREX_SPACEDIM] = {AMREX_D_DECL(dom_dx[0],dom_dx[1],m_dz_lsm)};
    Real lsm_z_hi = dom_rb.lo(2);
    Real lsm_z_lo = lsm_z_hi - Real(m_nz_lsm)*lsm_dx[2];
    lsm_rb.setHi(2,lsm_z_hi); lsm_rb.setLo(2,lsm_z_lo);
    m_lsm_geom.define( ba_lsm.minimalBox(), lsm_rb, m_geom.Coord(), m_geom.isPeriodic() );

    // Create the data
    for (auto ivar = 0; ivar < LsmData_NOAHMP::NumVars; ++ivar) {
        // State vars are CC
        lsm_fab_data[ivar] = std::make_shared<MultiFab>(ba_lsm, dm, 1, ng);

        // NOTE: Radiation steps first so we set values
        //       to reasonable initialization for coupling
        Real val_to_set = zero;
        if (ivar == LsmData_NOAHMP::t_sfc) {
            val_to_set = Real(300.0);
        } else if (ivar == LsmData_NOAHMP::sfc_emis) {
            val_to_set = Real(0.9);
        } else if ( (ivar>=LsmData_NOAHMP::sfc_alb_dir_vis) &&
                    (ivar<=LsmData_NOAHMP::sfc_alb_dif_nir) ) {
            val_to_set = Real(0.06);
        } else {
            val_to_set = zero;
        }
        lsm_fab_data[ivar]->setVal(val_to_set);
    }

    // Create the fluxes
    for (auto ivar = 0; ivar < LsmFlux_NOAHMP::NumVars; ++ivar) {
        // NOTE: Fluxes are CC with ghost cells for averaging
        lsm_fab_flux[ivar] = std::make_shared<MultiFab>(ba_lsm, dm, 1, IntVect(1,1,0));
        lsm_fab_flux[ivar]->setVal(0.);
    }

    Print() << "Noah-MP initialization started" << std::endl;

    // Set noahmpio_vect to the size of local blocks (boxes)
    noahmpio_vect.resize(cons_in.local_size(), lev);

    int klo = domain.smallEnd(2);

    // Create a 2D slab BoxArray from the 3D domain at the surface (Used by ALL builds)
    amrex::BoxArray ba2d = cons_in.boxArray();
    for (int i = 0; i < ba2d.size(); ++i) {
        amrex::Box bx2d = ba2d[i];
        bx2d.makeSlab(2, klo);
        ba2d.set(i, bx2d);
    }

    amrex::DistributionMapping dm2d = cons_in.DistributionMap();

    // Keep these as MultiFabs for MPMD::Copier, but place them in pinned host
    // memory because the Noah-MP coupling path accesses them on both GPU and CPU.
    amrex::MFInfo info;
    info.SetArena(amrex::The_Pinned_Arena());

    mf_noah_input  = std::make_unique<amrex::MultiFab>(ba2d, dm2d, NoahmpInputComp::NumComps, 0, info);
    mf_noah_output = std::make_unique<amrex::MultiFab>(ba2d, dm2d, NoahmpOutputComp::NumComps, 0, info);

    amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                      << " rank " << amrex::ParallelDescriptor::MyProc()
                      << "] NOAHMP::Init built ba2d size=" << ba2d.size()
                      << " input comps=" << mf_noah_input->nComp()
                      << " output comps=" << mf_noah_output->nComp()
                      << " input local size=" << mf_noah_input->local_size()
                      << std::endl;

#ifdef ERF_USE_NOAHMP_MPMD
    // Only initialize the Copier if doing an MPMD run
    mpmd_copier = std::make_unique<amrex::MPMD::Copier>(ba2d, dm2d);
#endif

    // Iterate over multifab and noahmpio object together
    int idb = 0;
    for (MFIter mfi(cons_in); mfi.isValid(); ++mfi, ++idb) {
        Box bx = mfi.tilebox();
        if (bx.smallEnd(2) != klo) { continue; }

        Box bx2d = bx;
        bx2d.makeSlab(2, klo);

        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Init block " << idb
                          << " tilebox lo=(" << bx.smallEnd(0) << "," << bx.smallEnd(1) << "," << bx.smallEnd(2) << ")"
                          << " hi=(" << bx.bigEnd(0) << "," << bx.bigEnd(1) << "," << bx.bigEnd(2) << ")"
                          << " slab lo=(" << bx2d.smallEnd(0) << "," << bx2d.smallEnd(1) << "," << bx2d.smallEnd(2) << ")"
                          << " hi=(" << bx2d.bigEnd(0) << "," << bx2d.bigEnd(1) << "," << bx2d.bigEnd(2) << ")"
                          << std::endl;

        // Get reference to the noahmpio object
        NoahmpIO_type* noahmpio = &noahmpio_vect[idb];

        // Pass idb context to noahmpio
        noahmpio->blkid = idb;

        // Pass level context to noahmpio
        noahmpio->level = lev;

        // Initialize scalar values
        noahmpio->ScalarInitDefault();

        // Store the rank of process for noahmp
        noahmpio->rank = ParallelDescriptor::MyProc();

        // Store parallel communicator for noahmp
        noahmpio->comm = MPI_Comm_c2f(ParallelDescriptor::Communicator());

        // Read namelist.erf file
        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Init block " << idb << " ReadNamelist begin" << std::endl;
        noahmpio->ReadNamelist();

        // Read the headers from the NetCDF land file
        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Init block " << idb << " ReadLandHeader begin" << std::endl;
        noahmpio->ReadLandHeader();

        noahmpio->xstart = bx.smallEnd(0);
        noahmpio->xend   = bx.bigEnd(0);
        noahmpio->ystart = bx.smallEnd(1);
        noahmpio->yend   = bx.bigEnd(1);

        // Domain bounds
        noahmpio->ids = noahmpio->xstart;
        noahmpio->ide = noahmpio->xend;
        noahmpio->jds = noahmpio->ystart;
        noahmpio->jde = noahmpio->yend;
        noahmpio->kds = 1;
        noahmpio->kde = 2;

        // Tile bounds
        noahmpio->its = noahmpio->xstart;
        noahmpio->ite = noahmpio->xend;
        noahmpio->jts = noahmpio->ystart;
        noahmpio->jte = noahmpio->yend;
        noahmpio->kts = 1;
        noahmpio->kte = 2;

        // Memory bounds
        noahmpio->ims = noahmpio->xstart;
        noahmpio->ime = noahmpio->xend;
        noahmpio->jms = noahmpio->ystart;
        noahmpio->jme = noahmpio->yend;
        noahmpio->kms = 1;
        noahmpio->kme = 2;

        // Allocate memory in Fortran for IO variables
        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Init block " << idb << " VarInitDefault begin" << std::endl;
        noahmpio->VarInitDefault();

        // Read NoahmpTable.TBL
        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Init block " << idb << " ReadTable begin" << std::endl;
        noahmpio->ReadTable();

        // Read and initialize data from the NetCDF land file
        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Init block " << idb << " ReadLandMain begin" << std::endl;
        noahmpio->ReadLandMain();

        // Compute additional initial values
        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Init block " << idb << " InitMain begin" << std::endl;
        noahmpio->InitMain();

        // Write initial plotfile
#ifdef ERF_USE_NOAHMP_MPMD
        if (amrex::MPMD::AppNum() == 0) {
#endif
        Print() << "Noah-MP writing lnd.nc file at lev: " << lev << std::endl;
        noahmpio->WriteLand(0);
#ifdef ERF_USE_NOAHMP_MPMD
        }
#endif
        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Init block " << idb << " completed" << std::endl;
    }

  Print() << "Noah-MP initialization completed" << std::endl;

};

void
NOAHMP::Plot_Landfile(const int& nstep)
{
    for (NoahmpIO_type &noahmpio : noahmpio_vect) {
        noahmpio.WriteLand(nstep);
    }
}

void
NOAHMP::Advance_With_State (const int& lev,
                            MultiFab& cons_in,
                            MultiFab& xvel_in,
                            MultiFab& yvel_in,
                            MultiFab* /*hfx3_out*/,
                            MultiFab* /*qfx3_out*/,
                            const Real& dt,
                            const int& nstep)
{
    Box domain = m_geom.Domain();

    Print () << "Noah-MP driver started at time step: " << nstep+1 << std::endl;
    amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                      << " rank " << amrex::ParallelDescriptor::MyProc()
                      << "] NOAHMP::Advance_With_State start lev=" << lev
                      << " step=" << nstep+1
                      << " domain lo=(" << domain.smallEnd(0) << "," << domain.smallEnd(1) << "," << domain.smallEnd(2) << ")"
                      << " hi=(" << domain.bigEnd(0) << "," << domain.bigEnd(1) << "," << domain.bigEnd(2) << ")"
                      << " cons local boxes=" << cons_in.local_size()
                      << std::endl;

    bool is_moist = (cons_in.nComp() > RhoQ1_comp);
    int klo = domain.smallEnd(2);

    int idb = 0;
    for (MFIter mfi(cons_in); mfi.isValid(); ++mfi, ++idb) {
        Box bx  = mfi.tilebox();
        Box gbx = mfi.tilebox(IntVect(0,0,0),IntVect(1,1,0));

        if (bx.smallEnd(2) != klo) { continue; }

        bx.makeSlab(2,klo);
        gbx.makeSlab(2,klo);

        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Advance_With_State block " << idb
                          << " bx lo=(" << bx.smallEnd(0) << "," << bx.smallEnd(1) << "," << bx.smallEnd(2) << ")"
                          << " hi=(" << bx.bigEnd(0) << "," << bx.bigEnd(1) << "," << bx.bigEnd(2) << ")"
                          << " gbx lo=(" << gbx.smallEnd(0) << "," << gbx.smallEnd(1) << "," << gbx.smallEnd(2) << ")"
                          << " hi=(" << gbx.bigEnd(0) << "," << gbx.bigEnd(1) << "," << gbx.bigEnd(2) << ")"
                          << std::endl;

        const Array4<const Real>& U_PHY  = xvel_in.const_array(mfi);
        const Array4<const Real>& V_PHY  = yvel_in.const_array(mfi);
        const Array4<const Real>& CONS   = cons_in.const_array(mfi);

        const Array4<const Real>& SWDOWN = lsm_fab_data[LsmData_NOAHMP::sw_flux_dn]->const_array(mfi);
        const Array4<const Real>& GLW    = lsm_fab_data[LsmData_NOAHMP::lw_flux_dn]->const_array(mfi);
        const Array4<const Real>& COSZEN = lsm_fab_data[LsmData_NOAHMP::cos_zenith_angle]->const_array(mfi);

        Array4<Real> TSK           = lsm_fab_data[LsmData_NOAHMP::t_sfc]->array(mfi);
        Array4<Real> EMISS         = lsm_fab_data[LsmData_NOAHMP::sfc_emis]->array(mfi);
        Array4<Real> ALBSFCDIR_VIS = lsm_fab_data[LsmData_NOAHMP::sfc_alb_dir_vis]->array(mfi);
        Array4<Real> ALBSFCDIR_NIR = lsm_fab_data[LsmData_NOAHMP::sfc_alb_dir_nir]->array(mfi);
        Array4<Real> ALBSFCDIF_VIS = lsm_fab_data[LsmData_NOAHMP::sfc_alb_dif_vis]->array(mfi);
        Array4<Real> ALBSFCDIF_NIR = lsm_fab_data[LsmData_NOAHMP::sfc_alb_dif_nir]->array(mfi);

        Array4<Real> q_flux_arr    = lsm_fab_flux[LsmFlux_NOAHMP::q_flux]->array(mfi);
        Array4<Real> t_flux_arr    = lsm_fab_flux[LsmFlux_NOAHMP::t_flux]->array(mfi);
        Array4<Real> tau13_arr     = lsm_fab_flux[LsmFlux_NOAHMP::tau13]->array(mfi);
        Array4<Real> tau23_arr     = lsm_fab_flux[LsmFlux_NOAHMP::tau23]->array(mfi);

        Array4<Real> noah_input_arr  = mf_noah_input->array(mfi);
        Array4<Real> noah_output_arr = mf_noah_output->array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real qv = (is_moist) ? CONS(i,j,k,RhoQ1_comp)/CONS(i,j,k,Rho_comp) : zero;
            noah_input_arr(i,j,0,NoahmpInputComp::u_phy)   = myhalf*(U_PHY(i,j,k)+U_PHY(i+1,j,k));
            noah_input_arr(i,j,0,NoahmpInputComp::v_phy)   = myhalf*(V_PHY(i,j,k)+V_PHY(i  ,j+1,k));
            noah_input_arr(i,j,0,NoahmpInputComp::t_phy)   = getTgivenRandRTh(CONS(i,j,k,Rho_comp),CONS(i,j,k,RhoTheta_comp),qv);
            noah_input_arr(i,j,0,NoahmpInputComp::qv_curr) = qv;
            noah_input_arr(i,j,0,NoahmpInputComp::p8w)     = getPgivenRTh(CONS(i,j,k,RhoTheta_comp),qv);
            noah_input_arr(i,j,0,NoahmpInputComp::swdown)  = SWDOWN(i,j,0);
            noah_input_arr(i,j,0,NoahmpInputComp::glw)     = GLW(i,j,0);
            noah_input_arr(i,j,0,NoahmpInputComp::coszen)  = COSZEN(i,j,0);
        });

        Gpu::streamSynchronize();
        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Advance_With_State block " << idb
                          << " completed ERF-to-Noahmp forcing pack" << std::endl;

#ifndef ERF_USE_NOAHMP_MPMD
        NoahmpIO_type* noahmpio = &noahmpio_vect[idb];

        LoopOnCpu(bx, [&] (int i, int j, int ) noexcept
        {
            noahmpio->U_PHY(i,1,j)   = noah_input_arr(i,j,0,NoahmpInputComp::u_phy);
            noahmpio->V_PHY(i,1,j)   = noah_input_arr(i,j,0,NoahmpInputComp::v_phy);
            noahmpio->T_PHY(i,1,j)   = noah_input_arr(i,j,0,NoahmpInputComp::t_phy);
            noahmpio->QV_CURR(i,1,j) = noah_input_arr(i,j,0,NoahmpInputComp::qv_curr);
            noahmpio->P8W(i,1,j)     = noah_input_arr(i,j,0,NoahmpInputComp::p8w);
            noahmpio->SWDOWN(i,j)    = noah_input_arr(i,j,0,NoahmpInputComp::swdown);
            noahmpio->GLW(i,j)       = noah_input_arr(i,j,0,NoahmpInputComp::glw);
            noahmpio->COSZEN(i,j)    = noah_input_arr(i,j,0,NoahmpInputComp::coszen);
        });

        noahmpio->itimestep = nstep+1;
        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Advance_With_State block " << idb
                          << " calling DriverMain locally" << std::endl;
        noahmpio->DriverMain();

        LoopOnCpu(bx, [&] (int i, int j, int ) noexcept
        {
            noah_output_arr(i,j,0,NoahmpOutputComp::hfx)           = noahmpio->HFX(i,j);
            noah_output_arr(i,j,0,NoahmpOutputComp::lh)            = noahmpio->LH(i,j);
            noah_output_arr(i,j,0,NoahmpOutputComp::tau_ew)        = noahmpio->TAU_EW(i,j);
            noah_output_arr(i,j,0,NoahmpOutputComp::tau_ns)        = noahmpio->TAU_NS(i,j);
            noah_output_arr(i,j,0,NoahmpOutputComp::tsk)           = noahmpio->TSK(i,j);
            noah_output_arr(i,j,0,NoahmpOutputComp::emiss)         = noahmpio->EMISS(i,j);
            noah_output_arr(i,j,0,NoahmpOutputComp::albsfcdir_vis) = noahmpio->ALBSFCDIRXY(i,1,j);
            noah_output_arr(i,j,0,NoahmpOutputComp::albsfcdir_nir) = noahmpio->ALBSFCDIRXY(i,2,j);
            noah_output_arr(i,j,0,NoahmpOutputComp::albsfcdif_vis) = noahmpio->ALBSFCDIFXY(i,1,j);
            noah_output_arr(i,j,0,NoahmpOutputComp::albsfcdif_nir) = noahmpio->ALBSFCDIFXY(i,2,j);
        });

        int i_lo = bx.smallEnd(0); int i_hi = bx.bigEnd(0);
        int j_lo = bx.smallEnd(1); int j_hi = bx.bigEnd(1);
        ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int ii = std::min(std::max(i,i_lo),i_hi);
            int jj = std::min(std::max(j,j_lo),j_hi);

            t_flux_arr(i,j,k)    = noah_output_arr(ii,jj,0,NoahmpOutputComp::hfx)/(CONS(ii,jj,k,Rho_comp)*Cp_d);
            q_flux_arr(i,j,k)    = noah_output_arr(ii,jj,0,NoahmpOutputComp::lh)/(CONS(ii,jj,k,Rho_comp)*L_v);
            tau13_arr(i,j,k)     = noah_output_arr(ii,jj,0,NoahmpOutputComp::tau_ew)/CONS(ii,jj,k,Rho_comp);
            tau23_arr(i,j,k)     = noah_output_arr(ii,jj,0,NoahmpOutputComp::tau_ns)/CONS(ii,jj,k,Rho_comp);

            TSK(i,j,0)           = noah_output_arr(ii,jj,0,NoahmpOutputComp::tsk);
            EMISS(i,j,0)         = noah_output_arr(ii,jj,0,NoahmpOutputComp::emiss);
            ALBSFCDIR_VIS(i,j,0) = noah_output_arr(ii,jj,0,NoahmpOutputComp::albsfcdir_vis);
            ALBSFCDIR_NIR(i,j,0) = noah_output_arr(ii,jj,0,NoahmpOutputComp::albsfcdir_nir);
            ALBSFCDIF_VIS(i,j,0) = noah_output_arr(ii,jj,0,NoahmpOutputComp::albsfcdif_vis);
            ALBSFCDIF_NIR(i,j,0) = noah_output_arr(ii,jj,0,NoahmpOutputComp::albsfcdif_nir);
        });

        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Advance_With_State block " << idb
                          << " copied NoahMP output back to ERF arrays" << std::endl;
#else
        amrex::ignore_unused(dt, TSK, EMISS, ALBSFCDIR_VIS, ALBSFCDIR_NIR,
                             ALBSFCDIF_VIS, ALBSFCDIF_NIR, q_flux_arr,
                             t_flux_arr, tau13_arr, tau23_arr, gbx,
                             noah_output_arr);
#endif
    }

#ifdef ERF_USE_NOAHMP_MPMD
    if (amrex::MPMD::AppNum() == 0) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(s_app0_root_global >= 0 && s_app1_root_global >= 0,
                                         "NOAHMP MPMD roots were not initialized before Advance_With_State");

        int keep_running = 1;
        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Advance_With_State entering MPMD exchange for step "
                          << nstep+1 << std::endl;

        if (amrex::ParallelDescriptor::MyProc() == 0) {
            MPI_Send(&keep_running, 1, MPI_INT, s_app1_root_global,
                     NOAHMP::MPMDControlTag, MPI_COMM_WORLD);
        }

        mpmd_copier->send(*mf_noah_input, 0, NoahmpInputComp::NumComps);
        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Advance_With_State completed MPMD send, waiting for recv"
                          << std::endl;
        mpmd_copier->recv(*mf_noah_output, 0, NoahmpOutputComp::NumComps);
        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Advance_With_State completed MPMD recv"
                          << std::endl;
    }

    idb = 0;
    for (MFIter mfi(cons_in); mfi.isValid(); ++mfi, ++idb) {
        Box bx  = mfi.tilebox();
        Box gbx = mfi.tilebox(IntVect(0,0,0),IntVect(1,1,0));

        if (bx.smallEnd(2) != klo) { continue; }

        bx.makeSlab(2,klo);
        gbx.makeSlab(2,klo);

        int i_lo = bx.smallEnd(0); int i_hi = bx.bigEnd(0);
        int j_lo = bx.smallEnd(1); int j_hi = bx.bigEnd(1);

        const Array4<const Real>& CONS   = cons_in.const_array(mfi);
        Array4<Real> TSK                 = lsm_fab_data[LsmData_NOAHMP::t_sfc]->array(mfi);
        Array4<Real> EMISS               = lsm_fab_data[LsmData_NOAHMP::sfc_emis]->array(mfi);
        Array4<Real> ALBSFCDIR_VIS       = lsm_fab_data[LsmData_NOAHMP::sfc_alb_dir_vis]->array(mfi);
        Array4<Real> ALBSFCDIR_NIR       = lsm_fab_data[LsmData_NOAHMP::sfc_alb_dir_nir]->array(mfi);
        Array4<Real> ALBSFCDIF_VIS       = lsm_fab_data[LsmData_NOAHMP::sfc_alb_dif_vis]->array(mfi);
        Array4<Real> ALBSFCDIF_NIR       = lsm_fab_data[LsmData_NOAHMP::sfc_alb_dif_nir]->array(mfi);
        Array4<Real> q_flux_arr          = lsm_fab_flux[LsmFlux_NOAHMP::q_flux]->array(mfi);
        Array4<Real> t_flux_arr          = lsm_fab_flux[LsmFlux_NOAHMP::t_flux]->array(mfi);
        Array4<Real> tau13_arr           = lsm_fab_flux[LsmFlux_NOAHMP::tau13]->array(mfi);
        Array4<Real> tau23_arr           = lsm_fab_flux[LsmFlux_NOAHMP::tau23]->array(mfi);
        Array4<const Real> noah_output_arr = mf_noah_output->const_array(mfi);

        ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int ii = std::min(std::max(i,i_lo),i_hi);
            int jj = std::min(std::max(j,j_lo),j_hi);

            t_flux_arr(i,j,k)    = noah_output_arr(ii,jj,0,NoahmpOutputComp::hfx)/(CONS(ii,jj,k,Rho_comp)*Cp_d);
            q_flux_arr(i,j,k)    = noah_output_arr(ii,jj,0,NoahmpOutputComp::lh)/(CONS(ii,jj,k,Rho_comp)*L_v);
            tau13_arr(i,j,k)     = noah_output_arr(ii,jj,0,NoahmpOutputComp::tau_ew)/CONS(ii,jj,k,Rho_comp);
            tau23_arr(i,j,k)     = noah_output_arr(ii,jj,0,NoahmpOutputComp::tau_ns)/CONS(ii,jj,k,Rho_comp);

            TSK(i,j,0)           = noah_output_arr(ii,jj,0,NoahmpOutputComp::tsk);
            EMISS(i,j,0)         = noah_output_arr(ii,jj,0,NoahmpOutputComp::emiss);
            ALBSFCDIR_VIS(i,j,0) = noah_output_arr(ii,jj,0,NoahmpOutputComp::albsfcdir_vis);
            ALBSFCDIR_NIR(i,j,0) = noah_output_arr(ii,jj,0,NoahmpOutputComp::albsfcdir_nir);
            ALBSFCDIF_VIS(i,j,0) = noah_output_arr(ii,jj,0,NoahmpOutputComp::albsfcdif_vis);
            ALBSFCDIF_NIR(i,j,0) = noah_output_arr(ii,jj,0,NoahmpOutputComp::albsfcdif_nir);
        });

        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Advance_With_State block " << idb
                          << " copied NoahMP output back to ERF arrays" << std::endl;
    }
#endif

    for (auto ivar = 0; ivar < LsmFlux_NOAHMP::NumVars; ++ivar) {
        lsm_fab_flux[ivar]->FillBoundary(m_geom.periodicity());
    }
    Print () << "Noah-MP driver completed" << std::endl;
    amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                      << " rank " << amrex::ParallelDescriptor::MyProc()
                      << "] NOAHMP::Advance_With_State end step=" << nstep+1
                      << std::endl;
};

#ifdef ERF_USE_NOAHMP_MPMD
void
NOAHMP::Advance_MPMD_Only (const int& nstep)
{
    Print() << "Noah-MP MPMD step started at time step: " << nstep+1 << std::endl;
    amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                      << " rank " << amrex::ParallelDescriptor::MyProc()
                      << "] NOAHMP::Advance_MPMD_Only start step=" << nstep+1
                      << " local boxes=" << mf_noah_input->local_size()
                      << std::endl;

    int idb = 0;
    for (MFIter mfi(*mf_noah_input); mfi.isValid(); ++mfi, ++idb) {
        const Box& bx = mfi.tilebox();

        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Advance_MPMD_Only block " << idb
                          << " bx lo=(" << bx.smallEnd(0) << "," << bx.smallEnd(1) << "," << bx.smallEnd(2) << ")"
                          << " hi=(" << bx.bigEnd(0) << "," << bx.bigEnd(1) << "," << bx.bigEnd(2) << ")"
                          << std::endl;

        NoahmpIO_type* noahmpio = &noahmpio_vect[idb];

        const Array4<const Real>& noah_input_arr  = mf_noah_input->const_array(mfi);
        Array4<Real>              noah_output_arr = mf_noah_output->array(mfi);

        LoopOnCpu(bx, [&] (int i, int j, int ) noexcept
        {
            noahmpio->U_PHY(i,1,j)   = noah_input_arr(i,j,0,NoahmpInputComp::u_phy);
            noahmpio->V_PHY(i,1,j)   = noah_input_arr(i,j,0,NoahmpInputComp::v_phy);
            noahmpio->T_PHY(i,1,j)   = noah_input_arr(i,j,0,NoahmpInputComp::t_phy);
            noahmpio->QV_CURR(i,1,j) = noah_input_arr(i,j,0,NoahmpInputComp::qv_curr);
            noahmpio->P8W(i,1,j)     = noah_input_arr(i,j,0,NoahmpInputComp::p8w);
            noahmpio->SWDOWN(i,j)    = noah_input_arr(i,j,0,NoahmpInputComp::swdown);
            noahmpio->GLW(i,j)       = noah_input_arr(i,j,0,NoahmpInputComp::glw);
            noahmpio->COSZEN(i,j)    = noah_input_arr(i,j,0,NoahmpInputComp::coszen);
        });

        noahmpio->itimestep = nstep+1;
        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Advance_MPMD_Only block " << idb
                          << " calling DriverMain" << std::endl;
        noahmpio->DriverMain();

        LoopOnCpu(bx, [&] (int i, int j, int ) noexcept
        {
            noah_output_arr(i,j,0,NoahmpOutputComp::hfx)           = noahmpio->HFX(i,j);
            noah_output_arr(i,j,0,NoahmpOutputComp::lh)            = noahmpio->LH(i,j);
            noah_output_arr(i,j,0,NoahmpOutputComp::tau_ew)        = noahmpio->TAU_EW(i,j);
            noah_output_arr(i,j,0,NoahmpOutputComp::tau_ns)        = noahmpio->TAU_NS(i,j);
            noah_output_arr(i,j,0,NoahmpOutputComp::tsk)           = noahmpio->TSK(i,j);
            noah_output_arr(i,j,0,NoahmpOutputComp::emiss)         = noahmpio->EMISS(i,j);
            noah_output_arr(i,j,0,NoahmpOutputComp::albsfcdir_vis) = noahmpio->ALBSFCDIRXY(i,1,j);
            noah_output_arr(i,j,0,NoahmpOutputComp::albsfcdir_nir) = noahmpio->ALBSFCDIRXY(i,2,j);
            noah_output_arr(i,j,0,NoahmpOutputComp::albsfcdif_vis) = noahmpio->ALBSFCDIFXY(i,1,j);
            noah_output_arr(i,j,0,NoahmpOutputComp::albsfcdif_nir) = noahmpio->ALBSFCDIFXY(i,2,j);
        });

        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Advance_MPMD_Only block " << idb
                          << " completed output pack" << std::endl;
    }

    Print() << "Noah-MP MPMD step completed" << std::endl;
    amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                      << " rank " << amrex::ParallelDescriptor::MyProc()
                      << "] NOAHMP::Advance_MPMD_Only end step=" << nstep+1
                      << std::endl;
}

void NOAHMP::Run_MPMD_Advance()
{
    amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                      << " rank " << amrex::ParallelDescriptor::MyProc()
                      << "] NOAHMP::Run_MPMD_Advance entering" << std::endl;

    // 1. Get Domain Size
    amrex::ParmParse pp_amr("amr");
    amrex::Vector<int> n_cell(3);
    pp_amr.getarr("n_cell", n_cell);

    // 2. Get Physical Bounds (using prob_extent)
    amrex::ParmParse pp_geom("geometry");
    amrex::Vector<amrex::Real> prob_lo  = {0.0, 0.0, 0.0}; 
    amrex::Vector<amrex::Real> prob_extent(3);
    pp_geom.queryarr("prob_lo", prob_lo);
    pp_geom.getarr("prob_extent", prob_extent);

    // Calculate prob_hi dynamically
    amrex::Vector<amrex::Real> prob_hi(3);
    for (int i = 0; i < 3; ++i) {
        prob_hi[i] = prob_lo[i] + prob_extent[i];
    }

    // 3. Construct RealBox and Geometry
    amrex::RealBox lb(prob_lo.dataPtr(), prob_hi.dataPtr());
    
    amrex::Box domain_bx(amrex::IntVect(0,0,0), amrex::IntVect(n_cell[0]-1, n_cell[1]-1, 0));
    amrex::Geometry geom(domain_bx, &lb, amrex::CoordSys::cartesian, nullptr);
    amrex::BoxArray ba(domain_bx);

    amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                      << " rank " << amrex::ParallelDescriptor::MyProc()
                      << "] NOAHMP::Run_MPMD_Advance geometry domain lo=("
                      << domain_bx.smallEnd(0) << "," << domain_bx.smallEnd(1) << "," << domain_bx.smallEnd(2)
                      << ") hi=("
                      << domain_bx.bigEnd(0) << "," << domain_bx.bigEnd(1) << "," << domain_bx.bigEnd(2)
                      << ") prob_lo=(" << prob_lo[0] << "," << prob_lo[1] << "," << prob_lo[2]
                      << ") prob_hi=(" << prob_hi[0] << "," << prob_hi[1] << "," << prob_hi[2]
                      << ")" << std::endl;

    amrex::DistributionMapping dm;
    dm.RoundRobinProcessorMap(ba.size(), amrex::ParallelDescriptor::NProcs());

    amrex::MultiFab cons_dummy(ba, dm, 1, 0);

    amrex::Real dt = 0.0;
    amrex::ParmParse pp_erf("erf");
    pp_erf.query("dt", dt);

    NOAHMP lsm;
    lsm.Init(0, cons_dummy, geom, dt); // this creates mf_noah_input, mf_noah_output, and mpmd_copier
    amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                      << " rank " << amrex::ParallelDescriptor::MyProc()
                      << "] NOAHMP::Run_MPMD_Advance finished Init" << std::endl;

    int step = 0;
    int keep_running = 1;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(s_app0_root_global >= 0 && s_app1_root_global >= 0,
                                     "NOAHMP MPMD roots were not initialized before Run_MPMD_Advance");

    // The Adaptive MPMD Loop
    while (true) {
        // Wait for App 0 signal (0 = stop, 1 = continue)
        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Run_MPMD_Advance waiting for keep_running at step "
                          << step+1 << std::endl;
        if (amrex::ParallelDescriptor::MyProc() == 0) {
            MPI_Recv(&keep_running, 1, MPI_INT, s_app0_root_global,
                     NOAHMP::MPMDControlTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        amrex::ParallelDescriptor::Bcast(&keep_running, 1, 0);
        if (!keep_running) {
            amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                              << " rank " << amrex::ParallelDescriptor::MyProc()
                              << "] NOAHMP::Run_MPMD_Advance received stop signal at step "
                              << step+1 << std::endl;
            break;
        }

        // 1. Receive forcing data from ERF App 0
        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Run_MPMD_Advance receiving forcing for step "
                          << step+1 << std::endl;
        lsm.mpmd_copier->recv(*(lsm.mf_noah_input), 0, NoahmpInputComp::NumComps);

        // 2. Run Noah-MP Physics directly from the copied forcing fields.
        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Run_MPMD_Advance launching Advance_MPMD_Only for step "
                          << step+1 << std::endl;
        lsm.Advance_MPMD_Only(step);

        // 3. Send calculated fluxes back to ERF App 0
        amrex::AllPrint() << "[app " << amrex::MPMD::AppNum()
                          << " rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Run_MPMD_Advance sending fluxes for step "
                          << step+1 << std::endl;
        lsm.mpmd_copier->send(*(lsm.mf_noah_output), 0, NoahmpOutputComp::NumComps);

        step++;
    }
    amrex::Print() << "Noah-MP MPMD driver completed " << step << " steps and exited cleanly." << std::endl;
}
#endif
