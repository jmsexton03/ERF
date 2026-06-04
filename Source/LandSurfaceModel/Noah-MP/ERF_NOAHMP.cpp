
#include<iostream>
#include<string>

#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_ParallelDescriptor.H>

#include <ERF_NOAHMP.H>
#include <ERF_Constants.H>
#include <ERF_EOS.H>

using namespace amrex;

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

    // Use Pinned Memory for fast Host <-> Device copies
    amrex::MFInfo info;
    info.SetArena(amrex::The_Pinned_Arena());

    mf_noah_input  = std::make_unique<amrex::MultiFab>(ba2d, dm2d, NoahmpInputComp::NumComps, 0, info);
    mf_noah_output = std::make_unique<amrex::MultiFab>(ba2d, dm2d, NoahmpOutputComp::NumComps, 0, info);

#ifdef ERF_USE_NOAHMP_MPMD
    // Only initialize the Copier if doing an MPMD run
    mpmd_copier = std::make_unique<amrex::MPMD::Copier>(ba2d, dm2d);
#endif

    // Iterate over multifab and noahmpio object together
    int idb = 0;
    for (MFIter mfi(cons_in); mfi.isValid(); ++mfi, ++idb) {
        Box bx = mfi.tilebox();
        if (bx.smallEnd(2) != klo) { continue; }

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
        noahmpio->ReadNamelist();

        // Read the headers from the NetCDF land file
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
        noahmpio->VarInitDefault();

        // Read NoahmpTable.TBL
        noahmpio->ReadTable();

        // Read and initialize data from the NetCDF land file
        noahmpio->ReadLandMain();

        // Compute additional initial values
        noahmpio->InitMain();

        // Write initial plotfile
        Print() << "Noah-MP writing lnd.nc file at lev: " << lev << std::endl;
        noahmpio->WriteLand(0);
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

    bool is_moist = (cons_in.nComp() > RhoQ1_comp);

    int klo = domain.smallEnd(2);

    // Loop over blocks to copy forcing data to Noahmp, drive the land model,
    // and copy data back to ERF Multifabs.
    int idb = 0;
    for (MFIter mfi(cons_in); mfi.isValid(); ++mfi, ++idb) {

        Box bx  = mfi.tilebox();
        Box gbx = mfi.tilebox(IntVect(0,0,0),IntVect(1,1,0));

        // Check if tile is at the lower boundary in lower z direction
        if (bx.smallEnd(2) != klo) { continue; }

        bx.makeSlab(2,klo);
        gbx.makeSlab(2,klo);

        // For limiting when populating ghost cells
        int i_lo = bx.smallEnd(0); int i_hi = bx.bigEnd(0);
        int j_lo = bx.smallEnd(1); int j_hi = bx.bigEnd(1);

        NoahmpIO_type* noahmpio = &noahmpio_vect[idb];

        const Array4<const Real>& U_PHY  = xvel_in.const_array(mfi);
        const Array4<const Real>& V_PHY  = yvel_in.const_array(mfi);
        const Array4<const Real>& CONS   = cons_in.const_array(mfi);

        // Into NOAH-MP
        const Array4<const Real>& SWDOWN = lsm_fab_data[LsmData_NOAHMP::sw_flux_dn]->const_array(mfi);
        const Array4<const Real>& GLW    = lsm_fab_data[LsmData_NOAHMP::lw_flux_dn]->const_array(mfi);
        const Array4<const Real>& COSZEN = lsm_fab_data[LsmData_NOAHMP::cos_zenith_angle]->const_array(mfi);

        // Out of NOAH-MP
        Array4<Real> TSK           = lsm_fab_data[LsmData_NOAHMP::t_sfc]->array(mfi);
        Array4<Real> EMISS         = lsm_fab_data[LsmData_NOAHMP::sfc_emis]->array(mfi);
        Array4<Real> ALBSFCDIR_VIS = lsm_fab_data[LsmData_NOAHMP::sfc_alb_dir_vis]->array(mfi);
        Array4<Real> ALBSFCDIR_NIR = lsm_fab_data[LsmData_NOAHMP::sfc_alb_dir_nir]->array(mfi);
        Array4<Real> ALBSFCDIF_VIS = lsm_fab_data[LsmData_NOAHMP::sfc_alb_dif_vis]->array(mfi);
        Array4<Real> ALBSFCDIF_NIR = lsm_fab_data[LsmData_NOAHMP::sfc_alb_dif_nir]->array(mfi);

        // NOTE: Need to expose stresses and get stresses from NOAHMP
        Array4<Real> q_flux_arr    = lsm_fab_flux[LsmFlux_NOAHMP::q_flux]->array(mfi);
        Array4<Real> t_flux_arr    = lsm_fab_flux[LsmFlux_NOAHMP::t_flux]->array(mfi);
        Array4<Real> tau13_arr     = lsm_fab_flux[LsmFlux_NOAHMP::tau13]->array(mfi);
        Array4<Real> tau23_arr     = lsm_fab_flux[LsmFlux_NOAHMP::tau23]->array(mfi);

        // Use Pinned MultiFabs
        Array4<Real> noah_input_arr  = mf_noah_input->array(mfi);
        Array4<Real> noah_output_arr = mf_noah_output->array(mfi);

        // Copy forcing data from ERF to Noahmp.
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

        // Synchronize to ensure GPU kernel is complete before host access
        Gpu::streamSynchronize();

        // Now on the host, copy data to NoahmpIO arrays
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

        // Call the noahmpio driver code. This runs the land model forcing for
        // each object in noahmpio_vect that represent a block in the domain.
#ifndef ERF_USE_NOAHMP_MPMD
        noahmpio->itimestep = nstep+1;
        noahmpio->DriverMain();
#else
        if (amrex::MPMD::AppNum() == 0) {
            // Signal App 1 to keep running for this step
            int keep_running = 1;
            if (amrex::ParallelDescriptor::MyProc() == 0) {
                MPI_Bcast(&keep_running, 1, MPI_INT, 0, MPI_COMM_WORLD);
            } else {
                MPI_Bcast(&keep_running, 1, MPI_INT, MPI_PROC_NULL, MPI_COMM_WORLD);
            }

            // Send atmospheric forcing to App 1
            mpmd_copier->send(*mf_noah_input, 0, NoahmpInputComp::NumComps);

            // Receive calculated fluxes back from App 1
            mpmd_copier->recv(*mf_noah_output, 0, NoahmpOutputComp::NumComps);
        } else {
            // If we are App 1, we just run the local physics driver
            noahmpio->itimestep = nstep+1;
            noahmpio->DriverMain();
        }
#endif

        // Copy results from NoahmpIO back to temporary arrays
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

        // Copy forcing data from Noahmp to ERF
        ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // Limit indices to the valid box. FillBoundary will pick these up below.
            int ii = std::min(std::max(i,i_lo),i_hi);
            int jj = std::min(std::max(j,j_lo),j_hi);

            // SurfaceLayer fluxes at CC
            t_flux_arr(i,j,k)    = noah_output_arr(ii,jj,0,NoahmpOutputComp::hfx)/(CONS(ii,jj,k,Rho_comp)*Cp_d);
            q_flux_arr(i,j,k)    = noah_output_arr(ii,jj,0,NoahmpOutputComp::lh)/(CONS(ii,jj,k,Rho_comp)*L_v);

            // NOTE: The following fluxes are nodal in xz/yz.
            //       The 2D MFs have 1 ghost cell so we can average these
            //       when using them in the surface layer class.
            tau13_arr(i,j,k)  = noah_output_arr(ii,jj,0,NoahmpOutputComp::tau_ew)/CONS(ii,jj,k,Rho_comp);
            tau23_arr(i,j,k)  = noah_output_arr(ii,jj,0,NoahmpOutputComp::tau_ns)/CONS(ii,jj,k,Rho_comp);

            // RRTMGP variables
            TSK(i,j,0)           = noah_output_arr(ii,jj,0,NoahmpOutputComp::tsk);
            EMISS(i,j,0)         = noah_output_arr(ii,jj,0,NoahmpOutputComp::emiss);
            ALBSFCDIR_VIS(i,j,0) = noah_output_arr(ii,jj,0,NoahmpOutputComp::albsfcdir_vis);
            ALBSFCDIR_NIR(i,j,0) = noah_output_arr(ii,jj,0,NoahmpOutputComp::albsfcdir_nir);
            ALBSFCDIF_VIS(i,j,0) = noah_output_arr(ii,jj,0,NoahmpOutputComp::albsfcdif_vis);
            ALBSFCDIF_NIR(i,j,0) = noah_output_arr(ii,jj,0,NoahmpOutputComp::albsfcdif_nir);
        });
    }

    // Fill the ghost cells
    for (auto ivar = 0; ivar < LsmFlux_NOAHMP::NumVars; ++ivar) {
        lsm_fab_flux[ivar]->FillBoundary(m_geom.periodicity());
    }
    Print () << "Noah-MP driver completed" << std::endl;
};

#ifdef ERF_USE_NOAHMP_MPMD
void NOAHMP::Run_MPMD_Advance()
{
    amrex::ParmParse pp_amr("amr");
    amrex::Vector<int> n_cell(3);
    pp_amr.getarr("n_cell", n_cell);

    amrex::ParmParse pp_geom("geometry");
    amrex::Vector<amrex::Real> prob_lo(3);
    amrex::Vector<amrex::Real> prob_hi(3);
    pp_geom.getarr("prob_lo", prob_lo);
    pp_geom.getarr("prob_hi", prob_hi);

    amrex::RealBox lb(prob_lo.dataPtr(), prob_hi.dataPtr());

    // Geometry needs the 2D Box first
    amrex::Box domain_bx(amrex::IntVect(0,0,0), amrex::IntVect(n_cell[0]-1, n_cell[1]-1, 0));
    amrex::Geometry geom(domain_bx, &lb, amrex::CoordSys::cartesian, nullptr);
    amrex::BoxArray ba(domain_bx);

    amrex::DistributionMapping dm;
    dm.RoundRobinProcessorMap(ba.size(), amrex::ParallelDescriptor::NProcs());

    amrex::MultiFab cons_dummy(ba, dm, 1, 0);
    amrex::MultiFab xvel_dummy(ba, dm, 1, 0);
    amrex::MultiFab yvel_dummy(ba, dm, 1, 0);

    amrex::Real dt = 0.0;
    amrex::ParmParse pp_erf("erf");
    pp_erf.query("dt", dt);

    NOAHMP lsm;
    lsm.Init(0, cons_dummy, geom, dt); // this creates mf_noah_input, mf_noah_output, and mpmd_copier

    int step = 0;
    int keep_running = 1;
    int root_app0 = 0;

    // The Adaptive MPMD Loop
    while (true) {
        // Wait for App 0 signal (0 = stop, 1 = continue)
        MPI_Bcast(&keep_running, 1, MPI_INT, root_app0, MPI_COMM_WORLD);
        if (!keep_running) {
            break;
        }

        // 1. Receive forcing data from ERF App 0
        lsm.mpmd_copier->recv(*(lsm.mf_noah_input), 0, NoahmpInputComp::NumComps);

        // 2. Run Noah-MP Physics
        lsm.Advance_With_State(0, cons_dummy, xvel_dummy, yvel_dummy, nullptr, nullptr, dt, step);

        // 3. Send calculated fluxes back to ERF App 0
        lsm.mpmd_copier->send(*(lsm.mf_noah_output), 0, NoahmpOutputComp::NumComps);

        step++;
    }
    amrex::Print() << "Noah-MP MPMD driver completed " << step << " steps and exited cleanly." << std::endl;
}
#endif
