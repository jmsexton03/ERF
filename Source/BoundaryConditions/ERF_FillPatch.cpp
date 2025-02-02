#include <ERF.H>
#include <ERF_PhysBCFunct.H>
#include <ERF_IndexDefines.H>
#include <ERF_TimeInterpolatedData.H>
#include <ERF_FillPatcher.H>
#include <ERF_Utils.H>

using namespace amrex;

/*
 * Fill valid and ghost data with the "state data" at the given time
 * NOTE: THIS OPERATES ON VELOCITY (MOMENTA ARE JUST TEMPORARIES)
 *
 * @param[in] lev  level of refinement at which to fill the data
 * @param[in] time time at which the data should be filled
 * @param[out] mfs_vel Vector of MultiFabs to be filled containing, in order: cons, xvel, yvel, and zvel
 * @param[out] mfs_mom Vector of MultiFabs to be filled containing, in order: cons, xmom, ymom, and zmom
 */
void
ERF::FillPatch (int lev, Real time,
                const Vector<MultiFab*>& mfs_vel,     // This includes cc quantities and VELOCITIES
                const Vector<MultiFab*>& mfs_mom,     // This includes cc quantities and MOMENTA
                const MultiFab& old_base_state,
                const MultiFab& new_base_state,
                bool fillset, bool cons_only)
{
    BL_PROFILE_VAR("ERF::FillPatch()",ERF_FillPatch);
    Interpolater* mapper = nullptr;

    AMREX_ALWAYS_ASSERT(lev > 0);

    PhysBCFunctNoOp null_bc;

    //
    // ***************************************************************************
    // The first thing we do is interpolate the momenta on the "valid" faces of
    // the fine grids (where the interface is coarse/fine not fine/fine) -- this
    // will not be over-written below because the FillPatch operators see these as
    // valid faces.
    //
    // Note that we interpolate momentum not velocity, but all the other boundary
    // conditions are imposed on velocity, so we convert to momentum here then
    // convert back.
    // ***************************************************************************
    if (fillset) {
        if (cf_set_width > 0) {
            FPr_c[lev-1].FillSet(*mfs_vel[Vars::cons], time, null_bc, domain_bcs_type);
        }
        if (cf_set_width >= 0 && !cons_only) {
            VelocityToMomentum(*mfs_vel[Vars::xvel], IntVect{0},
                               *mfs_vel[Vars::yvel], IntVect{0},
                               *mfs_vel[Vars::zvel], IntVect{0},
                               *mfs_vel[Vars::cons],
                               *mfs_mom[IntVars::xmom],
                               *mfs_mom[IntVars::ymom],
                               *mfs_mom[IntVars::zmom],
                               Geom(lev).Domain(),
                               domain_bcs_type);

            FPr_u[lev-1].FillSet(*mfs_mom[IntVars::xmom], time, null_bc, domain_bcs_type);
            FPr_v[lev-1].FillSet(*mfs_mom[IntVars::ymom], time, null_bc, domain_bcs_type);
            FPr_w[lev-1].FillSet(*mfs_mom[IntVars::zmom], time, null_bc, domain_bcs_type);

            MomentumToVelocity(*mfs_vel[Vars::xvel], *mfs_vel[Vars::yvel], *mfs_vel[Vars::zvel],
                               *mfs_vel[Vars::cons],
                               *mfs_mom[IntVars::xmom],
                               *mfs_mom[IntVars::ymom],
                               *mfs_mom[IntVars::zmom],
                               Geom(lev).Domain(),
                               domain_bcs_type);
        }
    }

    IntVect ngvect_cons = mfs_vel[Vars::cons]->nGrowVect();
    IntVect ngvect_vels = mfs_vel[Vars::xvel]->nGrowVect();

    {
        Vector<Real> ftime    = {t_old[lev], t_new[lev]};
        Vector<Real> ctime    = {t_old[lev-1], t_new[lev-1]};

        Vector<MultiFab*> fmf = {&vars_old[lev  ][Vars::cons], &vars_new[lev  ][Vars::cons]};
        Vector<MultiFab*> cmf = {&vars_old[lev-1][Vars::cons], &vars_new[lev-1][Vars::cons]};

        // We must fill a temporary then copy it back so we don't double add/subtract
        MultiFab mf_c(mfs_vel[Vars::cons]->boxArray(),mfs_vel[Vars::cons]->DistributionMap(),
                      mfs_vel[Vars::cons]->nComp()   ,mfs_vel[Vars::cons]->nGrowVect());

        mapper = &cell_cons_interp;

        if (interpolation_type == StateInterpType::Perturbational)
        {
            // Divide (rho theta) by rho to get theta (before we subtract rho0 from rho!)
            if (!amrex::almostEqual(time,ctime[1])) {
                MultiFab::Divide(vars_old[lev-1][Vars::cons],vars_old[lev-1][Vars::cons],
                                 Rho_comp,RhoTheta_comp,1,ngvect_cons);
                MultiFab::Subtract(vars_old[lev-1][Vars::cons],base_state[lev-1],
                                   BaseState::r0_comp,Rho_comp,1,ngvect_cons);
                MultiFab::Subtract(vars_old[lev-1][Vars::cons],base_state[lev-1],
                                   BaseState::th0_comp,RhoTheta_comp,1,ngvect_cons);
            }
            if (!amrex::almostEqual(time,ctime[0])) {
                 MultiFab::Divide(vars_new[lev-1][Vars::cons],vars_new[lev-1][Vars::cons],
                                  Rho_comp,RhoTheta_comp,1,ngvect_cons);
                 MultiFab::Subtract(vars_new[lev-1][Vars::cons],base_state[lev-1],
                                    BaseState::r0_comp,Rho_comp,1,ngvect_cons);
                 MultiFab::Subtract(vars_new[lev-1][Vars::cons],base_state[lev-1],
                                    BaseState::th0_comp,RhoTheta_comp,1,ngvect_cons);
            }

            if (!amrex::almostEqual(time,ftime[1])) {
                MultiFab::Divide(vars_old[lev  ][Vars::cons],vars_old[lev  ][Vars::cons],
                                 Rho_comp,RhoTheta_comp,1,IntVect{0});
                MultiFab::Subtract(vars_old[lev  ][Vars::cons],old_base_state,
                                   BaseState::r0_comp,Rho_comp,1,IntVect{0});
                MultiFab::Subtract(vars_old[lev  ][Vars::cons],old_base_state,
                                   BaseState::th0_comp,RhoTheta_comp,1,IntVect{0});
            }
            if (!amrex::almostEqual(time,ftime[0])) {
                MultiFab::Divide(vars_new[lev  ][Vars::cons],vars_new[lev  ][Vars::cons],
                                 Rho_comp,RhoTheta_comp,1,IntVect{0});
                MultiFab::Subtract(vars_new[lev  ][Vars::cons],old_base_state,
                                   BaseState::r0_comp,Rho_comp,1,IntVect{0});
                MultiFab::Subtract(vars_new[lev  ][Vars::cons],old_base_state,
                                   BaseState::th0_comp,RhoTheta_comp,1,IntVect{0});
            }
        }

        // Call FillPatchTwoLevels which ASSUMES that all ghost cells have already been filled
        FillPatchTwoLevels(mf_c, ngvect_cons, IntVect(0,0,0),
                           time, cmf, ctime, fmf, ftime,
                           0, 0, mf_c.nComp(), geom[lev-1], geom[lev],
                           refRatio(lev-1), mapper, domain_bcs_type,
                           BCVars::cons_bc);

        if (interpolation_type == StateInterpType::Perturbational)
        {
            // Restore the coarse values to what they were
            if (!amrex::almostEqual(time,ctime[1])) {
                MultiFab::Add(vars_old[lev-1][Vars::cons], base_state[lev-1],
                              BaseState::r0_comp,Rho_comp,1,ngvect_cons);
                MultiFab::Add(vars_old[lev-1][Vars::cons], base_state[lev-1],
                              BaseState::th0_comp,RhoTheta_comp,1,ngvect_cons);
                MultiFab::Multiply(vars_old[lev-1][Vars::cons], vars_old[lev-1][Vars::cons],
                                   Rho_comp,RhoTheta_comp,1,ngvect_cons);
            }
            if (!amrex::almostEqual(time,ctime[0])) {
                MultiFab::Add(vars_new[lev-1][Vars::cons], base_state[lev-1],
                              BaseState::r0_comp,Rho_comp,1,vars_new[lev-1][Vars::cons].nGrowVect());
                MultiFab::Add(vars_new[lev-1][Vars::cons], base_state[lev-1],
                              BaseState::th0_comp,RhoTheta_comp,1,vars_new[lev-1][Vars::cons].nGrowVect());
                MultiFab::Multiply(vars_new[lev-1][Vars::cons], vars_new[lev-1][Vars::cons],
                                   Rho_comp,RhoTheta_comp,1,ngvect_cons);
            }

            if (!amrex::almostEqual(time,ftime[1])) {
                MultiFab::Add(vars_old[lev][Vars::cons],base_state[lev  ],BaseState::r0_comp,Rho_comp,1,ngvect_cons);
                MultiFab::Add(vars_old[lev][Vars::cons],base_state[lev  ],BaseState::th0_comp,RhoTheta_comp,1,ngvect_cons);
                MultiFab::Multiply(vars_old[lev][Vars::cons], vars_old[lev][Vars::cons],
                                       Rho_comp,RhoTheta_comp,1,ngvect_cons);
            }
            if (!amrex::almostEqual(time,ftime[0])) {
                MultiFab::Add(vars_new[lev][Vars::cons], base_state[lev],BaseState::r0_comp,Rho_comp,1,ngvect_cons);
                MultiFab::Add(vars_new[lev][Vars::cons], base_state[lev],BaseState::th0_comp,RhoTheta_comp,1,ngvect_cons);
                MultiFab::Multiply(vars_new[lev][Vars::cons], vars_new[lev][Vars::cons],
                                   Rho_comp,RhoTheta_comp,1,ngvect_cons);
            }

            // Set values in the cells outside the domain boundary so that we can do the Add
            //     without worrying about uninitialized values outside the domain -- these
            //     will be filled in the physbcs call
            mf_c.setDomainBndry(1.234e20,0,2,geom[lev]); // Do both rho and (rho theta) together

            // Add rho_0 back to rho and theta_0 back to theta
            MultiFab::Add(mf_c, new_base_state,BaseState::r0_comp,Rho_comp,1,ngvect_cons);
            MultiFab::Add(mf_c, new_base_state,BaseState::th0_comp,RhoTheta_comp,1,ngvect_cons);

            // Multiply (theta) by rho to get (rho theta)
            MultiFab::Multiply(mf_c,mf_c,Rho_comp,RhoTheta_comp,1,ngvect_cons);
        }

        MultiFab::Copy(*mfs_vel[Vars::cons],mf_c,0,0,mf_c.nComp(),mf_c.nGrowVect());

        // ***************************************************************************************

        if (!cons_only)
        {
            mapper = &face_cons_linear_interp;

            MultiFab& mf_u = *mfs_vel[Vars::xvel];
            MultiFab& mf_v = *mfs_vel[Vars::yvel];
            MultiFab& mf_w = *mfs_vel[Vars::zvel];

            // **********************************************************************

            fmf = {&vars_old[lev  ][Vars::xvel], &vars_new[lev  ][Vars::xvel]};
            cmf = {&vars_old[lev-1][Vars::xvel], &vars_new[lev-1][Vars::xvel]};

            // Call FillPatchTwoLevels which ASSUMES that all ghost cells have already been filled
            FillPatchTwoLevels(mf_u, ngvect_vels, IntVect(0,0,0),
                               time, cmf, ctime, fmf, ftime,
                               0, 0, 1, geom[lev-1], geom[lev],
                               refRatio(lev-1), mapper, domain_bcs_type,
                               BCVars::xvel_bc);

            // **********************************************************************

            fmf = {&vars_old[lev  ][Vars::yvel], &vars_new[lev  ][Vars::yvel]};
            cmf = {&vars_old[lev-1][Vars::yvel], &vars_new[lev-1][Vars::yvel]};

            // Call FillPatchTwoLevels which ASSUMES that all ghost cells have already been filled
            FillPatchTwoLevels(mf_v, ngvect_vels, IntVect(0,0,0),
                               time, cmf, ctime, fmf, ftime,
                               0, 0, 1, geom[lev-1], geom[lev],
                               refRatio(lev-1), mapper, domain_bcs_type,
                               BCVars::yvel_bc);

            // **********************************************************************

            fmf = {&vars_old[lev  ][Vars::zvel], &vars_new[lev  ][Vars::zvel]};
            cmf = {&vars_old[lev-1][Vars::zvel], &vars_new[lev-1][Vars::zvel]};

            // Call FillPatchTwoLevels which ASSUMES that all ghost cells have already been filled
            FillPatchTwoLevels(mf_w, ngvect_vels, IntVect(0,0,0),
                               time, cmf, ctime, fmf, ftime,
                               0, 0, 1, geom[lev-1], geom[lev],
                               refRatio(lev-1), mapper, domain_bcs_type,
                               BCVars::zvel_bc);
        } // !cons_only
    } // lev > 0

    // ***************************************************************************
    // Physical bc's at domain boundary
    // ***************************************************************************
    int icomp_cons = 0;
    int ncomp_cons = mfs_vel[Vars::cons]->nComp();

    bool do_fb = true;

    if (m_r2d) fill_from_bndryregs(mfs_vel,time);

    // We call these even if init_type == InitType::Real because these will fill the vertical bcs
    // Note that we call FillBoundary inside the physbcs call
    (*physbcs_cons[lev])(*mfs_vel[Vars::cons],icomp_cons,ncomp_cons,ngvect_cons,time,BCVars::cons_bc, do_fb);
    if (!cons_only) {
        (*physbcs_u[lev])(*mfs_vel[Vars::xvel],0,1,ngvect_vels,time,BCVars::xvel_bc, do_fb);
        (*physbcs_v[lev])(*mfs_vel[Vars::yvel],0,1,ngvect_vels,time,BCVars::yvel_bc, do_fb);
        (*physbcs_w[lev])(*mfs_vel[Vars::zvel],*mfs_vel[Vars::xvel],*mfs_vel[Vars::yvel],
                          ngvect_vels,time,BCVars::zvel_bc, do_fb);
    }
}

void
ERF::FillPatch (int lev, Real time,
                const Vector<MultiFab*>& mfs_vel,     // This includes cc quantities and VELOCITIES
                bool cons_only)
{
    BL_PROFILE_VAR("ERF::FillPatch()",ERF_FillPatch);

    AMREX_ALWAYS_ASSERT(lev == 0);

    IntVect ngvect_cons = mfs_vel[Vars::cons]->nGrowVect();
    IntVect ngvect_vels = mfs_vel[Vars::xvel]->nGrowVect();

    Vector<Real> ftime    = {t_old[lev], t_new[lev]};

    //
    // Below we call FillPatchSingleLevel which does NOT fill ghost cells outside the domain
    //

    Vector<MultiFab*> fmf = {&vars_old[lev][Vars::cons], &vars_new[lev][Vars::cons]};
    const int  ncomp = mfs_vel[Vars::cons]->nComp();

    FillPatchSingleLevel(*mfs_vel[Vars::cons], ngvect_cons, time, fmf, IntVect(0,0,0), ftime,
                         0, 0, ncomp, geom[lev]);

    if (!cons_only) {
        fmf = {&vars_old[lev][Vars::xvel], &vars_new[lev][Vars::xvel]};
        FillPatchSingleLevel(*mfs_vel[Vars::xvel], ngvect_vels, time, fmf,
                             IntVect(0,0,0), ftime,  0, 0, 1, geom[lev]);

        fmf = {&vars_old[lev][Vars::yvel], &vars_new[lev][Vars::yvel]};
        FillPatchSingleLevel(*mfs_vel[Vars::yvel], ngvect_vels, time, fmf,
                             IntVect(0,0,0), ftime,  0, 0, 1, geom[lev]);

        fmf = {&vars_old[lev][Vars::zvel], &vars_new[lev][Vars::zvel]};
        FillPatchSingleLevel(*mfs_vel[Vars::zvel], ngvect_vels, time, fmf,
                             IntVect(0,0,0), ftime,  0, 0, 1, geom[lev]);
    } // !cons_only

    // ***************************************************************************
    // Physical bc's at domain boundary
    // ***************************************************************************
    int icomp_cons = 0;
    int ncomp_cons = mfs_vel[Vars::cons]->nComp();

    bool do_fb = true;

#ifdef ERF_USE_NETCDF
    // We call this here because it is an ERF routine
    if (use_real_bcs && (lev==0)) {
        fill_from_realbdy(mfs_vel,time,cons_only,icomp_cons,ncomp_cons,ngvect_cons,ngvect_vels);
        do_fb = false;
    }
#endif

    if (m_r2d) fill_from_bndryregs(mfs_vel,time);

    // We call these even if init_type == InitType::Real because these will fill the vertical bcs
    // Note that we call FillBoundary inside the physbcs call
    (*physbcs_cons[lev])(*mfs_vel[Vars::cons],icomp_cons,ncomp_cons,ngvect_cons,time,BCVars::cons_bc, do_fb);
    if (!cons_only) {
        (*physbcs_u[lev])(*mfs_vel[Vars::xvel],0,1,ngvect_vels,time,BCVars::xvel_bc, do_fb);
        (*physbcs_v[lev])(*mfs_vel[Vars::yvel],0,1,ngvect_vels,time,BCVars::yvel_bc, do_fb);
        (*physbcs_w[lev])(*mfs_vel[Vars::zvel],*mfs_vel[Vars::xvel],*mfs_vel[Vars::yvel],
                          ngvect_vels,time,BCVars::zvel_bc, do_fb);
    }
}
