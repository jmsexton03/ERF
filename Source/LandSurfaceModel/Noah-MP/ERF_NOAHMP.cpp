
#include<iostream>
#include<numeric>
#include<string>
#include<unordered_set>
#include<vector>

#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_ParallelDescriptor.H>

#include <ERF_NOAHMP.H>
#include <ERF_Constants.H>
#include <ERF_EOS.H>

using namespace amrex;

#ifdef ERF_USE_NOAHMP_SPMD
namespace {
int s_n_erf_ranks = 0;
int s_n_noah_ranks = 0;
bool s_is_erf_rank = true;
bool s_metadata_sent = false;
static int s_partner_world_rank = -1;
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

    amrex::AllPrint() << "[rank " << amrex::ParallelDescriptor::MyProc()
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

    amrex::AllPrint() << "[rank " << amrex::ParallelDescriptor::MyProc()
                      << "] NOAHMP::Init built ba2d size=" << ba2d.size()
                      << " input comps=" << mf_noah_input->nComp()
                      << " output comps=" << mf_noah_output->nComp()
                      << " input local size=" << mf_noah_input->local_size()
                      << std::endl;

    // Iterate over multifab and noahmpio object together
    int idb = 0;
    for (MFIter mfi(cons_in); mfi.isValid(); ++mfi, ++idb) {
        Box bx = mfi.tilebox();
        if (bx.smallEnd(2) != klo) { continue; }

        Box bx2d = bx;
        bx2d.makeSlab(2, klo);

        amrex::AllPrint() << "[rank " << amrex::ParallelDescriptor::MyProc()
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
        amrex::AllPrint() << "[rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Init block " << idb << " ReadNamelist begin" << std::endl;
        noahmpio->ReadNamelist();

        // Read the headers from the NetCDF land file
        amrex::AllPrint() << "[rank " << amrex::ParallelDescriptor::MyProc()
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
        amrex::AllPrint() << "[rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Init block " << idb << " VarInitDefault begin" << std::endl;
        noahmpio->VarInitDefault();

        // Read NoahmpTable.TBL
        amrex::AllPrint() << "[rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Init block " << idb << " ReadTable begin" << std::endl;
        noahmpio->ReadTable();

        // Read and initialize data from the NetCDF land file
        amrex::AllPrint() << "[rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Init block " << idb << " ReadLandMain begin" << std::endl;
        noahmpio->ReadLandMain();

        // Compute additional initial values
        amrex::AllPrint() << "[rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Init block " << idb << " InitMain begin" << std::endl;
        noahmpio->InitMain();

        Print() << "Noah-MP writing lnd.nc file at lev: " << lev << std::endl;
        noahmpio->WriteLand(0);
        amrex::AllPrint() << "[rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Init block " << idb << " completed" << std::endl;
    }

    AMREX_ALWAYS_ASSERT(m_dt <= noahmpio_vect[0].DTBL);

    Print() << "Noah-MP initialization completed" << std::endl;

};

void
NOAHMP::Plot_Landfile(const int& nstep)
{
    for (NoahmpIO_type &noahmpio : noahmpio_vect) {
        noahmpio.WriteLand(nstep);
    }
}

#ifdef ERF_USE_NOAHMP_SPMD
void
NOAHMP::ConfigureSPMD (int n_erf_ranks, int n_noah_ranks, bool is_erf_rank, int partner_world_rank)
{
    s_n_erf_ranks = n_erf_ranks;
    s_n_noah_ranks = n_noah_ranks;
    s_is_erf_rank = is_erf_rank;
    s_metadata_sent = false;
    s_partner_world_rank = partner_world_rank;
}

void
NOAHMP::ShutdownSPMDService ()
{
    const int done = 1;
    const int partner = s_partner_world_rank;
    MPI_Request request;
    MPI_Isend(const_cast<int*>(&done), 1, MPI_INT, partner, noahmp_spmd::SPMDControlTag, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
}

void
NOAHMP::SendSPMDMetadata () const
{
    const int partner = s_partner_world_rank;
    const int nlocal = mf_noah_input->local_size();

    std::vector<int> bounds;
    bounds.reserve(4 * nlocal);
    for (MFIter mfi(*mf_noah_input); mfi.isValid(); ++mfi) {
        Box const& bx = mfi.validbox();
        bounds.push_back(bx.smallEnd(0));
        bounds.push_back(bx.smallEnd(1));
        bounds.push_back(bx.bigEnd(0));
        bounds.push_back(bx.bigEnd(1));
    }

    MPI_Request requests[2];
    MPI_Isend(const_cast<int*>(&nlocal), 1, MPI_INT, partner, noahmp_spmd::SPMDMetaSizeTag, MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(bounds.data(), static_cast<int>(bounds.size()), MPI_INT, partner, noahmp_spmd::SPMDMetaBoxesTag, MPI_COMM_WORLD, &requests[1]);
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
}

void
NOAHMP::SendSPMDInput ()
{
    if (!s_metadata_sent) {
        SendSPMDMetadata();
        s_metadata_sent = true;
    }

    const int keep_running = 0;
    const int partner = s_partner_world_rank;
    const int nlocal = mf_noah_input->local_size();

    std::vector<MPI_Request> requests(1 + nlocal);
    int ireq = 0;
    MPI_Isend(const_cast<int*>(&keep_running), 1, MPI_INT, partner, noahmp_spmd::SPMDControlTag, MPI_COMM_WORLD, &requests[ireq++]);
    for (MFIter mfi(*mf_noah_input, MFItInfo().DisableDeviceSync()); mfi.isValid(); ++mfi) {
        auto const& fab = (*mf_noah_input)[mfi];
        MPI_Isend(fab.dataPtr(), static_cast<int>(fab.size()),
                  ParallelDescriptor::Mpi_typemap<Real>::type(),
                  partner, noahmp_spmd::SPMDInputTag, MPI_COMM_WORLD, &requests[ireq++]);
    }
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
}

void
NOAHMP::ReceiveSPMDOutput ()
{
    const int partner = s_partner_world_rank;
    const int nlocal = mf_noah_output->local_size();

    std::vector<MPI_Request> requests(nlocal);
    int ireq = 0;
    for (MFIter mfi(*mf_noah_output, MFItInfo().DisableDeviceSync()); mfi.isValid(); ++mfi) {
        auto& fab = (*mf_noah_output)[mfi];
        MPI_Irecv(fab.dataPtr(), static_cast<int>(fab.size()),
                  ParallelDescriptor::Mpi_typemap<Real>::type(),
                  partner, noahmp_spmd::SPMDOutputTag, MPI_COMM_WORLD, &requests[ireq++]);
    }
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
}

void
NOAHMP::Advance_SPMD_Only (const int& nstep)
{
    int idb = 0;
    for (MFIter mfi(*mf_noah_input); mfi.isValid(); ++mfi, ++idb) {
        auto const& bx = mfi.validbox();
        NoahmpIO_type* noahmpio = &noahmpio_vect[idb];
        auto const noah_input_arr = mf_noah_input->const_array(mfi);
        auto noah_output_arr = mf_noah_output->array(mfi);

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

        noahmpio->itimestep += 1;
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
    }
}
#endif

void
NOAHMP::Advance_With_State (const int& lev,
                            MultiFab& cons_in,
                            MultiFab& xvel_in,
                            MultiFab& yvel_in,
                            MultiFab* /*hfx3_out*/,
                            MultiFab* /*qfx3_out*/,
                            const Real& elapsed_time,
                            const Real& dt,
                            const int& nstep)
{
    Real noah_time = static_cast<Real>(noahmpio_vect[0].itimestep-1) *
                     static_cast<Real>(noahmpio_vect[0].DTBL);
    if (elapsed_time < noah_time) { return; }

    Box domain = m_geom.Domain();

    Print () << "Noah-MP driver started at time step: " << nstep+1 << std::endl;
    amrex::AllPrint() << "[rank " << amrex::ParallelDescriptor::MyProc()
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

        amrex::AllPrint() << "[rank " << amrex::ParallelDescriptor::MyProc()
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
        amrex::AllPrint() << "[rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Advance_With_State block " << idb
                          << " completed ERF-to-Noahmp forcing pack" << std::endl;

#ifndef ERF_USE_NOAHMP_SPMD
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

        noahmpio->itimestep += 1;
        amrex::AllPrint() << "[rank " << amrex::ParallelDescriptor::MyProc()
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

        amrex::AllPrint() << "[rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Advance_With_State block " << idb
                          << " copied NoahMP output back to ERF arrays" << std::endl;
#else
        amrex::ignore_unused(dt, TSK, EMISS, ALBSFCDIR_VIS, ALBSFCDIR_NIR,
                             ALBSFCDIF_VIS, ALBSFCDIF_NIR, q_flux_arr,
                             t_flux_arr, tau13_arr, tau23_arr, gbx,
                             noah_output_arr);
#endif
    }

#ifdef ERF_USE_NOAHMP_SPMD
    SendSPMDInput();
    ReceiveSPMDOutput();

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

        amrex::AllPrint() << "[rank " << amrex::ParallelDescriptor::MyProc()
                          << "] NOAHMP::Advance_With_State block " << idb
                          << " copied NoahMP output back to ERF arrays" << std::endl;
    }
#endif

    for (auto ivar = 0; ivar < LsmFlux_NOAHMP::NumVars; ++ivar) {
        lsm_fab_flux[ivar]->FillBoundary(m_geom.periodicity());
    }
    Print () << "Noah-MP driver completed" << std::endl;
    amrex::AllPrint() << "[rank " << amrex::ParallelDescriptor::MyProc()
                      << "] NOAHMP::Advance_With_State end step=" << nstep+1
                      << std::endl;
};
