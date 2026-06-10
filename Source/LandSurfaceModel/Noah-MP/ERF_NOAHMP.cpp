
#include<iostream>
#include<string>

#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_ParallelDescriptor.H>

#include <ERF_NOAHMP.H>
#include <ERF_NOAHMP_IO_Init.H>
#include <ERF_Constants.H>
#include <ERF_EOS.H>

using namespace amrex;

namespace {
constexpr int SPMDMetaSizeTag  = 100;
constexpr int SPMDMetaBoxesTag = 101;
constexpr int SPMDControlTag   = 200;
constexpr int SPMDInputTag     = 300;
constexpr int SPMDOutputTag    = 400;
}

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

    // Build 2D tiles in exactly the same MFIter order as before
    std::vector<NoahTile2D> tiles;
    tiles.reserve(cons_in.local_size());
    noahmp_input_tmp.resize(cons_in.local_size());
    noahmp_output_tmp.resize(cons_in.local_size());
    noahmp_partner_ranks.clear();

    int klo = domain.smallEnd(2);
    int idb = 0;
#ifdef ERF_USE_NOAHMP_SPMD
    // SPMD coupling requires a 1-to-1 mapping between boxes and ranks.
    bool use_tiling = false;
#else
    bool use_tiling = TilingIfNotGPU();
#endif
    for (MFIter mfi(cons_in, use_tiling); mfi.isValid(); ++mfi) {
        Box bx = mfi.tilebox();
        if (bx.smallEnd(2) != klo) { continue; }
        bx.makeSlab(2, klo);

        noahmp_input_tmp[idb] =
            std::make_unique<FArrayBox>(bx, NoahmpInputComp::NumComps, The_Pinned_Arena());
        noahmp_output_tmp[idb] =
            std::make_unique<FArrayBox>(bx, NoahmpOutputComp::NumComps, The_Pinned_Arena());

        tiles.push_back(NoahTile2D{
            bx.smallEnd(0), bx.smallEnd(1),
            bx.bigEnd(0),   bx.bigEnd(1)
        });
        ++idb;
    }

    // Optional safety: mirror prior expectation that we have work
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!tiles.empty(),
        "Noah-MP Init: no bottom-slab tiles found for this rank");

#ifdef ERF_USE_NOAHMP_SPMD
    // 1. Build the list of all global CPU ranks
    int nprocs_world;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs_world);

    // Recreate the same split logic here as amrex-spmd repo to find the CPU ranks
    int n_erf_ranks = amrex::ParallelDescriptor::NProcs();
    int stride = nprocs_world / n_erf_ranks;
    std::vector<int> ranks_other;
    for (int i = 0; i < nprocs_world; ++i) {
        if ((i % stride) != 0) { // If it's NOT an ERF rank
            ranks_other.push_back(i);
        }
    }

    // Safety check: this SPMD path assumes a 1-to-1 mapping between
    // global AMReX boxes and Noah-MP CPU ranks.
    const int total_global_boxes = static_cast<int>(cons_in.boxArray().size());
    const int total_cpu_ranks = static_cast<int>(ranks_other.size());
    if (total_global_boxes != total_cpu_ranks) {
        const amrex::Box domain_box = cons_in.boxArray().minimalBox();
        amrex::BoxArray suggested_ba =
            amrex::decompose(domain_box, total_cpu_ranks, {true, true, false});

        int suggested_mgs_x = 0;
        int suggested_mgs_y = 0;
        for (const auto& bx : suggested_ba) {
            suggested_mgs_x = std::max(suggested_mgs_x, bx.length(0));
            suggested_mgs_y = std::max(suggested_mgs_y, bx.length(1));
        }

        amrex::Print() << "\n[NOAHMP_SPMD][ERROR] Invalid ERF/Noah-MP box-to-rank mapping.\n"
                       << "  AMReX created " << total_global_boxes << " global boxes,\n"
                       << "  but only " << total_cpu_ranks << " Noah-MP CPU ranks are available.\n"
                       << "  This SPMD coupling path requires exactly one Noah-MP CPU rank per global box.\n"
                       << "  Adjust amr.max_grid_size_x/y/z or amr.blocking_factor in the inputs file\n"
                       << "  so the ERF BoxArray decomposes into exactly " << total_cpu_ranks
                       << " boxes.\n"
                       << "  Suggestion: Try setting amr.max_grid_size_x = " << suggested_mgs_x
                       << " and amr.max_grid_size_y = " << suggested_mgs_y
                       << " and amr.blocking_factor = 1\n" << std::endl;
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        total_global_boxes == total_cpu_ranks,
        "NOAHMP::Init requires cons_in.boxArray().size() == ranks_other.size() "
        "for 1:1 ERF-box to Noah-MP-rank mapping.");

    // 2. Loop through the local tiles, but use MFIter's *global* index
    //    to pick the specific CPU rank destination from the list.
    int dummy_num_tiles = 1; // amrex-spmd sends 1 box per CPU rank
    Vector<MPI_Request> requests(2 * tiles.size());
    int ireq = 0;
    int idb_send = 0;
    for (amrex::MFIter mfi(cons_in, use_tiling); mfi.isValid(); ++mfi) {
        amrex::Box bx = mfi.tilebox();
        if (bx.smallEnd(2) != klo) { continue; } // Keep your klo logic

        int global_box_index = mfi.index();
        int remote_cpu_rank = ranks_other[global_box_index];
        noahmp_partner_ranks.push_back(remote_cpu_rank);

        MPI_Isend(&dummy_num_tiles, 1, MPI_INT, remote_cpu_rank, SPMDMetaSizeTag,
                  MPI_COMM_WORLD, &requests[ireq++]);
        MPI_Isend(&tiles[idb_send], 4, MPI_INT, remote_cpu_rank, SPMDMetaBoxesTag,
                  MPI_COMM_WORLD, &requests[ireq++]);

        ++idb_send;
    }
    if (!requests.empty()) {
        Vector<MPI_Status> statuses(requests.size());
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), statuses.data());
    }
    AMREX_ALWAYS_ASSERT(noahmp_partner_ranks.size() == tiles.size());
#endif

    // Initialize ERF's copy
    InitNoahmpIOOnly(
        noahmpio_vect,
        lev,
        tiles,
        ParallelDescriptor::Communicator(),
        /*write_land0=*/true);

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

NOAHMP::~NOAHMP ()
{
#ifdef ERF_USE_NOAHMP_SPMD
    if (!noahmp_partner_ranks.empty()) {
        int done = 1;
        Vector<MPI_Request> requests(noahmp_partner_ranks.size());
        for (int ib = 0; ib < static_cast<int>(noahmp_partner_ranks.size()); ++ib) {
            MPI_Isend(&done, 1, MPI_INT, noahmp_partner_ranks[ib], SPMDControlTag,
                      MPI_COMM_WORLD, &requests[ib]);
        }
        Vector<MPI_Status> statuses(requests.size());
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), statuses.data());
    }
#endif
}

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
    // Verify we need to take another LSM step
    Real NOAH_time = static_cast<Real>(noahmpio_vect[0].itimestep-1) * static_cast<Real>(noahmpio_vect[0].DTBL);
    if (elapsed_time < NOAH_time) { return; }

    Box domain = m_geom.Domain();

    Print () << "Noah-MP driver started at time step: " << nstep+1 << std::endl;

    bool is_moist = (cons_in.nComp() > RhoQ1_comp);

    int klo = domain.smallEnd(2);

    // Loop over blocks to copy forcing data to Noahmp, drive the land model,
    // and copy data back to ERF Multifabs.
#ifdef ERF_USE_NOAHMP_SPMD
    bool use_tiling = false;

    int idb = 0;
    for (MFIter mfi(cons_in, use_tiling); mfi.isValid(); ++mfi) {
        Box bx = mfi.tilebox();
        if (bx.smallEnd(2) != klo) { continue; }
        bx.makeSlab(2,klo);

        const Array4<const Real>& U_PHY  = xvel_in.const_array(mfi);
        const Array4<const Real>& V_PHY  = yvel_in.const_array(mfi);
        const Array4<const Real>& CONS   = cons_in.const_array(mfi);
        const Array4<const Real>& SWDOWN = lsm_fab_data[LsmData_NOAHMP::sw_flux_dn]->const_array(mfi);
        const Array4<const Real>& GLW    = lsm_fab_data[LsmData_NOAHMP::lw_flux_dn]->const_array(mfi);
        const Array4<const Real>& COSZEN = lsm_fab_data[LsmData_NOAHMP::cos_zenith_angle]->const_array(mfi);
        Array4<Real> noah_input_arr = noahmp_input_tmp[idb]->array();

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

        ++idb;
    }

    // Ensure the GPU has finished writing the pinned host buffers before MPI reads them.
    Gpu::streamSynchronize();

    int done = 0;
    Vector<MPI_Request> requests(2 * noahmp_partner_ranks.size());
    int ireq = 0;
    for (int ib = 0; ib < static_cast<int>(noahmp_partner_ranks.size()); ++ib) {
        auto const& fab = *noahmp_input_tmp[ib];
        MPI_Isend(&done, 1, MPI_INT, noahmp_partner_ranks[ib], SPMDControlTag,
                  MPI_COMM_WORLD, &requests[ireq++]);
        MPI_Isend(fab.dataPtr(), static_cast<int>(fab.size()),
                  ParallelDescriptor::Mpi_typemap<Real>::type(),
                  noahmp_partner_ranks[ib], SPMDInputTag, MPI_COMM_WORLD,
                  &requests[ireq++]);
    }
    if (!requests.empty()) {
        Vector<MPI_Status> statuses(requests.size());
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), statuses.data());
    }

    requests.resize(noahmp_partner_ranks.size());
    ireq = 0;
    for (int ib = 0; ib < static_cast<int>(noahmp_partner_ranks.size()); ++ib) {
        auto& fab = *noahmp_output_tmp[ib];
        MPI_Irecv(fab.dataPtr(), static_cast<int>(fab.size()),
                  ParallelDescriptor::Mpi_typemap<Real>::type(),
                  noahmp_partner_ranks[ib], SPMDOutputTag, MPI_COMM_WORLD,
                  &requests[ireq++]);
    }
    if (!requests.empty()) {
        Vector<MPI_Status> statuses(requests.size());
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), statuses.data());
    }

    idb = 0;
    for (MFIter mfi(cons_in, use_tiling); mfi.isValid(); ++mfi) {
        Box bx  = mfi.tilebox();
        Box gbx = mfi.tilebox(IntVect(0,0,0),IntVect(1,1,0));
        if (bx.smallEnd(2) != klo) { continue; }

        bx.makeSlab(2,klo);
        gbx.makeSlab(2,klo);

        int i_lo = bx.smallEnd(0); int i_hi = bx.bigEnd(0);
        int j_lo = bx.smallEnd(1); int j_hi = bx.bigEnd(1);

        const Array4<const Real>& CONS = cons_in.const_array(mfi);
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
        Array4<Real> noah_output_arr = noahmp_output_tmp[idb]->array();

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

        ++idb;
    }
#else
    bool use_tiling = TilingIfNotGPU();
    int idb = 0;
    for (MFIter mfi(cons_in, use_tiling); mfi.isValid(); ++mfi, ++idb) {
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

        // Use The_Pinned_Arena() for host-accessible memory that can be used with GPU
        Array4<Real> noah_input_arr  =  noahmp_input_tmp[idb]->array();
        Array4<Real> noah_output_arr =  noahmp_output_tmp[idb]->array();

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
        noahmpio->itimestep += 1;
        noahmpio->DriverMain();

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
#endif

    // Fill the ghost cells
    for (auto ivar = 0; ivar < LsmFlux_NOAHMP::NumVars; ++ivar) {
        lsm_fab_flux[ivar]->FillBoundary(m_geom.periodicity());
    }
    Print () << "Noah-MP driver completed" << std::endl;
};
