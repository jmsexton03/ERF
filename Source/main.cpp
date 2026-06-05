#include <cstring>
#include <iostream>
#include <sstream>

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#include "ERF.H"
#include "ERF_InputsName.H"

#if defined(ERF_USE_WW3_COUPLING) || defined(ERF_USE_NOAHMP_SPMD)
#include <mpi.h>
#endif

#ifdef ERF_USE_WW3_COUPLING
#include <AMReX_MPMD.H>
#endif

#ifdef ERF_USE_NOAHMP_SPMD
#include <ERF_NOAHMP.H>
#include <ERF_NOAHMP_SPMD_Shared.H>
#endif

std::string inputs_name;

using namespace amrex;

namespace {

int find_dashdash (int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--") == 0) {
            return i;
        }
    }
    return argc;
}

}

/**
 * Function to set the refine_grid_layout flags to (1,1,0) by default
 * since the ERF default is different from the amrex default (1,1,1)
 * Also set max_grid_size to very large since the only reason for
 * chopping grids is if Nprocs > Ngrids
*/
void add_par () {
   ParmParse pp("amr");

   // Set the refine_grid_layout flags to (1,1,0) by default
   pp.add("refine_grid_layout_x",1);
   pp.add("refine_grid_layout_y",1);
   pp.add("refine_grid_layout_z",0);

   // n_proper is the minimum number of coarse cells between coarse-fine boundaries
   // between levels (ell and ell+1) and levels (ell-1 and ell).   We want this to be
   // greater than or equal to the stencil width (a function of spatial order) divided by
   // ref_ratio (which can be 2,3 or 4).  This ensures that fillpatch at level (ell)
   // does not need to reach beyond level (ell-1). Here to be conservative we set this to 2
   // (rather than the amrex default of 1).
   pp.add("n_proper",2);

   int max_grid_size = 2048;
   pp.queryAdd("max_grid_size",max_grid_size);

   // This will set the default value of blocking_factor to be 1, but will allow
   //     the user to override it in the inputs file or on command line
   int blocking_factor = 1;
   pp.queryAdd("blocking_factor",blocking_factor);

   int n_error_buf = 0;
   pp.queryAdd("n_error_buf",n_error_buf);
}

/**
 * Main driver -- creates the ERF object, calls ERF.InitData() and ERF.Evolve()
*/
int main (int argc, char* argv[])
{
    auto finalize_mpi_and_return = [](int code) {
#ifdef AMREX_USE_MPI
#ifdef ERF_USE_WW3_COUPLING
        amrex::MPMD::Finalize();
#else
        MPI_Finalize();
#endif
#endif
        return code;
    };

#if defined(AMREX_MPI_THREAD_MULTIPLE)
    int requested = MPI_THREAD_MULTIPLE;
    int provided = -1;
    MPI_Init_thread(&argc, &argv, requested, &provided);
#elif defined(AMREX_USE_MPI)
    MPI_Init(&argc, &argv);
#endif

    if (argc < 2) {
        // Print usage and exit with error code if no input file was provided.
        ERF::print_usage(MPI_COMM_WORLD, std::cout);
        ERF::print_error(MPI_COMM_WORLD, "No input file provided. Exiting!!");
        return finalize_mpi_and_return(1);
    }

    // Look for "-h" or "--help" flag and print usage
    for (auto i = 1; i < argc; i++) {
        const std::string param(argv[i]);
        if ((param == "--help") || (param == "-h") || (param == "--usage")) {
            ERF::print_banner(MPI_COMM_WORLD, std::cout);
            ERF::print_usage(MPI_COMM_WORLD, std::cout);
            return finalize_mpi_and_return(0);
        }
    }

    if (argc >= 2) {
        for (auto i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "--describe") {
                ERF::writeBuildInfo(std::cout);
                return finalize_mpi_and_return(0);
            }
        }
    }

    if (!strchr(argv[1], '=') && !amrex::FileSystem::Exists(std::string(argv[1])))
    {
        // Print usage and exit with error code if we cannot find the input file
        ERF::print_usage(MPI_COMM_WORLD, std::cout);
        ERF::print_error(
            MPI_COMM_WORLD, "Input file does not exist = " +
                                std::string(argv[1]) + ". Exiting!!");
        return finalize_mpi_and_return(1);
    }

  //  print_banner(MPI_COMM_WORLD, std::cout);
    // Check to see if the command line contains --describe
    if (argc >= 2) {
        for (auto i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "--describe") {
                ERF::writeBuildInfo(std::cout);
                return 0;
            }
        }
    }

    // Save the inputs file name for later.
    if (!strchr(argv[1], '=')) {
      inputs_name = argv[1];
    }

    int myproc_world = 0;
    int nprocs_world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &myproc_world);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs_world);

#ifdef ERF_USE_NOAHMP_SPMD
    // ---------------------------------------------------------
    // Node-aware SPMD Communicator Splitting
    // ---------------------------------------------------------
    
    // 1. Identify ranks on the same physical node
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);

    int local_rank = -1;
    int local_size = -1;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);
    MPI_Comm_free(&local_comm);

    // 2. Define splitting policy: 4 ERF ranks per node (for 4 GPUs)
    const int erf_ranks_per_node = 4;
    const int color = (local_rank < erf_ranks_per_node) ? 0 : 1; // 0 = ERF, 1 = NOAHMP
    const bool is_erf_rank = (color == 0);

    // 3. Create sub-communicators
    MPI_Comm comm_sub;
    MPI_Comm_split(MPI_COMM_WORLD, color, myproc_world, &comm_sub);

    int myproc_sub = -1;
    int nprocs_sub = -1;
    MPI_Comm_rank(comm_sub, &myproc_sub);
    MPI_Comm_size(comm_sub, &nprocs_sub);

    // 4. Calculate global total of ERF vs NoahMP ranks for ConfigureSPMD
    int my_erf_count = is_erf_rank ? 1 : 0;
    int global_erf_ranks = 0;
    MPI_Allreduce(&my_erf_count, &global_erf_ranks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    int global_noah_ranks = nprocs_world - global_erf_ranks;

    // 5. Diagnostics
    if (myproc_world == 0) {
        std::cout << "\n=======================================================\n";
        std::cout << "[SPMD INIT] Total World Ranks: " << nprocs_world << "\n";
        std::cout << "[SPMD INIT] Policy: " << erf_ranks_per_node << " ERF ranks per node\n";
        std::cout << "[SPMD INIT] Global Split: " << global_erf_ranks << " ERF, " << global_noah_ranks << " NoahMP\n";
        std::cout << "=======================================================\n\n";
    }

    // Print a quick summary from the "leader" rank of each node to avoid spam
    if (local_rank == 0) {
        std::cout << "[SPMD Node Leader] World Rank " << myproc_world 
                  << " reports local node size = " << local_size << "\n";
    }

    // Configure the SPMD service
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, myproc, MPI_INFO_NULL, &node_comm);

    int my_node_rank, node_size;
    MPI_Comm_rank(node_comm, &my_node_rank);
    MPI_Comm_size(node_comm, &node_size);

    // Gather all world ranks of NoahMP processes ON THIS NODE
    int local_noah_id = (!is_erf_rank) ? myproc : -1;
    std::vector<int> node_noah_ranks(node_size);
    MPI_Allgather(&local_noah_id, 1, MPI_INT, node_noah_ranks.data(), 1, MPI_INT, node_comm);

    std::vector<int> valid_node_noah_ranks;
    for (int r : node_noah_ranks) {
        if (r != -1) valid_node_noah_ranks.push_back(r);
    }

    if (valid_node_noah_ranks.empty()) {
        amrex::Abort("Error: No NoahMP rank found on this physical node! Check MPI task layout.");
    }

    // ERF assigns itself to an on-node NoahMP rank (round-robin on the node)
    // NoahMP service uses MPI_ANY_SOURCE, so it doesn't need a specific partner (-1)
    int partner_world_rank = -1;
    if (is_erf_rank) {
        partner_world_rank = valid_node_noah_ranks[my_node_rank % valid_node_noah_ranks.size()];
    }
    MPI_Comm_free(&node_comm);

    NOAHMP::ConfigureSPMD(global_erf_ranks, global_noah_ranks, is_erf_rank, partner_world_rank);


    if (is_erf_rank) {
        // Strip off extra arguments after '--' so AMReX doesn't parse them
        const int dashdash = find_dashdash(argc, argv);
        int erf_argc = dashdash;
        char** erf_argv = argv;

        if (myproc_sub == 0) {
            std::cout << "[SPMD][ERF] Initializing AMReX with " << nprocs_sub << " ranks.\n";
        }

        amrex::Initialize(erf_argc, erf_argv, true, comm_sub, add_par);

#ifdef ERF_USE_KOKKOS
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize(Kokkos::InitializationSettings()
                               .set_device_id(amrex::Gpu::Device::deviceId()));
        }
#endif

        {
            BL_PROFILE_VAR("main()", pmain);
            const Real strt_total = Real(amrex::second());
            ERF erf;
            erf.InitData();
            erf.Evolve();
            
            // Tell NOAHMP service ranks to terminate after Evolve() finishes
            NOAHMP::ShutdownSPMDService();

            Real end_total = Real(amrex::second()) - strt_total;
            ParallelDescriptor::ReduceRealMax(end_total ,ParallelDescriptor::IOProcessorNumber());
            if (erf.Verbose()) {
                amrex::Print() << "\nTotal Time: " << end_total << '\n';
            }

            BL_PROFILE_VAR_STOP(pmain);
        }

        MPI_Barrier(MPI_COMM_WORLD);

#ifdef ERF_USE_KOKKOS
        Kokkos::finalize();
#endif
        amrex::Finalize();

    } else {
        if (myproc_sub == 0) {
            std::cout << "[SPMD][NOAH] Starting RunNOAHMPSPMDService() with " << nprocs_sub << " ranks.\n";
        }
        
        RunNOAHMPSPMDService();
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    MPI_Comm_free(&comm_sub);

#else // standard ERF without NOAHMP SPMD

#ifdef ERF_USE_WW3_COUPLING
    MPI_Comm comm = amrex::MPMD::Initialize(argc, argv);
    amrex::Initialize(argc, argv, true, comm, add_par);
#else
    amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, add_par);
#endif

 #ifdef ERF_USE_KOKKOS
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize(Kokkos::InitializationSettings()
                           .set_device_id(amrex::Gpu::Device::deviceId()));
    }
#endif

    {
        BL_PROFILE_VAR("main()", pmain);
        const Real strt_total = Real(amrex::second());
        ERF erf;
        erf.InitData();
        erf.Evolve();

        Real end_total = Real(amrex::second()) - strt_total;
        ParallelDescriptor::ReduceRealMax(end_total ,ParallelDescriptor::IOProcessorNumber());
        if (erf.Verbose()) {
            amrex::Print() << "\nTotal Time: " << end_total << '\n';
        }

        BL_PROFILE_VAR_STOP(pmain);
    }

#ifdef ERF_USE_WW3_COUPLING
    MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef ERF_USE_KOKKOS
    Kokkos::finalize();
#endif

    amrex::Finalize();
#endif // ERF_USE_NOAHMP_SPMD

    return finalize_mpi_and_return(0);
}
