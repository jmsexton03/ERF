#include <cstring>
#include <iostream>

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
#include <sstream>
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

int parse_erf_ranks (int argc, char* argv[], int nprocs)
{
#ifdef ERF_USE_NOAHMP_SPMD
    const int dashdash = find_dashdash(argc, argv);
    if (dashdash >= argc-1) {
        amrex::Abort("NoahMP SPMD launch requires '-- <num_erf_ranks>' at the end of the command line");
    }

    int n_erf_ranks = 0;
    std::istringstream iss(argv[dashdash+1]);
    iss >> n_erf_ranks;

    if (!iss || n_erf_ranks <= 0 || n_erf_ranks >= nprocs) {
        amrex::Abort("Invalid NoahMP SPMD split: require 0 < num_erf_ranks < total MPI ranks");
    }

    const int n_noah_ranks = nprocs - n_erf_ranks;
    if (n_erf_ranks != n_noah_ranks) {
        amrex::Abort("Current NoahMP SPMD transport requires an equal ERF/Noah rank split");
    }
    return n_erf_ranks;
#else
    amrex::ignore_unused(argc, argv, nprocs);
    return nprocs;
#endif
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
        ERF::print_error(
            MPI_COMM_WORLD, "No input file provided. Exiting!!");
        return 1;
    }

    // Look for "-h" or "--help" flag and print usage
    for (auto i = 1; i < argc; i++) {
        const std::string param(argv[i]);
        if ((param == "--help") || (param == "-h") || (param == "--usage")) {
            ERF::print_banner(MPI_COMM_WORLD, std::cout);
            ERF::print_usage(MPI_COMM_WORLD, std::cout);
            return 0;
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
    const int n_erf_ranks = parse_erf_ranks(argc, argv, nprocs_world);
    const int dashdash = find_dashdash(argc, argv);
    const bool is_erf_rank = (myproc_world < n_erf_ranks);
    MPI_Comm comm_sub;
    MPI_Comm_split(MPI_COMM_WORLD, is_erf_rank ? 0 : 1, myproc_world, &comm_sub);

    NOAHMP::ConfigureSPMD(n_erf_ranks, nprocs_world - n_erf_ranks, is_erf_rank);

    if (is_erf_rank) {
        int erf_argc = dashdash;
	char** erf_argv = argv;
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
        RunNOAHMPSPMDService();
        MPI_Barrier(MPI_COMM_WORLD);
    }
#else
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
#endif

#ifdef ERF_USE_NOAHMP_SPMD
    MPI_Comm_free(&comm_sub);
#endif

#ifdef AMREX_USE_MPI
#ifdef ERF_USE_WW3_COUPLING
    amrex::MPMD::Finalize();
#else
    MPI_Finalize();
#endif
#endif
}
