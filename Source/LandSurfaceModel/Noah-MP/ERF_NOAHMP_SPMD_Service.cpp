#ifdef ERF_USE_NOAHMP_SPMD

#include <mpi.h>
#include <vector>
#include <iostream>

#include "ERF_NOAHMP_IO_Init.H"
#include "ERF_NOAHMP_SPMD_Shared.H"
#include "NoahmpIO.H"

namespace {

struct NoahServiceBlock {
    int ilo = 0, jlo = 0, ihi = -1, jhi = -1;
    int io_idx = -1;
    std::vector<double> input;
    std::vector<double> output;
};

inline int slab_index (int i, int j, int ilo, int jlo, int nx, int ncomp, int comp)
{
    const int ii = i - ilo;
    const int jj = j - jlo;
    return (jj * nx + ii) * ncomp + comp;
}

} // namespace

void RunNOAHMPSPMDService(MPI_Comm comm_sub)
{
    int myproc_sub = -1;
    MPI_Comm_rank(comm_sub, &myproc_sub);

    int num_tiles = 0;
    MPI_Status status;
    MPI_Recv(&num_tiles, 1, MPI_INT, MPI_ANY_SOURCE, noahmp_spmd::SPMDMetaSizeTag,
             MPI_COMM_WORLD, &status);
    const int erf_partner_rank = status.MPI_SOURCE;

    std::vector<NoahTile2D> tiles(num_tiles);
    if (num_tiles > 0) {
        MPI_Recv(tiles.data(), num_tiles * 4, MPI_INT, erf_partner_rank,
             noahmp_spmd::SPMDMetaBoxesTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    NoahmpIO_vector noahmpio_vect;
    std::vector<NoahServiceBlock> blocks(num_tiles);

    for (int ib = 0; ib < num_tiles; ++ib) {
        auto& b = blocks[ib];
        b.ilo = tiles[ib].xlo;
        b.jlo = tiles[ib].ylo;
        b.ihi = tiles[ib].xhi;
        b.jhi = tiles[ib].yhi;
        b.io_idx = ib;

        const int nx = b.ihi - b.ilo + 1;
        const int ny = b.jhi - b.jlo + 1;

        b.input.resize(nx * ny * NoahmpInputComp::NumComps);
        b.output.resize(nx * ny * NoahmpOutputComp::NumComps);
    }

    InitNoahmpIOOnly(
        noahmpio_vect,
        /*level=*/0,
        tiles,
        comm_sub,
        /*write_land0=*/false);

    std::cout << "NoahMP Rank " << myproc_sub
              << " successfully claimed by ERF Rank " << erf_partner_rank
              << " and initialized " << num_tiles << " tiles!" << std::endl;

    while (true) {
        int done = 0;
        MPI_Recv(&done, 1, MPI_INT, erf_partner_rank, noahmp_spmd::SPMDControlTag,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (done != 0) { break; }

        std::vector<MPI_Request> requests(blocks.size());
        int ireq = 0;
        for (auto& b : blocks) {
            MPI_Irecv(b.input.data(), static_cast<int>(b.input.size()), MPI_DOUBLE,
                      erf_partner_rank, noahmp_spmd::SPMDInputTag, MPI_COMM_WORLD,
                      &requests[ireq++]);
        }
        if (!requests.empty()) {
            std::vector<MPI_Status> statuses(requests.size());
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), statuses.data());
        }

        for (auto& b : blocks) {
            NoahmpIO_type& noah = noahmpio_vect[b.io_idx];
            const int nx = b.ihi - b.ilo + 1;

            for (int j = b.jlo; j <= b.jhi; ++j) {
                for (int i = b.ilo; i <= b.ihi; ++i) {
                    noah.U_PHY(i,1,j)   = b.input[slab_index(i,j,b.ilo,b.jlo,nx,NoahmpInputComp::NumComps,NoahmpInputComp::u_phy)];
                    noah.V_PHY(i,1,j)   = b.input[slab_index(i,j,b.ilo,b.jlo,nx,NoahmpInputComp::NumComps,NoahmpInputComp::v_phy)];
                    noah.T_PHY(i,1,j)   = b.input[slab_index(i,j,b.ilo,b.jlo,nx,NoahmpInputComp::NumComps,NoahmpInputComp::t_phy)];
                    noah.QV_CURR(i,1,j) = b.input[slab_index(i,j,b.ilo,b.jlo,nx,NoahmpInputComp::NumComps,NoahmpInputComp::qv_curr)];
                    noah.P8W(i,1,j)     = b.input[slab_index(i,j,b.ilo,b.jlo,nx,NoahmpInputComp::NumComps,NoahmpInputComp::p8w)];
                    noah.SWDOWN(i,j)    = b.input[slab_index(i,j,b.ilo,b.jlo,nx,NoahmpInputComp::NumComps,NoahmpInputComp::swdown)];
                    noah.GLW(i,j)       = b.input[slab_index(i,j,b.ilo,b.jlo,nx,NoahmpInputComp::NumComps,NoahmpInputComp::glw)];
                    noah.COSZEN(i,j)    = b.input[slab_index(i,j,b.ilo,b.jlo,nx,NoahmpInputComp::NumComps,NoahmpInputComp::coszen)];
                }
            }

            // Run Physics
            noah.itimestep += 1;
            noah.DriverMain();

            for (int j = b.jlo; j <= b.jhi; ++j) {
                for (int i = b.ilo; i <= b.ihi; ++i) {
                    b.output[slab_index(i,j,b.ilo,b.jlo,nx,NoahmpOutputComp::NumComps,NoahmpOutputComp::hfx)] = noah.HFX(i,j);
                    b.output[slab_index(i,j,b.ilo,b.jlo,nx,NoahmpOutputComp::NumComps,NoahmpOutputComp::lh)] = noah.LH(i,j);
                    b.output[slab_index(i,j,b.ilo,b.jlo,nx,NoahmpOutputComp::NumComps,NoahmpOutputComp::tau_ew)] = noah.TAU_EW(i,j);
                    b.output[slab_index(i,j,b.ilo,b.jlo,nx,NoahmpOutputComp::NumComps,NoahmpOutputComp::tau_ns)] = noah.TAU_NS(i,j);
                    b.output[slab_index(i,j,b.ilo,b.jlo,nx,NoahmpOutputComp::NumComps,NoahmpOutputComp::tsk)] = noah.TSK(i,j);
                    b.output[slab_index(i,j,b.ilo,b.jlo,nx,NoahmpOutputComp::NumComps,NoahmpOutputComp::emiss)] = noah.EMISS(i,j);
                    b.output[slab_index(i,j,b.ilo,b.jlo,nx,NoahmpOutputComp::NumComps,NoahmpOutputComp::albsfcdir_vis)] = noah.ALBSFCDIRXY(i,1,j);
                    b.output[slab_index(i,j,b.ilo,b.jlo,nx,NoahmpOutputComp::NumComps,NoahmpOutputComp::albsfcdir_nir)] = noah.ALBSFCDIRXY(i,2,j);
                    b.output[slab_index(i,j,b.ilo,b.jlo,nx,NoahmpOutputComp::NumComps,NoahmpOutputComp::albsfcdif_vis)] = noah.ALBSFCDIFXY(i,1,j);
                    b.output[slab_index(i,j,b.ilo,b.jlo,nx,NoahmpOutputComp::NumComps,NoahmpOutputComp::albsfcdif_nir)] = noah.ALBSFCDIFXY(i,2,j);
                }
            }
        }

        requests.resize(blocks.size());
        ireq = 0;
        for (auto& b : blocks) {
            MPI_Isend(b.output.data(), static_cast<int>(b.output.size()), MPI_DOUBLE,
                      erf_partner_rank, noahmp_spmd::SPMDOutputTag, MPI_COMM_WORLD,
                      &requests[ireq++]);
        }
        if (!requests.empty()) {
            std::vector<MPI_Status> statuses(requests.size());
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), statuses.data());
        }
    }
}
#endif
