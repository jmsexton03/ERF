#ifdef ERF_USE_NOAHMP_SPMD

#include <mpi.h>
#include <vector>

#include "ERF_NOAHMP_SPMD_Shared.H"
#include "NoahmpIO.H"

namespace {

struct NoahServiceBlock {
    int ilo = 0, jlo = 0, ihi = -1, jhi = -1;
    NoahmpIO_type io;
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

void RunNOAHMPSPMDService()
{
    int myrank_world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank_world);

    // 1. Receive metadata size from ANY source. The sender becomes our dedicated partner.
    int nblocks = 0;
    MPI_Status status;
    MPI_Recv(&nblocks, 1, MPI_INT, MPI_ANY_SOURCE, noahmp_spmd::SPMDMetaSizeTag,
             MPI_COMM_WORLD, &status);

    const int partner = status.MPI_SOURCE;

    // 2. Receive the bounding boxes
    std::vector<int> bounds(4 * nblocks, 0);
    MPI_Recv(bounds.data(), static_cast<int>(bounds.size()), MPI_INT, partner,
             noahmp_spmd::SPMDMetaBoxesTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 3. Create a dedicated sub-communicator for Noah-MP ranks (for NoahmpIO's internal use)
    MPI_Comm comm_sub;
    MPI_Comm_split(MPI_COMM_WORLD, /*color=*/1, myrank_world, &comm_sub);

    std::vector<NoahServiceBlock> blocks(nblocks);

    for (int ib = 0; ib < nblocks; ++ib) {
        auto& b = blocks[ib];
        b.ilo = bounds[4*ib+0];
        b.jlo = bounds[4*ib+1];
        b.ihi = bounds[4*ib+2];
        b.jhi = bounds[4*ib+3];

        const int nx = b.ihi - b.ilo + 1;
        const int ny = b.jhi - b.jlo + 1;

        b.input.resize(nx * ny * NoahmpInputComp::NumComps);
        b.output.resize(nx * ny * NoahmpOutputComp::NumComps);

        NoahmpIO_type& noah = b.io;
        noah.blkid = ib;
        noah.level = 0;
        noah.ScalarInitDefault();
        noah.rank = myrank_world;
        noah.comm = MPI_Comm_c2f(comm_sub);

        noah.ReadNamelist();
        noah.ReadLandHeader();

        noah.xstart = b.ilo; noah.xend = b.ihi;
        noah.ystart = b.jlo; noah.yend = b.jhi;

        noah.ids = noah.xstart; noah.ide = noah.xend;
        noah.jds = noah.ystart; noah.jde = noah.yend;
        noah.kds = 1;           noah.kde = 2;

        noah.its = noah.xstart; noah.ite = noah.xend;
        noah.jts = noah.ystart; noah.jte = noah.yend;
        noah.kts = 1;           noah.kte = 2;

        noah.ims = noah.xstart; noah.ime = noah.xend;
        noah.jms = noah.ystart; noah.jme = noah.yend;
        noah.kms = 1;           noah.kme = 2;

        noah.VarInitDefault();
        noah.ReadTable();
        noah.ReadLandMain();
        noah.InitMain();
    }

    int done = 0;
    int step = 0;

    // 4. Main Service Loop
    while (true) {
        // Wait for control signal (run step vs shutdown)
        MPI_Recv(&done, 1, MPI_INT, partner, noahmp_spmd::SPMDControlTag,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (done != 0) break;

        for (auto& b : blocks) {
            // Recv forcing data
            MPI_Recv(b.input.data(), static_cast<int>(b.input.size()), MPI_DOUBLE,
                     partner, noahmp_spmd::SPMDInputTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            NoahmpIO_type& noah = b.io;
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

            // Send Output back
            MPI_Send(b.output.data(), static_cast<int>(b.output.size()), MPI_DOUBLE,
                     partner, noahmp_spmd::SPMDOutputTag, MPI_COMM_WORLD);
        }

        ++step;
    }

    MPI_Comm_free(&comm_sub);
}
#endif
