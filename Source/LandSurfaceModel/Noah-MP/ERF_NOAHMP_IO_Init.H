#ifndef ERF_NOAHMP_IO_INIT_H
#define ERF_NOAHMP_IO_INIT_H

// AMReX-free: safe to include before amrex::Initialize()
// Only depends on NoahmpIO.H (STL + NoahArray) and MPI.

#include <vector>
#include <mpi.h>
#include <NoahmpIO.H>          // Submodules/Noah-MP/drivers/erf/NoahmpIO.H

struct NoahTile2D {
    int xlo, ylo, xhi, yhi;
};

// Initializes NoahmpIO_type entries in noahmpio_vect without any AMReX dependency.
// Caller is responsible for resizing noahmpio_vect to zero before calling if reuse is needed.
// write_land0: pass true only on ERF/AMReX side (requires parallel NetCDF + MPI-IO).
inline void InitNoahmpIOOnly(
    NoahmpIO_vector&            noahmpio_vect,
    int                         level,
    const std::vector<NoahTile2D>& tiles,
    MPI_Comm                    comm,
    bool                        write_land0 = false)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Uses NoahmpIO_vector::resize(size, level) which calls the Fortran helper
    noahmpio_vect.resize(static_cast<int>(tiles.size()), level);

    for (int idb = 0; idb < static_cast<int>(tiles.size()); ++idb) {
        NoahmpIO_type* n = &noahmpio_vect[idb];
        const NoahTile2D& t = tiles[idb];

        n->blkid = idb;
        n->level = level;
        n->rank  = rank;
        n->comm  = MPI_Comm_c2f(comm);

        n->ScalarInitDefault();
        n->ReadNamelist();
        n->ReadLandHeader();

        n->xstart = t.xlo;  n->xend = t.xhi;
        n->ystart = t.ylo;  n->yend = t.yhi;

        // Domain bounds
        n->ids = t.xlo;  n->ide = t.xhi;
        n->jds = t.ylo;  n->jde = t.yhi;
        n->kds = 1;      n->kde = 2;

        // Tile bounds (same as domain for current single-tile-per-box layout)
        n->its = t.xlo;  n->ite = t.xhi;
        n->jts = t.ylo;  n->jte = t.yhi;
        n->kts = 1;      n->kte = 2;

        // Memory bounds
        n->ims = t.xlo;  n->ime = t.xhi;
        n->jms = t.ylo;  n->jme = t.yhi;
        n->kms = 1;      n->kme = 2;

        n->VarInitDefault();
        n->ReadTable();
        n->ReadLandMain();
        n->InitMain();

        if (write_land0) { n->WriteLand(0); }
    }
}
#endif
