#ifndef ERF_NOAHMP_SPMD_SHARED_H
#define ERF_NOAHMP_SPMD_SHARED_H

// Pure C++ header. NO AMReX headers allowed here.

namespace NoahmpInputComp {
    enum {
        u_phy = 0, v_phy, t_phy, qv_curr, p8w, swdown, glw, coszen, NumComps
    };
}

namespace NoahmpOutputComp {
    enum {
        hfx = 0, lh, tau_ew, tau_ns, tsk, emiss,
        albsfcdir_vis, albsfcdir_nir, albsfcdif_vis, albsfcdif_nir, NumComps
    };
}

#ifdef ERF_USE_NOAHMP_SPMD
namespace noahmp_spmd {
    // MPI Communication Tags
    constexpr int SPMDMetaSizeTag  = 100;
    constexpr int SPMDMetaBoxesTag = 101;
    constexpr int SPMDControlTag   = 200;
    constexpr int SPMDInputTag     = 300;
    constexpr int SPMDOutputTag    = 400;
}

// Service entry point for NoahMP ranks (C-style linkage/free function)
void RunNOAHMPSPMDService(MPI_Comm comm_sub);
#endif

#endif // ERF_NOAHMP_SPMD_SHARED_H
