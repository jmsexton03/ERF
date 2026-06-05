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
    constexpr int SPMDControlTag   = 4242;
    constexpr int SPMDMetaSizeTag  = 4243;
    constexpr int SPMDMetaBoxesTag = 4244;
    constexpr int SPMDInputTag     = 4245;
    constexpr int SPMDOutputTag    = 4246;
}

// Service entry point for NoahMP ranks (C-style linkage/free function)
void RunNOAHMPSPMDService();
#endif

#endif // ERF_NOAHMP_SPMD_SHARED_H
