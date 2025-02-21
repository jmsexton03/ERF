#include <AMReX_REAL.H>

module ccpp_kind_types
  use iso_c_binding, only: c_float, c_int, c_double, c_bool
#ifdef BL_USE_FLOAT
   integer, parameter :: kind_phys = c_float
#else
   integer, parameter :: kind_phys = c_double
#endif
   contains
end module ccpp_kind_types
