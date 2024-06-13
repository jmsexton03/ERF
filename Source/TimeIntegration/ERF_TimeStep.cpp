#include <ERF.H>
#include <Utils.H>
#include <mpi.h>
#include <AMReX_MPMD.H>

using namespace amrex;

/**
 * Function that coordinates the evolution across levels -- this calls Advance to do the
 * actual advance at this level,  then recursively calls itself at finer levels
 *
 * @param[in] lev level of refinement (coarsest level is 0)
 * @param[in] time start time for time advance
 * @param[in] iteration time step counter
 */

void
ERF::timeStep (int lev, Real time, int /*iteration*/)
{
    if (regrid_int > 0)  // We may need to regrid
    {
        // help keep track of whether a level was already regridded
        // from a coarser level call to regrid
        static Vector<int> last_regrid_step(max_level+1, 0);

        // regrid changes level "lev+1" so we don't regrid on max_level
        // also make sure we don't regrid fine levels again if
        // it was taken care of during a coarser regrid
        if (lev < max_level)
        {
            if ( (istep[lev] % regrid_int == 0) && (istep[lev] > last_regrid_step[lev]) )
            {
                // regrid could add newly refine levels (if finest_level < max_level)
                // so we save the previous finest level index
                int old_finest = finest_level;

                regrid(lev, time);

#ifdef ERF_USE_PARTICLES
                if (finest_level != old_finest) {
                    particleData.Redistribute();
                }
#endif

                // mark that we have regridded this level already
                for (int k = lev; k <= finest_level; ++k) {
                    last_regrid_step[k] = istep[k];
                }

                // if there are newly created levels, set the time step
                for (int k = old_finest+1; k <= finest_level; ++k) {
                    dt[k] = dt[k-1] / MaxRefRatio(k-1);
                }
            } // if
        } // lev
    }

    // Update what we call "old" and "new" time
    t_old[lev] = t_new[lev];
    t_new[lev] += dt[lev];

    if (Verbose()) {
        amrex::Print() << "[Level " << lev << " step " << istep[lev]+1 << "] ";
        amrex::Print() << "ADVANCE from time = " << t_old[lev] << " to " << t_new[lev]
                       << " with dt = " << dt[lev] << std::endl;
    }

#ifdef ERF_USE_WW3_COUPLING
    for ( MFIter mfi(*Hwave[lev],false); mfi.isValid(); ++mfi)
    {
         //auto my_H_ptr = Hwave[lev]->array(mfi);
         //auto my_L_ptr = Lwave[lev]->array(mfi);
         const auto & bx = mfi.tilebox();
         // How to declare my_H_ptr directly?
         amrex::Array4<Real> my_H_arr = Hwave[lev]->array(mfi);
         amrex::Array4<Real> my_L_arr = Lwave[lev]->array(mfi);

         Real* my_H_ptr = my_H_arr.dataPtr();
         Real* my_L_ptr = my_L_arr.dataPtr();

         int rank_offset = amrex::MPMD::MyProc() - amrex::ParallelDescriptor::MyProc();
         int this_root, other_root;
         if (rank_offset == 0) { // First program
             this_root = 0;
             other_root = amrex::ParallelDescriptor::NProcs();
         } else {
             this_root = rank_offset;
             other_root = 0;
         }

    //    std::cout<<"My rank is "<<amrex::MPMD::MyProc()<<" out of "<<amrex::MPMD::NProcs()<<" total ranks in MPI_COMM_WORLD communicator "<<MPI_COMM_WORLD<< "and my rank is "<<amrex::ParallelDescriptor::MyProc()<<" out of "<<amrex::ParallelDescriptor::NProcs()<<" total ranks in my part of the split communicator for the appnum (color) "<< amrex::MPMD::AppNum()<<std::endl;
         int nx=2147483647; // sanity check
         int ny=2147483647; // sanity check

         //JUST RECV
         if (amrex::MPMD::MyProc() == this_root) {
             if (rank_offset == 0) // the first program
             {
                     MPI_Recv(&nx, 1, MPI_INT, other_root, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		     MPI_Recv(&ny, 1, MPI_INT, other_root, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
             }
             else // the second program
             {
                     MPI_Recv(&nx, 1, MPI_INT, other_root, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		     MPI_Recv(&ny, 1, MPI_INT, other_root, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
             }
         }
	 if((nx)*(ny) > 0) {
	     int nsealm = (nx)*ny;
	     Print()<<nsealm<<std::endl;
	     Print()<<nx<<std::endl;
	     Print()<<ny<<std::endl;
	     AMREX_ALWAYS_ASSERT_WITH_MESSAGE((nx)*ny <= bx.numPts(), "total number of points being filled exceeds the size of the current box\n");

         if (amrex::MPMD::MyProc() == this_root) {
             if (rank_offset == 0) // the first program
             {
                     MPI_Recv(my_H_ptr, nsealm, MPI_DOUBLE, other_root, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                     MPI_Recv(my_L_ptr, nsealm, MPI_DOUBLE, other_root, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
             }
             else // the second program
             {
                     MPI_Recv(my_H_ptr, nsealm, MPI_DOUBLE, other_root, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                     MPI_Recv(my_L_ptr, nsealm, MPI_DOUBLE, other_root, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
             }
         }
	 amrex::Print()<<"Just recieved "<<nsealm<<"as a double*"<<std::endl;
         amrex::Print()<<my_H_arr(192,92,0)<<std::endl;
         amrex::Print()<<my_L_arr(192,92,0)<<std::endl;
	 amrex::Print()<<my_H_ptr[192-0+(92-0)*193]<<std::endl;
         amrex::Print()<<my_L_ptr[192-0+(92-0)*193]<<std::endl;
         amrex::AllPrintToFile("output_HS_cpp.txt")<<FArrayBox(my_H_arr)<<std::endl;
         amrex::AllPrintToFile("output_L_cpp.txt")<<FArrayBox(my_L_arr)<<std::endl;
	 }
	 else {
	     finished_wave = true;
	 }
    }
    Hwave[lev]->FillBoundary();
    Lwave[lev]->FillBoundary();
#endif

// END MY EDITS

    // Advance a single level for a single time step
    Advance(lev, time, dt[lev], istep[lev], nsubsteps[lev]);

    ++istep[lev];

    if (Verbose()) {
        amrex::Print() << "[Level " << lev << " step " << istep[lev] << "] ";
        amrex::Print() << "Advanced " << CountCells(lev) << " cells" << std::endl;
    }

    if (lev < finest_level)
    {
        // recursive call for next-finer level
        for (int i = 1; i <= nsubsteps[lev+1]; ++i)
        {
            Real strt_time_for_fine = time + (i-1)*dt[lev+1];
            timeStep(lev+1, strt_time_for_fine, i);
        }
    }
}

/**
 * Function that advances the solution at one level for a single time step --
 * this does some preliminaries then calls erf_advance
 *
 * @param[in] lev level of refinement (coarsest level is 0)
 * @param[in] time start time for time advance
 * @param[in] dt_lev time step for this time advance
 */
