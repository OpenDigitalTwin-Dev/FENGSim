/*************************************************************************
 *
 * This file is part of the SAMRAI distribution.  For full copyright
 * information, see COPYRIGHT and COPYING.LESSER.
 *
 * Copyright:     (c) 1997-2016 Lawrence Livermore National Security, LLC
 * Description:   Main program for SAMRAI Linear Advection example problem.
 *
 ************************************************************************/

#include "SAMRAI/SAMRAI_config.h"

// Header for application-specific algorithm/data structure object

#include "Stencil.h"

// Headers for major algorithm/data structure objects

#include "SAMRAI/appu/VisItDataWriter.h"
#include "SAMRAI/mesh/BergerRigoutsos.h"
#include "SAMRAI/geom/CartesianGridGeometry.h"
#include "SAMRAI/mesh/GriddingAlgorithm.h"
#include "SAMRAI/algs/HyperbolicLevelIntegrator.h"
#include "SAMRAI/mesh/CascadePartitioner.h"
#include "SAMRAI/hier/PatchHierarchy.h"
#include "SAMRAI/mesh/StandardTagAndInitialize.h"
#include "SAMRAI/algs/TimeRefinementIntegrator.h"
#include "SAMRAI/algs/TimeRefinementLevelStrategy.h"
#include "SAMRAI/mesh/TileClustering.h"
#include "SAMRAI/tbox/NVTXUtilities.h"

// Headers for basic SAMRAI objects

#include "SAMRAI/hier/VariableDatabase.h"
#include "SAMRAI/hier/PatchLevel.h"
#include "SAMRAI/tbox/SAMRAIManager.h"
#include "SAMRAI/tbox/BalancedDepthFirstTree.h"
#include "SAMRAI/tbox/Database.h"
#include "SAMRAI/tbox/InputDatabase.h"
#include "SAMRAI/tbox/InputManager.h"
#if 1
#include "SAMRAI/tbox/SAMRAI_MPI.h"
#endif
#include "SAMRAI/tbox/PIO.h"
#include "SAMRAI/tbox/RestartManager.h"
#include "SAMRAI/tbox/Utilities.h"

#ifndef _MSC_VER
#include <unistd.h>
#endif

#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <fstream>
#include <memory>
#include <iomanip>

#include "SAMRAI/tbox/StartupShutdownManager.h"

#if defined(HAVE_CUDA)
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#endif

using namespace std;
using namespace SAMRAI;

int main(
    int argc,
    char* argv[])
{

  /*
   * Initialize tbox::MPI.
   */

  tbox::SAMRAI_MPI::init(&argc, &argv);
  tbox::SAMRAIManager::initialize();

  /*
   * Set tag allocator to use pinned memory.
   */
#if defined(HAVE_CUDA) || defined(HAVE_HIP)
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();
  rm.makeAllocator<umpire::strategy::QuickPool>("samrai::tag_allocator",
    rm.getAllocator(umpire::resource::Pinned));
#endif

  /*
   * Initialize SAMRAI, enable logging, and process command line.
   */
  tbox::SAMRAIManager::startup();
  const tbox::SAMRAI_MPI& mpi(tbox::SAMRAI_MPI::getSAMRAIWorld());

  bool success = true;

  {
    string input_filename;
    string restart_read_dirname;
    int restore_num = 0;

    bool is_from_restart = false;

    if ((argc != 2) && (argc != 4)) {
      tbox::pout << "USAGE:  " << argv[0] << " <input filename> "
        << "<restart dir> <restore number> [options]\n"
        << "  options:\n"
        << "  none at this time"
        << endl;
      tbox::SAMRAI_MPI::abort();
      return -1;
    } else {
      input_filename = argv[1];
      if (argc == 4) {
        restart_read_dirname = argv[2];
        restore_num = atoi(argv[3]);

        is_from_restart = true;
      }
    }

    tbox::plog << "input_filename = " << input_filename << endl;
    tbox::plog << "restart_read_dirname = " << restart_read_dirname << endl;
    tbox::plog << "restore_num = " << restore_num << endl;

    /*
     * Create input database and parse all data in input file.
     */

    std::shared_ptr<tbox::InputDatabase> input_db(
        new tbox::InputDatabase("input_db"));
    tbox::InputManager::getManager()->parseInputFile(input_filename, input_db);

    /*
     * Retrieve "Main" section of the input database.  First, read
     * dump information, which is used for writing plot files.
     * Second, if proper restart information was given on command
     * line, and the restart interval is non-zero, create a restart
     * database.
     */

    std::shared_ptr<tbox::Database> main_db(
        input_db->getDatabase("Main"));

    const tbox::Dimension dim(static_cast<unsigned short>(main_db->getInteger("dim")));

    const std::string base_name =
      main_db->getStringWithDefault("base_name", "unnamed");

    const std::string log_filename =
      main_db->getStringWithDefault("log_filename", base_name + ".log");

    bool log_all_nodes = false;
    if (main_db->keyExists("log_all_nodes")) {
      log_all_nodes = main_db->getBool("log_all_nodes");
    }
    if (log_all_nodes) {
      tbox::PIO::logAllNodes(log_filename);
    } else {
      tbox::PIO::logOnlyNodeZero(log_filename);
    }

    int viz_dump_interval = 1;
    if (main_db->keyExists("viz_dump_interval")) {
      viz_dump_interval = main_db->getInteger("viz_dump_interval");
    }

#ifdef HAVE_HDF5
    const std::string viz_dump_dirname =
      main_db->getStringWithDefault("viz_dump_dirname", base_name + ".visit");
    int visit_number_procs_per_file = 1;
#endif

    const std::string restart_write_dirname =
      main_db->getStringWithDefault("restart_write_dirname",
          base_name + ".restart");

    bool use_refined_timestepping = false;

    if (main_db->keyExists("timestepping")) {
      string timestepping_method = main_db->getString("timestepping");
      if (timestepping_method == "REFINED") {
        use_refined_timestepping = true;
      }
    }

    /*
     * Get the restart manager and root restart database.  If run is from
     * restart, open the restart file.
     */

    tbox::RestartManager* restart_manager = tbox::RestartManager::getManager();

    if (is_from_restart) {
      restart_manager->
        openRestartFile(restart_read_dirname, restore_num,
            mpi.getSize());
    }

    tbox::TimerManager::createManager(input_db->getDatabase("TimerManager"));

    /*
     * Create major algorithm and data objects which comprise application.
     * Each object will be initialized either from input data or restart
     * files, or a combination of both.  Refer to each class constructor
     * for details.  For more information on the composition of objects
     * for this application, see comments at top of file.
     */

    std::shared_ptr<geom::CartesianGridGeometry> grid_geometry(
        new geom::CartesianGridGeometry(
          dim,
          "CartesianGeometry",
          input_db->getDatabase("CartesianGeometry")));

    std::shared_ptr<hier::PatchHierarchy> patch_hierarchy(
        new hier::PatchHierarchy(
          "PatchHierarchy",
          grid_geometry,
          input_db->getDatabase("PatchHierarchy")));

    Stencil* stencil_model = new Stencil(
        "Stencil",
        dim,
        input_db->getDatabase("Stencil"),
        grid_geometry);

    std::shared_ptr<algs::HyperbolicLevelIntegrator> hyp_level_integrator(
        new algs::HyperbolicLevelIntegrator(
          "HyperbolicLevelIntegrator",
          input_db->getDatabase("HyperbolicLevelIntegrator"),
          stencil_model,
          use_refined_timestepping));

    std::shared_ptr<mesh::StandardTagAndInitialize> error_detector(
        new mesh::StandardTagAndInitialize(
          "StandardTagAndInitialize",
          hyp_level_integrator.get(),
          input_db->getDatabase("StandardTagAndInitialize")));

    bool use_tile_clustering = false;
    use_tile_clustering = main_db->getBoolWithDefault(
        "use_tile_clustering",
        use_tile_clustering);

    std::shared_ptr<mesh::BoxGeneratorStrategy> box_generator;

    if (use_tile_clustering) {
      box_generator = std::make_shared<mesh::TileClustering>(
          dim,
          input_db->getDatabaseWithDefault(
            "TileClustering",
            std::shared_ptr<tbox::Database>()));
    } else {
      box_generator = std::make_shared<mesh::BergerRigoutsos>(
            dim,
            input_db->getDatabaseWithDefault(
              "BergerRigoutsos",
              std::shared_ptr<tbox::Database>()));
    }

    std::shared_ptr<mesh::CascadePartitioner> load_balancer(
        new mesh::CascadePartitioner(
          dim,
          "LoadBalancer",
          input_db->getDatabase("LoadBalancer")));
    load_balancer->setSAMRAI_MPI(
        tbox::SAMRAI_MPI::getSAMRAIWorld());

    std::shared_ptr<mesh::GriddingAlgorithm> gridding_algorithm(
        new mesh::GriddingAlgorithm(
          patch_hierarchy,
          "GriddingAlgorithm",
          input_db->getDatabase("GriddingAlgorithm"),
          error_detector,
          box_generator,
          load_balancer));

    std::shared_ptr<algs::TimeRefinementIntegrator> time_integrator(
        new algs::TimeRefinementIntegrator(
          "TimeRefinementIntegrator",
          input_db->getDatabase("TimeRefinementIntegrator"),
          patch_hierarchy,
          hyp_level_integrator,
          gridding_algorithm));

#ifdef HAVE_HDF5
    std::shared_ptr<appu::VisItDataWriter> visit_data_writer(
       new appu::VisItDataWriter(
          dim,
          "Euler VisIt Writer",
          viz_dump_dirname,
          visit_number_procs_per_file));
    stencil_model->registerVisItDataWriter(visit_data_writer);
#endif

    /*
     * Initialize hierarchy configuration and data on all patches.
     * Then, close restart file and write initial state for visualization.
     */
    SAMRAI_CALI_MARK_BEGIN("initHierarchy");
    double dt_now = time_integrator->initializeHierarchy();
    SAMRAI_CALI_MARK_END("initHierarchy");

    tbox::RestartManager::getManager()->closeRestartFile();

    /*
     * After creating all objects and initializing their state, we
     * print the input database and variable database contents
     * to the log file.
     */

    tbox::plog << "\nCheck input data and variables before simulation:"
      << endl;
    tbox::plog << "Input database..." << endl;
    input_db->printClassData(tbox::plog);
    tbox::plog << "\nVariable database..." << endl;
    hier::VariableDatabase::getDatabase()->printClassData(tbox::plog);

#ifdef HAVE_HDF5
    if ((viz_dump_interval > 0))
      visit_data_writer->writePlotData(
          patch_hierarchy,
          time_integrator->getIntegratorStep(),
          time_integrator->getIntegratorTime());
#endif

    // tbox::plog << "\nCheck Linear Advection data... " << endl;
    // linear_advection_model->printClassData(tbox::plog);

    /*
     * Time step loop.  Note that the step count and integration
     * time are maintained by algs::TimeRefinementIntegrator.
     */

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    double loop_time = time_integrator->getIntegratorTime();
    double loop_time_end = time_integrator->getEndTime();

    int iteration_num = time_integrator->getIntegratorStep();

    while ((loop_time < loop_time_end) &&
        time_integrator->stepsRemaining()) {

      iteration_num = time_integrator->getIntegratorStep() + 1;

      tbox::pout << "++++++++++++++++++++++++++++++++++++++++++++" << endl;
      tbox::pout << "At begining of timestep # " << iteration_num - 1
        << endl;
      tbox::pout << "Simulation time is " << loop_time << endl;
//      char buf[50];
//      sprintf(buf, "Timestep %d", iteration_num);
//      RANGE_PUSH(buf, 2);
//#if defined(HAVE_CUDA)
//      if (iteration_num == 11)
//        cudaProfilerStart();
//#endif
      SAMRAI_CALI_MARK_BEGIN("advance");
      double dt_new = time_integrator->advanceHierarchy(dt_now);
      SAMRAI_CALI_MARK_END("advance");
//#if defined(HAVE_CUDA)
//      if (iteration_num == 13)
//        cudaProfilerStop();
//#endif
//      RANGE_POP;
      loop_time += dt_now;
      dt_now = dt_new;

      tbox::pout << "At end of timestep # " << iteration_num - 1 << endl;
      tbox::pout << "Simulation time is " << loop_time << endl;
      tbox::pout << "++++++++++++++++++++++++++++++++++++++++++++" << endl;

#ifdef HAVE_HDF5
       if ((viz_dump_interval > 0)
           && (iteration_num % viz_dump_interval) == 0) {
          visit_data_writer->writePlotData(patch_hierarchy,
             iteration_num,
             loop_time);
       }
#endif
       {
          // /*
          //  * Compute a solution norm as a check.
          //  */
           double norm = 0.0;
           int nlevels = patch_hierarchy->getNumberOfLevels();
           for (int ln = 0; ln < nlevels; ++ln) {
             const std::shared_ptr<hier::PatchLevel>& level(patch_hierarchy->getPatchLevel(ln));
             for (hier::PatchLevel::Iterator p(level->begin()); p != level->end(); ++p) {
               const std::shared_ptr<hier::Patch>& patch = *p;
               norm += stencil_model->computeNorm(hyp_level_integrator->getCurrentContext(), *patch);
             }
           }
           tbox::pout << "Solution norm: " << std::scientific << std::setprecision(12) << norm << std::endl;
       }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    tbox::pout << "Time elapsed = " << (end - start) << endl;

    // /*
    //  * Compute a solution norm as a check.
    //  */
     double norm = 0.0;
     int nlevels = patch_hierarchy->getNumberOfLevels();
     for (int ln = 0; ln < nlevels; ++ln) {
       const std::shared_ptr<hier::PatchLevel>& level(patch_hierarchy->getPatchLevel(ln));
       for (hier::PatchLevel::Iterator p(level->begin()); p != level->end(); ++p) {
         const std::shared_ptr<hier::Patch>& patch = *p;
         norm += stencil_model->computeNorm(hyp_level_integrator->getCurrentContext(), *patch);
       }

     }

    /*
     * Output timer results.
     */
    tbox::TimerManager::getManager()->print(tbox::pout);

    mpi.AllReduce(&norm, 1, MPI_SUM);

    tbox::pout << "Solution norm: " << std::scientific << std::setprecision(12) << norm << std::endl;

    if (main_db->keyExists("norm_baseline")) {
       double baseline = main_db->getDouble("norm_baseline");
       if (!tbox::MathUtilities<double>::equalEps(baseline, norm)) {
          tbox::pout << "Solution norm does not equal expected baseline: " << std::scientific << std::setprecision(12) << baseline << std::endl;
          success = false;
       }
    }

    /*
     * At conclusion of simulation, deallocate objects.
     */

    time_integrator.reset();
    gridding_algorithm.reset();
    load_balancer.reset();
    box_generator.reset();
    error_detector.reset();
    hyp_level_integrator.reset();

    if (stencil_model) delete stencil_model;

    patch_hierarchy.reset();
    grid_geometry.reset();

    input_db.reset();
    main_db.reset();

  }
  if (success) { 
    tbox::pout << "\nPASSED:  Stencil" << std::endl;
  } else {
    tbox::pout << "\nFAILED:  Stencil" << std::endl;
  }

  tbox::SAMRAIManager::shutdown();

  tbox::SAMRAIManager::finalize();
  tbox::SAMRAI_MPI::finalize();
  return(0);
}
