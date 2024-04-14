#include "TransportB1.h"

#include "Transport/B1/include/B1DetectorConstruction.h"
#include "Transport/B1/include/B1ActionInitialization.h"

#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "QBBC.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"

#include "Randomize.hh"


TransportB1::TransportB1()
{



}

void TransportB1::Run()
{

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

  // Detect interactive mode (if no arguments) and define UI session
  //

  //G4UIExecutive* ui = 0;
  //if ( argc == 1 ) {
  //ui = new G4UIExecutive(argc, argv);
  //}

  // Optionally: choose a different Random engine...
  // G4Random::setTheEngine(new CLHEP::MTwistEngine);

  // Construct the default run manager
  //
#ifdef G4MULTITHREADED
  G4MTRunManager* runManager = new G4MTRunManager;
#else
  G4RunManager* runManager = new G4RunManager;
#endif

  // Set mandatory initialization classes
  //



  // Detector construction
  runManager->SetUserInitialization(new B1DetectorConstruction());




  // Physics list
  G4VModularPhysicsList* physicsList = new QBBC;
  physicsList->SetVerboseLevel(1);
  runManager->SetUserInitialization(physicsList);



  // User action initialization
  runManager->SetUserInitialization(new B1ActionInitialization());

  // Initialize visualization
  //
  //G4VisManager* visManager = new G4VisExecutive;
  // G4VisExecutive can take a verbosity argument - see /vis/verbose guidance.
  // G4VisManager* visManager = new G4VisExecutive("Quiet");
  //visManager->Initialize();



  // Get the pointer to the User Interface manager
  G4UImanager* UImanager = G4UImanager::GetUIpointer();


  // Process macro or start UI session
  //
  //  if ( ! ui ) {
    // batch mode
    //G4String command = "/control/execute ";
    //G4String fileName = argv[1];
    //UImanager->ApplyCommand(command+fileName);








  //}
  //else {
    // interactive mode

    //UImanager->ApplyCommand("/control/execute init_vis.mac");



    UImanager->ApplyCommand("/control/verbose 2");
    UImanager->ApplyCommand("/control/saveHistory");
    UImanager->ApplyCommand("/run/verbose 2");
    UImanager->ApplyCommand("/run/initialize");
    UImanager->ApplyCommand("/run/beamOn 5");




    std::ifstream is("/home/jiping/software/geant4.10.06.p02/examples/basic/B1/build/output.vtk");
    std::ofstream out("/home/jiping/software/geant4.10.06.p02/examples/basic/B1/build/output2.vtk");

    const int len = 256;
    char L[len];
    int n = 0;
    while(is.getline(L,len)) n++;
    is.close();

    is.open("/home/jiping/software/geant4.10.06.p02/examples/basic/B1/build/output.vtk");
    out << "# vtk DataFile Version 2.0" << std::endl;
    out << "Unstructured Grid by M++" << std::endl;
    out << "ASCII" << std::endl;
    out << "DATASET UNSTRUCTURED_GRID" << std::endl;
    out << "POINTS " << 2*n << " float" << std::endl;
    while(is.getline(L,len)) {
    double z[6];
    sscanf(L, "%lf %lf %lf %lf %lf %lf", z, z + 1, z + 2, z + 3, z + 4, z + 5);
    out << z[0] << " " << z[1] << " " << z[2] << std::endl;
    out << z[3] << " " << z[4] << " " << z[5] << std::endl;
    }
    out << "CELLS " << n << " " << 3*n << std::endl;
    for (int i = 0; i < n; i++) {
    out << 2 << " " << i*2 << " " << i*2+1 << std::endl;
    }
    out << "CELL_TYPES " << n << std::endl;
    for (int i = 0; i < n; i++) {
    out << 3 << std::endl;
    }
    out.close();
    is.close();













    //ui->SessionStart();
    //    delete ui;
    //}


  // Job termination
  // Free the store: user actions, physics_list and detector_description are
  // owned and deleted by the run manager, so they should not be deleted
  // in the main() program !








  //  delete visManager;
  delete runManager;










}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....


