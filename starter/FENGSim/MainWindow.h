// this is the main window for fengsim
// created by jiping 2016
#ifndef MAINWINDOW_H
#define MAINWINDOW_H
// project define class
#include <QMainWindow>
#include <QComboBox>
#include <QProcess>
#include <QTimer>
#include <QFileDialog>
#include <QString>

#include "CAD/Primitive.h"
#include "CAD/OCCWidget.h"
#include "AboutDialog.h"
#include "CAD/CADDockWidget.h"
#include "CAD/PhysicsDockWidget.h"
#include "Mesh/MeshDockWidget.h"
#include "Mesh/MeshGeneration.h"
#include "Visual/VTKDockWidget.h"
#include "Visual/VTKWidget.h"
//#include "DataBase/DataBaseWindow.h"
#include "FEM/FEMDockWidget.h"
#include "Measure/MeasureDockWidget.h"
//#include "Measure/Registration.h"
#include "AdditiveManufacturing/AdditiveManufacturingDockWidget.h"
#include "AdditiveManufacturing/AMConfig.h"
#include "StatisticalProcessControl/SPCDockWidget.h"
#include "Machining/MachiningDockWidget.h"
#include "Machining/MachiningDockWidget2.h"
#include "Transport/TransportDockWidget.h"
#include "Measure/MeasureThread1.h"
#include "Measure/MeasureThread3.h"
#include "AdditiveManufacturing/AMThread1.h"
#include "AdditiveManufacturing/AMThread2.h"
#include "OCPoro/OCPoroDockWidget.h"
#include "OCPoro/OCPoroDialog.h"
#include "Mesh/MeshGeneration.h"
#include "Rivet/RivetDockWidget.h"
#include "Pipe/PipeDockWidget.h"


// ui design
namespace Ui {
class MainWindow;
}

#include <QThread>
#include <QProcess>
#include <QTextEdit>
#include "fstream"

class MeasureThread2;

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
private:
    Ui::MainWindow *ui;



    // ##############################################################################################
    // ##############################################################################################
    //
    // version 1.0 updated by Dr. Jiping Xin
    //
    // ##############################################################################################
    // ##############################################################################################

    void SetActionChecked (int n);



private slots:

    void OpenCADModule ();
    void OpenPhysicsModule ();
    void OpenMeshModule ();
    void OpenSolverModule ();
    void OpenVisualModule ();

    void NewProject ();
    void OpenProject ();



    // ###############################################
    // ###############################################










    // ##############################################################################################
    // ##############################################################################################
    // ##############################################################################################




















private:




    // cad dock widget
    CADDockWidget* cad_dock;
    PhysicsDockWidget* physics_dock;

    OCCWidget* OCCw;
    // mesh dock widget
    MeshDockWidget* mesh_dock;
    // visual dock widget
    VTKDockWidget* vtk_dock;
    VTKWidget* vtk_widget;
    // fem dock widget
    FEMDockWidget* fem_dock;
    // spc dock widget
    SPCDockWidget* spc_dock;
    // machining dock widget
    MachiningDockWidget* machining_dock;
    MachiningDockWidget2* machining_dock2;


    // cad data
    Primitives* parts;
    Boundaries* bnds;
    int boolean_part1;
    int boolean_part2;
    int boolean_type;
    vtkActor* Obj1;
    vtkActor* Obj2;
    Primitive* Prim1;
    Primitive* Prim2;
    Handle(AIS_InteractiveObject) BooleanFirst;
    Handle(AIS_InteractiveObject) BooleanSecond;
    QTimer* timer;

public slots:
    // view operations
    void Fit ();
    void Front ();
    void Back ();
    void Left ();
    void Right ();
    void Top ();
    void Bottom ();
    void Axo ();
    void ViewRotationH ();
    void SetViewRotationH ();
    void ViewRotationV ();
    void SetViewRotationV ();
    void Plot (TopoDS_Shape* S);
    void Plot (const TopoDS_Shape& S);
    // cad operations
    // remember all these operations are for data strucutres
    // new, select, change, remove
    // :-)
    // create a vertex, database add
    void MakeVertex (double x1=0, double x2=0, double x3=0);
    // create a line
    void MakeLine (double x11=0, double x12=0, double x13=0,
                   double x21=1, double x22=0, double x23=0);
    // create a plane
    void MakePlane (double p1=0, double p2=0, double p3=0,
                    double d1=0, double d2=0, double d3=1,
                    double x1 = 0, double x2 = 1,
                    double y1 = 0, double y2 = 1);
    // create a box
    void MakeBox (double x1=1, double x2=1, double x3=1,
                  double p1=0, double p2=0, double p3=0,
                  double d1=0, double d2=0, double d3=1);
    // create a sphere
    void MakeSphere (double r=1,
                     double p1=0, double p2=0, double p3=0,
                     double d1=0, double d2=0, double d3=1);
    // create a cylinder
    void MakeCylinder (double r=1, double h=1,
                       double p1=0, double p2=0, double p3=0,
                       double d1=0, double d2=0, double d3=1);
    // create a cone
    void MakeCone (double r1=1, double r2=0.0,
                   double h=1,
                   double p1=0, double p2=0, double p3=0,
                   double d1=0, double d2=0, double d3=1);
    // create a torus
    void MakeTorus (double r1=1, double r2=0.3,
                    double p1=0, double p2=0, double p3=0,
                    double d1=0, double d2=0, double d3=1);
    // remove an object, database delete
    void Remove ();
    // change an object, database edit
    void Change ();
    // boolean operations
    void SetBooleanPart1 ();
    void SetBooleanPart2 ();
    void BooleanUnion ();
    void BooleanSection ();
    void BooleanCut ();
    // sweep extrude mirror
    void CommonOperations ();
    // file
    void Save ();

    // windows

    void OpenMeasureModule ();
    void OpenSPCModule ();
    void OpenMachiningModule ();
    // mesh
public slots:

    void MeshGen ();
    void MeshPlot ();

    // fem
    // void ImportVTKFile () { vtk_dock->ImportVTKFile(); }
    // physical definition
    void SelectBnd ();
    void SelectDomain ();
    void UpdateBndValue ();
    void ShowBoundariesOrDomains ();
    void SetBoundaryTypeAndID ();
    //void CloudPointRegistration ();
    void CloudPointMove ();
    void CloudPointMoveType ();
    void CloudPointReset ();
    void CloudPointTargetReset ();
    void CloudPointSourceReset ();
    void SetTranSpinBoxStep ();
    void ShowCloudSourceAndTarget ();
    void BoxFit ();
public slots:
    void ImportAMStlModel ();
    void AMStlModelShow ();

    void ImportAMSlices ();
    void AMSlicesShow ();
    void AMStlModelToSlices ();
    void AMSlicesToPathPlanning();

    void AMReset ();
    void AMVoxelMeshGeneration ();
    void AMPathPlanning ();
    void AMMeshShow ();
    void AMPathPlanningShow ();

private:
    QString stl_file_name;
    QString cli_file_name;
    QString path_file_name;
    QString mesh_file_name;
    AMThread1* amtd1;
    AMThread2* amtd2;

private:
    //Registration reg;
    General* measure_cad;
private:
    TopoDS_Shape* CurrentBoundaryObj;
public slots:
    void BoundaryChecked ();
    void DomainChecked ();
private:
    AboutDialog* about_dialog;
public slots:
    // about dialog
    void OpenAboutDialog ();
    // Database Module
    //private:
    //    DataBaseWindow* dbwind;
    //public slots:
    //    void DataBaseWindowShow ();
public:
    MeshModule MM;

    // additive manufacturing
private:
public slots:
    void OpenAdditiveManufacturingModule ();









    // *******************************************************
    // measure
    // *******************************************************
private:
    MeasureDockWidget* measure_dock;
    Primitives* meas_parts;
    Boundaries* meas_bnds;
    bool measure_cloud_source_ischeck;
    QTimer* measure_timer =  new QTimer;
    int meas_scene_type = 0;
    MeasureThread1* meas_th1;
    MeasureThread2* meas_th2;
    MeasureThread3* meas_th3;
    int meas_line_num = 0;
    double meas_tran[6];
    int measure_exe_id;
public:
    QString meas_path;
public slots:
    void MeasureOpenCAD ();
    void MeasurePresBnd ();
    void MeasurePresLine ();
    void MeasureSelectedBndToPointCloud ();
    void MeasureSelectedBndToPointCloud2 ();
    void MeasureOpenPointCloud();
    void MeasureOpenPointCloud2();
    void MeasurePointCloudMove();
    void MeasurePointCloudReset();
    void MeasureRegistration ();
    void MeasurePlotResults ();
    void MeasurePCMove ();
    void MeasurePCMoveValue ();
    void MeasureSACIA ();
    void MeasureICP ();
    void MeasureGDT ();
    void MeasureCADHide ();
    void MeasureCADHide2 ();
    void MeasureCloudTargetHide ();
    void MeasureCloudSourceHide ();
    void MeasureCloudICPHide ();
    void MeasureCloudGDTHide ();
    void MeasureDeleteBnd ();
    void MeasureOpacity ();
    void MeasureSACIAPro ();


    // visualization
    void ImportVTKFile();








    // *******************************************************
    // additive manufacturing
private:
    AdditiveManufacturingDockWidget* additive_manufacturing_dock;
    Primitives* am_parts;
    Boundaries* am_bnds;
    General* am_cad;
    int am_sim_num;
    QTimer* am_timer =  new QTimer;
    AMConfig amconfig;
public slots:
    void AMOpenCAD ();
    void AMSetCADVisible ();
    void AMCAD2STL ();
    void AMSetSTLVisible ();
    void AMSTL2Slices ();
    void AMSetSlicesVisible ();
    void AMSlices2PathPlanning ();
    void AMSetPathPlanningVisible ();
    void AMSlices2Mesh ();
    void AMSetMeshVisible ();
    void AMSimulation ();
    void AMSimulationPlot ();
    void AMSimulationAnimation ();





    // *******************************************************
    // FEM
private:
    int fem_file_num;
    int fem_file_id;
    QTimer* fem_timer =  new QTimer;
public slots:
    void FEMConfigure ();
    void FEMCompute ();
    void FEMPlot ();
    void FEMExampleCompute ();
    void FEMAnimation ();



    // *******************************************************
    // multibody dynamic
private:
    int mbd_time_sum = 0;
    int mbd_time_step = 0;
    int mbd_speed = 0;
    QTimer* mbd_timer =  new QTimer;
    QString mbd_file_name;
public slots:
    void mbdOpenFile ();
    void mbdImportResults () {
        if (mbd_time_step*5 > mbd_time_sum) {
            mbd_time_step = 0;
            mbd_speed = 0;
            return;
        }
        vtk_widget->mbdImportResults(mbd_time_step,mbd_file_name);
        mbd_speed++;
        //mbd_time_step++;
        mbd_time_step = mbd_speed;
        mbd_timer->singleShot(1, this, SLOT(mbdImportResults()));
    }


    // *******************************************************
    // Machining
private:
    Cube* machining_part;
    Cube* machining_tool;
    Boundaries* machining_bnds;
    bool machining_part_done = false;
    bool machining_tool_done = false;
    double machining_tool_size[3];
    double machining_tool_pos[3];
    double machining_part_size[3];
    double machining_part_pos[3];
    int machining_ani_num;
    QTimer* machining_timer =  new QTimer;

public slots:
    void MachiningPartSet ();
    void MachiningToolSet ();
    void MachiningPartParametersUpdate ();
    void MachiningToolParametersUpdate ();
    void MachiningMakePart();
    void MachiningMakeTool();
    void MachiningSetDomainData();
    void MachiningSetBoundaryData();
    void MachiningMeshGeneration ();
    void MachiningSimulation ();
    void MachiningSimulationPlot ();
    void MachiningSimulationAnimation ();
    void MachiningRecompute ();


    // *******************************************************
    // transport
private:
    TransportDockWidget* transport_dock;
    Primitives* transport_parts;
    int transport_level;
    int transport_num;
public slots:
    void OpenTransportModule ();
    void TransportCADSave ();
    void TransportParaModel ();
    void TransportMakeSphere (double r,
                              double p1, double p2, double p3,
                              double d1, double d2, double d3);
    void TransportMakeCylinder (double r, double h,
                                double p1, double p2, double p3,
                                double d1, double d2, double d3, int color);
    void TransportMakeRing ();
    void TransportMakeDetectors ();
    void TransportSelect ();
    void TransportMCRun ();
    void TransportParaModelPlot ();

    // *******************************************************
    // general
    void TextOutput (QString str)
    {
        vtk_widget->TextOutput(str);
    }






    // *******************************************************
    // CAE poro
public:
    OCPoroDockWidget* ocporo_dock;
    int attnum;
    QString ocporofilename;
    OCPoroDialog* ocporosummary;
    OCPoroDialog* ocporosummary1;
    OCPoroDialog* ocporosummary2;
    std::vector<std::vector<double>> ocporosummarydata;
public slots:
    void OpenOCPoroModule ();
    void OCPoroImportVTKFile();
    void OCPoroSwitchAtt ();
    void OCPoroImportSummary ();


    /* !
  rivet app
  */

private:
    QTimer* rivet_timer =  new QTimer;
    int rivet_step = 1;
    int rivet_total_step = 0;
    QString rivet_file_name;
    double rivet_mesh_size = 10000;
    double rivet_mesh_refine = 0;
public:
    RivetDockWidget* rivet_dock;
public slots:
    void OpenRivetModule ();
    void rivetCreateModel ();
    void rivetModelRefresh ();
    void rivetMeshGen ();
    void rivetMeshPlot ();
    void rivetMeshRefresh ();
    void rivetSolver ();
    void rivetImportResults ();

    /* !
  pipe app
  */
private:
    QTimer* pipe_timer =  new QTimer;
    int pipe_step = 1;
    int pipe_total_step = 0;
    QString pipe_file_name;
    double pipe_mesh_size = 10000;
    double pipe_mesh_refine = 0;
public:
    PipeDockWidget* pipe_dock;
public slots:
    void OpenPipeModule ();
    void PipeCreateModel ();
    void PipeModelRefresh ();
    void PipeMeshGen ();
    void PipeMeshPlot ();
    void PipeSolver ();
    void PipeImportResults ();

    /* !
  machining2 app
  */
private:
    QTimer* machining2_timer =  new QTimer;
    int machining2_step = 0;
    int machining2_total_step = 0;
    QString machining2_file_name;
    double machining2_mesh_size = 10000;
    double machining2_mesh_refine = 0;
    General* machining2_part;
public slots:
    void Machining2Create3DModel ();
    void Machining2Mesh3DGen ();
    void Machining2Solver ();
    void Machining2ImportResults ();
    void Machining2ImportMPMResults ();



};

class MeasureThread2 : public QThread
{
    Q_OBJECT
public:
    double progress;
    VTKWidget* vtk_widget;
    Boundaries* meas_bnds;
    MeasureDockWidget* measure_dock;
    QTextEdit* measure_error = new QTextEdit;
    QString path;
    MeasureThread2() {}
    int measure_exe_id;
    void run ();
    void MeasureVariance (double mean, QString path);};


#endif // MAINWINDOW_H
