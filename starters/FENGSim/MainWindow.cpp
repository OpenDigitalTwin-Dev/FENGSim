//#include <QDesktopWidget>
#include <QComboBox>
#include <QFile>
//#include <QVTKWidget.h>
//#include "vtkSmartPointer.h"
//#include "vtkRenderer.h"
//#include "vtkRenderWindow.h"
// main window gui
#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "ui_CADDockWidget.h"
#include "ui_PhysicsDockWidget.h"
#include "ui_MeshDockWidget.h"
#include "ui_VTKDockWidget.h"
// vtk widget
//#include "DataVisual/VTKWidget.h"
// vtk render window
//#include "vtkGenericOpenGLRenderWindow.h"
#include "CAD/CADDockWidget.h"
//#include "ui_DataBaseWindow.h"
#include "ui_FEMDockWidget.h"
#include "Measure/MeasureDockWidget.h"
#include "ui_MeasureDockWidget.h"
//#include "Measure/Registration.h"
#include "ui_AdditiveManufacturingDockWidget.h"
#include "ui_MachiningDockWidget.h"
#include "ui_TransportDockWidget.h"
#include "ui_OCPoroDockWidget.h"
#include "ui_OCPoroDialog.h"
#include "Machining/MakeTools.h"
#include "QToolButton"
#include "QTreeWidget"
#include "STEPControl_Reader.hxx"
#include <QFileDialog>
#include "BRep_Builder.hxx"
#include "STEPControl_Writer.hxx"
#include "qcustomplot.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    // ui improve
    ui->setupUi(this);
    setWindowState(Qt::WindowMaximized);





    // ##############################################################################################
    // ##############################################################################################
    //
    // version 1.0 updated by Dr. Jiping Xin
    //
    // ##############################################################################################
    // ##############################################################################################

    connect(ui->actionNew, SIGNAL(triggered()), this, SLOT(NewProject()));
    connect(ui->actionCAD, SIGNAL(triggered()), this, SLOT(OpenCADModule()));
    connect(ui->actionPhysics, SIGNAL(triggered()), this, SLOT(OpenPhysicsModule()));
    connect(ui->actionMesh, SIGNAL(triggered()), this, SLOT(OpenMeshModule()));
    connect(ui->actionSolver, SIGNAL(triggered()), this, SLOT(OpenSolverModule()));
    connect(ui->actionVisual, SIGNAL(triggered()), this, SLOT(OpenVisualModule()));

    QToolButton *cadView = new QToolButton(this);
    cadView->setPopupMode(QToolButton::InstantPopup);
    cadView->setIcon(QIcon(":/main_wind/figure/main_wind/direction.png"));
    cadView->setMenu(ui->menuView);
    ui->toolBar->insertWidget(ui->actionAdditiveManufacturing,cadView);
    ui->toolBar->insertSeparator(ui->actionAdditiveManufacturing);

    connect(ui->actionOpen, SIGNAL(triggered()), this, SLOT(OpenProject()));







    // ##############################################################################################
    // ##############################################################################################



















    cad_dock = new CADDockWidget;
    physics_dock = new PhysicsDockWidget;
    mesh_dock = new MeshDockWidget;
    // dock for cad
    ui->dockWidget->move(100,200);
    // get a pencil and blackboard
    // ViewWidget = new QWidget(this);
    //    OCCw = new OCCWidget(this);
    //    OCCw->Initialize();
    // setCentralWidget(OCCw);
    // CAD operations
    parts =  new Primitives;
    bnds =  new Boundaries;
    // OCCw->SetPrimitivesDS(parts);
    // OCCw->SetTableWidget(cad_dock->ui->tableWidget);
    // set vtk widget
    vtk_dock = new VTKDockWidget;
    vtk_widget = new VTKWidget;
    vtk_widget->SetCADDockWidget(cad_dock);
    vtk_widget->SetPhyDockWidget(physics_dock);
    setCentralWidget(vtk_widget);
    vtk_widget->SetPrims(parts);
    vtk_widget->SetBnds(bnds);
    //vtk_widget->SetTableWidget(cad_dock->ui->tableWidget);

    timer = new QTimer;

    // fem dockwidget
    fem_dock = new FEMDockWidget;
    // view operations
    connect(ui->actionFit, SIGNAL(triggered()), this, SLOT(Fit()));
    connect(ui->actionFront, SIGNAL(triggered()), this, SLOT(Front()));
    connect(ui->actionBack, SIGNAL(triggered()), this, SLOT(Back()));
    connect(ui->actionLeft, SIGNAL(triggered()), this, SLOT(Left()));
    connect(ui->actionRight, SIGNAL(triggered()), this, SLOT(Right()));
    connect(ui->actionTop, SIGNAL(triggered()), this, SLOT(Top()));
    connect(ui->actionBottom, SIGNAL(triggered()), this, SLOT(Bottom()));
    connect(ui->actionAxo, SIGNAL(triggered()), this, SLOT(Axo()));
    connect(ui->actionViewRotationH, SIGNAL(triggered()), this, SLOT(SetViewRotationH()));
    connect(ui->actionViewRotationV, SIGNAL(triggered()), this, SLOT(SetViewRotationV()));



    // add
    connect(cad_dock->ui->pushButton, SIGNAL(clicked()), this, SLOT(MakeVertex()));
    connect(cad_dock->ui->actionLine, SIGNAL(triggered()), this, SLOT(MakeLine()));
    connect(cad_dock->ui->actionSquare, SIGNAL(triggered()), this, SLOT(MakePlane()));
    connect(cad_dock->ui->actionCube, SIGNAL(triggered()), this, SLOT(MakeBox()));
    connect(cad_dock->ui->actionSphere, SIGNAL(triggered()), this, SLOT(MakeSphere()));
    connect(cad_dock->ui->actionCylinder, SIGNAL(triggered()), this, SLOT(MakeCylinder()));
    connect(cad_dock->ui->actionCone, SIGNAL(triggered()), this, SLOT(MakeCone()));
    connect(cad_dock->ui->actionTorus, SIGNAL(triggered()), this, SLOT(MakeTorus()));
    // remove
    connect(cad_dock->ui->pushButton_9, SIGNAL(clicked()), this, SLOT(Remove()));
    // change
    // select
    connect(cad_dock->ui->pushButton_33, SIGNAL(clicked()), this, SLOT(Change()));
    // boolean operations;
    boolean_part1 = -1;
    boolean_part2 = -1;
    connect(cad_dock->ui->actionPart1, SIGNAL(triggered()), this, SLOT(SetBooleanPart1()));
    connect(cad_dock->ui->actionPart2, SIGNAL(triggered()), this, SLOT(SetBooleanPart2()));
    connect(cad_dock->ui->actionUnion, SIGNAL(triggered()), this, SLOT(BooleanUnion()));
    connect(cad_dock->ui->actionSection, SIGNAL(triggered()), this, SLOT(BooleanSection()));
    connect(cad_dock->ui->actionCut, SIGNAL(triggered()), this, SLOT(BooleanCut()));
    // more operations also change
    connect(cad_dock->ui->pushButton_36, SIGNAL(clicked()), this, SLOT(CommonOperations()));
    connect(cad_dock->ui->actionSelectBnd, SIGNAL(triggered(bool)), this, SLOT(SelectBnd()));
    connect(cad_dock->ui->actionSelectDomain, SIGNAL(triggered(bool)), this, SLOT(SelectDomain()));
    //connect(cad_dock->ui->pushButton_8, SIGNAL(clicked()), this, SLOT(UpdateBndValue()));
    //connect(cad_dock->ui->pushButton_10, SIGNAL(clicked()), this, SLOT(OpenProject()));
    connect(cad_dock->ui->pushButton_6, SIGNAL(clicked()), this, SLOT(OpenProject()));
    connect(physics_dock->ui->pushButton, SIGNAL(clicked()), this, SLOT(UpdateBndValue()));
    // module
    ui->dockWidget->hide();
    connect(ui->actionCAD, SIGNAL(triggered()), this, SLOT(OpenCADModule()));
    connect(ui->actionMesh, SIGNAL(triggered()), this, SLOT(OpenMeshModule()));
    connect(ui->actionSolver, SIGNAL(triggered()), this, SLOT(OpenSolverModule()));
    connect(ui->actionVisual, SIGNAL(triggered()), this, SLOT(OpenVisualModule()));
    // mesh
    connect(mesh_dock->ui->pushButton, SIGNAL(clicked(bool)), this, SLOT(MeshGen()));
    //connect(mesh_dock->ui->pushButton_2, SIGNAL(clicked(bool)), this, SLOT(ImportAMSlices()));
    //connect(mesh_dock->ui->pushButton_3, SIGNAL(clicked(bool)), this, SLOT(AMSlicesToMesh()));
    //connect(mesh_dock->ui->pushButton_3, SIGNAL(clicked(bool)), this, SLOT(AMStlModelShow()));
    //connect(mesh_dock->ui->pushButton_4, SIGNAL(clicked(bool)), this, SLOT(AMSlicesShow()));
    //connect(mesh_dock->ui->pushButton_6, SIGNAL(clicked(bool)), this, SLOT(AMReset()));
    // open and close
    connect(ui->actionSave, SIGNAL(triggered()), this, SLOT(Save()));
    // show about_dialog
    about_dialog = new AboutDialog;
    connect(ui->actionAbout, SIGNAL(triggered()), this, SLOT(OpenAboutDialog()));


    // visualzation

    connect(vtk_dock->ui->pushButton_2, SIGNAL(clicked(bool)), this, SLOT(ImportVTKFile()));
    // FEM

    connect(fem_dock->ui->pushButton, SIGNAL(clicked(bool)), this, SLOT(FEMCompute()));




    // add vtk widget
    //VTKWidget* vtkw = new VTKWidget;
    //vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    //renderer->SetBackground(.3, .3, .3); // Background color dark blue
    //vtkw->GetRenderWindow()->AddRenderer(renderer);
    //setCentralWidget(vtkw);
    //vtkw->Open();
    // database module
    //    dbwind = new DataBaseWindow;
    //    dbwind->ui->widget->setBackground(Qt::black);

    // measure module
    measure_dock = new MeasureDockWidget;
    connect(ui->actionMeasure, SIGNAL(triggered()), this, SLOT(OpenMeasureModule()));
    connect(measure_dock->ui->pushButton_10, SIGNAL(clicked(bool)), this, SLOT(MeasureGDT()));
    //connect(measure_dock->ui->pushButton_5, SIGNAL(clicked(bool)), this, SLOT(MeasureSACIA()));
    connect(measure_dock->ui->pushButton_19, SIGNAL(clicked(bool)), this, SLOT(MeasureICP()));
    connect(measure_dock->ui->horizontalScrollBar, SIGNAL(valueChanged(int)), this, SLOT(MeasureOpacity()));

    //connect(measure_dock->ui->doubleSpinBox_2, SIGNAL(valueChanged(double)), this, SLOT(CloudPointMove()));
    //connect(measure_dock->ui->comboBox_2, SIGNAL(currentIndexChanged(int)), this, SLOT(CloudPointMoveType()));
    //connect(measure_dock->ui->pushButton_5, SIGNAL(clicked(bool)), this, SLOT(CloudPointSourceReset()));
    //connect(measure_dock->ui->pushButton_6, SIGNAL(clicked(bool)), this, SLOT(CloudPointTargetReset()));
    //connect(measure_dock->ui->doubleSpinBox_3, SIGNAL(valueChanged(double)), this, SLOT(SetTranSpinBoxStep()));
    //connect(measure_dock->ui->pushButton_7, SIGNAL(clicked(bool)), this, SLOT(ShowCloudSourceAndTarget()));

    //reg.SetDockWidget(measure_dock);
    // views



    // additive manufacturing
    additive_manufacturing_dock = new AdditiveManufacturingDockWidget;
    connect(ui->actionAdditiveManufacturing, SIGNAL(triggered()), this, SLOT(OpenAdditiveManufacturingModule()));
    connect(additive_manufacturing_dock->ui->pushButton, SIGNAL(clicked(bool)), this, SLOT(ImportAMStlModel()));
    connect(additive_manufacturing_dock->ui->pushButton_2, SIGNAL(clicked(bool)), this, SLOT(AMStlModelToSlices()));
    connect(additive_manufacturing_dock->ui->pushButton_5, SIGNAL(clicked(bool)), this, SLOT(AMSlicesToPathPlanning()));
    connect(additive_manufacturing_dock->ui->pushButton_12, SIGNAL(clicked(bool)), this, SLOT(AMPathPlanningShow()));
    connect(additive_manufacturing_dock->ui->pushButton_3, SIGNAL(clicked(bool)), this, SLOT(AMStlModelShow()));
    connect(additive_manufacturing_dock->ui->pushButton_7, SIGNAL(clicked(bool)), this, SLOT(ImportAMSlices()));
    connect(additive_manufacturing_dock->ui->pushButton_9, SIGNAL(clicked(bool)), this, SLOT(AMSlicesShow()));
    //connect(additive_manufacturing_dock->ui->pushButton_8, SIGNAL(clicked(bool)), this, SLOT(AMSlicesToMesh()));
    connect(additive_manufacturing_dock->ui->pushButton_11, SIGNAL(clicked(bool)), this, SLOT(AMMeshShow()));
    //connect(additive_manufacturing_dock->ui->pushButton_8, SIGNAL(clicked(bool)), this, SLOT(AMVoxelMeshGeneration()));
    //connect(additive_manufacturing_dock->ui->pushButton_5, SIGNAL(clicked(bool)), this, SLOT(AMPathPlanning()));

    spc_dock = new SPCDockWidget;
    connect(ui->actionSPC, SIGNAL(triggered()), this, SLOT(OpenSPCModule()));







    // *******************************************************
    // measure
    meas_parts = new Primitives;
    meas_bnds =  new Boundaries;
    meas_th1 = new MeasureThread1;
    meas_th2 = new MeasureThread2;
    meas_th3 = new MeasureThread3;
    connect(measure_dock->ui->pushButton_3, SIGNAL(clicked(bool)), this, SLOT(MeasureOpenCAD()));
    //connect(measure_dock->ui->pushButton, SIGNAL(clicked(bool)), this, SLOT(MeasurePresBnd()));
    //connect(measure_dock->ui->pushButton_10, SIGNAL(clicked(bool)), this, SLOT(MeasurePresLine()));
    connect(measure_dock->ui->pushButton_4, SIGNAL(clicked(bool)), this, SLOT(MeasureSelectedBndToPointCloud()));
    connect(measure_dock->ui->pushButton_8, SIGNAL(clicked(bool)), this, SLOT(MeasureOpenPointCloud()));
    //connect(measure_dock->ui->pushButton_6, SIGNAL(clicked(bool)), this, SLOT(MeasureCADHide()));
    connect(measure_dock->ui->pushButton_5, SIGNAL(clicked(bool)), this, SLOT(MeasureCADHide2()));
    connect(measure_dock->ui->pushButton_7, SIGNAL(clicked(bool)), this, SLOT(MeasureCloudTargetHide()));
    connect(measure_dock->ui->pushButton_9, SIGNAL(clicked(bool)), this, SLOT(MeasureCloudSourceHide()));
    connect(measure_dock->ui->pushButton_11, SIGNAL(clicked(bool)), this, SLOT(MeasureCloudICPHide()));
    connect(measure_dock->ui->pushButton_13, SIGNAL(clicked(bool)), this, SLOT(MeasureCloudGDTHide()));

    //connect(measure_dock->ui->pushButton_6, SIGNAL(clicked(bool)), this, SLOT(MeasureDeleteBnd()));
    //connect(measure_dock->ui->pushButton_5, SIGNAL(clicked(bool)), this, SLOT(MeasurePointCloudReset()));
    //    connect(measure_dock->ui->doubleSpinBox_4, SIGNAL(valueChanged(double)), this, SLOT(MeasurePointCloudMove()));
    //    connect(measure_dock->ui->doubleSpinBox_5, SIGNAL(valueChanged(double)), this, SLOT(MeasurePointCloudMove()));
    //    connect(measure_dock->ui->doubleSpinBox_6, SIGNAL(valueChanged(double)), this, SLOT(MeasurePointCloudMove()));
    //    connect(measure_dock->ui->doubleSpinBox_7, SIGNAL(valueChanged(double)), this, SLOT(MeasurePointCloudMove()));
    //    connect(measure_dock->ui->doubleSpinBox_8, SIGNAL(valueChanged(double)), this, SLOT(MeasurePointCloudMove()));
    //    connect(measure_dock->ui->doubleSpinBox_9, SIGNAL(valueChanged(double)), this, SLOT(MeasurePointCloudMove()));
    //connect(measure_dock->ui->comboBox_2, SIGNAL(currentIndexChanged(int)), this, SLOT(MeasurePCMove()));
    //connect(measure_dock->ui->doubleSpinBox_4, SIGNAL(valueChanged(double)), this, SLOT(MeasurePCMoveValue()));
    meas_tran[0] = 0;
    meas_tran[1] = 0;
    meas_tran[2] = 0;
    meas_tran[3] = 0;
    meas_tran[4] = 0;
    meas_tran[5] = 0;

    //td2->measure_error = measure_dock->ui->textEdit;


    // *******************************************************
    // additive manufacturing
    am_parts = new Primitives;
    am_bnds =  new Boundaries;
    connect(additive_manufacturing_dock->ui->pushButton_4, SIGNAL(clicked(bool)), this, SLOT(AMOpenCAD()));
    connect(additive_manufacturing_dock->ui->pushButton_10, SIGNAL(clicked(bool)), this, SLOT(AMSetCADVisible()));
    connect(additive_manufacturing_dock->ui->pushButton_14, SIGNAL(clicked(bool)), this, SLOT(AMCAD2STL()));
    connect(additive_manufacturing_dock->ui->pushButton_17, SIGNAL(clicked(bool)), this, SLOT(AMSetSTLVisible()));
    connect(additive_manufacturing_dock->ui->pushButton_18, SIGNAL(clicked(bool)), this, SLOT(AMSTL2Slices()));
    connect(additive_manufacturing_dock->ui->pushButton_19, SIGNAL(clicked(bool)), this, SLOT(AMSetSlicesVisible()));
    connect(additive_manufacturing_dock->ui->pushButton_20, SIGNAL(clicked(bool)), this, SLOT(AMSlices2PathPlanning()));
    connect(additive_manufacturing_dock->ui->pushButton_21, SIGNAL(clicked(bool)), this, SLOT(AMSetPathPlanningVisible()));
    connect(additive_manufacturing_dock->ui->pushButton_22, SIGNAL(clicked(bool)), this, SLOT(AMSlices2Mesh()));
    connect(additive_manufacturing_dock->ui->pushButton_23, SIGNAL(clicked(bool)), this, SLOT(AMSetMeshVisible()));
    connect(additive_manufacturing_dock->ui->pushButton_26, SIGNAL(clicked(bool)), this, SLOT(AMSimulation()));
    connect(additive_manufacturing_dock->ui->pushButton_28, SIGNAL(clicked(bool)), this, SLOT(AMSimulationAnimation()));




    // *******************************************************
    // fem
    connect(fem_dock->ui->pushButton_2, SIGNAL(clicked()), this, SLOT(FEMExampleCompute()));



    // *******************************************************
    // machining
    machining_dock = new MachiningDockWidget;
    connect(ui->actionMachining,SIGNAL(triggered()), this, SLOT(OpenMachiningModule()));
    connect(machining_dock->ui->comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(MachiningPartSet()));
    connect(machining_dock->ui->comboBox_2, SIGNAL(currentIndexChanged(int)), this, SLOT(MachiningToolSet()));
    connect(machining_dock->ui->lineEdit, SIGNAL(textChanged(QString)), this, SLOT(MachiningPartParametersUpdate()));
    connect(machining_dock->ui->lineEdit_2, SIGNAL(textChanged(QString)), this, SLOT(MachiningToolParametersUpdate()));
    connect(machining_dock->ui->pushButton,SIGNAL(clicked()), this, SLOT(MachiningMakePart()));
    connect(machining_dock->ui->pushButton_2,SIGNAL(clicked()), this, SLOT(MachiningMakeTool()));
    connect(machining_dock->ui->pushButton_11,SIGNAL(clicked()), this, SLOT(MachiningSetDomainData()));
    connect(machining_dock->ui->pushButton_10,SIGNAL(clicked()), this, SLOT(MachiningSetBoundaryData()));
    connect(machining_dock->ui->pushButton_5,SIGNAL(clicked()), this, SLOT(MachiningMeshGeneration()));
    connect(machining_dock->ui->pushButton_9,SIGNAL(clicked()), this, SLOT(MachiningSimulation()));
    connect(machining_dock->ui->pushButton_12,SIGNAL(clicked()), this, SLOT(MachiningRecompute()));

    machining_bnds =  new Boundaries;









    // *******************************************************
    // transport
    transport_parts = new Primitives;
    transport_dock = new TransportDockWidget;
    connect(ui->actionTransport, SIGNAL(triggered()), this, SLOT(OpenTransportModule()));
    connect(transport_dock->ui->pushButton_2,SIGNAL(clicked()), this, SLOT(TransportCADSave()));
    connect(transport_dock->ui->pushButton_3,SIGNAL(clicked()), this, SLOT(TransportParaModel()));
    connect(transport_dock->ui->pushButton_6,SIGNAL(clicked()), this, SLOT(TransportSelect()));
    connect(transport_dock->ui->pushButton_4,SIGNAL(clicked()), this, SLOT(TransportMCRun()));

    // *******************************************************
    // *******************************************************
    // cae poro
    ocporo_dock = new OCPoroDockWidget;
    connect(ui->actionOCPoro, SIGNAL(triggered()), this, SLOT(OpenOCPoroModule()));
    connect(ocporo_dock->ui->pushButton, SIGNAL(clicked(bool)), this,
            SLOT(OCPoroImportVTKFile()));
    connect(ocporo_dock->ui->comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OCPoroSwitchAtt()));
    connect(ocporo_dock->ui->pushButton_3, SIGNAL(clicked(bool)), this,
            SLOT(OCPoroImportSummary()));
    ocporosummary = new OCPoroDialog;
    ocporosummary1 = new OCPoroDialog;
    ocporosummary2 = new OCPoroDialog;









    return;
}

MainWindow::~MainWindow()
{
    delete ui;
}












// ##############################################################################################
// ##############################################################################################
//
// version 1.0 updated by Jiping Xin
//
// ##############################################################################################
// ##############################################################################################



void MainWindow::SetActionChecked (int n) {
    ui->actionCAD->setChecked(false);
    ui->actionPhysics->setChecked(false);
    ui->actionMesh->setChecked(false);
    ui->actionSolver->setChecked(false);
    ui->actionVisual->setChecked(false);
    if (n == 0)
        ui->actionCAD->setChecked(true);
    else if (n == 1)
        ui->actionPhysics->setChecked(true);
    else if (n == 2)
        ui->actionMesh->setChecked(true);
    else if (n == 3)
        ui->actionSolver->setChecked(true);
    else if (n == 4)
        ui->actionVisual->setChecked(true);
}

void MainWindow::OpenCADModule()
{
    if (ui->actionCAD->isChecked())
    {
        SetActionChecked(0);
        // vtk setting
        vtk_widget->SetSelectable(true);
        vtk_widget->SetSelectDomain(true);
        //vtk_widget->Reset();
        // dock setting
        ui->dockWidget->setWidget(cad_dock);
        ui->dockWidget->show();
    }
    else
    {
        ui->dockWidget->hide();
    }
}

void MainWindow::OpenPhysicsModule()
{
    if (ui->actionPhysics->isChecked())
    {
        SetActionChecked(1);
        // vtk setting
        // dock setting
        //vtk_widget->SetSelectable(true);
        //vtk_widget->SetSelectDomain(true);
        //vtk_widget->Reset();
        // dock setting
        ui->dockWidget->setWidget(physics_dock);
        ui->dockWidget->show();
    }
    else
    {
        ui->dockWidget->hide();
    }
}

void MainWindow::OpenMeshModule()
{
    if (ui->actionMesh->isChecked())
    {
        SetActionChecked(2);
        // vtk setting
        vtk_widget->SetSelectable(false);
        // dock setting
        ui->dockWidget->setWidget(mesh_dock);
        ui->dockWidget->show();
    }
    else
    {
        ui->dockWidget->hide();
    }
}

void MainWindow::OpenSolverModule()
{
    if (ui->actionSolver->isChecked())
    {
        SetActionChecked(3);
        // vtk setting
        vtk_widget->SetSelectable(false);
        // dock setting
        ui->dockWidget->setWidget(fem_dock);
        ui->dockWidget->show();
    }
    else
    {
        ui->dockWidget->hide();
    }
}

void MainWindow::OpenVisualModule()
{
    if (ui->actionVisual->isChecked())
    {
        SetActionChecked(4);
        // vtk setting
        // dock setting
        ui->dockWidget->setWidget(vtk_dock);
        ui->dockWidget->show();
    }
    else
    {
        ui->dockWidget->hide();
    }
}

void MainWindow::NewProject ()
{
    parts->Clear();
    bnds->Clear();
    vtk_widget->Clear();

}

void MainWindow::OpenProject ()
{
    QString fileName = QFileDialog::getOpenFileName(this,tr("Open File"),"/home/jiping/OpenDigitalTwin/",
                                                    tr("CAD Files (*.stp *.step *.vtk)")
                                                    , 0 , QFileDialog::DontUseNativeDialog);
    cout << fileName.toStdString() << endl;
    if (fileName.isNull()) return;
    if (fileName.right(3)==QString("stp")||fileName.right(4)==QString("step")) {
        // file name
        char* ch;
        QByteArray ba = fileName.toLatin1();
        ch=ba.data();
        // reader
        STEPControl_Reader reader;
        reader.ReadFile(ch);
        Standard_Integer NbRoots = reader.NbRootsForTransfer();
        Standard_Integer NbTrans = reader.TransferRoots();
        General* S = new General(new TopoDS_Shape(reader.OneShape()));
        // add to ds and plot
        if (S != NULL)
        {
            vtk_widget->Plot(*(S->Value()));
            parts->Add(S);
        }
    }
    else if (fileName.right(3)==QString("vtk"))
    {
        vtk_widget->ImportVTKFile(fileName.toStdString());
    }
}





// ##############################################################################################
// ##############################################################################################
// ##############################################################################################









































// =======================================================================
//
// view operations
//
// =======================================================================
void MainWindow::Fit ()
{
    //OCCw->Fit();
    vtk_widget->Fit();
}
void MainWindow::Front ()
{
    //OCCw->Front();
    vtk_widget->Front();
}
void MainWindow::Back ()
{
    //OCCw->Back();
    vtk_widget->Back();
}
void MainWindow::Left ()
{
    //OCCw->Left();
    vtk_widget->Left();
}
void MainWindow::Right ()
{
    //OCCw->Right();
    vtk_widget->Right();
}
void MainWindow::Top ()
{
    //OCCw->Top();
    vtk_widget->Top();
}
void MainWindow::Bottom ()
{
    //OCCw->Bottom();
    vtk_widget->Bottom();
}
void MainWindow::Axo ()
{
    //OCCw->Axo();
    vtk_widget->Axo();
}
void MainWindow::ViewRotationH()
{
    vtk_widget->ViewRotationH();
}
void MainWindow::SetViewRotationH()
{
    if (ui->actionViewRotationH->isChecked())
    {
        ui->actionViewRotationV->setChecked(false);
        disconnect(timer, SIGNAL(timeout()), this, SLOT(ViewRotationV()));
        timer->stop();
        connect(timer, SIGNAL(timeout()), this, SLOT(ViewRotationH()));
        timer->start(10);
    }
    else
    {
        timer->stop();
    }
}
void MainWindow::ViewRotationV()
{
    vtk_widget->ViewRotationV();
}
void MainWindow::SetViewRotationV()
{
    if (ui->actionViewRotationV->isChecked())
    {
        ui->actionViewRotationH->setChecked(false);
        disconnect(timer, SIGNAL(timeout()), this, SLOT(ViewRotationH()));
        timer->stop();
        connect(timer, SIGNAL(timeout()), this, SLOT(ViewRotationV()));
        timer->start(10);
    }
    else
    {
        timer->stop();
    }
}
// =======================================================================
//
// plot
//
// =======================================================================
void MainWindow::Plot (TopoDS_Shape* S)
{
    // OCCw->Plot(S);
}
void MainWindow::Plot (const TopoDS_Shape& S)
{
    // OCCw->Plot(S);
}
// =======================================================================
//
// CAD operations
// for data structure, add, remove, change, select, ... operations
// first we give "add"
//
// =======================================================================
void MainWindow::MakeVertex(double x1, double x2, double x3)
{
    TopoDS_Shape* S = new TopoDS_Shape(BRepBuilderAPI_MakeVertex(gp_Pnt(x1,x2,x3)).Shape());
    Vertex* A = new Vertex(S);
    A->SetData(x1,x2,x3);
    Plot(A->Value());
    vtk_widget->Plot(*(A->Value()));
    parts->Add(A);
    cout << "geometry num: " << parts->size() << endl;
}
void MainWindow::MakeLine(double x11, double x12, double x13,
                          double x21, double x22, double x23)
{
    TopoDS_Shape* S = new TopoDS_Shape(BRepBuilderAPI_MakeEdge(gp_Pnt(x11,x12,x13),gp_Pnt(x21,x22,x23)));
    Line* A = new Line(S);
    A->SetData(x11,x12,x13,x21,x22,x23);
    Plot(A->Value());
    vtk_widget->Plot(*(A->Value()));
    parts->Add(A);
    cout << "geometry num: "  << parts->size() << endl;
}
void MainWindow::MakePlane(double p1, double p2, double p3,
                           double d1, double d2, double d3,
                           double x1, double x2,
                           double y1, double y2)
{

    TopoDS_Shape* S = new TopoDS_Shape(BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(p1,p2,p3),gp_Dir(d1,d2,d3)),x1,x2,y1,y2).Shape());
    Plane* A = new Plane(S);
    A->SetData(p1,p2,p3,
               d1,d2,d3,
               x1,x2,
               y1,y2);
    Plot(A->Value());
    vtk_widget->Plot(*(A->Value()));
    parts->Add(A);
    cout << "geometry num: " << parts->size() << endl;
}
void MainWindow::MakeBox(double x1, double x2, double x3,
                         double p1, double p2, double p3,
                         double d1, double d2, double d3)
{
    TopoDS_Shape* S = new TopoDS_Shape(BRepPrimAPI_MakeBox(gp_Ax2(gp_Pnt(p1,p2,p3),gp_Dir(d1,d2,d3)),x1,x2,x3).Shape());
    Cube* A = new Cube(S);
    A->SetData(x1,x2,x3,
               p1,p2,p3,
               d1,d2,d3);
    //Plot(A->Value());
    parts->Add(A);
    vtk_widget->Plot(*(A->Value()));
    cout << "geometry num: "  << parts->size() << endl;
}
#include "BRepPrimAPI_MakeRevol.hxx"
#include "gp_Cylinder.hxx"
void MainWindow::MakeSphere (double r,
                             double p1, double p2, double p3,
                             double d1, double d2, double d3)
{
    TopoDS_Shape* S = new TopoDS_Shape(BRepPrimAPI_MakeSphere(gp_Ax2(gp_Pnt(p1,p2,p3),gp_Dir(d1,d2,d3)),r).Shape());
    Sphere* A = new Sphere(S);
    A->SetData(r,p1,p2,p3,d1,d2,d3);
    //Plot(A->Value());
    parts->Add(A);
    vtk_widget->Plot(*(A->Value()));
    cout << "geometry num: "  << parts->size() << endl;
}
#include "StlAPI_Writer.hxx"


void MainWindow::MakeCylinder (double r, double h,
                               double p1, double p2, double p3,
                               double d1, double d2, double d3)
{
    TopoDS_Shape* S = new TopoDS_Shape(BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(p1,p2,p3),gp_Dir(d1,d2,d3)),r,h).Shape());
    Cylinder* A = new Cylinder(S);
    A->SetData(r,h,p1,p2,p3,d1,d2,d3);
    Plot(A->Value());
    parts->Add(A);
    vtk_widget->Plot(*(A->Value()));
    cout << "geometry num: "  << parts->size() << endl;

    StlAPI_Writer STLwriter;
    STLwriter.Write(*S,"data/cylinder.stl");

}
void MainWindow::MakeCone (double r1, double r2, double h,
                           double p1, double p2, double p3,
                           double d1, double d2, double d3)
{
    TopoDS_Shape* S = new TopoDS_Shape(BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(p1,p2,p3),gp_Dir(d1,d2,d3)),r1,r2,h).Shape());
    Cone* A = new Cone(S);
    A->SetData(r1,r2,h,p1,p2,p3,d1,d2,d3);
    Plot(A->Value());
    parts->Add(A);
    vtk_widget->Plot(*(A->Value()));
    cout << "geometry num: "  << parts->size() << endl;
}
void MainWindow::MakeTorus (double r1, double r2,
                            double p1, double p2, double p3,
                            double d1, double d2, double d3)
{
    TopoDS_Shape* S = new TopoDS_Shape(BRepPrimAPI_MakeTorus(gp_Ax2(gp_Pnt(p1,p2,p3),gp_Dir(d1,d2,d3)),r1,r2).Shape());
    Torus* A = new Torus(S);
    A->SetData(r1,r2,p1,p2,p3,d1,d2,d3);
    Plot(A->Value());
    parts->Add(A);
    vtk_widget->Plot(*(A->Value()));
    cout << "geometry num: "  << parts->size() << endl;
}
// =======================================================================
// remove an object from view and data structure
// =======================================================================
void MainWindow::Remove()
{
    //OCCw->Remove();
    vtk_widget->Remove();
    cad_dock->ClearPrimitives();
    //Reset();
}
// =======================================================================
// change
// create a new object from new parameters and
// remove an old object from view and data strcture
// =======================================================================
#include "BRepBuilderAPI_Transform.hxx"
void MainWindow::Change()
{
    if (cad_dock->PrimitiveType() == PrimitiveTYPE::POINT)
    {
        // create a new prim
        // position
        double p[3];
        p[0] = cad_dock->PointPosition()[0];
        p[1] = cad_dock->PointPosition()[1];
        p[2] = cad_dock->PointPosition()[2];
        MakeVertex(p[0],p[1],p[2]);
        vtk_widget->Remove();
        cad_dock->ClearPrimitives();
        cout << "geometry num: " << parts->size() << endl;
    }
    else if (cad_dock->PrimitiveType() == PrimitiveTYPE::LINE)
    {
        // create a new prim
        double p1[3];
        double p2[3];
        p1[0] = cad_dock->LineP1()[0];
        p1[1] = cad_dock->LineP1()[1];
        p1[2] = cad_dock->LineP1()[2];
        p2[0] = cad_dock->LineP2()[0];
        p2[1] = cad_dock->LineP2()[1];
        p2[2] = cad_dock->LineP2()[2];
        // make a new line
        MakeLine(p1[0],p1[1],p1[2],p2[0],p2[1],p2[2]);
        // remove an old object
        vtk_widget->Remove();
        cad_dock->ClearPrimitives();
        cout << "geometry num: " << parts->size() << endl;
    }
    else if (cad_dock->PrimitiveType() == PrimitiveTYPE::PLANE)
    {
        // create a new prim
        double p[3];
        double d[3];
        double x[2];
        double y[2];
        p[0] = cad_dock->PlanePos()[0];
        p[1] = cad_dock->PlanePos()[1];
        p[2] = cad_dock->PlanePos()[2];
        d[0] = cad_dock->PlaneDir()[0];
        d[1] = cad_dock->PlaneDir()[1];
        d[2] = cad_dock->PlaneDir()[2];
        x[0] = cad_dock->PlaneX()[0];
        x[1] = cad_dock->PlaneX()[1];
        y[0] = cad_dock->PlaneY()[0];
        y[1] = cad_dock->PlaneY()[1];
        // make a new vertex
        MakePlane(p[0],p[1],p[2],d[0],d[1],d[2],x[0],x[1],y[0],y[1]);
        // remove an old object
        vtk_widget->Remove();
        cad_dock->ClearPrimitives();
        cout << "geometry num: " << parts->size() << endl;
    }
    else if (cad_dock->PrimitiveType() == PrimitiveTYPE::BOX)
    {
        // create a new prim
        // length, width, height
        double s[3];
        s[0] = cad_dock->BoxSize()[0];
        s[1] = cad_dock->BoxSize()[1];
        s[2] = cad_dock->BoxSize()[2];
        // position
        double p[3];
        p[0] = cad_dock->BoxPos()[0];
        p[1] = cad_dock->BoxPos()[1];
        p[2] = cad_dock->BoxPos()[2];
        // direction
        double d[3];
        d[0] = cad_dock->BoxDir()[0];
        d[1] = cad_dock->BoxDir()[1];
        d[2] = cad_dock->BoxDir()[2];
        MakeBox(s[0],s[1],s[2],
                p[0],p[1],p[2],
                d[0],d[1],d[2]);
        vtk_widget->Remove();
        cad_dock->ClearPrimitives();
        cout  << "geometry num: " << parts->size() << endl;
    }
    else if (cad_dock->PrimitiveType() == PrimitiveTYPE::BALL)
    {
        // create a new prim
        // length, width, height
        double r;
        r = cad_dock->BallRadius();
        // position
        double p[3];
        p[0] = cad_dock->BallPos()[0];
        p[1] = cad_dock->BallPos()[1];
        p[2] = cad_dock->BallPos()[2];
        // direction
        double d[3];
        d[0] = cad_dock->BallDir()[0];
        d[1] = cad_dock->BallDir()[1];
        d[2] = cad_dock->BallDir()[2];
        MakeSphere(r, p[0], p[1], p[2], d[0], d[1], d[2]);
        vtk_widget->Remove();
        cad_dock->ClearPrimitives();
        cout  << "geometry num: " << parts->size() << endl;
    }
    else if (cad_dock->PrimitiveType() == PrimitiveTYPE::CYLINDER)
    {
        // create a new prim
        // length, width, height
        double r, h;
        r = cad_dock->CylinderRadius();
        h = cad_dock->CylinderHeight();
        // position
        double p[3];
        p[0] = cad_dock->CylinderPos()[0];
        p[1] = cad_dock->CylinderPos()[1];
        p[2] = cad_dock->CylinderPos()[2];
        // direction
        double d[3];
        d[0] = cad_dock->CylinderDir()[0];
        d[1] = cad_dock->CylinderDir()[1];
        d[2] = cad_dock->CylinderDir()[2];
        MakeCylinder(r, h, p[0], p[1], p[2], d[0], d[1], d[2]);
        vtk_widget->Remove();
        cad_dock->ClearPrimitives();
        cout  << "geometry num: " << parts->size() << endl;
    }
    else if (cad_dock->PrimitiveType() == PrimitiveTYPE::CONE)
    {
        // create a new prim
        // length, width, height
        double r1, r2, h;
        r1 = cad_dock->ConeRadius1();
        r2 = cad_dock->ConeRadius2();
        h = cad_dock->ConeHeight();
        // position
        double p[3];
        p[0] = cad_dock->ConePos()[0];
        p[1] = cad_dock->ConePos()[1];
        p[2] = cad_dock->ConePos()[2];
        // direction
        double d[3];
        d[0] = cad_dock->ConeDir()[0];
        d[1] = cad_dock->ConeDir()[1];
        d[2] = cad_dock->ConeDir()[2];
        MakeCone(r1, r2, h, p[0], p[1], p[2], d[0], d[1], d[2]);
        vtk_widget->Remove();
        cad_dock->ClearPrimitives();
        cout  << "geometry num: " << parts->size() << endl;
    }
    else if (cad_dock->PrimitiveType() == PrimitiveTYPE::TORUS)
    {
        // create a new prim
        // length, width, height
        double r1, r2;
        r1 = cad_dock->TorusRadius1();
        r2 = cad_dock->TorusRadius2();
        // position
        double p[3];
        p[0] = cad_dock->TorusPos()[0];
        p[1] = cad_dock->TorusPos()[1];
        p[2] = cad_dock->TorusPos()[2];
        // direction
        double d[3];
        d[0] = cad_dock->TorusDir()[0];
        d[1] = cad_dock->TorusDir()[1];
        d[2] = cad_dock->TorusDir()[2];
        MakeTorus(r1, r2, p[0], p[1], p[2], d[0], d[1], d[2]);
        vtk_widget->Remove();
        cad_dock->ClearPrimitives();
        cout  << "geometry num: " << parts->size() << endl;
    }
}
// =======================================================================
// boolean operations
// =======================================================================
void MainWindow::SetBooleanPart1()
{
    Obj1 = vtk_widget->GetSelectedActor();
    if (Obj1 != NULL)
    {
        //cad_dock->ui->pushButton_11->setChecked(true);
        cad_dock->ui->actionPart1->setChecked(true);
        Prim1 = vtk_widget->GetSelectedPrim();
    }
    else
    {
        //cad_dock->ui->pushButton_11->setChecked(false);
        cad_dock->ui->actionPart1->setChecked(false);
    }
}
void MainWindow::SetBooleanPart2()
{
    Obj2 = vtk_widget->GetSelectedActor();
    if (Obj2 != NULL)
    {
        //cad_dock->ui->pushButton_12->setChecked(true);
        cad_dock->ui->actionPart2->setChecked(true);
        Prim2 = vtk_widget->GetSelectedPrim();
    }
    else
    {
        //cad_dock->ui->pushButton_12->setChecked(false);
        cad_dock->ui->actionPart2->setChecked(false);
    }
}

#include "BRepAlgoAPI_Fuse.hxx"
#include "BRepAlgoAPI_Cut.hxx"
#include "BRepAlgoAPI_Section.hxx"
#include "BRepAlgoAPI_Common.hxx"
void MainWindow::BooleanUnion()
{
    if (Obj1 != NULL && Obj2 != NULL)
    {
        TopoDS_Shape* S = new TopoDS_Shape(BRepAlgoAPI_Fuse(*(Prim1->Value()),
                                                            *(Prim2->Value())
                                                            ).Shape());
        General* A = new General(S);
        parts->Add(A);
        vtk_widget->Plot(*(A->Value()),false);
        vtk_widget->Remove(Obj1);
        vtk_widget->Remove(Obj2);
        cad_dock->ui->actionPart1->setChecked(false);
        cad_dock->ui->actionPart2->setChecked(false);
    }
    cout  << "geometry num: " << parts->size() << endl;
}
void MainWindow::BooleanSection()
{
    if (Obj1 != NULL && Obj2 != NULL)
    {
        TopoDS_Shape* S = new TopoDS_Shape(BRepAlgoAPI_Common(*(Prim1->Value()),
                                                              *(Prim2->Value())
                                                              ).Shape());
        General* A = new General(S);
        parts->Add(A);
        vtk_widget->Plot(*(A->Value()),false);
        vtk_widget->Remove(Obj1);
        vtk_widget->Remove(Obj2);
        cad_dock->ui->actionPart1->setChecked(false);
        cad_dock->ui->actionPart2->setChecked(false);
    }
    cout  << "geometry num: " << parts->size() << endl;
}
void MainWindow::BooleanCut()
{
    if (Obj1 != NULL && Obj2 != NULL)
    {
        TopoDS_Shape* S = new TopoDS_Shape(BRepAlgoAPI_Cut(*(Prim1->Value()),
                                                           *(Prim2->Value())
                                                           ).Shape());
        General* A = new General(S);
        parts->Add(A);
        vtk_widget->Plot(*(A->Value()),false);
        vtk_widget->Remove(Obj1);
        vtk_widget->Remove(Obj2);
        cad_dock->ui->actionPart1->setChecked(false);
        cad_dock->ui->actionPart2->setChecked(false);
    }
    cout  << "geometry num: " << parts->size() << endl;
}
// =======================================================================
// more operations
// =======================================================================
#include "BRepPrimAPI_MakePrism.hxx"
void MainWindow::CommonOperations()
{
    if (cad_dock->OperationType() == OperationTYPE::SWEEP)
    {
        if (vtk_widget->GetSelectedActor() != NULL)
        {
            if (vtk_widget->GetSelectedPrim()->Value()->ShapeType() == TopAbs_VERTEX
                    || vtk_widget->GetSelectedPrim()->Value()->ShapeType() == TopAbs_EDGE ||
                    vtk_widget->GetSelectedPrim()->Value()->ShapeType() == TopAbs_FACE)
            {
                // position
                double p[3];
                p[0] = cad_dock->Pos()[0];
                p[1] = cad_dock->Pos()[1];
                p[2] = cad_dock->Pos()[2];
                // direction
                double d[3];
                d[0] = cad_dock->Dir_1()[0];
                d[1] = cad_dock->Dir_1()[1];
                d[2] = cad_dock->Dir_1()[2];
                // angle
                double angle;
                angle = cad_dock->Angle();
                TopoDS_Shape* S = new TopoDS_Shape(BRepPrimAPI_MakeRevol(*(vtk_widget->GetSelectedPrim()->Value()),
                                                                         gp_Ax1(gp_Pnt(p[0],p[1],p[2]),gp_Dir(d[0],d[1],d[2])),
                        angle/360*2*3.1415926).Shape());
                General* A = new General(S);
                // cout << "shape type: " << A->Value().ShapeType() << endl;
                vtk_widget->Remove();
                parts->Add(A);
                vtk_widget->Plot(*(A->Value()));
                cout  << "geometry num: " << parts->size() << endl;
                cad_dock->SetSweepModule();
            }
        }
    }
    else if (cad_dock->OperationType() == OperationTYPE::EXTRUDE)
    {
        if (vtk_widget->GetSelectedActor() != NULL)
        {
            if (vtk_widget->GetSelectedPrim()->Value()->ShapeType() == TopAbs_VERTEX
                    || vtk_widget->GetSelectedPrim()->Value()->ShapeType() == TopAbs_EDGE ||
                    vtk_widget->GetSelectedPrim()->Value()->ShapeType() == TopAbs_FACE)
            {
                double d[3];
                d[0] = cad_dock->Dir_1()[0];
                d[1] = cad_dock->Dir_1()[1];
                d[2] = cad_dock->Dir_1()[2];
                TopoDS_Shape* S = new TopoDS_Shape(BRepPrimAPI_MakePrism(*(vtk_widget->GetSelectedPrim()->Value()),
                                                                         gp_Vec(gp_XYZ(d[0],d[1],d[2]))));
                General* A= new General(S);
                // cout << "shape type: " << A->Value().ShapeType() << endl;
                vtk_widget->Remove();
                parts->Add(A);
                vtk_widget->Plot(*(A->Value()));
                cout  << "geometry num: " << parts->size() << endl;
                cad_dock->SetSweepModule();
            }
        }
    }
    else if (cad_dock->OperationType() == OperationTYPE::MIRROR)
    {
        if (vtk_widget->GetSelectedActor() != NULL)
        {
            double p[3];
            double d1[3];
            double d2[3];
            p[0] = cad_dock->Pos()[0];
            p[1] = cad_dock->Pos()[1];
            p[2] = cad_dock->Pos()[2];
            d1[0] = cad_dock->Dir_1()[0];
            d1[1] = cad_dock->Dir_1()[1];
            d1[2] = cad_dock->Dir_1()[2];
            d2[0] = cad_dock->Dir_2()[0];
            d2[1] = cad_dock->Dir_2()[1];
            d2[2] = cad_dock->Dir_2()[2];
            cout << p[0] << " " << p[1] << " " << p[2] << endl;
            cout << d1[0] << " " << d1[1] << " " << d1[2] << endl;
            cout << d2[0] << " " << d2[1] << " " << d2[2] << endl;
            gp_Trsf t;
            t.SetMirror(gp_Ax2(gp_Pnt(p[0],p[1],p[2]),gp_Dir(d1[0],d1[1],d1[2]),gp_Dir(d2[0],d2[1],d2[2])));
            TopoDS_Shape* S = new TopoDS_Shape(BRepBuilderAPI_Transform(*(vtk_widget->GetSelectedPrim()->Value()),t));
            // create a new mirrored object
            General* A = new General(S);
            // dont remove the old one
            parts->Add(A);
            vtk_widget->Plot(*(A->Value()));
            cout  << "geometry num: " << parts->size() << endl;
            cad_dock->SetSweepModule();
        }
    }
}
void MainWindow::SelectBnd()
{
    vtk_widget->PlotBnds();
    vtk_widget->SetSelectBnd(true);
    vtk_widget->SetSelectDomain(false);
    cad_dock->ui->actionSelectBnd->setChecked(true);
    cad_dock->ui->actionSelectDomain->setChecked(false);
    //cad_dock->ui->tabWidget->setCurrentIndex(2);
    //cad_dock->ui->tabWidget->setTabIcon(2,QIcon(":/cad_wind/figure/cad_wind/selection_face.png"));
    //cad_dock->ui->pushButton_5->setIcon(QIcon(":/cad_wind/figure/cad_wind/selection_face.png"));
}

void MainWindow::SelectDomain()
{
    vtk_widget->PlotDomains();
    vtk_widget->SetSelectDomain(true);
    vtk_widget->SetSelectBnd(false);
    cad_dock->ui->actionSelectBnd->setChecked(false);
    cad_dock->ui->actionSelectDomain->setChecked(true);
    //cad_dock->ui->tabWidget->setTabIcon(2,QIcon(":/cad_wind/figure/cad_wind/selection_domain.png"));
    //cad_dock->ui->pushButton_5->setIcon(QIcon(":/cad_wind/figure/cad_wind/selection_domain.png"));
}
// =======================================================================
//
// open and save
//
// =======================================================================

void MainWindow::Save ()
{
    QString fileName = QFileDialog::getSaveFileName(this,tr("Save File"),
                                                    "./output.stp",
                                                    tr("CAD Files (*.stp *.stl)"), 0 , QFileDialog::DontUseNativeDialog);
    QFileInfo fileinfo(fileName);
    QString file_suffix = fileinfo.suffix();
    std::cout << file_suffix.toStdString() << std::endl;


    TopoDS_Compound aRes;
    BRep_Builder aBuilder;
    aBuilder.MakeCompound (aRes);
    //for (std::list<TopoDS_Shape*>::iterator it = myCompound.begin(); it!=myCompound.end(); it++) {
    //for (std::list<Prim*>::iterator it = myCompound.begin(); it!=myCompound.end(); it++) {
    for (int i=0; i<parts->size(); i++)
    {
        aBuilder.Add(aRes,*((*parts)[i]->Value()));
    }
    if (file_suffix == "stp")
    {
        STEPControl_Writer writer;
        writer.Transfer(aRes,STEPControl_ManifoldSolidBrep);
        char* ch;
        QByteArray ba = fileName.toLatin1();
        ch=ba.data();
        writer.Write(ch);
    }
    else
    {
        StlAPI_Writer writer;
        writer.Write(aRes,fileName.toLatin1().data());
    }
}

// =======================================================================
//
// general module
//
// =======================================================================

void MainWindow::OpenMeasureModule()
{
    if (ui->actionMeasure->isChecked())
    {
        // set open and close
        ui->dockWidget->setWidget(measure_dock);
        ui->dockWidget->show();
        ui->actionCAD->setChecked(false);
        ui->actionMesh->setChecked(false);
        ui->actionSolver->setChecked(false);
        ui->actionVisual->setChecked(false);
        ui->actionMeasure->setChecked(true);
        ui->actionAdditiveManufacturing->setChecked(false);
        ui->actionMachining->setChecked(false);
    }
    else
    {
        ui->dockWidget->hide();
    }
}
void MainWindow::OpenSPCModule()
{
    if (ui->actionSPC->isChecked())
    {
        // set open and close
        ui->dockWidget->setWidget(spc_dock);
        ui->dockWidget->show();
        ui->actionCAD->setChecked(false);
        ui->actionMesh->setChecked(false);
        ui->actionSolver->setChecked(false);
        ui->actionVisual->setChecked(false);
        ui->actionMeasure->setChecked(false);
        ui->actionSPC->setChecked(true);
        ui->actionAdditiveManufacturing->setChecked(false);
        ui->actionMachining->setChecked(false);
        ui->actionSystem->setChecked(false);
    }
    else
    {
        ui->dockWidget->hide();
    }
}

#include "Mesh/MeshGeneration.h"
#include "vtkUnstructuredGridReader.h"
#include "vtkDataSetMapper.h"

void MainWindow::MeshGen()
{
    if (parts->size() == 0) return;
    std::cout << "mesh check " << parts->size() << std::endl;
    MM.MeshGeneration(parts->Union(),mesh_dock->ui->doubleSpinBox->value(),0,meas_path);
    vtk_widget->Hide();
    //vtk_widget->ImportVTKFile(std::string("/home/jiping/FENGSim/build-FENGSim-Desktop_Qt_5_12_10_GCC_64bit-Release/FENGSimDT.vtk"));
    MM.FileFormat();

    //        MM.FileFormatMeshToVTK2("/home/jiping/M++/ElastoPlasticity/conf/geo/InternallyPressurisedCylinder.mesh",
    //                                "/home/jiping/M++/ElastoPlasticity/conf/geo/InternallyPressurisedCylinder.vtk");
    vtk_widget->ImportVTKFile((meas_path+QString("/data/mesh/fengsim_mesh.vtk")).toStdString());


}

#include "Poly_Triangulation.hxx"
#include "TopExp_Explorer.hxx"
#include "BRep_TEdge.hxx"
#include "BRep_ListIteratorOfListOfCurveRepresentation.hxx"
#include "BRep_CurveRepresentation.hxx"
#include "BRep_Curve3D.hxx"
#include "Geom_Line.hxx"
#include "Geom_Plane.hxx"

void MainWindow::BoundaryChecked()
{
    //    if (machining_dock->ui->checkBox_2->isChecked())
    //    {
    //        machining_dock->ui->checkBox->setChecked(false);
    //        OCCw->SetSelectType(SelectBoundaryObj);
    //        OCCw->SetComboBox(machining_dock->ui->comboBox);
    //        OCCw->SetSpinBox(machining_dock->ui->spinBox);
    //    }
}
void MainWindow::DomainChecked()
{
    //    if (machining_dock->ui->checkBox->isChecked())
    //    {
    //        machining_dock->ui->checkBox_2->setChecked(false);
    //        OCCw->SetSelectType(SelectDomainObj);
    //    }
}
void MainWindow::ShowBoundariesOrDomains ()
{
    //    if (parts->size() == 0) return;
    //    if (machining_dock->ui->checkBox_2->isChecked())
    //    {
    //        OCCw->Clear();
    //        bndObjs = new Boundaries(*((*parts)[0]->Value()));
    //        OCCw->SetBoundarObjs(bndObjs);
    //        // here is very good
    //        // occ has already the function to load many objs and
    //        // use displayall to show
    //        for (int i = 0; i < bndObjs->Size(); i++)
    //        {
    //            //OCCw->Plot((*bndObjs)[i]->Value());
    //            OCCw->Load((*bndObjs)[i]->Value());
    //        }
    //        OCCw->DisplayAll();
    //        cout << "boundary num: " << bndObjs->Size() << endl;
    //    }
    //    else if (machining_dock->ui->checkBox->isChecked())
    //    {
    //        OCCw->Clear();
    //        OCCw->Plot((*parts)[0]->Value());
    //    }
}
void MainWindow::SetBoundaryTypeAndID()
{
    //    OCCw->SetCurrentBoundaryObj(CurrentBoundaryObj);
    //    if (CurrentBoundaryObj == NULL) return;
    //    int id = bndObjs->Include(*CurrentBoundaryObj);
    //    if (id == -1)
    //    {
    //        cout << "didn't choose a boundary obj" << endl;
    //        return;
    //    }
    //    if (machining_dock->ui->comboBox->currentText() == "Dirichlet")
    //    {
    //        (*bndObjs)[id]->SetType(Dirichlet);
    //    }
    //    else
    //    {
    //        (*bndObjs)[id]->SetType(Neumann);
    //    }
    //    (*bndObjs)[id]->SetId(machining_dock->ui->spinBox->text().toInt());
    //    //    for (int i = 0; i < bndObjs->Size(); i++)
    //    //    {
    //    //        cout << (*bndObjs)[i]->Type() << " " << (*bndObjs)[i]->Id() << endl;
    //    //    }
}
// =======================================================================
//
// about dialog
//
// =======================================================================
void MainWindow::OpenAboutDialog ()
{

    //about_dialog->ChangePicture(QString(":/main_wind/figure/main_wind/Fengsim_logo_hi.png"));
    about_dialog->show();
    //about_form->show();
}

// =======================================================================
//
// database module
//
// =======================================================================
//void MainWindow::DataBaseWindowShow()
//{
//    if (!ui->actionDataBase->isChecked())
//    {
//        dbwind->show();
//    }
//    else
//    {
//        dbwind->hide();
//    }
//}
void MainWindow::UpdateBndValue()
{
    vtk_widget->UpdateBndValue();
    bnds->ExportToFile();
}

#include "TopoDS.hxx"



void MainWindow::CloudPointMove()
{
    //reg.SetTran(measure_dock->ui->comboBox_2->currentIndex(),measure_dock->ui->doubleSpinBox_2->value());
    //reg.move();
    vtk_widget->MeasureClearCloudSource();
    vtk_widget->MeasureImportCloudSource(std::string("./data/cloud_source_move.vtk"));
}

void MainWindow::CloudPointMoveType ()
{
    //measure_dock->ui->doubleSpinBox_2->setValue(reg.GetTran(measure_dock->ui->comboBox_2->currentIndex()));
}

void MainWindow::CloudPointReset()
{
    vtk_widget->MeasureClearCloudSource();
    //vtk_widget->ClearCloudFinal();
    vtk_widget->ClearCloudFinalColor();
    std::fstream file1(std::string("./data/cloud_source.vtk"),ios::out);
}

void MainWindow::CloudPointTargetReset()
{
    vtk_widget->ClearCloudTarget();
    std::fstream file2(std::string("./data/cloud_target.vtk"),ios::out);
    //meas_selected_bnds.clear();
}

void MainWindow::CloudPointSourceReset()
{
    vtk_widget->MeasureClearCloudSource();
    std::fstream file2(std::string("./data/cloud_source.vtk"),ios::out);
}



void MainWindow::SetTranSpinBoxStep () {
    //measure_dock->ui->doubleSpinBox_2->setSingleStep(measure_dock->ui->doubleSpinBox_3->value());
}

void MainWindow::ShowCloudSourceAndTarget() {
    //if (measure_dock->ui->pushButton_7->isChecked())
    {
        vtk_widget->ShowCloudSource(true);
        vtk_widget->ShowCloudTarget(true);
    }
    //else if (!measure_dock->ui->pushButton_7->isChecked())
    {
        vtk_widget->ShowCloudSource(false);
        vtk_widget->ShowCloudTarget(false);
    }
}

void MainWindow::BoxFit() {
    //reg.box_fit();
    //reg.move();
    vtk_widget->MeasureClearCloudSource();
    vtk_widget->MeasureImportCloudSource(std::string("./data/cloud_source_move.vtk"));
}

void MainWindow::ImportAMStlModel()
{
    stl_file_name =  QFileDialog::getOpenFileName(0,"Open Stl Files",
                                                  QString("/home/jiping/OpenDT/FENGSim/FENGSim/data/"),
                                                  "Stl files (*.stl);;", 0 , QFileDialog::DontUseNativeDialog);
    //MM.ClearSlices();
    // MM.FileFormat2(cli_file_name);
    std::cout << " check " << std::endl;
    vtk_widget->Clear();
    //vtk_widget->ClearAMSlices();
    vtk_widget->ImportVTKFileAMStlModel(stl_file_name.toStdString());
    additive_manufacturing_dock->ui->pushButton_3->setChecked(true);
}

void MainWindow::AMStlModelShow()
{
    if (additive_manufacturing_dock->ui->pushButton_3->isChecked())
    {
        vtk_widget->ShowAMStlModel(true);
    }
    else if (!additive_manufacturing_dock->ui->pushButton_3->isChecked())
    {
        vtk_widget->ShowAMStlModel(false);
    }
}

void MainWindow::ImportAMSlices()
{
    cli_file_name =  QFileDialog::getOpenFileName(0,"Open Slices",
                                                  QString("/home/jiping/OpenDT/FENGSim/FENGSim/data"),
                                                  "Slice files (*.cli);;", 0 , QFileDialog::DontUseNativeDialog);
    MM.ClearSlices();
    MM.FileFormatCliToVTK(cli_file_name);
    vtk_widget->Clear();
    //vtk_widget->ClearAMSlices();
    vtk_widget->ImportVTKFileAMSlices("./Cura/data/slices.vtk");
    additive_manufacturing_dock->ui->pushButton_9->setChecked(true);
}

void MainWindow::AMStlModelToSlices()
{
    cli_file_name =  QFileDialog::getSaveFileName(0,"Open Slices",
                                                  QString("/home/jiping/OpenDT/FENGSim/FENGSim/data"),
                                                  "Slice files (*.cli);;", 0 , QFileDialog::DontUseNativeDialog);
    std::cout << "check cli file name: " << cli_file_name.toStdString() << std::endl;
    ofstream out;
    out.open("./../AM/build/solver/conf/cura.conf");
    out << "Model = SlicePhaseTest" << endl;
    out << stl_file_name.toStdString().c_str() << endl;
    out << cli_file_name.toStdString() + ".cli" << endl;
    out << cli_file_name.toStdString() + ".vtk" << endl;

    QProcess *proc = new QProcess();
    proc->setWorkingDirectory( "./../AM/build" );
    proc->start("./AMSolver");

    if (proc->waitForFinished(-1)) {
        //MM.ClearSlices();
        //MM.FileFormatCliToVTK(cli_file_name + ".cli");
        //vtk_widget->Clear();
        vtk_widget->ClearAMSlices();
        vtk_widget->ImportVTKFileAMSlices(cli_file_name.toStdString() + ".vtk");
        additive_manufacturing_dock->ui->pushButton_9->setChecked(true);
    }
}

void MainWindow::AMSlicesToPathPlanning()
{
    path_file_name =  QFileDialog::getSaveFileName(0,"Open Path Planning",
                                                   QString("/home/jiping/OpenDT/FENGSim/FENGSim/data"),
                                                   "Path files (*.vtk);;", 0 , QFileDialog::DontUseNativeDialog);
    ofstream out;
    out.open("./Cura/Cura/conf/cura.conf");
    out << "Model = InfillTest" << endl;
    out << cli_file_name.toStdString() + ".vtk" << endl;
    out << path_file_name.toStdString() << endl;

    QProcess *proc = new QProcess();
    proc->setWorkingDirectory( "./Cura" );
    proc->start("./CuraRun");

    if (proc->waitForFinished(-1)) {
        //MM.ClearSlices();
        //MM.FileFormatCliToVTK(cli_file_name + ".cli");
        //vtk_widget->Clear();
        vtk_widget->ClearAMPathPlanning();
        std::cout << path_file_name.toStdString() + "_outlines0.vtk" << std::endl;
        //vtk_widget->ImportVTKFileAMPathPlanning(path_file_name.toStdString() + "_outlines0.vtk");
        vtk_widget->ImportVTKFileAMPathPlanning(path_file_name.toStdString() + "_pathlines.vtk");
        additive_manufacturing_dock->ui->pushButton_12->setChecked(true);
    }
    proc->close();
}

void MainWindow::AMSlicesShow()
{
    if (additive_manufacturing_dock->ui->pushButton_9->isChecked())
    {
        vtk_widget->ShowAMSlices(true);
    }
    else if (!additive_manufacturing_dock->ui->pushButton_9->isChecked())
    {
        vtk_widget->ShowAMSlices(false);
    }
}


void MainWindow::AMReset()
{
    vtk_widget->ShowAMSlices(false);
    vtk_widget->ShowAMMesh(false);
    //mesh_dock->ui->pushButton_4->setChecked(false);
}



//void MainWindow::OpenCADModule()
//{
//    if (ui->actionCAD->isChecked())
//    {
//        vtk_widget->SetSelectable(true);
//        vtk_widget->SetSelectDomain(true);
//        vtk_widget->Reset();
//        // cout << parts->size() << endl;
//        // OCCw->Clear();
//        // OCCw->SetMachiningModule(false);
//        // OCCw->Fit();
//        ui->dockWidget->setWidget(cad_dock);
//        ui->dockWidget->show();
//        // set open and close
//        ui->actionCAD->setChecked(true);
//        ui->actionMesh->setChecked(false);
//        ui->actionSolver->setChecked(false);
//        ui->actionVisual->setChecked(false);
//        ui->actionMeasure->setChecked(false);
//    }
//    else
//    {
//        ui->dockWidget->hide();
//    }
//}

// additive manufacturing

void MainWindow::OpenAdditiveManufacturingModule()
{
    if (ui->actionAdditiveManufacturing->isChecked())
    {
        vtk_widget->SetSelectable(false);
        // set open and close
        ui->dockWidget->setWidget(additive_manufacturing_dock);
        ui->dockWidget->show();
        ui->actionCAD->setChecked(false);
        ui->actionMesh->setChecked(false);
        ui->actionSolver->setChecked(false);
        ui->actionVisual->setChecked(false);
        ui->actionAdditiveManufacturing->setChecked(true);
        ui->actionMeasure->setChecked(false);
        ui->actionSystem->setChecked(false);
        ui->actionMachining->setChecked(false);
    }
    else
    {
        ui->dockWidget->hide();
    }
}

void MainWindow::AMVoxelMeshGeneration()
{
    MM.VoxelMeshGeneration();
    //vtk_widget->Clear();
    vtk_widget->ClearAMMesh();
    vtk_widget->ImportVTKFileAMMesh("./Cura/data/am_voxel_mesh.vtk");
    additive_manufacturing_dock->ui->pushButton_11->setChecked(true);
}


void MainWindow::AMPathPlanning()
{
    MM.PathPlanning();
    vtk_widget->ImportVTKFileAMPathPlanning("./Cura/data/am_path_planning.vtk");
    additive_manufacturing_dock->ui->pushButton_12->setChecked(true);
}

void MainWindow::AMMeshShow()
{
    if (additive_manufacturing_dock->ui->pushButton_11->isChecked())
    {
        vtk_widget->ShowAMMesh(true);
    }
    else if (!additive_manufacturing_dock->ui->pushButton_11->isChecked())
    {
        vtk_widget->ShowAMMesh(false);
    }
}

void MainWindow::AMPathPlanningShow()
{
    if (additive_manufacturing_dock->ui->pushButton_12->isChecked())
    {
        vtk_widget->ShowAMPathPlanning(true);
    }
    else if (!additive_manufacturing_dock->ui->pushButton_12->isChecked())
    {
        vtk_widget->ShowAMPathPlanning(false);
    }
}























// *******************************************************
// measure
#include "IntTools_Context.hxx"
#include "BRep_Tool.hxx"
#include "GeomAPI_ProjectPointOnSurf.hxx"
#include "gp_Pnt.hxx"
#include "BRepClass3d.hxx"
#include "BRepClass_FaceClassifier.hxx"
#include "BRepClass3d_SolidClassifier.hxx"
#include "GeomAPI_ProjectPointOnSurf.hxx"
#include "BRep_Tool.hxx"
#include "BRepTools.hxx"
#include "Bnd_Box.hxx"
#include "BRepBndLib.hxx"
//#include "example2.h"
#include "TopoDS_Face.hxx"

void MainWindow::MeasureOpenCAD()
{
    TextOutput("Importing a CAD model. Please wait...");

    // file name
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), "/home/jiping/FENGSim/FENGSim/Measure/data/",
                                                    tr("CAD Files (*.stp *.step)"),
                                                    0 , QFileDialog::DontUseNativeDialog);
    if (fileName.isNull()) return;
    QFile::remove("./data/meas/fengsim_meas_scene.vtk");
    char* ch;
    QByteArray ba = fileName.toLatin1();
    ch=ba.data();
    // occ reader
    STEPControl_Reader reader;
    reader.ReadFile(ch);
    Standard_Integer NbRoots = reader.NbRootsForTransfer();
    Standard_Integer NbTrans = reader.TransferRoots();
    measure_cad = new General(new TopoDS_Shape(reader.OneShape()));
    if (measure_cad == 0) return;

    // reset all data
    //vtk_widget->Clear();
    meas_parts->Clear();
    meas_bnds->Clear();
    //vtk_widget->ClearSelectedBnd();
    //vtk_widget->SetPrims(meas_parts);
    //vtk_widget->SetBnds(meas_bnds);
    vtk_widget->MeasureClearAll();
    //vtk_widget->MeasurePlotCAD(*(measure_cad->Value()));
    meas_parts->Add(measure_cad);
    parts->Add(measure_cad);
    meas_bnds->Reset(meas_parts);
    vtk_widget->Plot(*(measure_cad->Value()));
    //vtk_widget->PlotBnds();
    vtk_widget->MeasurePlotBnds();
    vtk_widget->SetSelectable(true);
    vtk_widget->SetSelectBnd(true);
    vtk_widget->SetSelectDomain(false);


    //measure_dock->ui->pushButton_6->setChecked(true);
    //measure_dock->ui->pushButton->setChecked(false);
    measure_dock->ui->pushButton_5->setChecked(true);
    measure_dock->ui->pushButton_7->setChecked(false);
    measure_dock->ui->pushButton_9->setChecked(false);
    measure_dock->ui->pushButton_11->setChecked(false);
    measure_dock->ui->pushButton_13->setChecked(false);
    measure_dock->ui->progressBar->setValue(0);
    measure_dock->ui->horizontalScrollBar->setValue(100);
    measure_dock->ui->progressBar->setValue(0);
    measure_dock->ui->progressBar_2->setValue(0);
    //measure_dock->ui->pushButton_10->setChecked(false);
    std::cout << "parts num: " << meas_parts->size() << std::endl;


    // set vtk visual to meas_parts and meas_bnds

    TextOutput("It is done.");
    //measure_dock->ui->pushButton_3->setEnabled(false);

    QFile::remove("./data/meas/fengsim_meas_source.vtk");
    meas_line_num = 0;



    return;




    //MM.MeshGeneration(measure_cad->Value());
    //vtk_widget->Hide();
    //vtk_widget->ImportVTKFile(std::string("/home/jiping/FENGSim/build-FENGSim-Desktop_Qt_5_12_10_GCC_64bit-Release/FENGSimDT.vtk"));
    //MM.MeasureTarget();






    // OCC triangulation
    ofstream out;
    out.open(std::string("/home/jiping/FENGSim/FENGSim/Measure/data/cloud_target.vtk").c_str());
    TopExp_Explorer faceExplorer;
    for (faceExplorer.Init(*(measure_cad->Value()), TopAbs_FACE); faceExplorer.More(); faceExplorer.Next())
    {
        TopLoc_Location loc;
        TopoDS_Face aFace = TopoDS::Face(faceExplorer.Current());
        Handle_Poly_Triangulation triFace = BRep_Tool::Triangulation(aFace, loc);
        //        Standard_Integer nTriangles = triFace->NbTriangles();
        gp_Pnt vertex1;
        //        gp_Pnt vertex2;
        //        gp_Pnt vertex3;
        //        Standard_Integer nVertexIndex1 = 0;
        //        Standard_Integer nVertexIndex2 = 0;
        //        Standard_Integer nVertexIndex3 = 0;
        TColgp_Array1OfPnt nodes(1, triFace->NbNodes());
        //            Poly_Array1OfTriangle triangles(1, triFace->NbTriangles());
        nodes = triFace->Nodes();
        //            triangles = triFace->Triangles();
        for (int i = 0; i < triFace->NbNodes(); i++) {
            vertex1 = nodes.Value(i+1);
            out << vertex1.X() << " " << vertex1.Y() << " "<< vertex1.Z() << endl;
        }
    }
}

void MainWindow::MeasureCADHide()
{
    //if (!measure_dock->ui->pushButton_6->isChecked())
    vtk_widget->MeasureCADHide();
    //else if (measure_dock->ui->pushButton_6->isChecked())
    vtk_widget->MeasureCADOn();
}

void MainWindow::MeasureCADHide2()
{
    if (!measure_dock->ui->pushButton_5->isChecked())
        vtk_widget->MeasureBndsHide();
    else if (measure_dock->ui->pushButton_5->isChecked())
        vtk_widget->MeasureBndsOn();
}

void MainWindow::MeasurePresBnd()
{
    //if (measure_dock->ui->pushButton->isChecked()) {
    vtk_widget->MeasurePlotBnds();
    vtk_widget->SetSelectable(true);
    vtk_widget->SetSelectBnd(true);
    vtk_widget->SetSelectDomain(false);
    //vtk_widget->MeasureCADHide();
    //measure_dock->ui->pushButton_10->setChecked(false);
    //meas_selected_bnds.clear();
    //vtk_widget->ClearSelectedBnd();
    //measure_dock->ui->pushButton_6->setChecked(false);
    measure_dock->ui->pushButton_5->setChecked(true);
    MeasureCADHide();
    cout << "bnd num: " << meas_bnds->Size() << " selected bnd num: " << vtk_widget->selected_bnd_id.size() << endl;
    //}
    //else if (!measure_dock->ui->pushButton->isChecked()) {
    //        vtk_widget->MeasurePlotDomains();
    vtk_widget->SetSelectable(false);
    //        vtk_widget->ClearSelectedBnd();
    //        meas_bnds->Clear();
    cout << "bnd num: " << meas_bnds->Size() << " selected bnd num: " << vtk_widget->selected_bnd_id.size() << endl;
    //}
}

void MainWindow::MeasureOpacity()
{
    vtk_widget->MeasureOpacity(measure_dock->ui->horizontalScrollBar->value());
    std::cout << "check opacity working: " << measure_dock->ui->horizontalScrollBar->value() << std::endl;
}

void MainWindow::MeasureDeleteBnd()
{
    vtk_widget->MeasureDeleteBnd();
}

void MainWindow::MeasurePresLine()
{
    //if (measure_dock->ui->pushButton_10->isChecked()) {
    //  measure_dock->ui->pushButton->setChecked(false);
    //}
}

//void MainWindow::MeasureSelectBnd() {
//        if (meas_bnds->Size() == 0) return;

//        if (vtk_widget->GetSelectedBnd() == NULL) return;

//        int n = meas_bnds->Include(*(vtk_widget->GetSelectedBnd()->Value()));

//        bool sel = true;
//        for (int i = 0; i < meas_selected_bnds.size(); i++)
//                if (n == meas_selected_bnds[i])
//                        sel = false;
//        if (!sel) return;

//        meas_selected_bnds.push_back(n);
//        std::cout << "bnds num: " << meas_selected_bnds.size() << " new bnd id: " << n << std::endl;


//        return;

//        BRep_Builder builder;
//        TopoDS_Shell shell;
//        builder.MakeShell(shell);
//        for (int i = 0; i < meas_selected_bnds.size(); i++)
//                builder.Add(shell,*((*meas_bnds)[meas_selected_bnds[i]]->Value()));

//        vtk_widget->MeasPlot(shell);



//        return;

//        MM.MeshGeneration(&shell);

//        MM.MeasureTarget();

//        //    ofstream out;
//        //    out.open(std::string("/home/jiping/FENGSim/FENGSim/Measure/data/cloud_target.vtk").c_str(),ios::app);
//        //    TopLoc_Location loc;
//        //    TopoDS_Face aFace = TopoDS::Face(*((*bnds)[n]->Value()));
//        //    Handle_Poly_Triangulation triFace = BRep_Tool::Triangulation(aFace, loc);
//        //    //        Standard_Integer nTriangles = triFace->NbTriangles();
//        //    gp_Pnt vertex1;
//        //    //        gp_Pnt vertex2;
//        //    //        gp_Pnt vertex3;
//        //    //        Standard_Integer nVertexIndex1 = 0;
//        //    //        Standard_Integer nVertexIndex2 = 0;
//        //    //        Standard_Integer nVertexIndex3 = 0;
//        //    TColgp_Array1OfPnt nodes(1, triFace->NbNodes());
//        //    //            Poly_Array1OfTriangle triangles(1, triFace->NbTriangles());
//        //    nodes = triFace->Nodes();
//        //    //            triangles = triFace->Triangles();
//        //    for (int i = 0; i < triFace->NbNodes(); i++) {
//        //        vertex1 = nodes.Value(i+1);
//        //        out << vertex1.X() << " " << vertex1.Y() << " "<< vertex1.Z() << endl;
//        //    }



//        vtk_widget->ClearCloudTarget();
//        vtk_widget->MeasureImportCloudTarget(std::string("./data/meas/fengsim_meas_cloud_target.vtk"));

//        measure_dock->ui->pushButton->setChecked(false);
//        measure_dock->ui->pushButton_10->setChecked(false);

//}

void MainWindow::MeasureSelectedBndToPointCloud()
{
    TextOutput("Begin to change a CAD model to a point cloud model. Please wait...");
    meas_th3->vtk_widget = vtk_widget;
    meas_th3->meas_bnds = bnds;
    meas_th3->measure_dock = measure_dock;
    meas_th3->start();
    meas_th3->path = meas_path;
    measure_dock->ui->pushButton_4->setEnabled(false);
    measure_dock->ui->pushButton_7->setChecked(true);
    vtk_widget->MeasureCloudTargetHide(false);
    MeasureSelectedBndToPointCloud2();
}

void MainWindow::MeasureCloudTargetHide()
{
    if (!measure_dock->ui->pushButton_7->isChecked())
    {
        vtk_widget->MeasureCloudTargetHide(true);
    }
    else
    {
        vtk_widget->MeasureCloudTargetHide(false);
    }
}

void MainWindow::MeasureCloudSourceHide()
{
    if (!measure_dock->ui->pushButton_9->isChecked())
    {
        vtk_widget->MeasureCloudSourceHide(true);
    }
    else
    {
        vtk_widget->MeasureCloudSourceHide(false);
    }
}

void MainWindow::MeasureCloudICPHide()
{
    if (!measure_dock->ui->pushButton_11->isChecked())
    {
        vtk_widget->MeasureCloudICPHide(true);
    }
    else
    {
        vtk_widget->MeasureCloudICPHide(false);
    }
}

void MainWindow::MeasureCloudGDTHide()
{
    if (!measure_dock->ui->pushButton_13->isChecked())
    {
        vtk_widget->MeasureCloudGDTHide(true);
    }
    else
    {
        vtk_widget->MeasureCloudGDTHide(false);
    }
}

void MainWindow::MeasureSelectedBndToPointCloud2()
{
    if (meas_th3->isFinished())
    {
        //vtk_widget->MeasureSetSelectedBndsUnvisible();
        cout << "bnd num: " << meas_bnds->Size() << " selected bnd num: " << vtk_widget->selected_bnd_id.size() << endl;
        //vtk_widget->MeasureClearCloudTarget();
        //vtk_widget->MeasureClearICPFinal();
        //vtk_widget->MeasureImportCloudTarget((meas_path + QString("/data/meas/fengsim_meas_target.vtk")).toStdString());
        std::cout << (meas_path + QString("/data/mesh/fengsim_mesh.vtk")).toStdString() << std::endl;
        vtk_widget->MeasureImportCloudTarget((meas_path + QString("/data/mesh/fengsim_mesh.vtk")).toStdString());
        //measure_dock->ui->pushButton->setChecked(false);
        //vtk_widget->SetSelectable(false);
        measure_dock->ui->pushButton_4->setEnabled(true);
        if (vtk_widget->selected_bnd_id.size() > 0) {
            //measure_dock->ui->pushButton_4->setEnabled(false);
            //measure_dock->ui->pushButton->setEnabled(false);
        }
        else {
            //measure_dock->ui->pushButton_4->setEnabled(true);
        }

        TextOutput("It is done.");
        return;
    }
    machining_timer->singleShot(1, this, SLOT(MeasureSelectedBndToPointCloud2()));
}

void MainWindow::MeasureOpenPointCloud()
{
    TextOutput("Importing a point cloud model. Please wait...");
    QString fileName = QFileDialog::getOpenFileName(this,tr("Open File"),"/home/jiping/FENGSim/FENGSim/Measure/data/",tr("CMM Files (*.dxf *.vtk *.stl)")
                                                    , 0 , QFileDialog::DontUseNativeDialog);
    cout << fileName.toStdString() << endl;
    if (fileName.isNull()) return;
    QFileInfo fileinfo(fileName);
    QString file_suffix = fileinfo.suffix();
    // std::cout << file_suffix.toStdString() << std::endl;




    if (file_suffix == "dxf")
    {
        meas_scene_type = 111;
        ifstream is;
        is.open(fileName.toStdString());

        ofstream out;
        // app is appended
        out.open(std::string("./data/meas/fengsim_meas_scene.vtk").c_str(),ios::app);
        const int len = 256;
        char L[len];
        while (strncasecmp("EOF", L, 3) != 0)
        {
            is.getline(L,len);
            if (!strncasecmp("VERTEX", L, 6))
            {
                is.getline(L,len);
                is.getline(L,len);
                is.getline(L,len);

                double x = 0;
                double y = 0;
                double z = 0;
                is.getline(L,len);
                sscanf(L,"%lf", &x);
                is.getline(L,len);
                //sscanf(L,"%lf", &y);
                is.getline(L,len);
                sscanf(L,"%lf", &y);

                double pos[3];
                //int d = sscanf(measure_dock->ui->lineEdit_3->text().toStdString().c_str(), "(%lf,%lf,%lf)", pos, pos+1, pos+2);
                pos[0] = 0;
                pos[1] = 0;
                //pos[2] = measure_dock->ui->doubleSpinBox_11->text().toDouble();
                out << x + pos[0] << " " << y + pos[1] << " " << z + meas_line_num * pos[2] << endl;

                is.getline(L,len);
                is.getline(L,len);
                is.getline(L,len);
            }
        }
        out.close();
        meas_line_num++;
        measure_cloud_source_ischeck = vtk_widget->MeasureImportCloudSource(std::string("./data/meas/fengsim_meas_scene.vtk"));
        QFile::remove("./data/meas/fengsim_meas_scene_mesh_points.vtk");
        QFile::copy("./data/meas/fengsim_meas_scene.vtk", "./data/meas/fengsim_meas_scene_mesh_points.vtk");
        return;
    }
    else if (file_suffix == "vtk")
    {
        QFile::remove(meas_path + QString("/data/meas/fengsim_meas_cloud_source.vtk"));
        QFile::copy(fileName, meas_path + QString("/data/meas/fengsim_meas_cloud_source.vtk"));



        //                ofstream out;
        //                out.open("/home/jiping/FENGSim/FENGSim/Measure/data/example2/sphere_source.vtk");
        //                double r = 5;
        //                int n = 80;
        //                int m = 160;
        //                double pi = atan(1) * 4;
        //                int i = 0;
        //                while (i < n) {
        //                        int j = 0;
        //                        while (j < m) {
        //                                double x = r * sin(i * pi / n) * sin(j * 2 * pi / m);
        //                                double y = r * sin(i * pi / n) * cos(j * 2 * pi / m);
        //                                double z = r * cos(i * pi / n);
        //                                out << setiosflags(ios::fixed) << setprecision(16) << x << " " << y << " " << z << endl;
        //                                j = j + 1;
        //                        }
        //                        i = i + 1;
        //                }
        //                out.close();



    }
    else if (file_suffix == "stl")
    {
        //vtk_widget->SetSelectable(false);
        //vtk_widget->MeasureClearCloudSource();
        //vtk_widget->MeasureImportScene(fileName.toStdString());

        meas_th1->vtk_widget = vtk_widget;
        meas_th1->name = fileName.toStdString();
        meas_th1->path = meas_path;
        vtk_widget->MeasureClearCloudSource();
        meas_th1->start();
        measure_dock->ui->pushButton_8->setEnabled(false);
        measure_cloud_source_ischeck = true;
        MeasureOpenPointCloud2();
        return;
    }

    //    vtk_widget->SetSelectable(false);

    //    measure_cloud_source_ischeck = vtk_widget->MeasureImportCloudSource(std::string("./data/meas/fengsim_meas_cloud_source.vtk"));

    //    if (measure_cloud_source_ischeck) {
    //        vtk_widget->MeasureCloudSourceTransform();
    //    }

}

void MainWindow::MeasureOpenPointCloud2()
{
    if (meas_th1->isFinished())
    {
        vtk_widget->MeasureImportSource2(meas_path);
        measure_dock->ui->pushButton_8->setEnabled(true);
        measure_dock->ui->pushButton_9->setChecked(true);


        TextOutput("It is done.");

        return;
    }
    machining_timer->singleShot(1, this, SLOT(MeasureOpenPointCloud2()));
}

void MainWindow::MeasurePCMove()
{
    //    if (measure_dock->ui->comboBox_2->currentIndex() == 0)
    //        measure_dock->ui->doubleSpinBox_4->setValue(meas_tran[0]);
    //    else if (measure_dock->ui->comboBox_2->currentIndex() == 1)
    //        measure_dock->ui->doubleSpinBox_4->setValue(meas_tran[1]);
    //    else if (measure_dock->ui->comboBox_2->currentIndex() == 2)
    //        measure_dock->ui->doubleSpinBox_4->setValue(meas_tran[2]);
    //    else if (measure_dock->ui->comboBox_2->currentIndex() == 3)
    //        measure_dock->ui->doubleSpinBox_4->setValue(meas_tran[3]);
    //    else if (measure_dock->ui->comboBox_2->currentIndex() == 4)
    //        measure_dock->ui->doubleSpinBox_4->setValue(meas_tran[4]);
    //    else if (measure_dock->ui->comboBox_2->currentIndex() == 5)
    //        measure_dock->ui->doubleSpinBox_4->setValue(meas_tran[5]);
}

void MainWindow::MeasurePCMoveValue()
{
    //meas_tran[measure_dock->ui->comboBox_2->currentIndex()] = measure_dock->ui->doubleSpinBox_4->value();
    MeasurePointCloudMove();
}


void MainWindow::MeasurePointCloudMove () {
    //if (!measure_cloud_source_ischeck) return;
    //    double x = measure_dock->ui->doubleSpinBox_4->text().toDouble();
    //    double y = measure_dock->ui->doubleSpinBox_5->text().toDouble();
    //    double z = measure_dock->ui->doubleSpinBox_6->text().toDouble();
    //    double angle_x = measure_dock->ui->doubleSpinBox_7->text().toDouble();
    //    double angle_y = measure_dock->ui->doubleSpinBox_8->text().toDouble();
    //    double angle_z = measure_dock->ui->doubleSpinBox_9->text().toDouble();

    //   std::cout << x << " " << y << " " << z << std::endl;

    std::cout << meas_tran[0] << " "
                              << meas_tran[1] << " "
                              << meas_tran[2] << std::endl;
    vtk_widget->MeasureCloudSourceTransform(meas_tran[0],
            meas_tran[1],
            meas_tran[2],
            meas_tran[3],
            meas_tran[4],
            meas_tran[5],meas_path);



    //        return;
    //        reg.SetTran(0, measure_dock->ui->doubleSpinBox_4->text().toDouble());
    //        reg.SetTran(1, measure_dock->ui->doubleSpinBox_5->text().toDouble());
    //        reg.SetTran(2, measure_dock->ui->doubleSpinBox_6->text().toDouble());
    //        reg.SetTran(3, measure_dock->ui->doubleSpinBox_7->text().toDouble());
    //        reg.SetTran(4, measure_dock->ui->doubleSpinBox_8->text().toDouble());
    //        reg.SetTran(5, measure_dock->ui->doubleSpinBox_9->text().toDouble());
    //        reg.move();
    //        vtk_widget->MeasureClearCloudSource();
    //        vtk_widget->MeasureImportCloudSource(std::string("./data/meas/fengsim_meas_cloud_source2.vtk"));
}

void MainWindow::MeasurePointCloudReset() {

    vtk_widget->MeasureClearCloudSource();
    vtk_widget->MeasureClearICPFinal();
    //        //        reg.SetTran(0, 0);
    //        reg.SetTran(1, 0);
    //        reg.SetTran(2, 0);
    //        reg.SetTran(3, 0);
    //        reg.SetTran(4, 0);
    //        reg.SetTran(5, 0);
    QFile::remove("./data/meas/fengsim_meas_scene.vtk");
    QFile::remove("./data/meas/fengsim_meas_scene_2.vtk");
    QFile::remove("./data/meas/fengsim_meas_icp_final.vtk");
    //    measure_dock->ui->doubleSpinBox_2->setValue(1);
    //    measure_dock->ui->doubleSpinBox_4->setSingleStep(1);
    //    measure_dock->ui->doubleSpinBox_5->setSingleStep(1);
    //    measure_dock->ui->doubleSpinBox_6->setSingleStep(1);
    //    measure_dock->ui->doubleSpinBox_7->setSingleStep(1);
    //    measure_dock->ui->doubleSpinBox_8->setSingleStep(1);
    //    measure_dock->ui->doubleSpinBox_9->setSingleStep(1);
    //    measure_dock->ui->doubleSpinBox_4->setValue(0);
    //    measure_dock->ui->doubleSpinBox_5->setValue(0);
    //    measure_dock->ui->doubleSpinBox_6->setValue(0);
    //    measure_dock->ui->doubleSpinBox_7->setValue(0);
    //    measure_dock->ui->doubleSpinBox_8->setValue(0);
    //    measure_dock->ui->doubleSpinBox_9->setValue(0);
}

#include "Measure/ls.h"

//#include "Measure/MeasureThread1.h"

void MainWindow::MeasureRegistration()
{
    ofstream out;
    out.open((meas_path+QString("/data/meas/results")).toStdString().c_str());
    out << 0 << std::endl;
    out.close();


    if (measure_exe_id == 2)
        measure_dock->ui->pushButton_19->setEnabled(false);
    if (measure_exe_id == 3)
        measure_dock->ui->pushButton_10->setEnabled(false);


    meas_th2->vtk_widget = vtk_widget;
    meas_th2->meas_bnds = meas_bnds;
    meas_th2->measure_dock = measure_dock;
    meas_th2->path = meas_path;
    meas_th2->measure_exe_id = measure_exe_id;
    //meas_th2->measure_error = measure_dock->ui->textEdit;
    //measure_dock->ui->pushButton_12->setEnabled(false);
    TextOutput("Begin to do registration and calcualte error. Please wait...");
    meas_th2->start();

    MeasurePlotResults();


    return;


}


void MainWindow::MeasureSACIA()
{
    measure_exe_id = 1;
    MeasureRegistration();
}

void MainWindow::MeasureICP()
{
    measure_exe_id = 2;
    MeasureRegistration();
}

void MainWindow::MeasureGDT()
{
    measure_exe_id = 3;
    MeasureRegistration();
}

//void MeasureThread1::run () {
//    vtk_widget->SetSelectable(false);
//    vtk_widget->MeasureClearCloudSource();
//    vtk_widget->MeasureImportScene(name);
//    exit();
//    exec();
//}



void MeasureThread2::run () {
    std::cout << "begin registration" << std::endl;
    QProcess *proc = new QProcess();
    proc->setWorkingDirectory(path);
    if (measure_exe_id == 2)
    {
        std::cout << "surface profile run in " << path.toStdString() << std::endl;
        proc->start("../GDT/build/fengsim_meas");
    }
    else if (measure_exe_id == 3)
    {
        std::cout << "gdt error run in " << path.toStdString() << std::endl;
        proc->start("ls");
    }
    if (proc->waitForFinished(-1))
    {
        if (measure_exe_id==2)
        {
            measure_dock->ui->pushButton_19->setEnabled(true);
            vtk_widget->MeasureSTLTransform(path);
            quit();
            return;
        }
        else if (measure_exe_id==3)
        {
            measure_dock->ui->pushButton_10->setEnabled(false);
            std::cout << "check gdt error if do" << std::endl;
            vtk_widget->MeasureSTLTransform2(path);
            // *********************************
            // step 1
            // *********************************
            double xmax = -1e10;
            double xmin = 1e10;
            double ymax = -1e10;
            double ymin = 1e10;
            double zmax = -1e10;
            double zmin = 1e10;
            for (int i = 0; i < vtk_widget->selected_bnd_id.size(); i++)
            {
                TopExp_Explorer faceExplorer;
                for (faceExplorer.Init(*((*meas_bnds)[vtk_widget->selected_bnd_id[i]]->Value()), TopAbs_FACE);
                     faceExplorer.More(); faceExplorer.Next())
                {


                    TopoDS_Face aFace = TopoDS::Face(faceExplorer.Current());

                    double _xmax;
                    double _xmin;
                    double _ymax;
                    double _ymin;
                    double _zmax;
                    double _zmin;
                    Bnd_Box B;
                    BRepBndLib::Add(aFace, B);
                    B.Get(_xmin, _ymin, _zmin, _xmax, _ymax, _zmax);
                    if (_xmin < xmin) xmin = _xmin;
                    if (_ymin < ymin) ymin = _ymin;
                    if (_zmin < zmin) zmin = _zmin;
                    if (_xmax > xmax) xmax = _xmax;
                    if (_ymax > ymax) ymax = _ymax;
                    if (_zmax > zmax) zmax = _zmax;
                }
            }
            std::cout << xmin << " " << xmax << " "
                      << ymin << " " << ymax << " "
                      << zmin << " " << zmax << std::endl;

            double x[3];
            double x1[3];
            double x2[3];
            double x3[3];

            double up_error = -1e5;
            double down_error = 1e5;
            double average_error = 0;
            int average_num = 0;

            ofstream out;
            out.open((path + QString("/data/meas/fengsim_meas_gdt_results.vtk")).toStdString().c_str());
            std::cout << "vtk_widget->MeasureTransformCellsNumber(): " << vtk_widget->MeasureTransformCellsNumber() << std::endl;

            for (int i=0; i<vtk_widget->MeasureTransformCellsNumber(); )
            {
                vtk_widget->MeasureTransformCell(i,0,x1[0],x1[1],x1[2]);
                vtk_widget->MeasureTransformCell(i,1,x2[0],x2[1],x2[2]);
                vtk_widget->MeasureTransformCell(i,2,x3[0],x3[1],x3[2]);
                x[0] = (x1[0]+x2[0]+x3[0])/3.0;
                x[1] = (x1[1]+x2[1]+x3[1])/3.0;
                x[2] = (x1[2]+x2[2]+x3[2])/3.0;
                if (x[0] > xmin && x[0] < xmax && x[1] > ymin && x[1] < ymax && x[2] > zmin && x[2] < zmax)
                {
                    // *********************************
                    // step 2
                    // *********************************
                    for (int j = 0; j < vtk_widget->selected_bnd_id.size(); j++) {
                        TopExp_Explorer faceExplorer;
                        for (faceExplorer.Init(*((*meas_bnds)[vtk_widget->selected_bnd_id[j]]->Value()), TopAbs_FACE);
                             faceExplorer.More(); faceExplorer.Next()) {
                            TopoDS_Face aFace = TopoDS::Face(faceExplorer.Current());

                            IntTools_Context tool;
                            if (tool.IsValidPointForFace(gp_Pnt(x[0], x[1], x[2]), aFace, 0.1)) {
                                /* Returns true if the distance between point aP3D and
                                  face aF is less or equal to tolerance aTol and
                                  projection point is inside or on the boundaries of the face aF.*/
                                // *********************************
                                // step 3
                                // *********************************
                                double a[3];
                                a[0] = x2[0] - x1[0];
                                a[1] = x2[1] - x1[1];
                                a[2] = x2[2] - x1[2];
                                double b[3];
                                b[0] = x3[0] - x1[0];
                                b[1] = x3[1] - x1[1];
                                b[2] = x3[2] - x1[2];
                                double c[3];
                                c[0] = a[1]*b[2] - a[2]*b[1];
                                c[1] = a[2]*b[0] - a[0]*b[2];
                                c[2] = a[0]*b[1] - a[1]*b[0];

                                GeomAPI_ProjectPointOnSurf proj1(gp_Pnt(x[0], x[1], x[2]), BRep_Tool::Surface(aFace));
                                double d[3];
                                d[0] = proj1.NearestPoint().X() - x[0];
                                d[1] = proj1.NearestPoint().Y() - x[1];
                                d[2] = proj1.NearestPoint().Z() - x[2];
                                double angle = (c[0]*d[0] + c[1]*d[1] + c[2]*d[2]) / sqrt(c[0]*c[0]+c[1]*c[1]+c[2]*c[2])
                                        / sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);

                                if (angle>0.9)
                                {
                                    double error1 = proj1.LowerDistance();
                                    out << x[0] << " " << x[1] << " " << x[2] << " " << error1 << std::endl;
                                    if (error1 > up_error) up_error = error1;
                                    average_error += error1;
                                    average_num++;
                                }
                                else if (angle<-0.9)
                                {
                                    double error1 = -proj1.LowerDistance();
                                    out << x[0] << " " << x[1] << " " << x[2] << " " << error1 << std::endl;
                                    if (error1 < down_error) down_error = error1;
                                    average_error += error1;
                                    average_num++;
                                }
                            }
                        }
                    }
                }
                i+=2*vtk_widget->MeasureTransformDensity();

                if (i>vtk_widget->MeasureTransformCellsNumber())
                    progress = 1;
                else
                    progress = double(i+1)/double(vtk_widget->MeasureTransformCellsNumber());

            }
            out.close();
            average_error = average_error / average_num;

            MeasureVariance(average_error, path);



            quit();
            return;
        }


    }
    exec();
}

void MainWindow::MeasureSACIAPro()
{
    if (measure_exe_id==2)
    {
        ifstream is;
        is.open((meas_path + QString("/data/meas/sacia_pro.txt")).toStdString().c_str());
        const int len = 256;
        char L[len];
        double p;
        is.getline(L,len);
        sscanf(L,"%lf", &p);
        is.close();
        measure_dock->ui->progressBar->setValue(p*100.0);
    }
    else if (measure_exe_id==3)
    {
        measure_dock->ui->progressBar_2->setValue(meas_th2->progress*100);

    }
}



void MainWindow::MeasurePlotResults()
{
    if (meas_th2->isFinished()) {
        if (measure_exe_id == 2)
        {
            TextOutput("The registration is done.");
            measure_dock->ui->pushButton_11->setChecked(true);
            vtk_widget->MeasuresPlotICPtrans(0,255,0);
            measure_dock->ui->pushButton_11->setEnabled(true);
            vtk_widget->MeasureCloudICPHide(false);
        }
        else if (measure_exe_id == 3)
        {
            ifstream is;
            is.open((meas_path + QString("/data/meas/results")).toStdString().c_str());
            const int len = 256;
            char L[len];
            TextOutput("The calculation for gdt error is done.");
            double x1;
            double x2;
            double x3;
            is.getline(L,len);
            sscanf(L,"%lf", &x1);
            TextOutput(QString("up maximum distance: ") + QString::number(x1) + QString("."));
            is.getline(L,len);
            sscanf(L,"%lf", &x2);
            TextOutput(QString("down maximum distance: ") + QString::number(x2) + QString("."));
            is.getline(L,len);
            sscanf(L,"%lf", &x3);
            TextOutput(QString("mean distance: ") + QString::number(x3) + QString("."));
            TextOutput(QString("surface profile: ") + QString::number(x1+x2) + QString("."));
            is.close();

            measure_dock->ui->pushButton_7->setChecked(false);
            MeasureCloudTargetHide();
            measure_dock->ui->pushButton_9->setChecked(false);
            MeasureCloudSourceHide();
            measure_dock->ui->pushButton_11->setChecked(false);
            MeasureCloudICPHide();
            //measure_dock->ui->pushButton_6->setChecked(true);
            MeasureCADHide();
            measure_dock->ui->pushButton_5->setChecked(false);
            MeasureCADHide2();

            measure_dock->ui->progressBar_2->setValue(100);
            std::cout << "vtk_widget->MeasureTransformDensity(): " << vtk_widget->MeasureTransformDensity() << std::endl;
            measure_dock->ui->pushButton_10->setEnabled(true);
            measure_dock->ui->pushButton_13->setChecked(true);
            vtk_widget->MeasureImportICPFinal((meas_path + QString("/data/meas/fengsim_meas_gdt_results_variance.vtk")).toStdString());
        }
        return;
    }
    else {
        MeasureSACIAPro();
        measure_timer->singleShot(100, this, SLOT(MeasurePlotResults()));
        return;
    }
}

void MeasureThread2::MeasureVariance (double mean, QString path)
{
    ifstream is;
    is.open((path + QString("/data/meas/fengsim_meas_gdt_results.vtk")).toStdString().c_str());
    const int len = 256;
    char L[len];
    double x[4];

    int n = 0;
    double var = 0;
    double error = 0;
    while (is.getline(L,len)) {
        sscanf(L,"%lf %lf %lf %lf", x, x+1, x+2, x+3);
        n++;
        var += (x[3]-mean)*(x[3]-mean);
    }
    var = sqrt(var / n);
    is.close();
    std::cout << "var error: " << var << std::endl;

    double up_error = -1e5;
    double down_error = 1e5;
    double mean_dist = 0;
    ofstream out;
    out.open((path + QString("/data/meas/fengsim_meas_gdt_results_variance.vtk")).toStdString().c_str());
    is.open((path + QString("/data/meas/fengsim_meas_gdt_results.vtk")).toStdString().c_str());
    n = 0;
    while (is.getline(L,len)) {
        sscanf(L,"%lf %lf %lf %lf", x, x+1, x+2, x+3);
        if (abs(x[3]-mean)<var)
        {
            out << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << std::endl;
            if (x[3]>up_error) up_error = x[3];
            if (x[3]<down_error) down_error = x[3];
            mean_dist += abs(x[3]);
            n++;
        }
    }
    is.close();
    out.close();

    std::cout << "registration done: "  << std::endl;
    out.open((path+QString("/data/meas/results")).toStdString().c_str());
    out << abs(up_error) << std::endl;
    out << abs(down_error) << std::endl;
    out << abs(mean_dist/n) << std::endl;
    out.close();
}


















// *******************************************************
// additive manufacturing

void MainWindow::AMOpenCAD()
{
    ofstream out;
    out.open((meas_path+QString("../../CAM/Cura/conf/m++conf")).toStdString());
    out << "#loadconf = Poisson/conf/poisson.conf;" << endl;
    out << "#loadconf = Elasticity/conf/m++conf;" << endl;
    out << "#loadconf = ElastoPlasticity/conf/m++conf;" << endl;
    out << "#loadconf = ThermoElasticity/conf/m++conf;" << endl;
    out << "#loadconf = AdditiveManufacturing/conf/m++conf;" << endl;
    out << "loadconf = Cura/conf/cura.conf;" << endl;
    out.close();


    // file name
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), "/home/jiping/OpenDT/CAM/data/",
                                                    tr("CAD Files (*.stp *.step)"),
                                                    0 , QFileDialog::DontUseNativeDialog);
    if (fileName.isNull()) return;
    char* ch;
    QByteArray ba = fileName.toLatin1();
    ch=ba.data();

    // occ reader
    STEPControl_Reader reader;
    if (!reader.ReadFile(ch)) return;
    Standard_Integer NbRoots = reader.NbRootsForTransfer();
    Standard_Integer NbTrans = reader.TransferRoots();
    am_cad = new General(new TopoDS_Shape(reader.OneShape()));
    if (am_cad == 0) return;

    // reset all data
    vtk_widget->SetSelectable(false);
    vtk_widget->Clear();
    am_parts->Clear();
    am_bnds->Clear();
    vtk_widget->ClearSelectedBnd();
    vtk_widget->SetPrims(am_parts);
    vtk_widget->SetBnds(am_bnds);
    vtk_widget->AMImportCAD(*(am_cad->Value()));
    am_parts->Add(am_cad);
    //measure_dock->ui->pushButton->setChecked(false);
    //measure_dock->ui->pushButton_10->setChecked(false);
    std::cout << "parts num: " << am_parts->size() << std::endl;

    vtk_widget->AMSetSTLVisible(false);
    vtk_widget->AMSetSlicesVisible(false);
    QFile::remove("./Cura/data/am/am.stl");
    QFile::remove("./Cura/data/am/slices.cli");
    QFile::remove("./Cura/data/am/slices.vtk");

    // set vtk visual to meas_parts and meas_bnds

    additive_manufacturing_dock->ui->pushButton_10->setChecked(true);
    additive_manufacturing_dock->ui->pushButton_17->setChecked(false);
    additive_manufacturing_dock->ui->pushButton_19->setChecked(false);



    return;




    //MM.MeshGeneration(measure_cad->Value());
    //vtk_widget->Hide();
    //vtk_widget->ImportVTKFile(std::string("/home/jiping/FENGSim/build-FENGSim-Desktop_Qt_5_12_10_GCC_64bit-Release/FENGSimDT.vtk"));
    //MM.MeasureTarget();






    // OCC triangulation
    //        ofstream out;
    //        out.open(std::string("/home/jiping/FENGSim/FENGSim/Measure/data/cloud_target.vtk").c_str());
    //        TopExp_Explorer faceExplorer;
    //        for (faceExplorer.Init(*(measure_cad->Value()), TopAbs_FACE); faceExplorer.More(); faceExplorer.Next())
    //        {
    //                TopLoc_Location loc;
    //                TopoDS_Face aFace = TopoDS::Face(faceExplorer.Current());
    //                Handle_Poly_Triangulation triFace = BRep_Tool::Triangulation(aFace, loc);
    //                //        Standard_Integer nTriangles = triFace->NbTriangles();
    //                gp_Pnt vertex1;
    //                //        gp_Pnt vertex2;
    //                //        gp_Pnt vertex3;
    //                //        Standard_Integer nVertexIndex1 = 0;
    //                //        Standard_Integer nVertexIndex2 = 0;
    //                //        Standard_Integer nVertexIndex3 = 0;
    //                TColgp_Array1OfPnt nodes(1, triFace->NbNodes());
    //                //            Poly_Array1OfTriangle triangles(1, triFace->NbTriangles());
    //                nodes = triFace->Nodes();
    //                //            triangles = triFace->Triangles();
    //                for (int i = 0; i < triFace->NbNodes(); i++) {
    //                        vertex1 = nodes.Value(i+1);
    //                        out << vertex1.X() << " " << vertex1.Y() << " "<< vertex1.Z() << endl;
    //                }
    //        }
}

void MainWindow::AMSetCADVisible()
{
    if (additive_manufacturing_dock->ui->pushButton_10->isChecked())
    {
        vtk_widget->AMSetCADVisible(true);
    }
    else if (!additive_manufacturing_dock->ui->pushButton_10->isChecked())
    {
        vtk_widget->AMSetCADVisible(false);
    }
}

#include "StlAPI_Writer.hxx"
void MainWindow::AMCAD2STL()
{
    if (am_parts->size() == 0) return;
    StlAPI_Writer output;
    output.Write(*((*am_parts)[0]->Value()), (meas_path+QString("/data/am/am.stl")).toStdString().c_str());
    vtk_widget->AMImportSTL();
    additive_manufacturing_dock->ui->pushButton_17->setChecked(true);
}

void MainWindow::AMSetSTLVisible()
{
    if (additive_manufacturing_dock->ui->pushButton_17->isChecked())
    {
        vtk_widget->AMSetSTLVisible(true);
    }
    else if (!additive_manufacturing_dock->ui->pushButton_17->isChecked())
    {
        vtk_widget->AMSetSTLVisible(false);
    }
}


void MainWindow::AMSTL2Slices()
{
    if (am_parts->size() == 0) return;
    ofstream out;
    QString file = meas_path+QString("/../AM/build/solver/conf/cura.conf");
    std::cout << file.toStdString() << std::endl;
    out.open(file.toStdString());
    out << "Model = SlicePhaseTest" << endl;
    out << meas_path.toStdString() << "/data/am/am.stl"  << endl;
    out << meas_path.toStdString() << "/data/am/slices.vtk" << endl;
    out << meas_path.toStdString() << "/data/am/slices_pathplanning.vtk" << endl;
    out << meas_path.toStdString() << "/data/am/slices_meshing.cli" << endl;
    out << additive_manufacturing_dock->ui->doubleSpinBox->text().toDouble() << endl;
    out << additive_manufacturing_dock->ui->doubleSpinBox_2->text().toDouble() << endl;

    QProcess *proc = new QProcess();
    proc->setWorkingDirectory((meas_path+QString("/../AM/build")));
    proc->start("./AMSolver");

    if (proc->waitForFinished(-1)) {
        //MM.ClearSlices();
        //MM.FileFormatCliToVTK(cli_file_name + ".cli");
        //vtk_widget->Clear();
        //vtk_widget->ClearAMSlices();
        //vtk_widget->ImportVTKFileAMSlices(cli_file_name.toStdString() + ".vtk");
        vtk_widget->AMImportSlices();
        additive_manufacturing_dock->ui->pushButton_19->setChecked(true);
    }
}


void MainWindow::AMSetSlicesVisible()
{
    if (additive_manufacturing_dock->ui->pushButton_19->isChecked())
    {
        vtk_widget->AMSetSlicesVisible(true);
    }
    else if (!additive_manufacturing_dock->ui->pushButton_19->isChecked())
    {
        vtk_widget->AMSetSlicesVisible(false);
    }
}

void MainWindow::AMSlices2PathPlanning()
{
    ofstream out;
    QString file = meas_path+QString("/../AM/build/solver/conf/cura.conf");
    std::cout << meas_path.toStdString() << std::endl;
    out.open(file.toStdString());
    out << "Model = InfillTest" << endl;
    out << meas_path.toStdString() << "/data/am/slices_pathplanning.vtk" << endl;
    out << meas_path.toStdString() << "/data/am/pathplanning.vtk" << endl;

    QProcess *proc = new QProcess();
    proc->setWorkingDirectory((meas_path+QString("/../AM/build/")));
    proc->start("./AMSolver");

    if (proc->waitForFinished(-1)) {
        //MM.ClearSlices();
        //MM.FileFormatCliToVTK(cli_file_name + ".cli");
        //vtk_widget->Clear();
        //vtk_widget->ClearAMPathPlanning();
        //std::cout << path_file_name.toStdString() + "_outlines0.vtk" << std::endl;
        //vtk_widget->ImportVTKFileAMPathPlanning(path_file_name.toStdString() + "_outlines0.vtk");
        vtk_widget->AMImportPathPlanning();
        additive_manufacturing_dock->ui->pushButton_21->setChecked(true);
        QFile::remove(meas_path+QString("/../../AM/AdditiveManufacturing/conf/geo/pathplanning.vtk"));
        QFile::copy("./data/am/pathplanning.vtk",
                    meas_path+QString("/../../AM/AdditiveManufacturing/conf/geo/pathplanning.vtk"));
    }
    proc->close();
}

void MainWindow::AMSetPathPlanningVisible()
{
    if (additive_manufacturing_dock->ui->pushButton_21->isChecked())
    {
        vtk_widget->AMSetPathPlanningVisible(true);
    }
    else if (!additive_manufacturing_dock->ui->pushButton_21->isChecked())
    {
        vtk_widget->AMSetPathPlanningVisible(false);
    }
}

void MainWindow::AMSlices2Mesh()
{
    QProcess *proc = new QProcess(); 
    proc->setWorkingDirectory(meas_path+QString("/../../../software/slice2mesh-master/build-slice2mesh-Desktop_Qt_5_12_10_GCC_64bit-Debug"));
    std::cout << (QString("./slice2mesh ./data/am/slices_meshing.cli ") + additive_manufacturing_dock->ui->lineEdit_2->text()).toStdString() << std::endl;
    QString command(QString("./slice2mesh ")
                    +meas_path
                    +QString("/data/am/slices_meshing.cli ")
                    + additive_manufacturing_dock->ui->lineEdit_2->text());
    proc->start(command);

    if (proc->waitForFinished(-1)) {
        MM.FileFormatMeshToVTK(meas_path
                               +QString("/../../../software/slice2mesh-master/build-slice2mesh-Desktop_Qt_5_12_10_GCC_64bit-Debug")
                               +QString("/amslices2mesh.mesh"),
                               "./data/am/mesh.vtk");

        // this is old work for machining live and dead element
        //        MM.FileFormatMeshToGeo("/home/jiping/software/slice2mesh-master/build-slice2mesh-Desktop_Qt_5_12_10_GCC_64bit-Debug/amslices2mesh.mesh",
        //                               "/home/jiping/FENGSim/AM/Elasticity/conf/geo/thinwall.geo");
        MM.FileFormatMeshToGeo((meas_path
                               +QString("/../../../software/slice2mesh-master/build-slice2mesh-Desktop_Qt_5_12_10_GCC_64bit-Debug")
                               +QString("/amslices2mesh.mesh")).toStdString().c_str(),
                               (meas_path+QString("/../../AM/AdditiveManufacturing/conf/geo/thinwall.geo")).toStdString().c_str());

        vtk_widget->AMImportMesh();
        additive_manufacturing_dock->ui->pushButton_23->setChecked(true);
    }
}


void MainWindow::AMSetMeshVisible()
{
    if (additive_manufacturing_dock->ui->pushButton_23->isChecked())
    {
        vtk_widget->AMSetMeshVisible(true);
    }
    else if (!additive_manufacturing_dock->ui->pushButton_23->isChecked())
    {
        vtk_widget->AMSetMeshVisible(false);
    }
}

#include<QThread>
class MyThread1 : public QThread
{
public:
    void run();
    QString filename;
    void setfilename (QString name) {
        filename = name;
    }
};

void MyThread1::run()
{
    QProcess *proc = new QProcess();
    proc->setWorkingDirectory( QString("/home/jiping/OpenDT/AM") );
    QString command(QString("./AMRun"));
    proc->start(command);
    if (proc->waitForFinished(-1)) {
        quit();
    }
    exec();
}




void MainWindow::AMSimulation()
{





    additive_manufacturing_dock->ui->pushButton_26->setEnabled(false);
    additive_manufacturing_dock->ui->pushButton_27->setEnabled(false);
    additive_manufacturing_dock->ui->pushButton_28->setEnabled(false);
    additive_manufacturing_dock->ui->horizontalSlider->setEnabled(false);



    amconfig.clear();
    amconfig.am_source_v = additive_manufacturing_dock->ui->doubleSpinBox_5->value();
    amconfig.am_source_x = additive_manufacturing_dock->ui->doubleSpinBox_6->value();
    amconfig.am_source_y = additive_manufacturing_dock->ui->doubleSpinBox_7->value();
    amconfig.am_source_z = additive_manufacturing_dock->ui->doubleSpinBox_8->value();
    amconfig.am_source_h = additive_manufacturing_dock->ui->doubleSpinBox->value();
    amconfig.time = additive_manufacturing_dock->ui->doubleSpinBox_4->value();
    amconfig.time_num = additive_manufacturing_dock->ui->spinBox->value();
    amconfig.reset();


    QDir dir("./../../AM/data/vtk");
    QStringList stringlist_vtk;
    stringlist_vtk << "*.vtk";
    dir.setNameFilters(stringlist_vtk);
    QFileInfoList fileinfolist;
    fileinfolist = dir.entryInfoList();
    int files_num = fileinfolist.length();
    for (int i = 0; i < files_num; i++) {
        QFile file(fileinfolist[i].filePath());
        file.remove();
    }

    //QFile file("/home/jiping/FENGSim/AM/Elasticity/conf/geo/example7.geo");
    //file.remove();
    //file.setFileName("/home/jiping/FENGSim/AM/Elasticity/conf/geo/thinwall.geo");
    //file.copy("/home/jiping/FENGSim/AM/Elasticity/conf/geo/example7.geo");



    amtd1 = new AMThread1;
    amtd1->start();
    amtd2 = new AMThread2;
    amtd2->start();
    amtd2->timestep_num = amconfig.time_num + 1;
    amtd2->bar =  additive_manufacturing_dock->ui->progressBar;

    connect(amtd1, SIGNAL(finished()), this, SLOT(AMSimulationAnimation()));




}

void MainWindow::AMSimulationPlot()
{
    if (am_sim_num > amconfig.time_num) {
        return;
    }
    vtk_widget->AMImportSimulation(am_sim_num);
    vtk_widget->AMImportSource(am_sim_num);
    vtk_widget->RePlot();
    am_sim_num++;
    std::cout << "am_sim_num" << am_sim_num << std::endl;
    am_timer->singleShot(1, this, SLOT(AMSimulationPlot()));
}

void MainWindow::AMSimulationAnimation()
{
    am_sim_num = 0;
    additive_manufacturing_dock->ui->pushButton_26->setEnabled(true);
    additive_manufacturing_dock->ui->pushButton_27->setEnabled(true);
    additive_manufacturing_dock->ui->pushButton_28->setEnabled(true);
    additive_manufacturing_dock->ui->horizontalSlider->setEnabled(true);
    am_timer->singleShot(1, this, SLOT(AMSimulationPlot()));
}

















//        QThread* thread = new QThread;
//        // must start before movetothread
//        am_sim_timer->start(10);
//        am_sim_timer->moveToThread(thread);
//        thread->start();

//        MyThread1* myth1 = new MyThread1;
//        myth1->start();


//        QProcess *proc = new QProcess();
//        proc->setWorkingDirectory( "/home/jiping/M++/" );
//        QString command(QString("./AMRun"));
//        proc->start(command);


//        MyThread2* myth2 = new MyThread2;
//        myth2->start();

//        QThread* thread2 = new QThread(this);
//        thread2->start();
//        QProcess *proc = new QProcess();


//        proc->setWorkingDirectory( "/home/jiping/M++/" );
//        QString command(QString("./AMRun"));
//        proc->start(command);

//        proc->moveToThread(thread2);





//if (proc->waitForFinished(-1)) {
//                //                MM.FileFormatMeshToVTK("/home/jiping/software/slice2mesh-master/build-slice2mesh-Desktop_Qt_5_12_10_GCC_64bit-Debug/amslices2mesh.mesh",
//                //                                       "./data/am/mesh.vtk");
//                //                MM.FileFormatMeshToGeo("/home/jiping/software/slice2mesh-master/build-slice2mesh-Desktop_Qt_5_12_10_GCC_64bit-Debug/amslices2mesh.mesh",
//                //                                       "/home/jiping/M++/AdditiveManufacturing/conf/geo/thinwall.geo");
//                //                vtk_widget->AMImportMesh();
//                additive_manufacturing_dock->ui->pushButton_27->setChecked(true);
//    am_sim_timer->stop();
//}












// *******************************************************
// FEM



// #include "QDir"
// #include "QCoreApplication"
void MainWindow::FEMCompute()
{
    //vtk_widget->ImportVTKFile(std::string("/home/jiping/M++/data/vtk/linear_elasticity_deform_4.vtk"));

    //return;
    // cout << QDir::currentPath().toStdString().c_str() << endl;
    // cout << QCoreApplication::applicationDirPath().toStdString().c_str() << endl;
    QProcess *proc = new QProcess();
    proc->setWorkingDirectory( "../ALE/build" );
    // set equation
    //fem_dock->MainModule();
    //fem_dock->Configure();





    proc->start("./M++Solver");
    if (proc->waitForFinished(-1)) {
        vtk_widget->Hide();
        vtk_widget->ImportVTKFile(std::string("../ALE/build/data/vtk/elasticity_3_deform.vtk"));


    }
    return;







    if (fem_dock->ui->comboBox->currentText().toStdString() == "Poisson")
    {
        //proc->start("mpirun -np 4 ./PoissonRun");
    }
    else if (fem_dock->ui->comboBox->currentText().toStdString() == "Elasticity")
    {
        //proc->start("mpirun -np 4 ./ElasticityRun");
    }
    // Warning: Calling this function from the main (GUI) thread might cause your user interface to freeze.
    // If msecs is -1, this function will not time out.
    if (proc->waitForFinished(-1)) {
        vtk_widget->Hide();
        if (fem_dock->ui->comboBox->currentText().toStdString() == "Poisson")
        {
            //vtk_widget->ImportVTKFile(std::string("/home/jiping/M++/data/vtk/poisson_linear.vtk"));
        }
        else if (fem_dock->ui->comboBox->currentText().toStdString() == "Elasticity")
        {
            //vtk_widget->ImportVTKFile(std::string("/home/jiping/M++/data/vtk/linear_elasticity_3_deform.vtk"));
        }
    }
}

#include <QDir>

void MainWindow::FEMAnimation()
{
    if (fem_file_id > fem_file_num) {
        return;
    }
    vtk_widget->FEMImportResults2("/home/jiping/M++/data/vtk/heat_"+QString::number(fem_file_id)+".vtk");
    std::cout << "fem heat time step: " << "/home/jiping/M++/data/vtk/heat_"+QString::number(fem_file_id).toStdString() << std::endl;
    vtk_widget->RePlot();
    fem_file_id++;
    fem_timer->singleShot(1, this, SLOT(FEMAnimation()));
}

void MainWindow::FEMExampleCompute()
{

    vtk_widget->ImportVTKFile(std::string("/home/jiping/M++/data/vtk/linear_elasticity_deform_8.vtk"));

    return;
    // cout << QDir::currentPath().toStdString().c_str() << endl;
    // cout << QCoreApplication::applicationDirPath().toStdString().c_str() << endl;
    QProcess *proc = new QProcess();
    proc->setWorkingDirectory( "/home/jiping/M++/" );
    // set equation
    //fem_dock->MainModule();
    //fem_dock->Configure();



    if (fem_dock->ui->comboBox_4->currentText().toStdString() == "Poisson")
    {
        ofstream out;
        out.open("/home/jiping/FENGSim/Cura/conf/m++conf");
        out << "loadconf = Poisson/conf/poisson.conf;" << endl;
        out << "#loadconf = Heat/conf/heat.conf;" << endl;
        out << "#loadconf = Elasticity/conf/m++conf;" << endl;
        out << "loadconf = ElastoPlasticity/conf/m++conf;" << endl;
        out << "#loadconf = ThermoElasticity/conf/m++conf;" << endl;
        out << "#loadconf = AdditiveManufacturing/conf/m++conf;" << endl;
        out << "loadconf = Cura/conf/m++conf;" << endl;

        proc->start("mpirun -np 4 ./PoissonRun");
        if (proc->waitForFinished(-1)) {
            vtk_widget->FEMImportResults("/home/jiping/M++/data/vtk/poisson_linear.vtk");
        }
        return;
    }
    if (fem_dock->ui->comboBox_4->currentText().toStdString() == "Heat")
    {
        ofstream out;
        out.open("/home/jiping/FENGSim/Cura/conf/m++conf");
        out << "#loadconf = Poisson/conf/poisson.conf;" << endl;
        out << "loadconf = Heat/conf/heat.conf;" << endl;
        out << "#loadconf = Elasticity/conf/m++conf;" << endl;
        out << "loadconf = ElastoPlasticity/conf/m++conf;" << endl;
        out << "#loadconf = ThermoElasticity/conf/m++conf;" << endl;
        out << "#loadconf = AdditiveManufacturing/conf/m++conf;" << endl;
        out << "loadconf = Cura/conf/m++conf;" << endl;

        proc->start("mpirun -np 4 ./HeatRun");
        if (proc->waitForFinished(-1)) {
            QDir dir("/home/jiping/M++/data/vtk");
            QStringList stringlist_vtk;
            stringlist_vtk << "heat_*.vtk";
            dir.setNameFilters(stringlist_vtk);
            QFileInfoList fileinfolist;
            fileinfolist = dir.entryInfoList();
            fem_file_num = fileinfolist.size();
            fem_file_id = 0;
            fem_timer->singleShot(1, this, SLOT(FEMAnimation()));
        }
        return;
    }




    return;






    if (fem_dock->ui->comboBox->currentText().toStdString() == "Poisson")
    {
        proc->start("mpirun -np 4 ./PoissonRun");
    }
    else if (fem_dock->ui->comboBox->currentText().toStdString() == "Elasticity")
    {
        proc->start("mpirun -np 4 ./ElasticityRun");
    }
    // Warning: Calling this function from the main (GUI) thread might cause your user interface to freeze.
    // If msecs is -1, this function will not time out.
    if (proc->waitForFinished(-1)) {
        vtk_widget->Hide();
        if (fem_dock->ui->comboBox->currentText().toStdString() == "Poisson")
        {
            vtk_widget->ImportVTKFile(std::string("/home/jiping/M++/data/vtk/poisson_linear.vtk"));
        }
        else if (fem_dock->ui->comboBox->currentText().toStdString() == "Elasticity")
        {
            vtk_widget->ImportVTKFile(std::string("/home/jiping/M++/data/vtk/linear_elasticity_3_deform.vtk"));
        }
    }
}


// *******************************************************
// *******************************************************
//      Machining

void MainWindow::OpenMachiningModule()
{
    if (ui->actionMachining->isChecked())
    {
        // set open and close

        //                vtk_widget->Reset();



        ui->dockWidget->setWidget(machining_dock);
        ui->dockWidget->show();
        ui->actionCAD->setChecked(false);
        ui->actionMesh->setChecked(false);
        ui->actionSolver->setChecked(false);
        ui->actionVisual->setChecked(false);
        ui->actionMeasure->setChecked(false);
        ui->actionSPC->setChecked(false);
        ui->actionAdditiveManufacturing->setChecked(false);
        ui->actionSystem->setChecked(false);
        ui->actionMachining->setChecked(true);
        machining_part_size[0] = 10;
        machining_part_size[1] = 1;
        machining_part_size[2] = 1;
        machining_part_pos[0] = 0;
        machining_part_pos[1] = 0;
        machining_part_pos[2] = 0;
        machining_tool_size[0] = 1;
        machining_tool_size[1] = 1;
        machining_tool_size[2] = 1;
        machining_tool_pos[0] = 10;
        machining_tool_pos[1] = 0;
        machining_tool_pos[2] = 0.5;
    }
    else
    {
        ui->dockWidget->hide();
    }
}

void MainWindow::MachiningPartSet()
{
    if (machining_dock->ui->comboBox->currentIndex()==0)
        machining_dock->ui->lineEdit->setText(QString::number(machining_part_size[0]) + QString(",") +
                QString::number(machining_part_size[1]) + QString(",") +
                QString::number(machining_part_size[2]));
    if (machining_dock->ui->comboBox->currentIndex()==1)
        machining_dock->ui->lineEdit->setText(QString::number(machining_part_pos[0]) + QString(",") +
                QString::number(machining_part_pos[1]) + QString(",") +
                QString::number(machining_part_pos[2]));
}

void MainWindow::MachiningToolSet()
{
    if (machining_dock->ui->comboBox_2->currentIndex()==0)
        machining_dock->ui->lineEdit_2->setText(QString::number(machining_tool_size[0]) + QString(",") +
                QString::number(machining_tool_size[1]) + QString(",") +
                QString::number(machining_tool_size[2]));
    if (machining_dock->ui->comboBox_2->currentIndex()==1)
        machining_dock->ui->lineEdit_2->setText(QString::number(machining_tool_pos[0]) + QString(",") +
                QString::number(machining_tool_pos[1]) + QString(",") +
                QString::number(machining_tool_pos[2]));
}

void MainWindow::MachiningPartParametersUpdate()
{
    double z[3];
    if (sscanf(machining_dock->ui->lineEdit->text().toStdString().data(), "%lf,%lf,%lf", z, z + 1, z + 2)==3)
    {
        if (machining_dock->ui->comboBox->currentIndex()==0)
        {
            machining_part_size[0] = z[0];
            machining_part_size[1] = z[1];
            machining_part_size[2] = z[2];
        }
        if (machining_dock->ui->comboBox->currentIndex()==1)
        {
            machining_part_pos[0] = z[0];
            machining_part_pos[1] = z[1];
            machining_part_pos[2] = z[2];
        }
    }
}

void MainWindow::MachiningToolParametersUpdate()
{
    double z[3];
    if (sscanf(machining_dock->ui->lineEdit_2->text().toStdString().data(), "%lf,%lf,%lf", z, z + 1, z + 2)==3)
    {
        if (machining_dock->ui->comboBox_2->currentIndex()==0)
        {
            machining_tool_size[0] = z[0];
            machining_tool_size[1] = z[1];
            machining_tool_size[2] = z[2];
        }
        if (machining_dock->ui->comboBox_2->currentIndex()==1)
        {
            machining_tool_pos[0] = z[0];
            machining_tool_pos[1] = z[1];
            machining_tool_pos[2] = z[2];
        }
    }
}

void MainWindow::MachiningMakePart()
{
    TopoDS_Shape* S = new TopoDS_Shape(BRepPrimAPI_MakeBox(gp_Ax2(gp_Pnt(machining_part_pos[0],machining_part_pos[1],machining_part_pos[2]),gp_Dir(0,0,1)),machining_part_size[0],machining_part_size[1],machining_part_size[2]).Shape());
    machining_part = new Cube(S);
    machining_part->SetData(machining_part_size[0],machining_part_size[1],machining_part_size[2],
            machining_part_pos[0],machining_part_pos[1],machining_part_pos[2],
            0,0,1);
    vtk_widget->MachiningPartPlot(*(machining_part->Value()));
    machining_bnds->Clear();
    machining_bnds->Add(*(machining_part->Value()));
    machining_part_done = true;
}

void MainWindow::MachiningMakeTool()
{
    if (machining_dock->ui->comboBox->currentIndex()==2)
    {
        STEPControl_Writer writer;
        writer.Transfer(MakeTools(20,1.5),STEPControl_ManifoldSolidBrep);
        writer.Write("/home/jiping/tool.stp");
        //vtk_widget->Plot(MakeTools(20,1.5));
        vtk_widget->MachiningToolPlot(MakeTools(20,1.5));
    }
    else {
        TopoDS_Shape* S = new TopoDS_Shape(BRepPrimAPI_MakeBox(gp_Ax2(gp_Pnt(machining_tool_pos[0],machining_tool_pos[1],machining_tool_pos[2]),gp_Dir(0,0,1)),machining_tool_size[0],machining_tool_size[1],machining_tool_size[2]).Shape());
        machining_tool = new Cube(S);
        machining_tool->SetData(machining_tool_size[0], machining_tool_size[1], machining_tool_size[2],
                machining_tool_pos[0], machining_tool_pos[1], machining_tool_pos[2],
                0,0,1);
        vtk_widget->MachiningToolPlot(*(machining_tool->Value()));
        machining_tool_done = true;
    }
}

void MainWindow::MachiningSetDomainData()
{
    if (machining_dock->ui->pushButton_11->isChecked())
    {
        std::cout << machining_part << " " << machining_tool << std::endl;
        if (machining_part_done == false || machining_tool_done == false) return;
        vtk_widget->SetSelectable(true);
        vtk_widget->SetSelectDomain(true);
        vtk_widget->Clear();
        vtk_widget->MachiningPartPlot(*(machining_part->Value()));
        vtk_widget->MachiningToolPlot(*(machining_tool->Value()));
    }
    else
    {
        vtk_widget->SetSelectable(false);
        vtk_widget->SetSelectDomain(false);
    }

}



void MainWindow::MachiningSetBoundaryData()
{
    if (machining_dock->ui->pushButton_10->isChecked())
    {
        vtk_widget->SetSelectable(true);
        vtk_widget->SetSelectBnd(true);
        vtk_widget->machining_set_part_bnds(machining_bnds);
        vtk_widget->Clear();
        vtk_widget->MachiningPlotPartBnds();
        vtk_widget->MachiningToolPlot(*(machining_tool->Value()));
    }
    else
    {
        vtk_widget->SetSelectable(false);
        vtk_widget->SetSelectBnd(false);
    }
}


void MainWindow::MachiningMeshGeneration()
{
    MM.MeshGeneration(machining_part->Value(),machining_dock->ui->doubleSpinBox->value(),0);
    vtk_widget->Clear();
    vtk_widget->ImportVTKFile(std::string("./data/mesh/fengsim_mesh.vtk"));
    vtk_widget->MachiningToolPlot(*(machining_tool->Value()));
}


#include "Machining/MachiningThread1.h"
#include "Machining/MachiningThread2.h"


void MainWindow::MachiningSimulation()
{







    if (machining_dock->ui->progressBar->value()<100) {

        machining_ani_num = 0;
        machining_dock->ui->progressBar->setValue(0);
        QDir dir("/home/jiping/OpenDT/M++/data/vtk");
        QStringList stringlist_vtk;
        stringlist_vtk << "*.vtk";
        dir.setNameFilters(stringlist_vtk);
        QFileInfoList fileinfolist;
        fileinfolist = dir.entryInfoList();
        int files_num = fileinfolist.length();
        for (int i = 0; i < files_num; i++) {
            QFile file(fileinfolist[i].filePath());
            file.remove();
        }
        QFile file("/home/jiping/OpenDT/M++/Machining/conf/geo/mesh_3.geo");
        file.remove();
        file.setFileName("/home/jiping/OpenDT/M++/Machining/conf/geo/machining.geo");
        file.copy("/home/jiping/OpenDT/M++/Machining/conf/geo/mesh_3.geo");



        MachiningThread1* td1 = new MachiningThread1;
        td1->start();
        MachiningThread2* td2 = new MachiningThread2;
        td2->start();
        td2->bar =  machining_dock->ui->progressBar;
        td2->timestep_num = machining_dock->ui->spinBox->value();
        connect(td1, SIGNAL(finished()), this, SLOT(MachiningSimulationAnimation()));
    }
    else {
        MachiningSimulationAnimation();
    }



}


void MainWindow::MachiningSimulationPlot()
{
    if (machining_ani_num > machining_dock->ui->spinBox->value()) {
        return;
    }
    vtk_widget->Clear();
    vtk_widget->MachiningImportPart(machining_ani_num);
    vtk_widget->MachiningImportTool(machining_ani_num);
    vtk_widget->RePlot();
    machining_ani_num++;
    std::cout << "machining_sim_num" << machining_ani_num << std::endl;
    machining_timer->singleShot(1, this, SLOT(MachiningSimulationPlot()));
}

void MainWindow::MachiningSimulationAnimation()
{
    machining_ani_num = 0;
    machining_timer->singleShot(1, this, SLOT(MachiningSimulationPlot()));
}
#include <QDirIterator>
void MainWindow::MachiningRecompute()
{
    machining_ani_num = 0;
    machining_dock->ui->progressBar->setValue(0);


    QDir dir("/home/jiping/OpenDT/M++/data/vtk");
    QStringList stringlist_vtk;
    stringlist_vtk << "*.vtk";
    dir.setNameFilters(stringlist_vtk);
    QFileInfoList fileinfolist;
    fileinfolist = dir.entryInfoList();
    int files_num = fileinfolist.length();
    for (int i = 0; i < files_num; i++) {
        QFile file(fileinfolist[i].filePath());
        file.remove();
    }

    QFile file("/home/jiping/OpenDT/M++/Machining/conf/geo/mesh_3.geo");
    file.remove();
    file.setFileName("/home/jiping/OpenDT/M++/Machining/conf/geo/machining.geo");
    file.copy("/home/jiping/OpenDT/M++/Machining/conf/geo/mesh_3.geo");


    //        //std::cout << QProcess::execute("rm -rf /home/jiping/FENGSim/M++/data/vtk/*.vtk") << std::endl;
    //        QString filePath = "/home/jiping/FENGSim/M++/data/vtk/*.vtk";
    //        QFile file(filePath);
    //        file.remove();


    //        QDir Dir("/home/jiping/FENGSim/M++/data/vtk/");
    //        if(Dir.isEmpty())
    //        {
    //                return;
    //        }
    //        QDirIterator DirsIterator("/home/jiping/FENGSim/M++/data/vtk/");
    //        while(DirsIterator.hasNext())
    //        {
    //                Dir.remove(DirsIterator.next());
    //        }




    //        QProcess::execute("cp /home/jiping/FENGSim/M++/Machining/conf/geo/machining.geo /home/jiping/FENGSim/M++/Machining/conf/geo/mesh_3.geo");
    /*
        QProcess *proc = new QProcess();
        QString command(QString("rm -rf data/vtk/*.*"));
        proc->setWorkingDirectory("/home/jiping/M++/");
        proc->start(command);*/


}


// =======================================================================
//
// transport module
//
// =======================================================================
#include "Transport/data_analyze.h"
void MainWindow::OpenTransportModule()
{




    if (ui->actionTransport->isChecked())
    {
        vtk_widget->SetSelectable(false);
        vtk_widget->SetSelectDomain(false);
        vtk_widget->Reset();
        // cout << parts->size() << endl;
        // OCCw->Clear();
        // OCCw->SetMachiningModule(false);
        // OCCw->Fit();
        ui->dockWidget->setWidget(transport_dock);
        ui->dockWidget->show();
        // set open and close
        ui->actionCAD->setChecked(true);
        ui->actionMesh->setChecked(false);
        ui->actionSolver->setChecked(false);
        ui->actionVisual->setChecked(false);
        ui->actionMeasure->setChecked(false);

        transport_dock->ui->tableWidget->setRowCount(2);
        transport_dock->ui->tableWidget->setRowHeight(0,30);
        transport_dock->ui->tableWidget->setRowHeight(1,30);
        transport_dock->ui->tableWidget_2->setRowCount(9);
        transport_dock->ui->tableWidget_2->setRowHeight(0,30);
        transport_dock->ui->tableWidget_2->setRowHeight(1,30);
        transport_dock->ui->tableWidget_2->setRowHeight(2,30);
        transport_dock->ui->tableWidget_2->setRowHeight(3,30);
        transport_dock->ui->tableWidget_2->setRowHeight(4,30);
        transport_dock->ui->tableWidget_2->setRowHeight(5,30);
        transport_dock->ui->tableWidget_2->setRowHeight(6,30);
        transport_dock->ui->tableWidget_2->setRowHeight(7,30);
        transport_dock->ui->tableWidget_2->setRowHeight(8,30);


    }
    else
    {
        ui->dockWidget->hide();
    }
}

void MainWindow::TransportCADSave()
{
    QString fileName = QFileDialog::getSaveFileName(this,tr("Save File"),
                                                    "/home/jiping/FENGSim/FENGSim/Transport/data/output.stp",
                                                    tr("CAD Files (*.stp)"), 0 , QFileDialog::DontUseNativeDialog);
    TopoDS_Compound aRes;
    BRep_Builder aBuilder;
    aBuilder.MakeCompound (aRes);
    //for (std::list<TopoDS_Shape*>::iterator it = myCompound.begin(); it!=myCompound.end(); it++) {
    //for (std::list<Prim*>::iterator it = myCompound.begin(); it!=myCompound.end(); it++) {
    for (int i=0; i<transport_parts->size(); i++)
    {
        aBuilder.Add(aRes,*((*transport_parts)[i]->Value()));
    }
    STEPControl_Writer writer;
    writer.Transfer(aRes,STEPControl_ManifoldSolidBrep);
    char* ch;
    QByteArray ba = fileName.toLatin1();
    ch=ba.data();
    writer.Write(ch);
}

void MainWindow::TransportMakeSphere (double r,
                                      double p1, double p2, double p3,
                                      double d1, double d2, double d3)
{
    TopoDS_Shape* S = new TopoDS_Shape(BRepPrimAPI_MakeSphere(gp_Ax2(gp_Pnt(p1,p2,p3),gp_Dir(d1,d2,d3)),r).Shape());
    Sphere* A = new Sphere(S);
    A->SetData(r,p1,p2,p3,d1,d2,d3);
    //Plot(A->Value());
    transport_parts->Add(A);
    vtk_widget->TransportPlot(*(A->Value()),true,6);
    cout << "geometry num: "  << transport_parts->size() << endl;
}

void MainWindow::TransportMakeCylinder (double r, double h,
                                        double p1, double p2, double p3,
                                        double d1, double d2, double d3, int color)
{
    TopoDS_Shape* S = new TopoDS_Shape(BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(p1,p2,p3),gp_Dir(d1,d2,d3)),r,h).Shape());
    Cylinder* A = new Cylinder(S);
    A->SetData(r,h,p1,p2,p3,d1,d2,d3);
    //Plot(A->Value());
    transport_parts->Add(A);
    vtk_widget->TransportPlot(*(A->Value()),true,color);
    cout << "geometry num: "  << transport_parts->size() << endl;

}

void MainWindow::TransportMakeRing ()
{
    TopoDS_Shape* S1 = new TopoDS_Shape(BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0,0,-140*transport_level/2),gp_Dir(0,0,1)),200,140*transport_level).Shape());
    TopoDS_Shape* S2 = new TopoDS_Shape(BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0,0,-140*transport_level/2),gp_Dir(0,0,1)),260.4,140*transport_level).Shape());
    TopoDS_Shape* S = new TopoDS_Shape(BRepAlgoAPI_Cut(*S2,*S1).Shape());
    General* A = new General(S);
    transport_parts->Add(A);
    vtk_widget->TransportPlot(*(A->Value()),false,0);

}

void MainWindow::TransportMakeDetectors()
{
    for (double j=0; j<transport_level; j++) {
        TopoDS_Shape* S = new TopoDS_Shape(BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt((200+260.4)/2.0-50.8/2.0,0,70+j*140-140*transport_level/2),gp_Dir(1,0,0)),40,50.8).Shape());
        Cylinder* A = new Cylinder(S);
        transport_parts->Add(A);
        vtk_widget->TransportPlot(*(A->Value()),true,8);

        for (double i = 1; i < transport_num; i++) {
            double angle = i * 360.0 / transport_num / 360 * 2 * 3.1415926;
            std::cout << angle << std::endl;
            gp_Trsf t;
            //t.SetMirror(gp_Ax2(gp_Pnt(0,0,0),gp_Dir(1,0,0),gp_Dir(0,cos(angle/2.0),sin(angle/2.0))));
            t.SetRotation(gp_Ax1(gp_Pnt(0,0,0),gp_Dir(0,0,1)),angle);
            TopoDS_Shape* SS = new TopoDS_Shape(BRepBuilderAPI_Transform(*S,t));
            General* AA = new General(SS);
            transport_parts->Add(AA);
            vtk_widget->TransportPlot(*(AA->Value()),true,8);
        }
    }
}

void MainWindow::TransportParaModel ()
{

    transport_parts->Clear();
    vtk_widget->Clear();

    //transport_dock->ui->tableWidget->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    //transport_dock->ui->tableWidget->verticalHeader()->setSectionResizeMode(QHeaderView::Stretch);

    transport_level = transport_dock->ui->tableWidget->item(0,0)->text().toInt();
    transport_num = transport_dock->ui->tableWidget->item(1,0)->text().toInt();
    std::cout << transport_level << std::endl;
    std::cout << transport_num << std::endl;
    TransportMakeRing();
    TransportMakeDetectors();
    TransportMakeSphere(20,0,0,0,1,0,0);

    vtk_widget->SetPrims(transport_parts);


    //        TransportMakeCylinder (40, 50.8,
    //                               0, 0, 0,
    //                               1, 1, 1,8);

    //        TopoDS_Shape* S = new TopoDS_Shape(BRepPrimAPI_MakeRevol(*(vtk_widget->GetSelectedPrim()->Value()),
    //                                                                 gp_Ax1(gp_Pnt(p[0],p[1],p[2]),gp_Dir(d[0],d[1],d[2])),
    //                        angle/360*2*3.1415926).Shape());
    //        General* A = new General(S);



    //        G4double LiqSci_rmin =  0*mm, LiqSci_rmax = 40*mm;//探测器半径
    //        G4double LiqSci_hz = 50.8*mm;//厚度
    //        G4double LiqSci_phimin = 0.*deg, LiqSci_phimax = 360.*deg;
    //        G4int nb_cryst = 8;//每层八个
    //        G4int nb_rings = 3;//三层
    //        G4double dPhi = twopi/nb_cryst;//旋转角度
    //        G4double ring_R1= 200*mm;//内半径
    //        G4double ring_R2= 260.4*mm;//外半径
    //        G4double ring_hz= 140.*mm;//层高
    //        G4double detector_dZ = nb_rings*ring_hz;//总高

}

void MainWindow::TransportParaModelPlot ()
{
    int n = 0;
    vtk_widget->Clear();
    vtk_widget->TransportPlot(*((*transport_parts)[n]->Value()),false,0);
    n++;
    std::cout << transport_level << " " << transport_num << " " << transport_parts->size() << std::endl;
    for (double j=0; j<transport_level; j++) {
        for (double i = 0; i < transport_num; i++) {
            vtk_widget->TransportPlot(*((*transport_parts)[n]->Value()),false,8);
            n++;
        }
    }
    vtk_widget->TransportPlot(*((*transport_parts)[n]->Value()),false,6);
}

void MainWindow::TransportSelect()
{
    if (transport_dock->ui->pushButton_6->isChecked()) {
        vtk_widget->SetSelectable(true);
        vtk_widget->SetSelectDomain(true);
        return;
    }
    else if (!transport_dock->ui->pushButton_6->isChecked()) {
        vtk_widget->SetSelectable(false);
        vtk_widget->SetSelectDomain(false);
        TransportParaModelPlot ();
        return;
    }
}

#include "Transport/B1/include/TransportB1.h"

void MainWindow::TransportMCRun()
{


    //    TransportB1 B1;
    //    B1.Run();
    //    return;

    QFile::remove("/home/jiping/OpenDT/MPU1.2/output.vtk");
    QFile::remove("/home/jiping/OpenDT/MPU1.2/output2.vtk");
    QProcess *proc = new QProcess();
    proc->setWorkingDirectory("/home/jiping/OpenDT/MPU1.2/build");
    std::cout << proc->workingDirectory().toStdString() << std::endl;
    proc->start("./exampleTOF");







    if (proc->waitForFinished(-1)) {

        std::cout << "transport check" << std::endl;

        std::ifstream is("/home/jiping/OpenDT/MPU1.2/output.vtk");
        std::ofstream out("/home/jiping/OpenDT/MPU1.2/output2.vtk");

        const int len = 256;
        char L[len];
        int n = 0;
        while(is.getline(L,len)) n++;
        is.close();

        is.open("/home/jiping/OpenDT/MPU1.2/output.vtk");
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
        vtk_widget->TransportImportVTKFile("/home/jiping/OpenDT/MPU1.2/output2.vtk",9);



//        data_analyze results;
//        std::vector<double> results_show(results.analyze());

//        std::cout << results_show.size() << std::endl;
//        for (int i=0; i<results_show.size(); i++)
//            std::cout << results_show[i] << " ";
//        std::cout << std::endl;
//        for (int i=0; i<9; i++)
//            transport_dock->ui->tableWidget_2->item(i,0)->setText(QString::number(results_show[i]));

        return;

    }
}


void MainWindow::ImportVTKFile()
{
    QString stl_file_name =  QFileDialog::getOpenFileName(0,"Open VTK Files",
                                                          QString("./home/jiping/"),
                                                          "VTK files (*.vtk);;", 0 , QFileDialog::DontUseNativeDialog);
    vtk_widget->Clear();
    vtk_widget->ImportVTKFile(stl_file_name.toStdString());
}


void MainWindow::OCPoroImportVTKFile()
{
    ocporofilename =  QFileDialog::getOpenFileName(0,"Open VTK Files",
                                                   QString("./home/jiping/"),
                                                   "VTK files (*.vtk);;", 0 , QFileDialog::DontUseNativeDialog);
    vtk_widget->Clear();
    ocporo_dock->ui->comboBox->clear();
    attnum = vtk_widget->OCPoroImportVTKFile(ocporofilename.toStdString());
    for (int i=0; i<attnum; i++) {
        ocporo_dock->ui->comboBox->addItem(QString::number(i));
    }
}

void MainWindow::OCPoroSwitchAtt() {
    //std::cout << attnum << std::endl;
    if (attnum!=0) {
        vtk_widget->OCPoroImportVTKFile(ocporofilename.toStdString(),
                                        ocporo_dock->ui->comboBox->currentIndex());
        //std::cout << ocporo_dock->ui->comboBox->currentIndex() << std::endl;
    }
}
#include "qcustomplot.h"
#include "QVector"
void MainWindow::OCPoroImportSummary()
{
    QString fileName =  QFileDialog::getOpenFileName(0,"Open OUT Files",
                                                     QString("./home/jiping/"),
                                                     "OUT files (*.out);;", 0 , QFileDialog::DontUseNativeDialog);


    ifstream is;
    is.open(fileName.toStdString());
    const int len = 256;
    char L[len];
    is.getline(L,len);
    is.getline(L,len);
    is.getline(L,len);
    is.getline(L,len);
    bool stop = true;
    while (stop)
    {
        is.getline(L,len);
        double x[9];
        int d = sscanf(L,"%lf %lf %lf %lf %lf %lf %lf %lf %lf", x, x+1, x+2
                       , x+3, x+4, x+5, x+6, x+7, x+8);
        if (d<9) stop = false;
        else {
            std::vector<double> vv;
            for (int i=0; i<9; i++)
                vv.push_back(x[i]);
            ocporosummarydata.push_back(vv);
        }
    }

    QVector<double> x(ocporosummarydata.size());
    QVector<double> y(ocporosummarydata.size());
    QVector<double> y1(ocporosummarydata.size());
    QVector<double> y2(ocporosummarydata.size());
    for (int i=0; i<ocporosummarydata.size(); i++) {
        x[i] = ocporosummarydata[i][0];
        y[i] = ocporosummarydata[i][1];
        y1[i] = ocporosummarydata[i][2];
        y2[i] = ocporosummarydata[i][3];
        std::cout << x[i] << " " << y[i] << std::endl;
    }
    ocporosummary->ui->customplot->addGraph();
    ocporosummary->ui->customplot->graph(0)->setData(x,y);
    ocporosummary->ui->customplot->graph(0)->rescaleAxes();
    ocporosummary->show();



    ocporosummary1->ui->customplot->addGraph();
    ocporosummary1->ui->customplot->graph(0)->setData(x,y1);
    ocporosummary1->ui->customplot->graph(0)->rescaleAxes();
    ocporosummary1->show();

    ocporosummary2->ui->customplot->addGraph();
    ocporosummary2->ui->customplot->graph(0)->setData(x,y2);
    ocporosummary2->ui->customplot->graph(0)->rescaleAxes();
    ocporosummary2->show();


}
void MainWindow::OpenOCPoroModule()
{
    if (ui->actionOCPoro->isChecked())
    {
        vtk_widget->SetSelectable(false);
        vtk_widget->SetSelectDomain(false);
        vtk_widget->Reset();
        // cout << parts->size() << endl;
        // OCCw->Clear();
        // OCCw->SetMachiningModule(false);
        // OCCw->Fit();
        ui->dockWidget->setWidget(ocporo_dock);
        ui->dockWidget->show();
        // set open and close
        ui->actionCAD->setChecked(false);
        ui->actionMesh->setChecked(false);
        ui->actionSolver->setChecked(false);
        ui->actionVisual->setChecked(false);
        ui->actionMeasure->setChecked(false);
    }
    else
    {
        ui->dockWidget->hide();
    }
}
