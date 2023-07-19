#include "FEMWindow.h"
#include "ui_FEMWindow.h"

#include <QProcess>

#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>

#include <vtkGenericDataObjectReader.h>
#include <vtkStructuredGridReader.h>
#include <vtkUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <vtkStructuredGridGeometryFilter.h>
#include <vtkUnstructuredGridGeometryFilter.h>
#include <vtkDataSetMapper.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkLookupTable.h>
#include <vtkLight.h>
#include <vtkCamera.h>
#include <vtkProperty.h>

#include <QDesktopWidget>

//#include "dolfin.h"

FEMWindow::FEMWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::FEMWindow)
{
    ui->setupUi(this);

    qvtk = new QVTKWidget;
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->SetBackground(.3, .3, .3); // Background color dark blue
    qvtk->GetRenderWindow()->AddRenderer(renderer);
    setCentralWidget(qvtk);

    ui->tableWidget->verticalHeader()->hide();
    ui->tableWidget->setColumnCount(2);
    //ui->tableWidget->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    QStringList header;
    header<<"Parameter"<<"Value";
    ui->tableWidget->setHorizontalHeaderLabels(header);
    ui->tableWidget->setRowCount(5);
    ui->tableWidget->setItem(0,0,new QTableWidgetItem("Element"));
    ui->tableWidget->setItem(0,1,new QTableWidgetItem("Tet"));
    ui->tableWidget->setItem(1,0,new QTableWidgetItem("Order"));
    ui->tableWidget->setItem(1,1,new QTableWidgetItem(QString::number(1)));
    ui->tableWidget->setItem(2,0,new QTableWidgetItem("Iterative Solver"));
    ui->tableWidget->setItem(2,1,new QTableWidgetItem("CG"));
    ui->tableWidget->setItem(3,0,new QTableWidgetItem("PreConditioner"));
    ui->tableWidget->setItem(3,1,new QTableWidgetItem("Jacobi"));
    ui->tableWidget->setItem(4,0,new QTableWidgetItem("Parallel"));
    ui->tableWidget->setItem(4,1,new QTableWidgetItem(QString::number(2)));

    connect(ui->pushButton,SIGNAL(pressed()),this,SLOT(Solve()));

    //this->showMaximized();

    ui->menubar->hide();
 //   this->show();
//    QDesktopWidget *desk=QApplication::desktop();
//    int wd = desk->width();
//    int ht = desk->height();
//    this->move(0,ht/2-100);
}

FEMWindow::~FEMWindow()
{
    delete ui;
}

void FEMWindow::Solve() {
    QProcess *proc = new QProcess();
    proc->start("mpirun -np 2 M++");

    if (proc->waitForFinished()) {
        vtkSmartPointer<vtkUnstructuredGridReader> reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
        reader->SetFileName(std::string("/home/jiping/FEngSIM/build-FEngSIM-Desktop_Qt_5_6_2_GCC_64bit-Debug/data/vtk/out_put.vtk").c_str());
        reader->Update();

        vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
        mapper->SetInputConnection(reader->GetOutputPort());

        vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();

        lut->SetHueRange(0.666667,0.0);
        double * range = reader->GetOutput()->GetScalarRange();

        lut->Build();
        mapper->SetScalarRange(range);
        mapper->SetLookupTable(lut);
        mapper->Update();

        vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        actor->GetProperty()->FrontfaceCullingOn();

        vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();

        renderer->AddActor(actor);
        renderer->SetBackground(.3, .3, .3); // Background color dark blue
        renderer->ResetCamera();

        qvtk->GetRenderWindow()->AddRenderer(renderer);
        qvtk->update();
    }
}

void FEMWindow::SolveDolfin() {

}
