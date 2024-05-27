#include "FEMDockWidget.h"
#include "ui_FEMDockWidget.h"

FEMDockWidget::FEMDockWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::FEMDockWidget)
{
    ui->setupUi(this);
    //ui->comboBox->addItem("Poisson");
    //ui->comboBox->addItem("Heat");
    ui->comboBox->addItem("Elasticity");
    ui->comboBox->addItem("Dynamic Elasticity");
    ui->comboBox->addItem("Dynamic Elastoplasticity");
    //ui->comboBox_4->addItem("Poisson");
    //ui->comboBox_4->addItem("Heat");
    //ui->comboBox_4->addItem("Elasticity");
    //ui->comboBox_4->addItem("ElastoPlasticity");
    //connect(ui->lineEdit, SIGNAL(selectionChanged()), this, SLOT(OpenMeshFile()));
}

FEMDockWidget::~FEMDockWidget()
{
    delete ui;
}

#include <iostream>
#include <fstream>
#include <QFileDialog>
#include <QFileInfo>

void FEMDockWidget::OpenMeshFile()
{
    QString filename =  QFileDialog::getOpenFileName(0,"Open Mesh",
                                                     QString("/home/jiping/M++"),
                                                     "Mesh files (*.geo);;", 0 , QFileDialog::DontUseNativeDialog);
    QFileInfo fi = QFileInfo(filename);
    //ui->lineEdit->setText(fi.baseName());
    mesh_file = fi.baseName();
}

void FEMDockWidget::MainModule()
{
    std::ofstream out;
    out.open(std::string("/home/jiping/M++/conf/m++conf").c_str());
    if (ui->comboBox->currentText().toStdString() == "Poisson")
    {
        out << "loadconf = PoissonModule/conf/m++conf;" << std::endl;
    }
    else if (ui->comboBox->currentText().toStdString() == "Elasticity")
    {
        out << "loadconf = ElasticityModule/conf/m++conf;" << std::endl;
    }
    out.close();
}

void FEMDockWidget::Configure()
{
    std::ofstream out;
    if (ui->comboBox->currentText().toStdString() == "Poisson")
    {
        out.open(std::string("/home/jiping/M++/PoissonModule/conf/poisson_qt.conf").c_str());
        //out << "Model = " + ui->comboBox_2->currentText().toStdString() + ";" << std::endl;
        out << "Model = NonLinear;" << std::endl;
        out << "GeoPath = PoissonModule/;" << std::endl;

        //out << "EXAMPLE = " + ui->comboBox_4->currentText().toStdString() + "; " << std::endl;
        //out << "Mesh = example" + ui->comboBox_4->currentText().toStdString() + ";" << std::endl;
        out << "plevel = 0;" << std::endl;
        //out << "level = " + ui->comboBox_3->currentText().toStdString() + ";" << std::endl;
        out << "Discretization = linear;" << std::endl;
        out << "Overlap_Distribution = 0;" << std::endl;
        out << "Overlap = none;" << std::endl;
        out << "Distribution = Stripes;" << std::endl;

        out << "NewtonSteps = 10;" << std::endl;
        out << "NewtonResidual = 1e-20;" << std::endl;

        out << "LinearSolver = CG;" << std::endl;
        out << "Preconditioner = Jacobi;" << std::endl;
        out << "LinearSteps = 50000;" << std::endl;
        out << "LinearEpsilon = 1e-15;" << std::endl;
        out << "LinearReduction = 1e-15;" << std::endl;

        out << "QuadratureCell = 3;" << std::endl;
        out << "QuadratureBoundary = 2;" << std::endl;

        out << "precision = 5;" << std::endl;
    }
    else if (ui->comboBox->currentText().toStdString() == "Elasticity")
    {
        out.open(std::string("../Elasticity/build/solver/conf/m++conf").c_str());
        out << "loadconf = solver/conf/elasticity.conf;" << std::endl;
        out.close();
    }
    else if (ui->comboBox->currentText().toStdString() == "Dynamic Elasticity")
    {
        out.open(std::string("../Elasticity/build/solver/conf/m++conf").c_str());
        out << "loadconf = solver/conf/telasticity.conf;" << std::endl;
        out.close();
    }
    else if (ui->comboBox->currentText().toStdString() == "Dynamic Elastoplasticity")
    {
        out.open(std::string("../Elasticity/build/solver/conf/m++conf").c_str());
        out << "loadconf = solver/conf/telastoplasticity.conf;" << std::endl;
        out.close();
    }
}

