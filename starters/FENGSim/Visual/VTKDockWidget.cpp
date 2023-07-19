#include "VTKDockWidget.h"
#include "ui_VTKDockWidget.h"

VTKDockWidget::VTKDockWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::VTKDockWidget)
{
    ui->setupUi(this);
}

VTKDockWidget::~VTKDockWidget()
{
    delete ui;
}

