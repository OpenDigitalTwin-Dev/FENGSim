#include "MeshDockWidget.h"
#include "ui_MeshDockWidget.h"

MeshDockWidget::MeshDockWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MeshDockWidget)
{
    ui->setupUi(this);
}

MeshDockWidget::~MeshDockWidget()
{
    delete ui;
}
