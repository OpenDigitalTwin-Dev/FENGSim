#include "MachiningDockWidget2.h"
#include "ui_MachiningDockWidget2.h"

MachiningDockWidget2::MachiningDockWidget2(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MachiningDockWidget2)
{
    ui->setupUi(this);
}

MachiningDockWidget2::~MachiningDockWidget2()
{
    delete ui;
}
