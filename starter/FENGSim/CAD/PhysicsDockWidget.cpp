#include "PhysicsDockWidget.h"
#include "ui_PhysicsDockWidget.h"

PhysicsDockWidget::PhysicsDockWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::PhysicsDockWidget)
{
    ui->setupUi(this);
}

PhysicsDockWidget::~PhysicsDockWidget()
{
    delete ui;
}
