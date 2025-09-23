#include "RobotDockWidget.h"
#include "ui_RobotDockWidget.h"

RobotDockWidget::RobotDockWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::RobotDockWidget)
{
    ui->setupUi(this);
}

RobotDockWidget::~RobotDockWidget()
{
    delete ui;
}
