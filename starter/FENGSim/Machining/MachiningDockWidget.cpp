#include "MachiningDockWidget.h"
#include "ui_MachiningDockWidget.h"

MachiningDockWidget::MachiningDockWidget(QWidget *parent) :
        QWidget(parent),
        ui(new Ui::MachiningDockWidget)
{
        ui->setupUi(this);
        //ui->comboBox_5->view()->setFixedWidth(200);
        //ui->comboBox_6->view()->setFixedWidth(150);

}

MachiningDockWidget::~MachiningDockWidget()
{
        delete ui;
}
