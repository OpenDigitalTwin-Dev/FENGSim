#include "TransportDockWidget.h"
#include "ui_TransportDockWidget.h"

TransportDockWidget::TransportDockWidget(QWidget *parent) :
        QWidget(parent),
        ui(new Ui::TransportDockWidget)
{
        ui->setupUi(this);
}

TransportDockWidget::~TransportDockWidget()
{
        delete ui;
}
