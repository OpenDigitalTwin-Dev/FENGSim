#include "PipeDockWidget.h"
#include "ui_PipeDockWidget.h"

PipeDockWidget::PipeDockWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::PipeDockWidget)
{
    ui->setupUi(this);
}

PipeDockWidget::~PipeDockWidget()
{
    delete ui;
}
