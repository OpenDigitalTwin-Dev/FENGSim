#include "RivetDockWidget.h"
#include "ui_RivetDockWidget.h"

RivetDockWidget::RivetDockWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::RivetDockWidget)
{
    ui->setupUi(this);
}

RivetDockWidget::~RivetDockWidget()
{
    delete ui;
}
