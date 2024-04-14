#include "OCPoroDockWidget.h"
#include "ui_OCPoroDockWidget.h"

OCPoroDockWidget::OCPoroDockWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::OCPoroDockWidget)
{
    ui->setupUi(this);
}

OCPoroDockWidget::~OCPoroDockWidget()
{
    delete ui;
}
