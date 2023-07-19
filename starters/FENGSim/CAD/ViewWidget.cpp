#include "ViewWidget.h"
#include "ui_ViewWidget.h"

ViewWidget::ViewWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ViewWidget)
{
    ui->setupUi(this);
}

ViewWidget::~ViewWidget()
{
    delete ui;
}
