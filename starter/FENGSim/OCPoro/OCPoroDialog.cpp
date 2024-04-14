#include "OCPoroDialog.h"
#include "ui_OCPoroDialog.h"

OCPoroDialog::OCPoroDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::OCPoroDialog)
{
    ui->setupUi(this);
}

OCPoroDialog::~OCPoroDialog()
{
    delete ui;
}
