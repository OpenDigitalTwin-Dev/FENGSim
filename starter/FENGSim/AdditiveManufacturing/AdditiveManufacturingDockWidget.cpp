#include "AdditiveManufacturingDockWidget.h"
#include "ui_AdditiveManufacturingDockWidget.h"

AdditiveManufacturingDockWidget::AdditiveManufacturingDockWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::AdditiveManufacturingDockWidget)
{
    ui->setupUi(this);
    ui->toolBox->setVisible(false);


    ui->tableWidget->setRowCount(6);
    ui->tableWidget->setColumnCount(2);/*
    ui->tableWidget->setColumnWidth(0,20);
    ui->tableWidget->setColumnWidth(0,50);*/
    ui->tableWidget->setRowHeight(0,30);
    ui->tableWidget->setRowHeight(1,30);
    ui->tableWidget->setRowHeight(2,30);
    ui->tableWidget->setRowHeight(3,30);
    ui->tableWidget->setRowHeight(4,30);
    ui->tableWidget->setRowHeight(5,30);
    ui->tableWidget->setCellWidget(0, 0, ui->label_15);
    ui->tableWidget->setCellWidget(0, 1, ui->doubleSpinBox_5);
    ui->tableWidget->setCellWidget(1, 0, ui->label_16);
    ui->tableWidget->setCellWidget(1, 1, ui->doubleSpinBox_6);
    ui->tableWidget->setCellWidget(2, 0, ui->label_17);
    ui->tableWidget->setCellWidget(2, 1, ui->doubleSpinBox_7);
    ui->tableWidget->setCellWidget(3, 0, ui->label_18);
    ui->tableWidget->setCellWidget(3, 1, ui->doubleSpinBox_8);
    ui->tableWidget->setCellWidget(4, 0, ui->label_13);
    ui->tableWidget->setCellWidget(4, 1, ui->doubleSpinBox_4);
    ui->tableWidget->setCellWidget(5, 0, ui->label_14);
    ui->tableWidget->setCellWidget(5, 1, ui->spinBox);
    ui->tableWidget->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    ui->tableWidget->verticalHeader()->setSectionResizeMode(QHeaderView::Stretch);

}

AdditiveManufacturingDockWidget::~AdditiveManufacturingDockWidget()
{
    delete ui;
}
