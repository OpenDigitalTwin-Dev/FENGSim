#include "MeasureDockWidget.h"
#include "ui_MeasureDockWidget.h"

MeasureDockWidget::MeasureDockWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MeasureDockWidget)
{
    ui->setupUi(this);
    menu_objects =  new QMenu;
    menu_objects->addAction(ui->actionSurface);
    menu_objects->addAction(ui->actionLine);
    //ui->pushButton->setMenu(menu_objects);
    menu_cdt =  new QMenu;
    menu_cdt->addAction(ui->actionSurfaceProfile);
    menu_cdt->addAction(ui->actionLineProfile);
    menu_cdt->addAction(ui->actionFlatness);
    menu_cdt->addAction(ui->actionStraightness);
//    menu_cdt->addAction(ui->actionCircularity);
//    menu_cdt->addAction(ui->actionCylindricity);
    //ui->pushButton_2->setMenu(menu_cdt);
//    gdt_objects_type = new QMenu();
//    gdt_objects_type->addAction(ui->actionSurface);
//    gdt_objects_type->addAction(ui->actionLine);
//    ui->pushButton->setMenu(gdt_objects_type);

    connect(ui->actionLine, SIGNAL(triggered()), this, SLOT(SetObject1()));
    connect(ui->actionSurface, SIGNAL(triggered()), this, SLOT(SetObject2()));

    connect(ui->actionStraightness, SIGNAL(triggered()), this, SLOT(SetType1()));
    connect(ui->actionFlatness, SIGNAL(triggered()), this, SLOT(SetType2()));
    connect(ui->actionCircularity, SIGNAL(triggered()), this, SLOT(SetType3()));
    connect(ui->actionCylindricity, SIGNAL(triggered()), this, SLOT(SetType4()));
    connect(ui->actionLineProfile, SIGNAL(triggered()), this, SLOT(SetType5()));
    connect(ui->actionSurfaceProfile, SIGNAL(triggered()), this, SLOT(SetType6()));

    //connect(ui->doubleSpinBox_2, SIGNAL(valueChanged(double)), this, SLOT(SetSingleStep()));


    //ui->tableWidget->setRowCount(7);
    //ui->tableWidget->setColumnCount(2);
    /*
        ui->tableWidget->setColumnWidth(0,20);
        ui->tableWidget->setColumnWidth(0,50);*/
    //        ui->tableWidget->setRowHeight(0,30);
    //        ui->tableWidget->setRowHeight(1,30);
    //        ui->tableWidget->setRowHeight(2,30);
    //        ui->tableWidget->setRowHeight(3,30);
    //        ui->tableWidget->setRowHeight(4,30);
    //        ui->tableWidget->setRowHeight(5,30);
    //        ui->tableWidget->setRowHeight(6,30);
    //        ui->tableWidget->setCellWidget(0, 0, ui->label_10);
    //        ui->tableWidget->setCellWidget(0, 1, ui->doubleSpinBox_2);
    //        ui->tableWidget->setCellWidget(1, 0, ui->label_11);
    //        ui->tableWidget->setCellWidget(1, 1, ui->doubleSpinBox_4);
    //        ui->tableWidget->setCellWidget(2, 0, ui->label_12);
    //        ui->tableWidget->setCellWidget(2, 1, ui->doubleSpinBox_5);
    //        ui->tableWidget->setCellWidget(3, 0, ui->label_13);
    //        ui->tableWidget->setCellWidget(3, 1, ui->doubleSpinBox_6);
    //        ui->tableWidget->setCellWidget(4, 0, ui->label_7);
    //        ui->tableWidget->setCellWidget(4, 1, ui->doubleSpinBox_7);
    //        ui->tableWidget->setCellWidget(5, 0, ui->label_14);
    //        ui->tableWidget->setCellWidget(5, 1, ui->doubleSpinBox_8);
    //        ui->tableWidget->setCellWidget(6, 0, ui->label_15);
    //        ui->tableWidget->setCellWidget(6, 1, ui->doubleSpinBox_9);
    //        ui->tableWidget->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    //        ui->tableWidget->verticalHeader()->setSectionResizeMode(QHeaderView::Stretch);


}

MeasureDockWidget::~MeasureDockWidget()
{
    delete ui;
}

void MeasureDockWidget::SetAllObjectsUnchecked()
{
    ui->actionLine->setChecked(false);
    ui->actionSurface->setChecked(false);
}

void MeasureDockWidget::SetObject1 ()
{
    if (ui->actionLine->isChecked())
    {
        measure_obj = MeasureObject::measure_line;
        //ui->pushButton->setIcon(QIcon(":/new/measure/figure/measure_wind/line.png"));
        SetAllObjectsUnchecked();
        ui->actionLine->setChecked(true);
    }
    ui->actionLine->setChecked(true);
}

void MeasureDockWidget::SetObject2 ()
{
    if (ui->actionSurface->isChecked())
    {
        measure_obj = MeasureObject::measure_surface;
        //ui->pushButton->setIcon(QIcon(":/new/measure/figure/measure_wind/face.png"));
        SetAllObjectsUnchecked();
        ui->actionSurface->setChecked(true);
    }
    ui->actionSurface->setChecked(true);
}

void MeasureDockWidget::SetAllTypesUnchecked()
{
    ui->actionStraightness->setChecked(false);
    ui->actionFlatness->setChecked(false);
    ui->actionCircularity->setChecked(false);
    ui->actionCylindricity->setChecked(false);
    ui->actionLineProfile->setChecked(false);
    ui->actionSurfaceProfile->setChecked(false);
}

void MeasureDockWidget::SetType1 ()
{
    if (ui->actionStraightness->isChecked())
    {
        measure_type = MeasureType::straightness;
        //ui->pushButton_2->setIcon(QIcon(":/new/measure/figure/measure_wind/straight.png"));
        SetAllTypesUnchecked();
        ui->actionStraightness->setChecked(true);
    }
    ui->actionStraightness->setChecked(true);
}

void MeasureDockWidget::SetType2 ()
{
    if (ui->actionFlatness->isChecked())
    {
        measure_type = MeasureType::flatness;
        //ui->pushButton_2->setIcon(QIcon(":/new/measure/figure/measure_wind/flatness.png"));
        SetAllTypesUnchecked();
        ui->actionFlatness->setChecked(true);
    }
    ui->actionFlatness->setChecked(true);
}

void MeasureDockWidget::SetType3 ()
{
    if (ui->actionCircularity->isChecked())
    {
        measure_type = MeasureType::circularity;
        //ui->pushButton_2->setIcon(QIcon(":/new/measure/figure/measure_wind/circularity.png"));
        SetAllTypesUnchecked();
        ui->actionCircularity->setChecked(true);
    }
    ui->actionCircularity->setChecked(true);
}

void MeasureDockWidget::SetType4 ()
{
    if (ui->actionCylindricity->isChecked())
    {
        measure_type = MeasureType::cylindricity;
        //ui->pushButton_2->setIcon(QIcon(":/new/measure/figure/measure_wind/cylindricity.png"));
        SetAllTypesUnchecked();
        ui->actionCylindricity->setChecked(true);
    }
    ui->actionCylindricity->setChecked(true);
}

void MeasureDockWidget::SetType5 ()
{
    if (ui->actionLineProfile->isChecked())
    {
        measure_type = MeasureType::lineprofile;
        //ui->pushButton_2->setIcon(QIcon(":/new/measure/figure/measure_wind/lineprofile.png"));
        SetAllTypesUnchecked();
        ui->actionLineProfile->setChecked(true);
    }
    ui->actionLineProfile->setChecked(true);
}

void MeasureDockWidget::SetType6 ()
{
    if (ui->actionSurfaceProfile->isChecked())
    {
        measure_type = MeasureType::straightness;
        //ui->pushButton_2->setIcon(QIcon(":/new/measure/figure/measure_wind/surfaceprofile.png"));
        SetAllTypesUnchecked();
        ui->actionSurfaceProfile->setChecked(true);
    }
    ui->actionSurfaceProfile->setChecked(true);
}

void MeasureDockWidget::SetSingleStep()
{
    //        ui->doubleSpinBox_4->setSingleStep(ui->doubleSpinBox_2->text().toDouble());
    //        ui->doubleSpinBox_5->setSingleStep(ui->doubleSpinBox_2->text().toDouble());
    //        ui->doubleSpinBox_6->setSingleStep(ui->doubleSpinBox_2->text().toDouble());
    //        ui->doubleSpinBox_7->setSingleStep(ui->doubleSpinBox_2->text().toDouble());
    //        ui->doubleSpinBox_8->setSingleStep(ui->doubleSpinBox_2->text().toDouble());
    //        ui->doubleSpinBox_9->setSingleStep(ui->doubleSpinBox_2->text().toDouble());
}
