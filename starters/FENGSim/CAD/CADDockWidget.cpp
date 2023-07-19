#include "CADDockWidget.h"
#include "ui_CADDockWidget.h"

CADDockWidget::CADDockWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::CADDockWidget)
{
    ui->setupUi(this);
    // curve
    menu_curve = new QMenu();
    menu_curve->addAction(ui->actionLine);
    ui->pushButton_2->setMenu(menu_curve);
    // surface
    menu_face = new QMenu();
    menu_face->addAction(ui->actionSquare);
    ui->pushButton_3->setMenu(menu_face);
    // primitives
    menu_prims = new QMenu();
    menu_prims->addAction(ui->actionSphere);
    menu_prims->addAction(ui->actionCube);
    menu_prims->addAction(ui->actionCylinder);
    menu_prims->addAction(ui->actionCone);
    menu_prims->addAction(ui->actionTorus);
    ui->pushButton_4->setMenu(menu_prims);
    // boolean operations
    menu_boolean = new QMenu();
    menu_boolean->addAction(ui->actionPart1);
    menu_boolean->addAction(ui->actionPart2);
    menu_boolean->addAction(ui->actionUnion);
    menu_boolean->addAction(ui->actionSection);
    menu_boolean->addAction(ui->actionCut);
    ui->pushButton_0->setMenu(menu_boolean);
    // sweep, extrude, mirror
    menu_sweep = new QMenu();
    menu_sweep->addAction(ui->actionSweep);
    menu_sweep->addAction(ui->actionExtrude);
    menu_sweep->addAction(ui->actionMirror);
    ui->pushButton_13->setMenu(menu_sweep);
    // select edge, face, domain
    menu_selection = new QMenu();
    menu_selection->addAction(ui->actionSelectBnd);
    menu_selection->addAction(ui->actionSelectDomain);
    ui->pushButton_5->setMenu(menu_selection);
    // choose sweep, extrude, mirror
    connect(ui->actionSweep, SIGNAL(triggered()), this, SLOT(SetSweepModule()));
    connect(ui->actionExtrude, SIGNAL(triggered()), this, SLOT(SetExtrudeModule()));
    connect(ui->actionMirror, SIGNAL(triggered()), this, SLOT(SetMirrorModule()));
    connect(ui->comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OperationComboBoxIndexChange()));
    connect(ui->doubleSpinBox, SIGNAL(valueChanged(double)), this, SLOT(OperationValueChange1()));
    connect(ui->doubleSpinBox_2, SIGNAL(valueChanged(double)), this, SLOT(OperationValueChange2()));
    connect(ui->doubleSpinBox_3, SIGNAL(valueChanged(double)), this, SLOT(OperationValueChange3()));
    // point line plane box ball torus cylinder cone
    connect(ui->comboBox_2, SIGNAL(currentIndexChanged(int)), this, SLOT(PrimitiveComboBoxIndexChange()));
    connect(ui->doubleSpinBox_4, SIGNAL(valueChanged(double)), this, SLOT(PrimitiveValueChange4()));
    connect(ui->doubleSpinBox_5, SIGNAL(valueChanged(double)), this, SLOT(PrimitiveValueChange5()));
    connect(ui->doubleSpinBox_6, SIGNAL(valueChanged(double)), this, SLOT(PrimitiveValueChange6()));
    // initial
    //SetSweepModule();
    ClearPrimitives();



}

CADDockWidget::~CADDockWidget()
{
    delete ui;
}

void CADDockWidget::ClearOperations()
{
    ui->comboBox->clear();
    ui->doubleSpinBox->setValue(0);
    ui->doubleSpinBox_2->setValue(0);
    ui->doubleSpinBox_3->setValue(0);
    pos[0] = 0;
    pos[1] = 0;
    pos[2] = 0;
    dir1[0] = 1;
    dir1[1] = 0;
    dir1[2] = 0;
    dir2[0] = 0;
    dir2[1] = 1;
    dir2[2] = 0;
    angle = 360;
}

void CADDockWidget::SetSweepModule()
{
    operation_type = OperationTYPE::SWEEP;
    if (!ui->actionSweep->isChecked())
        ui->actionSweep->setChecked(true);
    ClearOperations();
    ui->tabWidget->setCurrentIndex(2);
    ui->comboBox->addItem("Position");
    ui->comboBox->addItem("Direction 1");
    ui->comboBox->addItem("Angle");
    ui->actionExtrude->setChecked(false);
    ui->actionMirror->setChecked(false);
    ui->pushButton_13->setIcon(QIcon(":/cad_wind/figure/cad_wind/sweep.png"));
    ui->tabWidget->setTabIcon(2,QIcon(":/cad_wind/figure/cad_wind/sweep.png"));
    ui->comboBox->setCurrentIndex(0);
    ui->doubleSpinBox->setValue(pos[0]);
    ui->doubleSpinBox_2->setValue(pos[1]);
    ui->doubleSpinBox_3->setValue(pos[2]);
}

void CADDockWidget::SetExtrudeModule()
{
    operation_type = OperationTYPE::EXTRUDE;
    if (!ui->actionExtrude->isChecked())
        ui->actionExtrude->setChecked(true);
    ClearOperations();
    ui->tabWidget->setCurrentIndex(2);
    ui->comboBox->addItem("Direction 1");
    ui->actionSweep->setChecked(false);
    ui->actionMirror->setChecked(false);
    ui->pushButton_13->setIcon(QIcon(":/cad_wind/figure/cad_wind/extrude.png"));
    ui->tabWidget->setTabIcon(2,QIcon(":/cad_wind/figure/cad_wind/extrude.png"));
    ui->comboBox->setCurrentIndex(0);
    ui->doubleSpinBox->setValue(dir1[0]);
    ui->doubleSpinBox_2->setValue(dir1[1]);
    ui->doubleSpinBox_3->setValue(dir1[2]);
}

void CADDockWidget::SetMirrorModule()
{
    operation_type = OperationTYPE::MIRROR;
    if (!ui->actionMirror->isChecked())
        ui->actionMirror->setChecked(true);
    ClearOperations();
    ui->tabWidget->setCurrentIndex(2);
    ui->comboBox->addItem("Position");
    ui->comboBox->addItem("Direction 1");
    ui->comboBox->addItem("Direction 2");
    ui->actionSweep->setChecked(false);
    ui->actionExtrude->setChecked(false);
    ui->pushButton_13->setIcon(QIcon(":/cad_wind/figure/cad_wind/mirror.png"));
    ui->tabWidget->setTabIcon(2,QIcon(":/cad_wind/figure/cad_wind/mirror.png"));
    ui->comboBox->setCurrentIndex(0);
    ui->doubleSpinBox->setValue(pos[0]);
    ui->doubleSpinBox_2->setValue(pos[1]);
    ui->doubleSpinBox_3->setValue(pos[2]);
}

void CADDockWidget::OperationComboBoxIndexChange()
{
    if (ui->comboBox->currentText() == "Position")
    {
        ui->doubleSpinBox->setValue(pos[0]);
        ui->doubleSpinBox_2->setValue(pos[1]);
        ui->doubleSpinBox_3->setValue(pos[2]);
    }
    else if (ui->comboBox->currentText() == "Direction 1")
    {
        ui->doubleSpinBox->setValue(dir1[0]);
        ui->doubleSpinBox_2->setValue(dir1[1]);
        ui->doubleSpinBox_3->setValue(dir1[2]);
    }
    else if (ui->comboBox->currentText() == "Direction 2")
    {
        ui->doubleSpinBox->setValue(dir2[0]);
        ui->doubleSpinBox_2->setValue(dir2[1]);
        ui->doubleSpinBox_3->setValue(dir2[2]);
    }
    else if (ui->comboBox->currentText() == "Angle")
    {
        ui->doubleSpinBox->setValue(angle);
        ui->doubleSpinBox_2->setValue(0);
        ui->doubleSpinBox_3->setValue(0);
    }
}

void CADDockWidget::OperationValueChange1()
{
    if (ui->comboBox->currentText() == "Position")
    {
        pos[0] = ui->doubleSpinBox->value();
    }
    else if (ui->comboBox->currentText() == "Direction 1")
    {
        dir1[0] = ui->doubleSpinBox->value();
    }
    else if (ui->comboBox->currentText() == "Direction 2")
    {
        dir2[0] = ui->doubleSpinBox->value();
    }
    else if (ui->comboBox->currentText() == "Angle")
    {
        angle = ui->doubleSpinBox->value();
    }
}

void CADDockWidget::OperationValueChange2()
{
    if (ui->comboBox->currentText() == "Position")
    {
        pos[1] = ui->doubleSpinBox_2->value();
    }
    else if (ui->comboBox->currentText() == "Direction 1")
    {
        dir1[1] = ui->doubleSpinBox_2->value();
    }
    else if (ui->comboBox->currentText() == "Direction 2")
    {
        dir2[1] = ui->doubleSpinBox_2->value();
    }
}

void CADDockWidget::OperationValueChange3()
{
    if (ui->comboBox->currentText() == "Position")
    {
        pos[2] = ui->doubleSpinBox_3->value();
    }
    else if (ui->comboBox->currentText() == "Direction 1")
    {
        dir1[2] = ui->doubleSpinBox_3->value();
    }
    else if (ui->comboBox->currentText() == "Direction 2")
    {
        dir2[2] = ui->doubleSpinBox_3->value();
    }
}

void CADDockWidget::ClearPrimitives()
{
    primitive_type = PrimitiveTYPE::NONE;
    ui->comboBox_2->clear();
    ui->doubleSpinBox_4->setValue(0);
    ui->doubleSpinBox_5->setValue(0);
    ui->doubleSpinBox_6->setValue(0);
    //ui->tabWidget->setCurrentIndex(0);
}

void CADDockWidget::SetAnyModule (double *pos, double *dir, double angle)
{
    ClearPrimitives();
    primitive_type = PrimitiveTYPE::ANY;
    //ui->tabWidget->setCurrentIndex(0);
    for (int i = 0; i < 3; i++)
    {
        any_pos[i] = pos[i];
        any_dir[i] = dir[i];
    }
    any_angle =  angle;
    ui->comboBox_2->addItem("Position");
    ui->comboBox_2->addItem("Direction");
    ui->comboBox_2->addItem("Angle");
    ui->doubleSpinBox_4->setValue(any_pos[0]);
    ui->doubleSpinBox_5->setValue(any_pos[1]);
    ui->doubleSpinBox_6->setValue(any_pos[2]);
}

void CADDockWidget::SetPointModule (double* x) {
    ClearPrimitives();
    primitive_type = PrimitiveTYPE::POINT;
    for (int i = 0; i < 3; i++)
    {
        point[i] = x[i];
    }

    ui->comboBox_2->addItem("Position");
    ui->doubleSpinBox_4->setValue(point[0]);
    ui->doubleSpinBox_5->setValue(point[1]);
    ui->doubleSpinBox_6->setValue(point[2]);
}

void CADDockWidget::SetLineModule (double *x1, double *x2)
{
    ClearPrimitives();
    primitive_type = PrimitiveTYPE::LINE;
    for (int i = 0; i < 3; i++)
    {
        line_p1[i] = x1[i];
        line_p2[i] = x2[i];
    }
    ui->comboBox_2->addItem("x1");
    ui->comboBox_2->addItem("x2");
    ui->doubleSpinBox_4->setValue(line_p1[0]);
    ui->doubleSpinBox_5->setValue(line_p1[1]);
    ui->doubleSpinBox_6->setValue(line_p1[2]);
}

void CADDockWidget::SetPlaneModule (double* pos, double* dir, double* x, double* y) {
    ClearPrimitives();
    primitive_type = PrimitiveTYPE::PLANE;
    for (int i = 0; i < 3; i++)
    {
        plane_pos[i] = pos[i];
        plane_dir[i] = dir[i];
    }
    for (int i = 0; i < 2; i++)
    {
        plane_x[i] = x[i];
        plane_y[i] = y[i];
    }
    ui->comboBox_2->addItem("Position");
    ui->comboBox_2->addItem("Direction");
    ui->comboBox_2->addItem("x");
    ui->comboBox_2->addItem("y");
    ui->doubleSpinBox_4->setValue(plane_pos[0]);
    ui->doubleSpinBox_5->setValue(plane_pos[1]);
    ui->doubleSpinBox_6->setValue(plane_pos[2]);
}

void CADDockWidget::SetBoxModule (double* x, double* pos, double* dir) {
    ClearPrimitives();
    primitive_type = PrimitiveTYPE::BOX;
    for (int i = 0; i < 3; i++)
    {
        box_size[i] = x[i];
        box_pos[i] = pos[i];
        box_dir[i] = dir[i];
    }
    ui->comboBox_2->addItem("Parameters");
    ui->comboBox_2->addItem("Position");
    ui->comboBox_2->addItem("Direction");
    ui->doubleSpinBox_4->setValue(box_size[0]);
    ui->doubleSpinBox_5->setValue(box_size[1]);
    ui->doubleSpinBox_6->setValue(box_size[2]);
}

void CADDockWidget::SetSphereModule (double r, double* pos, double* dir) {
    ClearPrimitives();
    primitive_type = PrimitiveTYPE::BALL;
    ball_radius = r;
    for (int i = 0; i < 3; i++)
    {
        ball_pos[i] = pos[i];
        ball_dir[i] = dir[i];
    }
    ui->comboBox_2->addItem("Radius");
    ui->comboBox_2->addItem("Position");
    ui->comboBox_2->addItem("Direction");
    ui->doubleSpinBox_4->setValue(ball_radius);
    ui->doubleSpinBox_5->setValue(0);
    ui->doubleSpinBox_6->setValue(0);
}

void CADDockWidget::SetCylinderModule (double r, double h, double* pos, double* dir) {
    ClearPrimitives();
    primitive_type = PrimitiveTYPE::CYLINDER;
    cylinder_radius = r;
    cylinder_height = h;
    for (int i = 0; i < 3; i++)
    {
        cylinder_pos[i] = pos[i];
        cylinder_dir[i] = dir[i];
    }
    ui->comboBox_2->addItem("Radius");
    ui->comboBox_2->addItem("Height");
    ui->comboBox_2->addItem("Position");
    ui->comboBox_2->addItem("Direction");
    ui->doubleSpinBox_4->setValue(cylinder_radius);
    ui->doubleSpinBox_5->setValue(0);
    ui->doubleSpinBox_6->setValue(0);
}

void CADDockWidget::SetConeModule (double r1, double r2, double h, double* pos, double* dir) {
    ClearPrimitives();
    primitive_type = PrimitiveTYPE::CONE;
    cone_radius1 = r1;
    cone_radius2 = r2;
    cone_height = h;
    for (int i = 0; i < 3; i++)
    {
        cone_pos[i] = pos[i];
        cone_dir[i] = dir[i];
    }
    ui->comboBox_2->addItem("Radius Top");
    ui->comboBox_2->addItem("Radius Bottom");
    ui->comboBox_2->addItem("Height");
    ui->comboBox_2->addItem("Position");
    ui->comboBox_2->addItem("Direction");
    ui->doubleSpinBox_4->setValue(cone_radius1);
    ui->doubleSpinBox_5->setValue(0);
    ui->doubleSpinBox_6->setValue(0);
}

void CADDockWidget::SetTorusModule (double r1, double r2, double* pos, double* dir) {
    ClearPrimitives();
    primitive_type = PrimitiveTYPE::TORUS;
    torus_radius1 = r1;
    torus_radius2 = r2;
    for (int i = 0; i < 3; i++)
    {
        torus_pos[i] = pos[i];
        torus_dir[i] = dir[i];
    }
    ui->comboBox_2->addItem("Radius Inner");
    ui->comboBox_2->addItem("Radius Outer");
    ui->comboBox_2->addItem("Position");
    ui->comboBox_2->addItem("Direction");
    ui->doubleSpinBox_4->setValue(torus_radius1);
    ui->doubleSpinBox_5->setValue(0);
    ui->doubleSpinBox_6->setValue(0);
}

void CADDockWidget::PrimitiveComboBoxIndexChange()
{
    if (primitive_type == PrimitiveTYPE::ANY)
    {
        if (ui->comboBox_2->currentText() == "Position")
        {
            ui->doubleSpinBox_4->setValue(any_pos[0]);
            ui->doubleSpinBox_5->setValue(any_pos[1]);
            ui->doubleSpinBox_6->setValue(any_pos[2]);
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            ui->doubleSpinBox_4->setValue(any_dir[0]);
            ui->doubleSpinBox_5->setValue(any_dir[1]);
            ui->doubleSpinBox_6->setValue(any_dir[2]);
        }
        else if (ui->comboBox_2->currentText() == "Angle")
        {
            ui->doubleSpinBox_4->setValue(any_angle);
        }
    }
    else if (primitive_type == PrimitiveTYPE::POINT)
    {
        ui->doubleSpinBox_4->setValue(point[0]);
        ui->doubleSpinBox_5->setValue(point[1]);
        ui->doubleSpinBox_6->setValue(point[2]);
    }
    else if (primitive_type == PrimitiveTYPE::LINE)
    {
        if (ui->comboBox_2->currentText() == "x1")
        {
            ui->doubleSpinBox_4->setValue(line_p1[0]);
            ui->doubleSpinBox_5->setValue(line_p1[1]);
            ui->doubleSpinBox_6->setValue(line_p1[2]);
        }
        else if (ui->comboBox_2->currentText() == "x2")
        {
            ui->doubleSpinBox_4->setValue(line_p2[0]);
            ui->doubleSpinBox_5->setValue(line_p2[1]);
            ui->doubleSpinBox_6->setValue(line_p2[2]);
        }
    }
    else if (primitive_type == PrimitiveTYPE::PLANE)
    {
        if (ui->comboBox_2->currentText() == "Position")
        {
            ui->doubleSpinBox_4->setValue(plane_pos[0]);
            ui->doubleSpinBox_5->setValue(plane_pos[1]);
            ui->doubleSpinBox_6->setValue(plane_pos[2]);
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            ui->doubleSpinBox_4->setValue(plane_dir[0]);
            ui->doubleSpinBox_5->setValue(plane_dir[1]);
            ui->doubleSpinBox_6->setValue(plane_dir[2]);
        }
        else if (ui->comboBox_2->currentText() == "x")
        {
            ui->doubleSpinBox_4->setValue(plane_x[0]);
            ui->doubleSpinBox_5->setValue(plane_x[1]);
        }
        else if (ui->comboBox_2->currentText() == "y")
        {
            ui->doubleSpinBox_4->setValue(plane_y[0]);
            ui->doubleSpinBox_5->setValue(plane_y[1]);
        }
    }
    else if (primitive_type == PrimitiveTYPE::BOX)
    {
        if (ui->comboBox_2->currentText() == "Parameters")
        {
            ui->doubleSpinBox_4->setValue(box_size[0]);
            ui->doubleSpinBox_5->setValue(box_size[1]);
            ui->doubleSpinBox_6->setValue(box_size[2]);
        }
        else if (ui->comboBox_2->currentText() == "Position")
        {
            ui->doubleSpinBox_4->setValue(box_pos[0]);
            ui->doubleSpinBox_5->setValue(box_pos[1]);
            ui->doubleSpinBox_6->setValue(box_pos[2]);
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            ui->doubleSpinBox_4->setValue(box_dir[0]);
            ui->doubleSpinBox_5->setValue(box_dir[1]);
            ui->doubleSpinBox_6->setValue(box_dir[2]);
        }
    }
    else if (primitive_type == PrimitiveTYPE::BALL)
    {
        if (ui->comboBox_2->currentText() == "Radius")
        {
            ui->doubleSpinBox_4->setValue(ball_radius);
            ui->doubleSpinBox_5->setValue(0);
            ui->doubleSpinBox_6->setValue(0);
        }
        else if (ui->comboBox_2->currentText() == "Position")
        {
            ui->doubleSpinBox_4->setValue(ball_pos[0]);
            ui->doubleSpinBox_5->setValue(ball_pos[1]);
            ui->doubleSpinBox_6->setValue(ball_pos[2]);
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            ui->doubleSpinBox_4->setValue(ball_dir[0]);
            ui->doubleSpinBox_5->setValue(ball_dir[1]);
            ui->doubleSpinBox_6->setValue(ball_dir[2]);
        }
    }
    else if (primitive_type == PrimitiveTYPE::CYLINDER)
    {
        if (ui->comboBox_2->currentText() == "Radius")
        {
            ui->doubleSpinBox_4->setValue(cylinder_radius);
            ui->doubleSpinBox_5->setValue(0);
            ui->doubleSpinBox_6->setValue(0);
        }
        else if (ui->comboBox_2->currentText() == "Height")
        {
            ui->doubleSpinBox_4->setValue(cylinder_height);
            ui->doubleSpinBox_5->setValue(0);
            ui->doubleSpinBox_6->setValue(0);
        }
        else if (ui->comboBox_2->currentText() == "Position")
        {
            ui->doubleSpinBox_4->setValue(cylinder_pos[0]);
            ui->doubleSpinBox_5->setValue(cylinder_pos[1]);
            ui->doubleSpinBox_6->setValue(cylinder_pos[2]);
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            ui->doubleSpinBox_4->setValue(cylinder_dir[0]);
            ui->doubleSpinBox_5->setValue(cylinder_dir[1]);
            ui->doubleSpinBox_6->setValue(cylinder_dir[2]);
        }
    }
    else if (primitive_type == PrimitiveTYPE::CONE)
    {
        if (ui->comboBox_2->currentText() == "Radius Top")
        {
            ui->doubleSpinBox_4->setValue(cone_radius1);
            ui->doubleSpinBox_5->setValue(0);
            ui->doubleSpinBox_6->setValue(0);
        }
        else if (ui->comboBox_2->currentText() == "Radius Bottom")
        {
            ui->doubleSpinBox_4->setValue(cone_radius2);
            ui->doubleSpinBox_5->setValue(0);
            ui->doubleSpinBox_6->setValue(0);
        }
        else if (ui->comboBox_2->currentText() == "Height")
        {
            ui->doubleSpinBox_4->setValue(cone_height);
            ui->doubleSpinBox_5->setValue(0);
            ui->doubleSpinBox_6->setValue(0);
        }
        else if (ui->comboBox_2->currentText() == "Position")
        {
            ui->doubleSpinBox_4->setValue(cone_pos[0]);
            ui->doubleSpinBox_5->setValue(cone_pos[1]);
            ui->doubleSpinBox_6->setValue(cone_pos[2]);
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            ui->doubleSpinBox_4->setValue(cone_dir[0]);
            ui->doubleSpinBox_5->setValue(cone_dir[1]);
            ui->doubleSpinBox_6->setValue(cone_dir[2]);
        }
    }
    else if (primitive_type == PrimitiveTYPE::TORUS)
    {
        if (ui->comboBox_2->currentText() == "Radius Inner")
        {
            ui->doubleSpinBox_4->setValue(torus_radius1);
            ui->doubleSpinBox_5->setValue(0);
            ui->doubleSpinBox_6->setValue(0);
        }
        else if (ui->comboBox_2->currentText() == "Radius Outer")
        {
            ui->doubleSpinBox_4->setValue(torus_radius2);
            ui->doubleSpinBox_5->setValue(0);
            ui->doubleSpinBox_6->setValue(0);
        }
        else if (ui->comboBox_2->currentText() == "Position")
        {
            ui->doubleSpinBox_4->setValue(torus_pos[0]);
            ui->doubleSpinBox_5->setValue(torus_pos[1]);
            ui->doubleSpinBox_6->setValue(torus_pos[2]);
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            ui->doubleSpinBox_4->setValue(torus_dir[0]);
            ui->doubleSpinBox_5->setValue(torus_dir[1]);
            ui->doubleSpinBox_6->setValue(torus_dir[2]);
        }
    }
}

void CADDockWidget::PrimitiveValueChange4()
{
    if (primitive_type == PrimitiveTYPE::ANY)
    {
        if (ui->comboBox_2->currentText() == "Position")
        {
            any_pos[0] = ui->doubleSpinBox_4->value();
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            any_pos[0] = ui->doubleSpinBox_4->value();
        }
        else if (ui->comboBox_2->currentText() == "Angle")
        {
            any_angle = ui->doubleSpinBox_4->value();
        }
    }
    else if (primitive_type == PrimitiveTYPE::POINT)
    {
        point[0] = ui->doubleSpinBox_4->value();
    }
    else if (primitive_type == PrimitiveTYPE::LINE)
    {
        if (ui->comboBox_2->currentText() == "x1")
        {
            line_p1[0] = ui->doubleSpinBox_4->value();
        }
        else if (ui->comboBox_2->currentText() == "x2")
        {
            line_p2[0] = ui->doubleSpinBox_4->value();
        }
    }
    else if (primitive_type == PrimitiveTYPE::PLANE)
    {
        if (ui->comboBox_2->currentText() == "Position")
        {
            plane_pos[0] = ui->doubleSpinBox_4->value();
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            plane_dir[0] = ui->doubleSpinBox_4->value();
        }
        else if (ui->comboBox_2->currentText() == "x")
        {
            plane_x[0] = ui->doubleSpinBox_4->value();
        }
        else if (ui->comboBox_2->currentText() == "y")
        {
            plane_y[0] = ui->doubleSpinBox_4->value();
        }
    }
    else if (primitive_type == PrimitiveTYPE::BOX)
    {
        if (ui->comboBox_2->currentText() == "Parameters")
        {
            box_size[0] = ui->doubleSpinBox_4->value();
        }
        else if (ui->comboBox_2->currentText() == "Position")
        {
            box_pos[0] = ui->doubleSpinBox_4->value();
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            box_dir[0] = ui->doubleSpinBox_4->value();
        }
    }
    else if (primitive_type == PrimitiveTYPE::BALL)
    {
        if (ui->comboBox_2->currentText() == "Radius")
        {
            ball_radius = ui->doubleSpinBox_4->value();
        }
        else if (ui->comboBox_2->currentText() == "Position")
        {
            ball_pos[0] = ui->doubleSpinBox_4->value();
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            ball_dir[0] = ui->doubleSpinBox_4->value();
        }
    }
    else if (primitive_type == PrimitiveTYPE::CYLINDER)
    {
        if (ui->comboBox_2->currentText() == "Radius")
        {
            cylinder_radius = ui->doubleSpinBox_4->value();
        }
        else if (ui->comboBox_2->currentText() == "Height")
        {
            cylinder_height = ui->doubleSpinBox_4->value();
        }
        else if (ui->comboBox_2->currentText() == "Position")
        {
            cylinder_pos[0] = ui->doubleSpinBox_4->value();
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            cylinder_dir[0] = ui->doubleSpinBox_4->value();
        }
    }
    else if (primitive_type == PrimitiveTYPE::CONE)
    {
        if (ui->comboBox_2->currentText() == "Radius Top")
        {
            cone_radius1 = ui->doubleSpinBox_4->value();
        }
        else if (ui->comboBox_2->currentText() == "Radius Bottom")
        {
            cone_radius2 = ui->doubleSpinBox_4->value();
        }
        else if (ui->comboBox_2->currentText() == "Height")
        {
            cone_height = ui->doubleSpinBox_4->value();
        }
        else if (ui->comboBox_2->currentText() == "Position")
        {
            cone_pos[0] = ui->doubleSpinBox_4->value();
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            cone_dir[0] = ui->doubleSpinBox_4->value();
        }
    }
    else if (primitive_type == PrimitiveTYPE::TORUS)
    {
        if (ui->comboBox_2->currentText() == "Radius Inner")
        {
            torus_radius1 = ui->doubleSpinBox_4->value();
        }
        else if (ui->comboBox_2->currentText() == "Radius Outer")
        {
            torus_radius2 = ui->doubleSpinBox_4->value();
        }
        else if (ui->comboBox_2->currentText() == "Position")
        {
            torus_pos[0] = ui->doubleSpinBox_4->value();
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            torus_dir[0] = ui->doubleSpinBox_4->value();
        }
    }
}

void CADDockWidget::PrimitiveValueChange5()
{
    if (primitive_type == PrimitiveTYPE::ANY)
    {
        if (ui->comboBox_2->currentText() == "Position")
        {
            any_pos[1] = ui->doubleSpinBox_5->value();
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            any_pos[1] = ui->doubleSpinBox_5->value();
        }
    }
    else if (primitive_type == PrimitiveTYPE::POINT)
    {
        point[1] = ui->doubleSpinBox_5->value();
    }
    else if (primitive_type == PrimitiveTYPE::LINE)
    {
        if (ui->comboBox_2->currentText() == "x1")
        {
            line_p1[1] = ui->doubleSpinBox_5->value();
        }
        else if (ui->comboBox_2->currentText() == "x2")
        {
            line_p2[1] = ui->doubleSpinBox_5->value();
        }
    }
    else if (primitive_type == PrimitiveTYPE::PLANE)
    {
        if (ui->comboBox_2->currentText() == "Position")
        {
            plane_pos[1] = ui->doubleSpinBox_5->value();
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            plane_dir[1] = ui->doubleSpinBox_5->value();
        }
        else if (ui->comboBox_2->currentText() == "x")
        {
            plane_x[1] = ui->doubleSpinBox_5->value();
        }
        else if (ui->comboBox_2->currentText() == "y")
        {
            plane_y[1] = ui->doubleSpinBox_5->value();
        }
    }
    else if (primitive_type == PrimitiveTYPE::BOX)
    {
        if (ui->comboBox_2->currentText() == "Parameters")
        {
            box_size[1] = ui->doubleSpinBox_5->value();
        }
        else if (ui->comboBox_2->currentText() == "Position")
        {
            box_pos[1] = ui->doubleSpinBox_5->value();
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            box_dir[1] = ui->doubleSpinBox_5->value();
        }
    }
    else if (primitive_type == PrimitiveTYPE::BALL)
    {
        if (ui->comboBox_2->currentText() == "Position")
        {
            ball_pos[1] = ui->doubleSpinBox_5->value();
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            ball_dir[1] = ui->doubleSpinBox_5->value();
        }
    }
    else if (primitive_type == PrimitiveTYPE::CYLINDER)
    {
        if (ui->comboBox_2->currentText() == "Position")
        {
            cylinder_pos[1] = ui->doubleSpinBox_5->value();
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            cylinder_dir[1] = ui->doubleSpinBox_5->value();
        }
    }
    else if (primitive_type == PrimitiveTYPE::CONE)
    {
        if (ui->comboBox_2->currentText() == "Position")
        {
            cone_pos[1] = ui->doubleSpinBox_5->value();
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            cone_dir[1] = ui->doubleSpinBox_5->value();
        }
    }
    else if (primitive_type == PrimitiveTYPE::TORUS)
    {
       if (ui->comboBox_2->currentText() == "Position")
        {
            torus_pos[1] = ui->doubleSpinBox_5->value();
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            torus_dir[1] = ui->doubleSpinBox_5->value();
        }
    }
}

void CADDockWidget::PrimitiveValueChange6()
{
    if (primitive_type == PrimitiveTYPE::ANY)
    {
        if (ui->comboBox_2->currentText() == "Position")
        {
            any_pos[2] = ui->doubleSpinBox_6->value();
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            any_pos[2] = ui->doubleSpinBox_6->value();
        }
    }
    else if (primitive_type == PrimitiveTYPE::POINT)
    {
        point[2] = ui->doubleSpinBox_6->value();
    }
    else if (primitive_type == PrimitiveTYPE::LINE)
    {
        if (ui->comboBox_2->currentText() == "x1")
        {
            line_p1[2] = ui->doubleSpinBox_6->value();
        }
        else if (ui->comboBox_2->currentText() == "x2")
        {
            line_p2[2] = ui->doubleSpinBox_6->value();
        }
    }
    else if (primitive_type == PrimitiveTYPE::PLANE)
    {
        if (ui->comboBox_2->currentText() == "Position")
        {
            plane_pos[2] = ui->doubleSpinBox_6->value();
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            plane_dir[2] = ui->doubleSpinBox_6->value();
        }
    }
    else if (primitive_type == PrimitiveTYPE::BOX)
    {
        if (ui->comboBox_2->currentText() == "Parameters")
        {
            box_size[2] = ui->doubleSpinBox_6->value();
        }
        else if (ui->comboBox_2->currentText() == "Position")
        {
            box_pos[2] = ui->doubleSpinBox_6->value();
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            box_dir[2] = ui->doubleSpinBox_6->value();
        }
    }
    else if (primitive_type == PrimitiveTYPE::BALL)
    {
        if (ui->comboBox_2->currentText() == "Position")
        {
            ball_pos[2] = ui->doubleSpinBox_6->value();
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            ball_dir[2] = ui->doubleSpinBox_6->value();
        }
    }
    else if (primitive_type == PrimitiveTYPE::CYLINDER)
    {
        if (ui->comboBox_2->currentText() == "Position")
        {
            cylinder_pos[2] = ui->doubleSpinBox_6->value();
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            cylinder_dir[2] = ui->doubleSpinBox_6->value();
        }
    }
    else if (primitive_type == PrimitiveTYPE::CONE)
    {
        if (ui->comboBox_2->currentText() == "Position")
        {
            cone_pos[2] = ui->doubleSpinBox_6->value();
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            cone_dir[2] = ui->doubleSpinBox_6->value();
        }
    }
    else if (primitive_type == PrimitiveTYPE::TORUS)
    {
       if (ui->comboBox_2->currentText() == "Position")
        {
            torus_pos[2] = ui->doubleSpinBox_6->value();
        }
        else if (ui->comboBox_2->currentText() == "Direction")
        {
            torus_dir[2] = ui->doubleSpinBox_6->value();
        }
    }
}
