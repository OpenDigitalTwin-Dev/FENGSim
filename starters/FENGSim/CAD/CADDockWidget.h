#ifndef CADDOCKWIDGET_H
#define CADDOCKWIDGET_H

#include <QWidget>
#include <QMenu>
#include <QComboBox>

namespace Ui {
class CADDockWidget;
}

enum PrimitiveTYPE {
    BOX = 4, BALL = 5, TORUS = 6, CYLINDER = 7, CONE = 8, NONE = -1,
    POINT = 1, LINE = 2, PLANE = 3, ANY = 9
};

enum OperationTYPE {
    SWEEP = 1, EXTRUDE = 2, MIRROR = 3
};

class CADDockWidget : public QWidget
{
    Q_OBJECT

public:
    explicit CADDockWidget(QWidget *parent = 0);
    ~CADDockWidget();

public:
    Ui::CADDockWidget *ui;

private:
    QMenu* menu_prims;
    QMenu* menu_curve;
    QMenu* menu_face;
    QMenu* menu_boolean;
    QMenu* menu_sweep;
    QMenu* menu_view;
    QMenu* menu_coordinate_system;
    QMenu* menu_selection;
    // sweep extrude mirror
    double pos[3];
    double dir1[3];
    double dir2[3];
    double angle;
    // primitives
    // any
    double any_pos[3];
    double any_dir[3];
    double any_angle;
    // point
    double point[3];
    // line
    double line_p1[3];
    double line_p2[3];
    // plane
    double plane_pos[3];
    double plane_dir[3];
    double plane_x[2];
    double plane_y[2];
    // box
    double box_size[3];
    double box_pos[3];
    double box_dir[3];
    // ball
    double ball_radius;
    double ball_pos[3];
    double ball_dir[3];
    // cylinder
    double cylinder_radius;
    double cylinder_height;
    double cylinder_pos[3];
    double cylinder_dir[3];
    // cylinder
    double cone_radius1;
    double cone_radius2;
    double cone_height;
    double cone_pos[3];
    double cone_dir[3];
    // torus
    double torus_radius1;
    double torus_radius2;
    double torus_pos[3];
    double torus_dir[3];

public slots:
    // sweep extrude mirror
    void ClearOperations ();
    void SetSweepModule ();
    void SetExtrudeModule ();
    void SetMirrorModule ();
    void OperationComboBoxIndexChange();
    void OperationValueChange1();
    void OperationValueChange2();
    void OperationValueChange3();
    // primitives
    // point line plane cube ball cylinder cone torus
    void ClearPrimitives ();
    void SetAnyModule (double* pos, double* dir, double angle);
    void SetPointModule (double* x);
    void SetLineModule (double* x1, double* x2);
    void SetPlaneModule (double* pos, double* dir, double* x, double* y);
    void SetBoxModule (double* x, double* pos, double* dir);
    void SetSphereModule (double r, double* pos, double* dir);
    void SetCylinderModule (double r, double h, double* pos, double* dir);
    void SetConeModule (double r1, double r2, double h, double* pos, double* dir);
    void SetTorusModule (double r1, double r2, double* pos, double* dir);
    void PrimitiveComboBoxIndexChange();
    void PrimitiveValueChange4();
    void PrimitiveValueChange5();
    void PrimitiveValueChange6();

public:
    int operation_type;
    int OperationType () { return operation_type; }
    // sweep extrude mirror
    double* Pos () { return pos; }
    double* Dir_1 () { return dir1; }
    double* Dir_2 () { return dir2; }
    double Angle () { return angle; }
    int primitive_type;
    int PrimitiveType () { return primitive_type; }
    // primitives
    // any
    double* AnyPosition () { return any_pos; }
    double* AnyDirection () { return any_dir; }
    double AnyAngle () { return angle; }
    // point
    double* PointPosition () { return point; }
    // line
    double* LineP1 () { return line_p1; }
    double* LineP2 () { return line_p2; }
    // plane
    double* PlanePos () { return plane_pos; }
    double* PlaneDir () { return plane_dir; }
    double* PlaneX() { return plane_x; }
    double* PlaneY() { return plane_y; }
    // box
    double* BoxSize () { return box_size; }
    double* BoxPos () { return box_pos; }
    double* BoxDir () { return box_dir; }
    // ball
    double BallRadius () { return ball_radius; }
    double* BallPos () { return ball_pos; }
    double* BallDir () { return ball_dir; }
    // cylinder
    double CylinderRadius () { return cylinder_radius; }
    double CylinderHeight () { return cylinder_height; }
    double* CylinderPos () { return cylinder_pos; }
    double* CylinderDir () { return cylinder_dir; }
    // cone
    double ConeRadius1 () { return cone_radius1; }
    double ConeRadius2 () { return cone_radius2; }
    double ConeHeight () { return cone_height; }
    double* ConePos () { return cone_pos; }
    double* ConeDir () { return cone_dir; }
    // torus
    double TorusRadius1 () { return torus_radius1; }
    double TorusRadius2 () { return torus_radius2; }
    double* TorusPos () { return torus_pos; }
    double* TorusDir () { return torus_dir; }
};

#endif // CADDOCKWIDGET_H
