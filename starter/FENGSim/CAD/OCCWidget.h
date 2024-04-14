#ifndef OCCWIDGET_H
#define OCCWIDGET_H

#include <QObject>
// =======================================================================
//
// AIS_InteractiveContext is from OCC and the definition is :
//    " The Interactive Context allows you to manage graphic behavior
//      and selection of Interactive Objects in one or more viewers. "
//
// =======================================================================
#include <AIS_InteractiveContext.hxx>
// =======================================================================
//
// V3d_View is from OCC and the definition is :
//    " Defines the application object VIEW for the VIEWER application. "
//
// =======================================================================
#include <V3d_View.hxx>
// =======================================================================
//
// AIS_Shape is from OCC and the definition is :
//    " A framework to manage presentation and selection of shapes.
//      AIS_Shape is the interactive object which is used the most by
//      applications. "
//
// =======================================================================
#include "AIS_Shape.hxx"
#include "Prs3d_LineAspect.hxx"
// =======================================================================
//
// use OCCWidget to do the operations to data strctures
//
// =======================================================================
#include "Primitive.h"
#include "QComboBox"
#include "QSpinBox"
#include "CAD/ViewWidget.h"

class OCCWidget;
extern ViewWidget* vw;
extern OCCWidget* OCCw;

class MainWindow;

enum SelectOjbectType
{
    SelectBoundaryObj, SelectDomainObj
};

//class OCCWidget: public QGLWidget {
class OCCWidget: public QWidget {
    Q_OBJECT
public:
    OCCWidget (QWidget* parent=NULL);
    ~OCCWidget();
    // initial
public:
    virtual void Initialize();
    virtual void paintEvent( QPaintEvent* );
    virtual void resizeEvent( QResizeEvent* );
    // view operations for zoom, move, rotate
    virtual void mousePressEvent( QMouseEvent* );
    virtual void mouseReleaseEvent(QMouseEvent* );
    virtual void mouseMoveEvent( QMouseEvent* );
    virtual void wheelEvent( QWheelEvent* );
    virtual void MouseMove(Qt::MouseButtons nFlags, const QPoint point );
    virtual void mouseDoubleClickEvent( QMouseEvent* );
    // eight view dirctions for fit, axo, front, back, left, right,
    // top and bottom
public slots:
    void Fit();
    void Front();
    void Back();
    void Top();
    void Bottom();
    void Left();
    void Right();
    void Axo();
    // context operations
private:
    SelectOjbectType select_type;
    Primitives* prims;
    QTableWidget* table_prop;
public:
    void SetSelectType (SelectOjbectType type)
    {
        select_type = type;
    }
    SelectOjbectType SelectType ()
    {
        return select_type;
    }
    void SetPrimitivesDS (Primitives* p)
    {
        prims = p;
    }
public:
    void SetTableWidget (QTableWidget* t)
    {
        table_prop = t;
    }
private:
    Boundaries* bndObjs;
    TopoDS_Shape* currentBndObj;
    QComboBox* comboBox1;
    QSpinBox* spinBox1;
public:
    void SetCurrentBoundaryObj (TopoDS_Shape*& S)
    {
        S = currentBndObj;
    }
    void SetBoundarObjs (Boundaries* bnd)
    {
        bndObjs = bnd;
    }
    void SetComboBox (QComboBox* box)
    {
        comboBox1 = box;
    }
    void SetSpinBox (QSpinBox* box)
    {
        spinBox1 = box;
    }
    void Select ();
    void SelectBoundary ();
    void SelectEdge ();
    void SelectFace ();
    void SelectBody ();
    int CurrentId ();
    Handle(AIS_InteractiveObject) CurrentAISObject ();
    void ClearCurrents ()
    {
        Context->ClearCurrents(true);
    }
    void Clear ()
    {
        Context->RemoveAll(true);
    }
    void RemoveAISObject (Handle(AIS_InteractiveObject) obj);
    void Remove ();
    void Plot (const TopoDS_Shape* S);
    void Plot (const TopoDS_Shape& S);
    void Load (const TopoDS_Shape& S);
    void DisplayAll ()
    {
        Context->DisplayAll(true);
    }
    void MoveTo (const Standard_Integer theXPix, const Standard_Integer theYPix,
                 const Graphic3d_Mat4d aProj, const Graphic3d_Mat4d anOrient,
                 const Standard_Integer aWidth, const Standard_Integer aHeight, const Standard_Boolean theToRedrawOnUpdate = Standard_True)
    {
        //Context->MoveTo(theXPix, theYPix, aProj, anOrient, aWidth, aHeight, theToRedrawOnUpdate);
    }

    void PlotAIS (Handle(AIS_Shape) S);
    void Erase2 (Handle(AIS_Shape) S);
    // set machining module
private:
    bool machining_module_checked;
public:
    void SetMachiningModule (bool t = false)
    {
        machining_module_checked = t;
    }
    // viewer is a pencil and view is a blackboard
public:
    Handle(Graphic3d_GraphicDriver) GraphicDriver;
    Handle(V3d_View) View;
    Handle(V3d_Viewer) Viewer;
    Handle(AIS_InteractiveContext) Context;
    Standard_Integer Xmin;
    Standard_Integer Ymin;
    Standard_Integer Xmax;
    Standard_Integer Ymax;
    // old
    // virtual void onLButtonDown(const QPoint point );
    // virtual void onLButtonUp(const QPoint point);
    void DragEvent( const int x, const int y, const int TheState );
    void InputEvent ();
    void MoveEvent( const int x, const int y );
    Handle(V3d_View) view ()
    {
        return View;
    }
    WId WinId;
    WId GetWinId ()
    {
        return WinId;
    }
protected:
    friend class MainWindow;

};

#endif // OCCWIDGET_H
