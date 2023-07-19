#include <QApplication>
#include <QMouseEvent>
// =======================================================================
// we need to include Properties.h before the other *.h which below it.
// =======================================================================
// need to check, must be included
#include <QTableWidget>
// =======================================================================
// from OCC and the definition is
//        This class defines an OpenGl graphic driver.
// =======================================================================
#include "OpenGl_GraphicDriver.hxx"
#include <Xw_Window.hxx>
#include "Aspect_DisplayConnection.hxx"
#include "OCCWidget.h"
#include "Primitive.h"
// =======================================================================
// define an external OCC view widget
// =======================================================================
#include "BRepBuilderAPI_MakeFace.hxx"
#include "BRepBuilderAPI_Sewing.hxx"
#include "BRepAlgo_Fuse.hxx"
OCCWidget* OCCw = NULL;
ViewWidget* vw = NULL;
//OCCWidget::OCCWidget (QWidget* parent) : QGLWidget(parent)

OCCWidget::OCCWidget (QWidget* parent) : QWidget(parent)
{
    setBackgroundRole( QPalette::NoRole );
    setMouseTracking( true );
    //Initialize();



}
OCCWidget::~OCCWidget()
{
}
#include "V3d_AmbientLight.hxx"
#include "V3d_DirectionalLight.hxx"
void OCCWidget::Initialize()
{    



    // from OCC and the definition is
    //        "This class creates and provides connection with X server."
    Handle(Aspect_DisplayConnection) Connection = new Aspect_DisplayConnection();
    //
    // display
    //
    // from OCC and the definition is
    //       "This class allows the definition of a graphic driver for
    //        3d interface (currently only OpenGl driver is used). "
    GraphicDriver = new OpenGl_GraphicDriver(Connection);
    // from Qt and the definition is
    //        "Returns the window's platform id."
    WinId = (WId) winId();
    //WinId = ViewWidget->winId();
    // from OCC and the definition is
    //        "This class defines XLib window intended for creation of OpenGL context."
    Handle(Xw_Window) xwind = new Xw_Window(Connection, (Window) WinId);
    // x window
    // graphic driver is a pencil and qt widget is a blackboard. occ build a connection between them.
    // V3d_Viewer is the pencil and V3d_View is the blackboard.
    Viewer = new V3d_Viewer(GraphicDriver, (short* const)"viewer");
    View = Viewer->CreateView();
    View->SetWindow(xwind);
    if (!xwind->IsMapped()) xwind->Map();



    View->Camera()->SetCenter(gp_Pnt(0,0,0));
    View->Camera()->SetUp(gp_Dir(0,0,1));
    View->Camera()->SetEye(gp_Pnt(1,0,0));

    /*
    cout << "****************************************" << endl;
    cout << "eye center" << endl;
    cout << View->Camera()->Center().X() << " "
         << View->Camera()->Center().Y() << " "
         << View->Camera()->Center().Z() << endl;
    cout << "eye up" << endl;
    cout << View->Camera()->Up().X() << " "
         << View->Camera()->Up().Y() << " "
         << View->Camera()->Up().Z() << endl;
    cout << "eye position" << endl;
    cout << View->Camera()->Eye().X() << " "
         << View->Camera()->Eye().Y() << " "
         << View->Camera()->Eye().Z() << endl;
    cout << "near and far" << endl;
    cout << View->Camera()->ZNear() << " " << View->Camera()->ZFar() << endl;
    cout << "aspect and scale" << endl;
    cout << View->Camera()->Aspect() << " " << View->Camera()->Scale() << endl;
    cout << "occ projection matrix" << endl;
    for (int i=0; i<View->Camera()->ProjectionMatrix().Rows(); i++)
    {
        for (int j=0; j<View->Camera()->ProjectionMatrix().Cols(); j++)
        {
            cout << View->Camera()->ProjectionMatrix().GetValue(i,j) << " ";
        }
        cout << endl;
    }
    cout << "occ orientation matrix" << endl;
    for (int i=0; i<View->Camera()->OrientationMatrix().Rows(); i++)
    {
        for (int j=0; j<View->Camera()->OrientationMatrix().Cols(); j++)
        {
            cout << View->Camera()->OrientationMatrix().GetValue(i,j) << " ";
        }
        cout << endl;
    }*/








    // AIS_InteractiveContext is from OCC and the definition is :
    //    "The Interactive Context allows you to manage graphic behavior
    //     and selection of Interactive Objects in one or more viewers. "
    Context = new AIS_InteractiveContext(Viewer);
    //
    Viewer->SetDefaultLights();
    Viewer->SetLightOn();
    //   Handle(V3d_AmbientLight) aLight1 = new V3d_AmbientLight (Viewer, Quantity_NOC_GRAY50);
    //   Handle(V3d_DirectionalLight) aLight2 = new V3d_DirectionalLight (Viewer, V3d_YposZpos,Quantity_NOC_GRAY90,true);
    //   View->SetBackgroundColor(Quantity_NOC_BLACK);
    //   Viewer->SetLightOff(aLight2);
    //   View->MustBeResized();
    //   View->TriedronDisplay(Aspect_TOTP_LEFT_LOWER, Quantity_NOC_GOLD, 0.1, V3d_ZBUFFER);
    //   View->Camera()->SetProjectionType (Graphic3d_Camera::Projection_Perspective);

    Context->SetDisplayMode(AIS_Shaded);
    Context->SetHilightColor(Quantity_NOC_PALEGREEN);
    Context->SelectionColor(Quantity_NOC_NAVAJOWHITE1);
    Context->SetAutoActivateSelection(true);
    Context->DefaultDrawer()->SetFaceBoundaryDraw(true);
    Handle(Prs3d_LineAspect) theAspect = new Prs3d_LineAspect(Quantity_NOC_BLACK,Aspect_TOL_SOLID,2);
    Context->DefaultDrawer()->SetFaceBoundaryAspect(theAspect);
    // others
    select_type = SelectDomainObj;



}
void OCCWidget::paintEvent( QPaintEvent*  )
{
    if (Context.IsNull())
    {
        // init();
    }
    // View->Redraw();
    // View->MustBeResized();
}
void OCCWidget::resizeEvent( QResizeEvent*  )
{
    if( !View.IsNull() )
    {
        View->MustBeResized();
    }
}
void OCCWidget::mousePressEvent( QMouseEvent* e )
{
    if ( e->button() == Qt::LeftButton )
    {
        // from OCC and the definition is
        //         " Begin the rotation of the view around the screen axis
        //           according to the mouse position <X,Y>. "
        View->StartRotation(e->pos().x(),e->pos().y());
    }
    else if ( e->button() == Qt::RightButton )
    {
        Xmax = e->pos().x();
        Ymax = e->pos().y();
    }
}
void OCCWidget::mouseReleaseEvent(QMouseEvent* event)
{
    if ( event->button() == Qt::LeftButton )
    {
    }
}
void OCCWidget::mouseMoveEvent(QMouseEvent* event)
{
    MouseMove(event->buttons(),event->pos());
}
void OCCWidget::wheelEvent(QWheelEvent * event)
{
    if(event->delta() > 0)
    {
        // from OCC and the definition is
        //         " Zooms the view by a factor relative to the initial value
        //           expressed by Start = Standard_True Updates the view. "
        View->SetZoom(1.1, Standard_True);

    }
    else
    {
        View->SetZoom(0.9, Standard_True);
    }
}
void OCCWidget::MouseMove( Qt::MouseButtons nFlags, const QPoint point )
{
    if ( nFlags == Qt::LeftButton )
    {
        if (!machining_module_checked)
        {
            View->Rotation( point.x(), point.y() );
            View->Redraw();







        }
    }
    else if ( nFlags == Qt::RightButton )
    {
        View->Pan( point.x() - Xmax, Ymax - point.y() );
        Xmax = point.x();
        Ymax = point.y();





    }
    else
    {




        Context->MoveTo(point.x(),point.y(),View);




    }

}
void OCCWidget::mouseDoubleClickEvent(QMouseEvent *event)
{
    // from OCC and the definition is
    //         " Stores and hilights the previous detected; Unhilights
    //           the previous picked. "
    if (select_type == SelectDomainObj)
    {
        Select();
    }
    else if (select_type == SelectBoundaryObj)
    {
        SelectBoundary();
    }
}
void OCCWidget::Fit()
{
    View->FitAll(0.3);
    View->ZFitAll();
    View->Redraw();
}
void OCCWidget::Front()
{
    View->SetProj(V3d_Xpos);
    View->FitAll(0.3);
    View->ZFitAll();
}
void OCCWidget::Back()
{
    View->SetProj( V3d_Xneg );
    View->FitAll(0.3);
    View->ZFitAll();
}
void OCCWidget::Top()
{
    View->SetProj( V3d_Zpos );
    View->FitAll(0.3);
    View->ZFitAll();
}
void OCCWidget::Bottom()
{
    View->SetProj( V3d_Zneg );
    View->FitAll(0.3);
    View->ZFitAll();
}
void OCCWidget::Left()
{
    View->SetProj( V3d_Ypos );
    View->FitAll(0.3);
    View->ZFitAll();
}
void OCCWidget::Right()
{
    View->SetProj( V3d_Yneg );
    View->FitAll(0.3);
    View->ZFitAll();
}
void OCCWidget::Axo()
{
    View->SetProj(V3d_XposYnegZpos);
    View->FitAll(0.3);
    View->ZFitAll();
}
#include "Poly_Triangulation.hxx"
void OCCWidget::Plot (const TopoDS_Shape* S)
{
    Handle(AIS_Shape) AisShape = new AIS_Shape(*S);
    AisShape->SetSelectionMode(0);
    AisShape->SetHilightMode(0);
    Context->SetColor(AisShape,Quantity_Color(0.6,0.6,0.6,Quantity_TOC_RGB));
    Context->Display(AisShape);
    //Fit();
}
void OCCWidget::Plot (const TopoDS_Shape& S)
{
    Handle(AIS_Shape) AisShape = new AIS_Shape(S);
    AisShape->SetSelectionMode(0);
    AisShape->SetHilightMode(0);
    Context->SetColor(AisShape,Quantity_Color(0.6,0.6,0.6,Quantity_TOC_RGB));
    Context->Display(AisShape);
    //Fit();
}
void OCCWidget::Load (const TopoDS_Shape& S)
{
    Handle(AIS_Shape) AisShape = new AIS_Shape(S);
    AisShape->SetSelectionMode(0);
    AisShape->SetHilightMode(0);
    Context->SetColor(AisShape,Quantity_Color(0.6,0.6,0.6,Quantity_TOC_RGB));
    Context->Load(AisShape);
    //Fit();
}
void OCCWidget::PlotAIS (Handle(AIS_Shape) S)
{
    S->SetSelectionMode(0);
    S->SetHilightMode(0);
    Context->SetColor(S,Quantity_Color(0.8,0.8,0.8,Quantity_TOC_RGB));
    Context->SetTransparency(S,0.9);
    Context->Display(S);
}
void OCCWidget::Erase2(Handle(AIS_Shape) S)
{
    Context->Erase(S);
}
// =======================================================================
//
// Context get the position of the screen from MouseMove fuction
// choose a AIS_Shape
//
// =======================================================================
void OCCWidget::SelectEdge()
{
    //    Context->CloseAllContexts();
    //    Context->OpenLocalContext();
    //    Context->ActivateStandardMode(TopAbs_EDGE);
}
void OCCWidget::SelectFace()
{
    //    Context->CloseAllContexts();
    //    Context->OpenLocalContext();
    //    Context->ActivateStandardMode(TopAbs_FACE);
}
void OCCWidget::SelectBody()
{
    //    Context->CloseAllContexts(true);
    //    Context->OpenLocalContext(true, true, false, false);
    //    Context->ActivateStandardMode(TopAbs_SOLID);
}
void OCCWidget::Select()
{
    Context->Select();

    return;
    if (Context->NbSelected() == 0)
    {
        table_prop->clearContents();
        table_prop->setRowCount(0);
    }
    else
    {
        for (Context->InitCurrent(); Context->MoreCurrent(); Context->NextCurrent())
        {
            TopoDS_Shape S(Handle(AIS_Shape)::DownCast(Context->Current())->Shape());
            int i = prims->Include(S);
            if (i != -1)
            {
                //(*prims)[i]->ShowProperties(table_prop);
            }
        }
    }
}
void OCCWidget::SelectBoundary ()
{
    Context->Select();
    if (Context->NbSelected() == 0)
    {
        cout << "didn't choose a obj" << endl;
        currentBndObj = NULL;
    }
    else
    {
        for (Context->InitCurrent(); Context->MoreCurrent(); Context->NextCurrent())
        {
            currentBndObj = new TopoDS_Shape(Handle(AIS_Shape)::DownCast(Context->Current())->Shape());
            int id = bndObjs->Include(*currentBndObj);
            int type = (*bndObjs)[id]->Type();
            int n = (*bndObjs)[id]->Id();
            if (type == 0)
            {
                comboBox1->setCurrentIndex(0);
            }
            else if (type == 1)
            {
                comboBox1->setCurrentIndex(1);
            }
            spinBox1->setValue(n);
        }
    }
}
void OCCWidget::Remove ()
{
    // since context will delete a current object, can't use
    // for (context->initcurrent; context->morecurrent(); context->nextcurrent)
    int n = Context->NbSelected();
    if (n > 0)
    {
        for (int i=0; i<n; i++)
        {
            Context->InitCurrent();
            // delete from data structure
            TopoDS_Shape S(Handle(AIS_Shape)::DownCast(Context->Current())->Shape());
            int m = prims->Include(S);
            if (m == -1) return;
            prims->Delete((*prims)[m]);
            // clear object from window
            Context->Remove(Context->Current(),true);
            // clear table
            table_prop->clearContents();
            table_prop->setRowCount(0);
        }
    }
    cout << "geometry num: " << prims->size() << endl;
}
int OCCWidget::CurrentId()
{
    for (Context->InitCurrent(); Context->MoreCurrent(); Context->NextCurrent())
    {
        TopoDS_Shape S(Handle(AIS_Shape)::DownCast(Context->Current())->Shape());
        int i = prims->Include(S);
        return i;
    }
    return -1;
}

Handle(AIS_InteractiveObject) OCCWidget::CurrentAISObject ()
{
    for (Context->InitCurrent(); Context->MoreCurrent(); Context->NextCurrent())
    {
        TopoDS_Shape S(Handle(AIS_Shape)::DownCast(Context->Current())->Shape());
        int i = prims->Include(S);
        if (i != -1)
            return Context->Current();
    }
    return NULL;
}

void OCCWidget::RemoveAISObject(Handle(AIS_InteractiveObject) obj)
{
    TopoDS_Shape S(Handle(AIS_Shape)::DownCast(obj)->Shape());
    int m = prims->Include(S);
    prims->Delete((*prims)[m]);
    // clear object from window
    Context->Remove(obj);
    // clear table
    table_prop->clearContents();
    table_prop->setRowCount(0);
}
// =======================================================================
//
// old codes need to be changed
//
// =======================================================================
void OCCWidget::DragEvent( const int x, const int y, const int TheState )
{
    // TheState == -1  button down
    // TheState ==  0  move
    // TheState ==  1  button up
    static Standard_Integer theButtonDownX = 0;
    static Standard_Integer theButtonDownY = 0;
    if ( TheState == -1 )
    {
        theButtonDownX = x;
        theButtonDownY = y;
    }
    if ( TheState == 1 )
    {
        Context->Select( theButtonDownX, theButtonDownY, x, y, View );
    }
}
void OCCWidget::MoveEvent (const int x, const int y)
{
    Context->MoveTo(x,y,View);
}
