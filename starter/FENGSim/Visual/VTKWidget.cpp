#include "VTKWidget.h"
#include "vtkLookupTable.h"
#include "vtkProperty.h"
#include "vtkUnstructuredGridReader.h"
#include "vtkDataSetMapper.h"
#include "vtkUnstructuredGrid.h"
#include "vtkAlgorithmOutput.h"
#include "vtkAlgorithm.h"
#include "vtkCamera.h"
#include "vtkGenericOpenGLRenderWindow.h"
#include <vtkAutoInit.h>
#include "vtkOrientationMarkerWidget.h"
#include "vtkLight.h"
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);
VTK_MODULE_INIT(vtkRenderingFreeType)
#include "IVtkOCC_SelectableObject.hxx"
#include "IVtkTools_ShapeObject.hxx"
#include <QGraphicsView>

#include <vtkSmartPointer.h>
#include <vtkSimplePointsReader.h>


//double COLOR0[3] = {0.75, 0.75, 0.75};
//double COLOR0[3] = {64.0/255.0, 158.0/255.0, 166.0/255.0};
//double COLOR0[3] = {105.0/255.0, 143.0/255.0, 162.0/255.0};
//double COLOR0[3] = {209.0/255.0, 222.0/255.0, 215.0/255.0};
double COLOR0[3] = {230.0/255.0, 225.0/255.0, 216.0/255.0};
double COLOR1[3] = {255.0/255.0, 153.0/255.0, 153.0/255.0};
double COLOR2[3] = {255.0/255.0, 204.0/255.0, 153.0/255.0};
double COLOR3[3] = {102.0/255.0, 102.0/255.0, 153.0/255.0};
double COLOR4[3] = {204.0/255.0, 102.0/255.0, 153.0/255.0};
double COLOR5[3] = {255.0/255.0, 255.0/255.0, 204.0/255.0};
double COLOR6[3] = {255.0/255.0, 0.0/255.0, 0.0/255.0};
double COLOR7[3] = {255.0/255.0, 170.0/255.0, 0.0/255.0};
double COLOR8[3] = {0.0/255.0, 0.0/255.0, 255.0/255.0};
double COLOR9[3] = {0.0/255.0, 255.0/255.0, 0.0/255.0};
double COLOR10[3] = {255.0/255.0, 0.0/255.0, 0.0/255.0};



VTKWidget::VTKWidget (QWidget *parent) : QVTKOpenGLWidget(parent)
{
    // qvtkopenglwidget is different with qvtkwidget,
    // it need to create a generic render window by user
    vtkSmartPointer<vtkGenericOpenGLRenderWindow> wind = vtkGenericOpenGLRenderWindow::New();
    SetRenderWindow(wind);
    // vtk renderer
    renderer = vtkSmartPointer<vtkRenderer>::New();

    scalarBar->SetBarRatio(0.2);



    //        vtkSmartPointer<vtkLight> myLight = vtkSmartPointer<vtkLight>::New();
    //        myLight->SetColor(1, 1, 1);
    //        myLight->SetPosition(0, 0, 100);
    //        myLight->SetFocalPoint(renderer->GetActiveCamera()->GetFocalPoint());
    //        renderer->AddLight(myLight);
    //        vtkSmartPointer<vtkLight> myLight2 = vtkSmartPointer<vtkLight>::New();
    //        myLight2->SetColor(1, 1, 1);
    //        myLight2->SetPosition(100, 0, 0);
    //        myLight2->SetFocalPoint(renderer->GetActiveCamera()->GetFocalPoint());
    //        renderer->AddLight(myLight2);
    //        vtkSmartPointer<vtkLight> myLight3 = vtkSmartPointer<vtkLight>::New();
    //        myLight3->SetColor(1, 1, 1);
    //        myLight3->SetPosition(0, 100, 0);
    //        myLight3->SetFocalPoint(renderer->GetActiveCamera()->GetFocalPoint());
    //        renderer->AddLight(myLight3);
    //        vtkSmartPointer<vtkLight> myLight4 = vtkSmartPointer<vtkLight>::New();
    //        myLight4->SetColor(1, 1, 1);
    //        myLight4->SetPosition(0, 0, -100);
    //        myLight4->SetFocalPoint(renderer->GetActiveCamera()->GetFocalPoint());
    //        renderer->AddLight(myLight4);
    //        vtkSmartPointer<vtkLight> myLight5 = vtkSmartPointer<vtkLight>::New();
    //        myLight5->SetColor(1, 1, 1);
    //        myLight5->SetPosition(-100, 0, 0);
    //        myLight5->SetFocalPoint(renderer->GetActiveCamera()->GetFocalPoint());
    //        renderer->AddLight(myLight5);
    //        vtkSmartPointer<vtkLight> myLight6 = vtkSmartPointer<vtkLight>::New();
    //        myLight6->SetColor(1, 1, 1);
    //        myLight6->SetPosition(0, -100, 0);
    //        myLight6->SetFocalPoint(renderer->GetActiveCamera()->GetFocalPoint());
    //        renderer->AddLight(myLight6);



    renderer->SetBackground(33.0/255.0, 40.0/255.0, 48.0/255.0);
    // Background color dark blue
    GetRenderWindow()->AddRenderer(renderer);
    // occ picker
    aPicker = vtkSmartPointer<IVtkTools_ShapePicker>::New();
    aPicker->SetRenderer(renderer);
    ObjectId = 0;
    SelectedVtkActor = NULL;
    Selectable = true;
    SelectDomain = true;
    SelectBnd = false;
    //BoundarySelectable = false;
    // axes
    vtkAxesActor* axesActor = vtkAxesActor::New();
    vtkOrientationMarkerWidget* axesWidget = vtkOrientationMarkerWidget::New();
    axesWidget->SetOrientationMarker(axesActor);
    axesWidget->SetInteractor(GetInteractor());
    axesWidget->SetEnabled(true);
    axesWidget->SetInteractive(0);

    // data structure
    prims = new Primitives;
    bnds = new Boundaries;
    // measurement


    actor_am_stl_model = vtkSmartPointer<vtkActor>::New();
    actor_am_slices = vtkSmartPointer<vtkActor>::New();
    actor_am_mesh = vtkSmartPointer<vtkActor>::New();








    // *******************************************************
    // measure
    meas_cad_actor = vtkSmartPointer<vtkActor>::New();
    meas_cloud_source_actor = vtkSmartPointer<vtkActor>::New();
    meas_cloud_target_actor = vtkSmartPointer<vtkActor>::New();
    meas_cloud_icp_actor = vtkSmartPointer<vtkActor>::New();
    meas_icp_final = vtkSmartPointer<vtkActor>::New();
    meas_icp_final_color = vtkSmartPointer<vtkActor>::New();
    meas_model = vtkSmartPointer<vtkActor>::New();
    meas_scene = vtkSmartPointer<vtkActor>::New();
    renderer->AddActor(meas_cad_actor);
    renderer->AddActor(meas_cloud_target_actor);
    renderer->AddActor(meas_cloud_source_actor);
    renderer->AddActor(meas_cloud_icp_actor);
    for (int i = 0; i < 3; i++)
    {
        meas_source[i] = 0;
        meas_target[i] = 0;
    }




    // *******************************************************
    // additive manufacturing
    am_cad_actor = vtkSmartPointer<vtkActor>::New();
    am_stl_actor = vtkSmartPointer<vtkActor>::New();
    am_slices_actor = vtkSmartPointer<vtkActor>::New();
    am_pathplanning_actor = vtkSmartPointer<vtkActor>::New();
    am_mesh_actor = vtkSmartPointer<vtkActor>::New();
    am_simulation_actor = vtkSmartPointer<vtkActor>::New();
    am_source_actor = vtkSmartPointer<vtkActor>::New();




    // *******************************************************
    // fem
    fem_simulation_actor = vtkSmartPointer<vtkActor>::New();


    // *******************************************************
    // machining
    machining_part_bnds = new Boundaries;


    // *******************************************************
    // text output
    textActor->GetTextProperty()->SetFontSize ( 18 );
    //textActor->GetTextProperty()->SetBold(true);
    textActor->GetTextProperty()->SetColor (1, 1, 1);
    textActor->GetTextProperty()->SetJustificationToLeft();
    textActor->SetWidth(300);
    textActor->SetHeight(300);
    SetTextPosition();
    renderer->AddActor2D ( textActor );
    TextOutput();

}

void VTKWidget::SetTextPosition()
{
    int textwidth = geometry().width();
    textActor->SetPosition(textwidth-textActor->GetWidth()/2,50);

}

void VTKWidget::Plot (TopoDS_Shape S, bool t)
{
    IVtkOCC_Shape::Handle aShapeImpl = new IVtkOCC_Shape(S);
    ObjectId += 1;


    aShapeImpl->SetId(ObjectId);



    // vtkPolyDataAlgorithm
    vtkSmartPointer<IVtkTools_ShapeDataSource> aDS = vtkSmartPointer<IVtkTools_ShapeDataSource>::New();
    aDS->SetShape(aShapeImpl);

    // vtkAlgorithmOutput -> vtkAlgorithm
    vtkSmartPointer<vtkPolyDataMapper> Mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    Mapper->SetInputConnection(aDS->GetOutputPort());

    // actor
    vtkSmartPointer<vtkActor> Actor = vtkSmartPointer<vtkActor>::New();
    Actor->SetMapper(Mapper);


    Actor->SetId(ObjectId);



    Actor->GetProperty()->SetColor(COLOR0);
    Actor->SetSelected(false);

    // renderer
    renderer->AddActor(Actor);

    if (t)
    {
        renderer->ResetCamera();
    }


    switch (S.ShapeType()) {
    case 7:
        aPicker->SetSelectionMode(aShapeImpl, SM_Vertex);
        Actor->GetProperty()->SetPointSize(10);
        break;
    case 6:
        aPicker->SetSelectionMode(aShapeImpl, SM_Edge);
        Actor->GetProperty()->SetLineWidth(3);
        break;
    case 4:
        aPicker->SetSelectionMode(aShapeImpl, SM_Face);
        break;
    default:
        aPicker->SetSelectionMode(aShapeImpl, SM_Solid);
        break;
    }
    // qvtkopenglwidget update has some problems, it seems didn't use render again.
    // update();
    GetRenderWindow()->Render();
    return;
}

void VTKWidget::PlotBnds ()
{
    Clear();
    bnds->Reset(prims);
    for (int i = 0; i < bnds->Size(); i++)
    {
        Plot(*((*bnds)[i]->Value()), false);
    }
}

void VTKWidget::PlotDomains ()
{
    Clear();
    selected_bnd_id.clear();
    for (int i = 0; i < prims->size(); i++)
    {
        Plot(*((*prims)[i]->Value()));
    }
}

void VTKWidget::mouseDoubleClickEvent( QMouseEvent* event)
{
    double x, y;
    x = event->pos().x();
    y = event->pos().y();
    if (Selectable == true)
    {
        if (SelectDomain == true)
        {
            Pick(x,y);
            if (SelectedVtkActor != NULL)
            {
                int n = prims->Include(IVtkTools_ShapeDataSource::SafeDownCast(
                                           SelectedVtkActor->GetMapper()->GetInputConnection(0,0)->GetProducer()
                                           )->GetShape()->GetShape());
                (*prims)[n]->ShowProperties(cad_dw);
            }
            else
            {
                cad_dw->ClearPrimitives();
            }
        }
        else if (SelectBnd == true)
        {
            Pick(x,y);
            if (SelectedVtkActor != NULL)
            {
                int n = bnds->Include(IVtkTools_ShapeDataSource::SafeDownCast(
                                          SelectedVtkActor->GetMapper()->GetInputConnection(0,0)->GetProducer()
                                          )->GetShape()->GetShape());
                //cad_dw->ui->tabWidget->setCurrentIndex(2);
                if ((*bnds)[n]->Type() == Dirichlet)
                    phy_dw->ui->comboBox->setCurrentIndex(0);
                else
                    phy_dw->ui->comboBox->setCurrentIndex(1);
                phy_dw->ui->doubleSpinBox->setValue((*bnds)[n]->GetValue()[0]);
                phy_dw->ui->doubleSpinBox_2->setValue((*bnds)[n]->GetValue()[1]);
                phy_dw->ui->doubleSpinBox_3->setValue((*bnds)[n]->GetValue()[2]);
            }
            else
            {
                phy_dw->ui->comboBox->setCurrentIndex(0);
                phy_dw->ui->doubleSpinBox->setValue(0);
                phy_dw->ui->doubleSpinBox_2->setValue(0);
                phy_dw->ui->doubleSpinBox_3->setValue(0);
            }
        }
    }
}

void VTKWidget::Pick (double x, double y)
{
    cout << "selected num " << aPicker->Pick(x,y,0,renderer) << endl;
    IVtk_ShapeIdList ids = aPicker->GetPickedShapesIds();
    vtkActorCollection* acts = renderer->GetActors();
    int num = acts->GetNumberOfItems();
    acts->InitTraversal();
    for (int i = 0; i < num; ++i)
    {
        vtkActor* actor = acts->GetNextActor();
        bool IsSelected = false;
        for (IVtk_ShapeIdList::Iterator anIt (ids); anIt.More(); anIt.Next())
        {
            if (actor->GetId() == anIt.Value())
            {
                if (actor->IsSelected() == true)
                {
                    if (SelectBnd)
                    {
                        SelectedVtkActor = actor;
                        int n = GetSelectedBndId();
                        for (int j = 0; j < selected_bnd_id.size(); j++) {
                            if (selected_bnd_id[j] == n) {
                                selected_bnd_id.erase(selected_bnd_id.begin()+j);
                                break;
                            }
                        }
                        std::cout << "selected bnd " << selected_bnd_id.size() << ": ";
                        for (int j = 0; j < selected_bnd_id.size(); j++)
                            std::cout << " " << selected_bnd_id[j];
                        std::cout << std::endl;
                    }


                    actor->GetProperty()->SetColor(COLOR5);
                    actor->GetProperty()->SetOpacity(measure_op);
                    actor->SetSelected(false);
                    SelectedVtkActor = NULL;
                }
                else if (actor->IsSelected() == false)
                {
                    actor->GetProperty()->SetColor(COLOR5);
                    actor->SetSelected(true);
                    SelectedVtkActor = actor;

                    if (SelectBnd)
                    {
                        selected_bnd_id.push_back(GetSelectedBndId());
                        std::cout << "selected bnd " << selected_bnd_id.size() << ": ";
                        for (int j = 0; j < selected_bnd_id.size(); j++)
                            std::cout << " " << selected_bnd_id[j];
                        std::cout << std::endl;
                    }
                }
                IsSelected = true;
            }
        }


        // actor not selected
        if (ids.Size() > 0)
        {
            if (IsSelected == false)
            {
                if (SelectDomain)
                {
                    if (actor->IsSelected() == true)
                    {
                        // *** if could choose only one use ***
                        actor->GetProperty()->SetColor(COLOR0);
                        actor->SetSelected(false);
                    }
                }
            }
        }

        if (ids.Size() == 0)
        {
            actor->GetProperty()->SetColor(COLOR0);
            actor->SetSelected(false);
            SelectedVtkActor = NULL;
        }

    }


    //actor_meas_selected_bnd->GetProperty()->SetColor(COLOR1);


    GetRenderWindow()->Render();
}

int VTKWidget::SelectedId()
{
    if (SelectedVtkActor == NULL)
    {
        return -1;
    }
    else
    {
        return IVtkTools_ShapeDataSource::SafeDownCast(
                    SelectedVtkActor->GetMapper()->GetInputConnection(0,0)->GetProducer()
                    )->GetShape()->GetId();
    }
}

Primitive* VTKWidget::GetSelectedPrim ()
{
    int n = prims->Include(IVtkTools_ShapeDataSource::SafeDownCast(
                               SelectedVtkActor->GetMapper()->GetInputConnection(0,0)->GetProducer()
                               )->GetShape()->GetShape());
    return (*prims)[n];
}

Boundary* VTKWidget::GetSelectedBnd ()
{
    if (SelectedVtkActor == NULL) return NULL;

    int n = bnds->Include(IVtkTools_ShapeDataSource::SafeDownCast(
                              SelectedVtkActor->GetMapper()->GetInputConnection(0,0)->GetProducer()
                              )->GetShape()->GetShape());
    return (*bnds)[n];
}

int VTKWidget::GetSelectedBndId ()
{
    if (SelectedVtkActor == NULL) return -1;
    int n = bnds->Include(IVtkTools_ShapeDataSource::SafeDownCast(
                              SelectedVtkActor->GetMapper()->GetInputConnection(0,0)->GetProducer()
                              )->GetShape()->GetShape()
                          );
    return n;
}

void VTKWidget::ExportCurrentCamera()
{
    double a[3];
    double b[3];
    double c[3];
    for (int i = 0; i < 3; i++) {
        a[i] = renderer->GetActiveCamera()->GetFocalPoint()[i];
        b[i] = renderer->GetActiveCamera()->GetPosition()[i];
        c[i] = renderer->GetActiveCamera()->GetViewUp()[i];
    }
    QFile::remove("/home/jiping/FENGSim/webcad/Sample-of-WebGL-with-STL-loader-master/WebGLViewer/Models/camera.txt");
    ofstream out;
    out.open("/home/jiping/FENGSim/webcad/Sample-of-WebGL-with-STL-loader-master/WebGLViewer/Models/camera.txt");
    for (int i = 0; i<3; i++)
        out << a[i] << " ";
    for (int i = 0; i<3; i++)
        out << b[i] << " ";
    for (int i = 0; i<3; i++)
        out << c[i] << " ";
    out.close();

}
void VTKWidget::mouseMoveEvent(QMouseEvent* event)
{
    if ( event->buttons() == Qt::LeftButton )
    {
        QVTKOpenGLWidget::mouseMoveEvent(event);
        ExportCurrentCamera();
    }
    else if ( event->buttons() == Qt::RightButton )
    {
        QVTKOpenGLWidget::mouseMoveEvent(event);
        ExportCurrentCamera();
    }
    else if ( event->buttons() == Qt::MidButton )
    {
        QVTKOpenGLWidget::mouseMoveEvent(event);
        ExportCurrentCamera();
    }
    else
    {
        if (Selectable == true)
        {
            MouseMove(event->buttons(),event->pos());
        }
        else
        {
            QVTKOpenGLWidget::mouseMoveEvent(event);
        }
    }
}

void VTKWidget::MouseMove( Qt::MouseButtons nFlags, const QPoint point )
{
    double x = point.x();
    double y = point.y();
    aPicker->Pick(x,y,0,renderer);
    IVtk_ShapeIdList ids = aPicker->GetPickedShapesIds();

    vtkActorCollection* acts = renderer->GetActors();
    int num = acts->GetNumberOfItems();
    //std::cout << num << std::endl;


    acts->InitTraversal();
    for (int i = 0; i < num; ++i)
    {
        vtkActor* actor = acts->GetNextActor();
        bool IsSelected = false;
        for (IVtk_ShapeIdList::Iterator anIt (ids); anIt.More(); anIt.Next())
        {
            if (actor->GetId() == anIt.Value())
            {
                IsSelected = true;
                if (actor->IsSelected() == true)
                {
                    actor->GetProperty()->SetColor(actor->GetProperty()->GetColor());
                }
                else
                {
                    actor->GetProperty()->SetColor(COLOR5);
                }
            }
        }
        if (IsSelected == false)
        {
            if (actor->IsSelected() == true)
            {
                //actor->GetProperty()->SetColor(COLOR5);
                actor->GetProperty()->SetColor(actor->GetProperty()->GetColor());
            }
            else if (actor->IsSelected() == false)
            {
                //actor->GetProperty()->SetColor(actor->GetProperty()->GetColor());
                //actor->GetProperty()->SetColor(COLOR0);
                if (actor==meas_cloud_target_actor)
                    actor->GetProperty()->SetColor(0,0,255);
                else if (actor==meas_cloud_source_actor)
                    actor->GetProperty()->SetColor(255,0,0);
                else if (actor==meas_cloud_icp_actor)
                    actor->GetProperty()->SetColor(0,255,0);
                else
                    actor->GetProperty()->SetColor(COLOR0);
            }
        }
    }
    //actor_meas_selected_bnd->GetProperty()->SetColor(COLOR1);
    GetRenderWindow()->Render();
}

void VTKWidget::Remove (vtkActor* A)
{
    if (A == NULL)
    {
        return;
    }
    else
    {
        // delete topods_shape
        prims->Delete(IVtkTools_ShapeDataSource::SafeDownCast(
                          A->GetMapper()->GetInputConnection(0,0)->GetProducer())->GetShape()->GetShape());
        // delete selectable_object in viewselector
        int type = IVtkTools_ShapeDataSource::SafeDownCast(
                    A->GetMapper()->GetInputConnection(0,0)->GetProducer())->GetShape()->GetShape().ShapeType();
        switch (type) {
        case 7:
            aPicker->SetSelectionMode(
                        IVtkTools_ShapeDataSource::SafeDownCast(
                            A->GetMapper()->GetInputConnection(0,0)->GetProducer())->GetShape(),
                        SM_Vertex,false);
            break;
        case 6:
            aPicker->SetSelectionMode(
                        IVtkTools_ShapeDataSource::SafeDownCast(
                            A->GetMapper()->GetInputConnection(0,0)->GetProducer())->GetShape(),
                        SM_Edge,false);
            break;
        case 4:
            aPicker->SetSelectionMode(
                        IVtkTools_ShapeDataSource::SafeDownCast(
                            A->GetMapper()->GetInputConnection(0,0)->GetProducer())->GetShape(),
                        SM_Face,false);
            break;
        default:
            aPicker->SetSelectionMode(
                        IVtkTools_ShapeDataSource::SafeDownCast(
                            A->GetMapper()->GetInputConnection(0,0)->GetProducer())->GetShape(),
                        SM_Solid,false);
            break;
        }
        // delete actor
        renderer->RemoveActor(A);
        cout << "geometry num: " << prims->size() << endl;
    }
    GetRenderWindow()->Render();
}

void VTKWidget::Remove ()
{
    Remove(SelectedVtkActor);
    SelectedVtkActor = NULL;
}

void VTKWidget::Hide ()
{
    vtkActorCollection* acts = renderer->GetActors();
    int num = acts->GetNumberOfItems();
    acts->InitTraversal();
    for (int i = 0; i < num; ++i)
    {
        vtkActor* actor = acts->GetNextActor();
        actor->VisibilityOff();
    }
}

void VTKWidget::Reset ()
{
    vtkActorCollection* acts = renderer->GetActors();
    int num = acts->GetNumberOfItems();
    acts->InitTraversal();
    for (int i = 0; i < num; ++i)
    {
        vtkActor* actor = acts->GetNextActor();
        renderer->RemoveActor(actor);
    }
    prims->Clear();
    aPicker = vtkSmartPointer<IVtkTools_ShapePicker>::New();
    aPicker->SetRenderer(renderer);
    GetRenderWindow()->Render();
}

void VTKWidget::Clear ()
{
    vtkActorCollection* acts = renderer->GetActors();
    int num = acts->GetNumberOfItems();
    acts->InitTraversal();
    for (int i = 0; i < num; ++i)
    {
        vtkActor* actor = acts->GetNextActor();
        renderer->RemoveActor(actor);
    }
    aPicker = vtkSmartPointer<IVtkTools_ShapePicker>::New();
    aPicker->SetRenderer(renderer);
    GetRenderWindow()->Render();
    selected_bnd_id.clear();
}

#include "vtkSTLReader.h"
void VTKWidget::ImportSTLFile(std::string name)
{
    vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();
    reader->SetFileName(name.c_str());
    reader->Update();
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
}
#include "vtkPointData.h"
#include "vtkCellData.h"
#include "vtkScalarBarActor.h"
#include "vtkExtractTensorComponents.h"
#include "vtkDoubleArray.h"

void VTKWidget::ImportVTKFile(std::string name, int type, int n)
{
    fstream _file;
    _file.open(name, ios::in);
    if (!_file) return;
    // read a vtk file
    vtkSmartPointer<vtkUnstructuredGridReader> reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(name.c_str());
    //reader->ReadAllScalarsOn();
    reader->Update();

    //std::cout << reader->GetOutput()->GetCellData()->GetTensors()->GetSize() << std::endl;


    //    vtkSmartPointer<vtkExtractTensorComponents> extract = vtkSmartPointer<vtkExtractTensorComponents>::New();
    //    extract->SetInputConnection(reader->GetOutputPort());
    //    extract->ExtractScalarsOn();
    //    extract->SetScalarModeToComponent();
    //    extract->SetScalarComponents(0,0);
    //    extract->Update();
    //    std::cout << extract->GetOutput()->GetPointData()->GetScalars()->GetSize() << std::endl;
    //    reader->GetOutput()->GetCellData()->SetScalars(extract->GetOutput()->GetCellData()->GetScalars());

    QString color_name;
    color_name = QString("displacement");

    if (n>0) {
        vtkSmartPointer<vtkDoubleArray> values =
                vtkSmartPointer<vtkDoubleArray>::New();
        int num = reader->GetOutput()->GetCellData()->GetTensors()->GetSize()/9-1;
        std::cout << "tensor numbers: " << num << std::endl;
        values->SetNumberOfValues(num);
        values->SetName ("strains");
        for (int i=0; i<num; i++) {
            double s = reader->GetOutput()->GetCellData()->GetTensors()->GetComponent(i,n-1);
            values->SetValue(i,s);
        }
        reader->GetOutput()->GetCellData()->SetScalars(values);
        reader->Update();
        if (type == 1)
            color_name = QString("strain");
        else if (type == 2)
            color_name = QString("stress");
    }







    // mapper
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->SetVectorModeToMagnitude();
    lut->Build();
    mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();

    // colorbar

    scalarBar->SetLookupTable(mapper->GetLookupTable());
    scalarBar->SetTitle(color_name.toStdString().c_str());
    scalarBar->SetNumberOfLabels(10);
    scalarBar->SetDragable(true);


    // actor
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->EdgeVisibilityOn();
    actor->GetProperty()->SetAmbient(0.25);
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    // actor->GetProperty()->SetOpacity(100.0);
    //actor->GetProperty()->SetEdgeColor(255.0/255.0,255.0/255.0,255.0/255.0);
    // renderer
    renderer->AddActor(actor);
    renderer->AddActor2D(scalarBar);
    //renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    GetRenderWindow()->Render();
}

void VTKWidget::Fit()
{
    renderer->ResetCamera();
    GetRenderWindow()->Render();
}

void VTKWidget::Front()
{
    renderer->GetActiveCamera()->SetFocalPoint(0,0,0);
    renderer->GetActiveCamera()->SetPosition(100,0,0);
    renderer->GetActiveCamera()->SetViewUp(0,0,1);
    renderer->ResetCamera();
    GetRenderWindow()->Render();
}

void VTKWidget::Back()
{
    renderer->GetActiveCamera()->SetFocalPoint(0,0,0);
    renderer->GetActiveCamera()->SetPosition(-100,0,0);
    renderer->GetActiveCamera()->SetViewUp(0,0,1);
    renderer->ResetCamera();
    GetRenderWindow()->Render();
}

void VTKWidget::Top()
{
    renderer->GetActiveCamera()->SetFocalPoint(0,0,0);
    renderer->GetActiveCamera()->SetPosition(0,0,100);
    renderer->GetActiveCamera()->SetViewUp(0,1,0);
    renderer->ResetCamera();
    GetRenderWindow()->Render();
}

void VTKWidget::Bottom()
{
    renderer->GetActiveCamera()->SetFocalPoint(0,0,0);
    renderer->GetActiveCamera()->SetPosition(0,0,-100);
    renderer->GetActiveCamera()->SetViewUp(0,1,0);
    renderer->ResetCamera();
    GetRenderWindow()->Render();
}

void VTKWidget::Left()
{
    renderer->GetActiveCamera()->SetFocalPoint(0,0,0);
    renderer->GetActiveCamera()->SetPosition(0,-100,0);
    renderer->GetActiveCamera()->SetViewUp(0,0,1);
    renderer->ResetCamera();
    GetRenderWindow()->Render();
}

void VTKWidget::Right()
{
    renderer->GetActiveCamera()->SetFocalPoint(0,0,0);
    renderer->GetActiveCamera()->SetPosition(0,100,0);
    renderer->GetActiveCamera()->SetViewUp(0,0,1);
    renderer->ResetCamera();
    GetRenderWindow()->Render();
}

void VTKWidget::Axo()
{
    renderer->GetActiveCamera()->SetFocalPoint(0,0,0);
    renderer->GetActiveCamera()->SetPosition(100,100,100);
    renderer->GetActiveCamera()->SetViewUp(-1,-1,2);
    renderer->ResetCamera();
    GetRenderWindow()->Render();
}

void VTKWidget::ViewRotationH()
{
    renderer->GetActiveCamera()->Azimuth(1);
    GetRenderWindow()->Render();
}

void VTKWidget::ViewRotationV()
{
    renderer->GetActiveCamera()->Elevation(1);
    GetRenderWindow()->Render();
}

void VTKWidget::UpdateBndValue()
{
    if (SelectedVtkActor != NULL)
    {
        int n = bnds->Include(IVtkTools_ShapeDataSource::SafeDownCast(
                                  SelectedVtkActor->GetMapper()->GetInputConnection(0,0)->GetProducer()
                                  )->GetShape()->GetShape());

        if (phy_dw->ui->comboBox->currentIndex() == 0)
            (*bnds)[n]->SetType(Dirichlet);
        else if (phy_dw->ui->comboBox->currentIndex() == 1)
            (*bnds)[n]->SetType(Neumann);
        double t[3] = {phy_dw->ui->doubleSpinBox->value(),
                       phy_dw->ui->doubleSpinBox_2->value(),
                       phy_dw->ui->doubleSpinBox_3->value()};
        (*bnds)[n]->SetValue(t);
    }
}


#include "vtkVertexGlyphFilter.h"
#include "vtkUnsignedCharArray.h"
#include "vtkPointData.h"
#include "vtkFloatArray.h"

void VTKWidget::ImportVTKFileCloudColorFinal(double tol)
{
    vtkSmartPointer<vtkPoints> points =
            vtkSmartPointer<vtkPoints>::New();
    ifstream is;
    is.open(std::string("./data/final.vtk").c_str());
    const int len = 256;
    char L[len];
    double x[3];
    int num = 0;
    while (is.getline(L,len))
    {
        sscanf(L,"%lf %lf %lf", x, x+1, x+2);
        points->InsertNextPoint (x[0], x[1], x[2]);
        num++;
    }
    vtkSmartPointer<vtkPolyData> pointsPolydata =
            vtkSmartPointer<vtkPolyData>::New();

    pointsPolydata->SetPoints(points);

    vtkSmartPointer<vtkVertexGlyphFilter> vertexFilter =
            vtkSmartPointer<vtkVertexGlyphFilter>::New();
#if VTK_MAJOR_VERSION <= 5
    vertexFilter->SetInputConnection(pointsPolydata->GetProducerPort());
#else
    vertexFilter->SetInputData(pointsPolydata);
#endif
    vertexFilter->Update();

    vtkSmartPointer<vtkPolyData> polydata =
            vtkSmartPointer<vtkPolyData>::New();
    polydata->ShallowCopy(vertexFilter->GetOutput());

    polydata->BuildCells();
    // Setup colors
    double red[3] = {255, 0, 0};
    double green[3] = {0, 255, 0};
    double blue[3] = {0, 0, 255};

    vtkSmartPointer<vtkUnsignedCharArray> colors =
            vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(3);
    colors->SetName ("Colors");
    is.close();
    is.open(std::string("./data/color.vtk").c_str());
    while (is.getline(L,len))
    {
        sscanf(L,"%lf", x);
        if (x[0] > tol) {
            colors->InsertNextTuple(red);
        }
        else {
            colors->InsertNextTuple(green);
        }
    }
    polydata->GetPointData()->SetScalars(colors);
    polydata->Modified();

    // Visualization
    vtkSmartPointer<vtkPolyDataMapper> mapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
#if VTK_MAJOR_VERSION <= 5
    mapper->SetInputConnection(polydata->GetProducerPort());
#else
    mapper->SetInputData(polydata);
#endif


    meas_icp_final_color->SetMapper(mapper);
    meas_icp_final_color->GetProperty()->SetPointSize(5);



    renderer->AddActor(meas_icp_final_color);
    GetRenderWindow()->Render();


}

#include <vtkSTLReader.h>
void VTKWidget::ImportVTKFileAMStlModel(std::string name)
{
    fstream _file;
    _file.open(name, ios::in);
    if (!_file) return;
    // read a vtk file
    vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();
    reader->SetFileName(name.c_str());
    reader->Update();
    // mapper
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->Build();
    mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();
    // actor
    actor_am_stl_model = vtkSmartPointer<vtkActor>::New();
    actor_am_stl_model->SetMapper(mapper);
    actor_am_stl_model->GetProperty()->EdgeVisibilityOn();
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    // actor->GetProperty()->SetOpacity(100.0);
    // renderer
    renderer->AddActor(actor_am_stl_model);
    renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    GetRenderWindow()->Render();
}

#include "vtkPolyDataReader.h"

void VTKWidget::ImportVTKFileAMSlices(std::string name)
{
    fstream _file;
    _file.open(name, ios::in);
    if (!_file) return;
    // read a vtk file
    vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
    reader->SetFileName(name.c_str());
    reader->Update();
    // mapper
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->Build();
    mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();
    // actor
    actor_am_slices = vtkSmartPointer<vtkActor>::New();
    actor_am_slices->SetMapper(mapper);
    actor_am_slices->GetProperty()->EdgeVisibilityOn();
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    // actor->GetProperty()->SetOpacity(100.0);
    // renderer
    renderer->AddActor(actor_am_slices);
    renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    GetRenderWindow()->Render();
}

void VTKWidget::ImportVTKFileAMMesh(std::string name)
{
    fstream _file;
    _file.open(name, ios::in);
    if (!_file) return;
    // read a vtk file
    vtkSmartPointer<vtkUnstructuredGridReader> reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(name.c_str());
    reader->Update();
    // mapper
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->Build();
    mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();
    // actor
    actor_am_mesh = vtkSmartPointer<vtkActor>::New();
    actor_am_mesh->SetMapper(mapper);
    actor_am_mesh->GetProperty()->EdgeVisibilityOn();
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    // actor->GetProperty()->SetOpacity(100.0);
    // renderer
    renderer->AddActor(actor_am_mesh);
    //renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    GetRenderWindow()->Render();
}

void VTKWidget::ImportVTKFileAMPathPlanning(std::string name)
{
    fstream _file;
    _file.open(name, ios::in);
    if (!_file) return;
    // read a vtk file
    vtkSmartPointer<vtkUnstructuredGridReader> reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(name.c_str());
    reader->Update();
    // mapper
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->Build();
    mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();
    // actor
    actor_am_path_planning = vtkSmartPointer<vtkActor>::New();
    actor_am_path_planning->SetMapper(mapper);
    actor_am_path_planning->GetProperty()->EdgeVisibilityOn();
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    // actor->GetProperty()->SetOpacity(100.0);
    actor_am_path_planning->GetProperty()->SetColor(COLOR6);

    // renderer
    renderer->AddActor(actor_am_path_planning);
    //renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    GetRenderWindow()->Render();
}

//void VTKWidget::ShowAMPathPlanning (bool t)
//{
//    if (t) {
//        actor_am_path_planning->GetProperty()->SetEdgeVisibility(true);
//    }
//    else {
//        actor_am_path_planning->GetProperty()->SetEdgeVisibility(false);
//    }
//    GetRenderWindow()->Render();
//}































// *******************************************************
// measure
//void VTKWidget::MeasPlot (TopoDS_Shape S, bool t)
//{
//        IVtkOCC_Shape::Handle aShapeImpl = new IVtkOCC_Shape(S);
//        ObjectId += 1;


//        aShapeImpl->SetId(ObjectId);



//        // vtkPolyDataAlgorithm
//        vtkSmartPointer<IVtkTools_ShapeDataSource> aDS = vtkSmartPointer<IVtkTools_ShapeDataSource>::New();
//        aDS->SetShape(aShapeImpl);

//        // vtkAlgorithmOutput -> vtkAlgorithm
//        vtkSmartPointer<vtkPolyDataMapper> Mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
//        Mapper->SetInputConnection(aDS->GetOutputPort());

//        // actor
//        //renderer->RemoveActor(actor_meas_selected_bnd);
//        //        actor_meas_selected_bnd  = vtkSmartPointer<vtkActor>::New();
//        actor_meas_selected_bnd->SetMapper(Mapper);


//        actor_meas_selected_bnd->SetId(ObjectId);



//        actor_meas_selected_bnd->GetProperty()->SetColor(COLOR1);
//        actor_meas_selected_bnd->SetSelected(false);

//        // renderer
//        renderer->AddActor(actor_meas_selected_bnd);

//        if (t)
//        {
//                renderer->ResetCamera();
//        }

//        switch (S.ShapeType()) {
//        case 7:
//                aPicker->SetSelectionMode(aShapeImpl, SM_Vertex);
//                actor_meas_selected_bnd->GetProperty()->SetPointSize(10);
//                break;
//        case 6:
//                aPicker->SetSelectionMode(aShapeImpl, SM_Edge);
//                actor_meas_selected_bnd->GetProperty()->SetLineWidth(3);
//                break;
//        case 4:
//                aPicker->SetSelectionMode(aShapeImpl, SM_Face);
//                break;
//        default:
//                aPicker->SetSelectionMode(aShapeImpl, SM_Solid);
//                break;
//        }
//        // qvtkopenglwidget update has some problems, it seems didn't use render again.
//        // update();
//        GetRenderWindow()->Render();
//        return;
//}


void VTKWidget::ClearSelectedBnd()
{
    vtkActorCollection* acts = renderer->GetActors();
    int num = acts->GetNumberOfItems();
    acts->InitTraversal();
    for (int i = 0; i < num; ++i)
    {
        vtkActor* actor = acts->GetNextActor();
        actor->GetProperty()->SetColor(COLOR0);
        actor->SetSelected(false);
    }
    SelectedVtkActor = NULL;
    GetRenderWindow()->Render();
    selected_bnd_id.clear();
}





























// *******************************************************
// *******************************************************
//                        measure
// *******************************************************
// *******************************************************

#include "vtkTextActor.h"
#include "vtkTextProperty.h"

void VTKWidget::MeasurePlotCAD (TopoDS_Shape S, bool t)
{
    IVtkOCC_Shape::Handle aShapeImpl = new IVtkOCC_Shape(S);
    ObjectId += 1;


    aShapeImpl->SetId(ObjectId);



    // vtkPolyDataAlgorithm
    vtkSmartPointer<IVtkTools_ShapeDataSource> aDS = vtkSmartPointer<IVtkTools_ShapeDataSource>::New();
    aDS->SetShape(aShapeImpl);

    // vtkAlgorithmOutput -> vtkAlgorithm
    vtkSmartPointer<vtkPolyDataMapper> Mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    Mapper->SetInputConnection(aDS->GetOutputPort());

    // actor
    //vtkSmartPointer<vtkActor> Actor = vtkSmartPointer<vtkActor>::New();

    meas_cad_actor->SetMapper(Mapper);


    meas_cad_actor->SetId(ObjectId);

    meas_cad_actor->GetProperty()->SetOpacity(measure_op);


    meas_cad_actor->GetProperty()->SetColor(COLOR0);
    meas_cad_actor->SetVisibility(false);
    meas_cad_actor->SetSelected(false);

    // renderer

    Axo();





    if (t)
    {
        renderer->ResetCamera();
    }

    switch (S.ShapeType()) {
    case 7:
        aPicker->SetSelectionMode(aShapeImpl, SM_Vertex);
        meas_cad_actor->GetProperty()->SetPointSize(10);
        break;
    case 6:
        aPicker->SetSelectionMode(aShapeImpl, SM_Edge);
        meas_cad_actor->GetProperty()->SetLineWidth(3);
        break;
    case 4:
        aPicker->SetSelectionMode(aShapeImpl, SM_Face);
        break;
    default:
        aPicker->SetSelectionMode(aShapeImpl, SM_Solid);
        break;
    }
    SetSelectable(false);
    // qvtkopenglwidget update has some problems, it seems didn't use render again.
    // update();
    GetRenderWindow()->Render();
    return;
}

void VTKWidget::MeasurePlotCAD2 (TopoDS_Shape S, bool t)
{
    IVtkOCC_Shape::Handle aShapeImpl = new IVtkOCC_Shape(S);
    ObjectId += 1;


    aShapeImpl->SetId(ObjectId);



    // vtkPolyDataAlgorithm
    vtkSmartPointer<IVtkTools_ShapeDataSource> aDS = vtkSmartPointer<IVtkTools_ShapeDataSource>::New();
    aDS->SetShape(aShapeImpl);

    // vtkAlgorithmOutput -> vtkAlgorithm
    vtkSmartPointer<vtkPolyDataMapper> Mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    Mapper->SetInputConnection(aDS->GetOutputPort());

    // actor
    vtkSmartPointer<vtkActor> Actor = vtkSmartPointer<vtkActor>::New();
    meas_bnd_actors.push_back(Actor);

    Actor->SetMapper(Mapper);


    Actor->SetId(ObjectId);

    Actor->GetProperty()->SetOpacity(measure_op);


    Actor->GetProperty()->SetColor(COLOR0);
    Actor->SetSelected(false);

    // renderer
    renderer->AddActor(Actor);






    if (t)
    {
        renderer->ResetCamera();
    }


    switch (S.ShapeType()) {
    case 7:
        aPicker->SetSelectionMode(aShapeImpl, SM_Vertex);
        Actor->GetProperty()->SetPointSize(10);
        break;
    case 6:
        aPicker->SetSelectionMode(aShapeImpl, SM_Edge);
        Actor->GetProperty()->SetLineWidth(3);
        break;
    case 4:
        aPicker->SetSelectionMode(aShapeImpl, SM_Face);
        break;
    default:
        aPicker->SetSelectionMode(aShapeImpl, SM_Solid);
        break;
    }
    // qvtkopenglwidget update has some problems, it seems didn't use render again.
    // update();
    return;
}

void VTKWidget::MeasurePlotDomains ()
{
    //Clear();
    for (int i = 0; i < prims->size(); i++)
    {
        MeasurePlotCAD(*((*prims)[i]->Value()));
    }
}


void VTKWidget::MeasurePlotBnds ()
{
    Clear();
    bnds->Reset(prims);
    if (meas_bnd_actors.size()!=0)
        return;
    for (int i = 0; i < bnds->Size(); i++)
    {
        MeasurePlotCAD2(*((*bnds)[i]->Value()), false);
    }
    renderer->ResetCamera();
    //GetRenderWindow()->Render();




}

void VTKWidget::MeasureCADHide ()
{
    meas_cad_actor->VisibilityOff();
    GetRenderWindow()->Render();
}

void VTKWidget::MeasureCADOn ()
{
    meas_cad_actor->VisibilityOn();
    GetRenderWindow()->Render();
}

void VTKWidget::MeasureBndsHide ()
{
    for (int i=0; i<meas_bnd_actors.size(); i++)
        meas_bnd_actors[i]->VisibilityOff();
    GetRenderWindow()->Render();
}

void VTKWidget::MeasureBndsOn ()
{
    for (int i=0; i<meas_bnd_actors.size(); i++)
        meas_bnd_actors[i]->VisibilityOn();
    GetRenderWindow()->Render();
}

void VTKWidget::MeasureDeleteBnd()
{
    if (selected_bnd_id.size()==0) return;
    //meas_bnd_actors[selected_bnd_id[selected_bnd_id.size()-1]]->GetProperty()->SetColor(COLOR10);
    meas_bnd_actors[selected_bnd_id[selected_bnd_id.size()-1]]->GetProperty()->SetOpacity(0.05);
    meas_deleted_bnd.push_back(selected_bnd_id[selected_bnd_id.size()-1]);
    GetRenderWindow()->Render();
}

void VTKWidget::MeasureImportCloudTarget(std::string name, int r, int g, int b)
{
    if (selected_bnd_id.size()==0)
    {
        meas_cloud_target_actor->SetVisibility(false);
        return;
    }
    fstream _file;
    _file.open(name, ios::in);
    if (!_file) return;
    // read a vtk file
    //vtkSmartPointer<vtkSimplePointsReader> reader = vtkSmartPointer<vtkSimplePointsReader>::New();
    vtkSmartPointer<vtkUnstructuredGridReader> reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(name.c_str());
    std::cout << "check target file: " << name.c_str() << std::endl;
    reader->Update();
    // mapper
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->Build();
    mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();
    // actor
    //renderer->RemoveActor(meas_cloud_target_actor);
    //meas_cloud_target_actor = vtkSmartPointer<vtkActor>::New();
    meas_cloud_target_actor->SetMapper(mapper);
    //meas_cloud_target_actor->GetProperty()->EdgeVisibilityOn();
    meas_cloud_target_actor->GetProperty()->SetRepresentationToPoints();
    meas_cloud_target_actor->GetProperty()->SetPointSize(6);
    meas_cloud_target_actor->GetProperty()->SetColor(r,g,b);
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    // actor->GetProperty()->SetOpacity(100.0);
    // renderer
    renderer->AddActor(meas_cloud_target_actor);
    //renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    GetRenderWindow()->Render();




}

void VTKWidget::MeasureImportCloudICP(std::string name, int r, int g, int b)
{
    if (selected_bnd_id.size()==0)
    {
        meas_cloud_target_actor->SetVisibility(false);
        return;
    }
    fstream _file;
    _file.open(name, ios::in);
    if (!_file) return;
    // read a vtk file
    vtkSmartPointer<vtkSimplePointsReader> reader = vtkSmartPointer<vtkSimplePointsReader>::New();
    reader->SetFileName(name.c_str());
    reader->Update();
    // mapper
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->Build();
    mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();
    // actor
    //renderer->RemoveActor(meas_cloud_target_actor);
    //meas_cloud_target_actor = vtkSmartPointer<vtkActor>::New();
    meas_cloud_icp_actor->SetMapper(mapper);
    meas_cloud_icp_actor->GetProperty()->EdgeVisibilityOn();
    meas_cloud_icp_actor->GetProperty()->SetPointSize(6);
    meas_cloud_icp_actor->GetProperty()->SetColor(r,g,b);
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    // actor->GetProperty()->SetOpacity(100.0);
    // renderer
    //renderer->AddActor(meas_cloud_target_actor);
    //renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    GetRenderWindow()->Render();




}

void VTKWidget::MeasureCloudTargetHide(bool hide)
{
    if (hide)
        meas_cloud_target_actor->SetVisibility(false);
    else
        meas_cloud_target_actor->SetVisibility(true);
    GetRenderWindow()->Render();
}

void VTKWidget::MeasureCloudSourceHide(bool hide)
{
    if (hide)
        meas_cloud_source_actor->SetVisibility(false);
    else
        meas_cloud_source_actor->SetVisibility(true);
    GetRenderWindow()->Render();
}

void VTKWidget::MeasureCloudICPHide(bool hide)
{
    if (hide)
        meas_cloud_icp_actor->SetVisibility(false);
    else
        meas_cloud_icp_actor->SetVisibility(true);
    GetRenderWindow()->Render();
}

void VTKWidget::MeasureCloudGDTHide(bool hide)
{
    if (hide)
        meas_gdt_actor->SetVisibility(false);
    else
        meas_gdt_actor->SetVisibility(true);
    GetRenderWindow()->Render();
}

void VTKWidget::MeasureImportCloudSource2(std::string name, int r, int g, int b)
{
    fstream _file;
    _file.open(name, ios::in);
    if (!_file) return;
    // read a vtk file
    vtkSmartPointer<vtkSimplePointsReader> reader = vtkSmartPointer<vtkSimplePointsReader>::New();
    reader->SetFileName(name.c_str());
    reader->Update();
    // mapper
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->Build();
    mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();
    // actor
    renderer->RemoveActor(meas_cloud_source_actor);
    meas_cloud_source_actor = vtkSmartPointer<vtkActor>::New();
    meas_cloud_source_actor->SetMapper(mapper);
    meas_cloud_source_actor->GetProperty()->EdgeVisibilityOn();
    meas_cloud_source_actor->GetProperty()->SetPointSize(6);
    meas_cloud_source_actor->GetProperty()->SetColor(r,g,b);
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    // actor->GetProperty()->SetOpacity(100.0);
    // renderer
    renderer->AddActor(meas_cloud_source_actor);
    //renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    GetRenderWindow()->Render();




}


bool VTKWidget::MeasureImportCloudSource(std::string name)
{
    //    fstream _file;
    //    _file.open("./data/meas/fengsim_meas_cloud_source.vtk", ios::in);
    //    if (!_file) {
    //        return false;
    //    }

    // reader source
    vtkSmartPointer<vtkSimplePointsReader> reader = vtkSmartPointer<vtkSimplePointsReader>::New();
    reader->SetFileName("./data/meas/fengsim_meas_scene.vtk");
    reader->Update();

    // mapper
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());

    // actor
    renderer->RemoveActor(meas_cloud_source_actor);
    meas_cloud_source_actor = vtkSmartPointer<vtkActor>::New();
    meas_cloud_source_actor->SetMapper(mapper);
    //meas_cloud_source_actor->SetVisibility(false);
    meas_cloud_source_actor->GetProperty()->SetPointSize(3);
    meas_cloud_source_actor->GetProperty()->SetColor(255,0,0);

    meas_target[0] = meas_cloud_target_actor->GetCenter()[0];
    meas_target[1] = meas_cloud_target_actor->GetCenter()[1];
    meas_target[2] = meas_cloud_target_actor->GetCenter()[2];

    meas_source[0] = meas_cloud_source_actor->GetCenter()[0];
    meas_source[1] = meas_cloud_source_actor->GetCenter()[1];
    meas_source[2] = meas_cloud_source_actor->GetCenter()[2];

    std::cout << "target center: " << meas_target[0] << " "
              << meas_target[1] << " "
              << meas_target[2] << std::endl;
    std::cout << "source center: " << meas_source[0] << " "
              << meas_source[1] << " "
              << meas_source[2] << std::endl;





    // renderer
    renderer->AddActor(meas_cloud_source_actor);

    return true;
}

#include "vtkTransformPolyDataFilter.h"
#include "vtkTransform.h"
#include "vtkTransformFilter.h"

void VTKWidget::MeasureCloudSourceTransform(double x, double y, double z,
                                            double angle_x, double angle_y, double angle_z,
                                            QString filename)
{
    fstream _file;
    _file.open((filename+QString("/data/meas/fengsim_meas_scene2.vtk")).toStdString(), ios::in);
    if (!_file) {
        return;
    }


    // transform
    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    //transform->Translate(meas_target[0]+x, meas_target[1]+y, meas_target[2]+z);
    transform->Translate(x, y, z);
    transform->RotateZ(angle_z);
    transform->RotateY(angle_y);
    transform->RotateX(angle_x);
    //transform->Translate(-meas_source[0], -meas_source[1], -meas_source[2]);




    vtkSmartPointer<vtkMatrix4x4> T = vtkSmartPointer<vtkMatrix4x4>::New();
    transform->GetMatrix(T);



    ifstream is1;
    ofstream out1;
    is1.open((filename+QString("/data/meas/fengsim_meas_scene2.vtk")).toStdString());
    out1.open((filename+QString("/data/meas/_fengsim_meas_scene2.vtk")).toStdString());
    const int len = 256;
    char L[len];
    while (is1.getline(L,len))
    {
        double z1[4];
        z1[0] = 0;
        z1[1] = 0;
        z1[2] = 0;
        z1[3] = 1;
        sscanf(L, "%lf %lf %lf", z1, z1 + 1, z1 + 2);
        double z2[4];
        z2[0] = 0;
        z2[1] = 0;
        z2[2] = 0;
        z2[3] = 0;
        for (int i = 0; i<4; i++) {
            for (int j = 0; j<4; j++) {
                z2[i] += T->Element[i][j]*z1[j];
            }
        }
        out1 << setprecision(16) << z2[0] << " " << z2[1] << " " << z2[2] << endl;

        //                int k = 0;
        //                while (k < 100) {
        //                        is1.getline(L,len);
        //                        k++;
        //                }

    }
    is1.close();
    out1.close();

    out1.open((filename+QString("/data/meas/matrix.txt")).toStdString());
    for (int i=0; i<4; i++) {
        for (int j=0; j<4; j++) {
            out1 << T->Element[i][j] << " ";
        }
        out1 << std::endl;
    }
    out1.close();



    // reader source
    vtkSmartPointer<vtkSimplePointsReader> reader = vtkSmartPointer<vtkSimplePointsReader>::New();
    reader->SetFileName((filename+QString("/data/meas/_fengsim_meas_scene2.vtk")).toStdString().c_str());
    reader->Update();



    //        // transformfilter
    //        vtkSmartPointer<vtkTransformFilter> transformFilter =
    //                        vtkSmartPointer<vtkTransformFilter>::New();
    //        transformFilter->SetInputConnection(reader->GetOutputPort());
    //        transformFilter->SetTransform(transform);
    //        transformFilter->Update();

    // mapper
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    //mapper->SetInputConnection(transformFilter->GetOutputPort());
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->Build();
    mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();

    // actor
    renderer->RemoveActor(meas_cloud_source_actor);
    meas_cloud_source_actor = vtkSmartPointer<vtkActor>::New();
    meas_cloud_source_actor->SetMapper(mapper);
    meas_cloud_source_actor->GetProperty()->EdgeVisibilityOn();
    meas_cloud_source_actor->GetProperty()->SetPointSize(6);
    meas_cloud_source_actor->GetProperty()->SetColor(255,0,0);

    // renderer
    renderer->AddActor(meas_cloud_source_actor);
    //renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    GetRenderWindow()->Render();













    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    // actor->GetProperty()->SetOpacity(100.0);
    // renderer











    // filter
    //        vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter =    vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    //        transformFilter->SetInputConnection(meas_cloud_source_reader->GetOutputPort());
    //        transformFilter->SetTransform(transform);
    //        transformFilter->Update();
    //        // mapper
    //        vtkSmartPointer<vtkPolyDataMapper> transformedMapper =
    //                        vtkSmartPointer<vtkPolyDataMapper>::New();
    //        transformedMapper->SetInputConnection(transformFilter->GetOutputPort());
    //        vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    //        lut->SetHueRange(0.666667,0.0);
    //        double * range = meas_cloud_source_reader->GetOutput()->GetScalarRange();
    //        lut->Build();
    //        transformedMapper->SetScalarRange(range);
    //        transformedMapper->SetLookupTable(lut);
    //        transformedMapper->Update();
    //        // actor
    //        //renderer->RemoveActor(meas_cloud_source_actor);
    //        //meas_cloud_source_actor = vtkSmartPointer<vtkActor>::New();
    //        meas_cloud_source_actor->SetMapper(transformedMapper);
    //        meas_cloud_source_actor->GetProperty()->EdgeVisibilityOn();
    //        meas_cloud_source_actor->GetProperty()->SetPointSize(6);
    //        meas_cloud_source_actor->GetProperty()->SetColor(255,0,0);



    // renderer
    //        renderer->AddActor(meas_cloud_source_actor);

}

#include "vtkPolyDataWriter.h"

bool VTKWidget::MeasureExportCloudSource(std::string name, double x, double y, double z, double angle_x, double angle_y, double angle_z)
{

    fstream _file;
    _file.open("./data/meas/fengsim_meas_cloud_source.vtk", ios::in);
    if (!_file) {
        return false;
    }


    QFile::remove("./data/meas/fengsim_meas_cloud_source2.vtk");


    // reader source
    vtkSmartPointer<vtkSimplePointsReader> reader = vtkSmartPointer<vtkSimplePointsReader>::New();
    reader->SetFileName("./data/meas/fengsim_meas_cloud_source.vtk");
    reader->Update();
    //        for (int i = 0; i < reader->GetOutput()->GetNumberOfPoints(); i++)
    //        {
    //                double pp[3];
    //                reader->GetOutput()->GetPoint(i,pp);
    //                //std::cout << setprecision(16) << pp[0] << " " << pp[1] << " " << pp[2] << std::endl;
    //        }





    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    points->SetDataTypeToDouble();
    ifstream is;
    is.open("./data/meas/fengsim_meas_cloud_source.vtk");
    const int len = 256;
    char L[len];
    double tol1 = 0;
    while (is.getline(L,len))
    {
        double z[3];
        z[0] = 0;
        z[1] = 0;
        z[2] = 0;
        sscanf(L, "%lf %lf %lf", z, z + 1, z + 2);
        points->InsertNextPoint(z[0], z[1], z[2]);
        double tol2 = abs(sqrt(z[0]*z[0]+z[1]*z[1]+z[2]*z[2])-5);
        if (tol2>tol1) tol1 = tol2;
        //std::cout << L << std::endl;
        //std::cout << setiosflags(ios::fixed) << setprecision(16) << z[0] << " " << z[1] << " " << z[2] << std::endl;
    }
    is.close();
    std::cout << "source tol: " << tol1 << std::endl;
    vtkSmartPointer<vtkPolyData> polygonPolyData = vtkSmartPointer<vtkPolyData>::New();
    polygonPolyData->SetPoints(points);





    // transform



    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->Translate(meas_target[0]+x, meas_target[1]+y, meas_target[2]+z);
    transform->RotateZ(angle_z);
    transform->RotateY(angle_y);
    transform->RotateX(angle_x);
    transform->Translate(-meas_source[0], -meas_source[1], -meas_source[2]);

    vtkSmartPointer<vtkMatrix4x4> T = vtkSmartPointer<vtkMatrix4x4>::New();
    transform->GetMatrix(T);

    for (int i = 0; i<4; i++) {
        for (int j = 0; j<4; j++) {
            std::cout << "T: " << T->Element[i][j] << " ";
        }
        std::cout << std::endl;
    }





    ifstream is1;
    ofstream out1;
    is1.open("./data/meas/fengsim_meas_cloud_source.vtk");
    out1.open("./data/meas/fengsim_meas_cloud_source2.vtk");
    while (is1.getline(L,len))
    {
        double z1[4];
        z1[0] = 0;
        z1[1] = 0;
        z1[2] = 0;
        z1[3] = 1;
        sscanf(L, "%lf %lf %lf", z1, z1 + 1, z1 + 2);
        double z2[4];
        z2[0] = 0;
        z2[1] = 0;
        z2[2] = 0;
        z2[3] = 0;
        for (int i = 0; i<4; i++) {
            for (int j = 0; j<4; j++) {
                z2[i] += T->Element[i][j]*z1[j];
            }
        }
        out1 << setprecision(16) << z2[0] << " " << z2[1] << " " << z2[2] << endl;


        //                int k = 0;
        //                while (k < 100) {
        //                        is1.getline(L,len);
        //                        k++;
        //                }

    }
    is1.close();
    out1.close();


    return true;

















    // transform filter
    vtkSmartPointer<vtkTransformFilter> transformFilter =
            vtkSmartPointer<vtkTransformFilter>::New();
    transformFilter->SetInputConnection(reader->GetOutputPort());
    //transformFilter->SetInputData(polygonPolyData);
    transformFilter->SetTransform(transform);
    transformFilter->Update();

    // mapper
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(transformFilter->GetOutputPort());
    mapper->Update();

    // writer
    vtkSmartPointer<vtkSimplePointsWriter> meas_cloud_source_writer =  vtkSmartPointer<vtkSimplePointsWriter>::New();
    meas_cloud_source_writer->SetDecimalPrecision(16);
    //meas_cloud_source_writer->SetInputConnection(reader->GetOutputPort());
    //meas_cloud_source_writer->SetInputData(reader->GetOutput());
    //meas_cloud_source_writer->SetInputData(transformFilter->GetOutput());
    //meas_cloud_source_writer->SetInputConnection(transformFilter->GetOutputPort());
    meas_cloud_source_writer->SetInputData(polygonPolyData);
    meas_cloud_source_writer->SetFileName(name.c_str());
    meas_cloud_source_writer->Write();




    return true;
}


void VTKWidget::MeasureClearCloudSource ()
{

    renderer->RemoveActor(meas_cloud_source_actor);
    meas_cloud_source_actor = vtkSmartPointer<vtkActor>::New();
    renderer->AddActor(meas_cloud_source_actor);
    GetRenderWindow()->Render();

}

void VTKWidget::MeasureClearCloudTarget()
{

    renderer->RemoveActor(meas_cloud_target_actor);
    meas_cloud_target_actor = vtkSmartPointer<vtkActor>::New();
    renderer->AddActor(meas_cloud_target_actor);
    GetRenderWindow()->Render();

}

void VTKWidget::MeasureClearAll()
{
    renderer->RemoveActor(meas_cloud_target_actor);
    renderer->RemoveActor(meas_cloud_source_actor);
    renderer->RemoveActor(meas_cad_actor);
    renderer->RemoveActor(meas_cloud_icp_actor);
    renderer->RemoveActor(meas_gdt_actor);
    for (int i=0; i<meas_bnd_actors.size(); i++)
        renderer->RemoveActor(meas_bnd_actors[i]);
    meas_bnd_actors.clear();
    selected_bnd_id.clear();
    meas_cloud_target_actor = vtkSmartPointer<vtkActor>::New();
    meas_cloud_source_actor = vtkSmartPointer<vtkActor>::New();
    meas_cad_actor = vtkSmartPointer<vtkActor>::New();
    meas_cloud_icp_actor = vtkSmartPointer<vtkActor>::New();
    meas_gdt_actor = vtkSmartPointer<vtkActor>::New();
    renderer->AddActor(meas_cloud_target_actor);
    renderer->AddActor(meas_cloud_source_actor);
    renderer->AddActor(meas_cad_actor);
    renderer->AddActor(meas_cloud_icp_actor);
    renderer->AddActor(meas_gdt_actor);
    ObjectId = 0;
    measure_op = 1.0;
    SelectedVtkActor = NULL;
    // strange must redefine aPicker again ????????????
    aPicker = vtkSmartPointer<IVtkTools_ShapePicker>::New();
    aPicker->SetRenderer(renderer);
    GetRenderWindow()->Render();
}



#include "vtkParticleReader.h".h"
void VTKWidget::MeasureImportICPFinal(std::string name)
{
    fstream _file;
    _file.open(name, ios::in);
    if (!_file) return;
    // read a vtk file
    vtkSmartPointer<vtkParticleReader> reader = vtkSmartPointer<vtkParticleReader>::New();
    //vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
    reader->SetFileName(name.c_str());
    reader->Update();
    // mapper
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->Build();
    mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();
    // actor
    renderer->RemoveActor(meas_gdt_actor);
    meas_gdt_actor = vtkSmartPointer<vtkActor>::New();
    meas_gdt_actor->SetMapper(mapper);
    meas_gdt_actor->GetProperty()->EdgeVisibilityOn();
    meas_gdt_actor->GetProperty()->SetPointSize(6);
    //meas_icp_final->GetProperty()->SetColor(0,255,0);
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    // actor->GetProperty()->SetOpacity(100.0);
    // renderer
    renderer->AddActor(meas_gdt_actor);
    //renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    GetRenderWindow()->Render();
}

void VTKWidget::MeasureClearICPFinal ()
{
    //        Clear(meas_icp_final);
    renderer->RemoveActor(meas_icp_final);
    GetRenderWindow()->Render();

}



#include "vtkImplicitPolyDataDistance.h"
void VTKWidget::MeasureImportColorICPFinal(std::string name)
{
    //        fstream _file;
    //        _file.open(name, ios::in);
    //        if (!_file) return;
    //        // read a vtk file
    //        vtkSmartPointer<vtkUnstructuredGridReader> reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    //        reader->SetFileName("./data/mesh/fengsim_mesh.vtk");
    //        reader->Update();

    //        vtkSmartPointer<vtkImplicitPolyDataDistance> implicitPolyDataDistance =
    //                        vtkSmartPointer<vtkImplicitPolyDataDistance>::New();
    //        implicitPolyDataDistance->SetInput(reader->GetOutput());

    //        fstream _file;
    //        _file.open(name, ios::in);
    //        if (!_file) return;
    //        // read a vtk file
    //        vtkSmartPointer<vtkSimplePointsReader> reader = vtkSmartPointer<vtkSimplePointsReader>::New();
    //        //vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
    //        reader->SetFileName(name.c_str());
    //        reader->Update();
    //        // mapper
    //        vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    //        mapper->SetInputConnection(reader->GetOutputPort());
    //        vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    //        lut->SetHueRange(0.666667,0.0);
    //        double * range = reader->GetOutput()->GetScalarRange();
    //        lut->Build();
    //        mapper->SetScalarRange(range);
    //        mapper->SetLookupTable(lut);
    //        mapper->Update();
    //        // actor
    //        meas_icp_final = vtkSmartPointer<vtkActor>::New();
    //        meas_icp_final->SetMapper(mapper);
    //        meas_icp_final->GetProperty()->EdgeVisibilityOn();
    //        meas_icp_final->GetProperty()->SetPointSize(6);
    //        meas_icp_final->GetProperty()->SetColor(0,255,0);
    //        // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    //        // actor->GetProperty()->SetOpacity(100.0);
    //        // renderer
    //        renderer->AddActor(meas_icp_final);
    //        //renderer->ResetCamera();
    //        // Automatically set up the camera based on the visible actors.
    //        // The camera will reposition itself to view the center point of the actors,
    //        // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    //        // so that all of the actors can be seen.
    //        renderer->ResetCameraClippingRange();
    //        // Reset the camera clipping range based on the bounds of the visible actors.
    //        // This ensures that no props are cut off
    //        // redraw
    //        GetRenderWindow()->Render();
}

void VTKWidget::MeasureSetSelectedBndsUnvisible()
{
    vtkActorCollection* acts = renderer->GetActors();
    int num = acts->GetNumberOfItems();
    acts->InitTraversal();
    for (int i = 0; i < num; ++i)
    {

        vtkActor* actor = acts->GetNextActor();

        if (actor == meas_cloud_source_actor) continue;



        actor->GetProperty()->SetColor(COLOR0);



        actor->SetSelected(false);
    }
    SelectedVtkActor = NULL;
    GetRenderWindow()->Render();
    //selected_bnd_id.clear();
}

#include "vtkIterativeClosestPointTransform.h"
#include "vtkLandmarkTransform.h"
void VTKWidget::MeasureVTKICP()
{
    vtkSmartPointer<vtkSimplePointsReader> reader_target = vtkSmartPointer<vtkSimplePointsReader>::New();
    reader_target->SetFileName("./data/meas/fengsim_meas_cloud_target.vtk");
    reader_target->Update();
    vtkSmartPointer<vtkSimplePointsReader> reader_source = vtkSmartPointer<vtkSimplePointsReader>::New();
    reader_source->SetFileName("./data/meas/fengsim_meas_cloud_source2.vtk");
    reader_source->Update();






    vtkSmartPointer<vtkPoints> points_target = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkPoints> points_source = vtkSmartPointer<vtkPoints>::New();
    points_target->SetDataTypeToDouble();
    points_source->SetDataTypeToDouble();
    ifstream is;
    is.open("./data/meas/fengsim_meas_cloud_target.vtk");
    const int len = 256;
    char L[len];
    while (is.getline(L,len))
    {
        double z[3];
        z[0] = 0;
        z[1] = 0;
        z[2] = 0;
        sscanf(L, "%lf %lf %lf", z, z + 1, z + 2);
        points_target->InsertNextPoint(z[0], z[1], z[2]);
    }
    is.close();
    is.open("./data/meas/fengsim_meas_cloud_source2.vtk");
    while (is.getline(L,len))
    {
        double z[3];
        z[0] = 0;
        z[1] = 0;
        z[2] = 0;
        sscanf(L, "%lf %lf %lf", z, z + 1, z + 2);
        points_source->InsertNextPoint(z[0], z[1], z[2]);
    }
    is.close();
    vtkSmartPointer<vtkPolyData> polygonPolyData_target = vtkSmartPointer<vtkPolyData>::New();
    polygonPolyData_target->SetPoints(points_target);
    vtkSmartPointer<vtkPolyData> polygonPolyData_source = vtkSmartPointer<vtkPolyData>::New();
    polygonPolyData_source->SetPoints(points_source);







    vtkSmartPointer<vtkIterativeClosestPointTransform> icp =
            vtkSmartPointer<vtkIterativeClosestPointTransform>::New();
    icp->SetSource(reader_source->GetOutput());
    icp->SetTarget(reader_target->GetOutput());
    //        icp->SetSource(polygonPolyData_source);
    //        icp->SetTarget(polygonPolyData_target);
    icp->GetLandmarkTransform()->SetModeToRigidBody();
    icp->SetMaximumNumberOfIterations(1000);
    icp->StartByMatchingCentroidsOn();
    icp->Modified();
    icp->Update();

    vtkSmartPointer<vtkMatrix4x4> m = icp->GetMatrix();
    std::cout << "The resulting matrix is: " << *m << std::endl;

    vtkSmartPointer<vtkTransformPolyDataFilter> icpTransformFilter =
            vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    //icpTransformFilter->SetInputData(reader_source->GetOutput());
    icpTransformFilter->SetInputConnection(reader_source->GetOutputPort());
    //icpTransformFilter->SetInputData(polygonPolyData_source);
    icpTransformFilter->SetTransform(icp);
    icpTransformFilter->Update();

    vtkSmartPointer<vtkPolyDataMapper> solutionMapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
    solutionMapper->SetInputConnection(icpTransformFilter->GetOutputPort());

    meas_icp_final = vtkSmartPointer<vtkActor>::New();
    meas_icp_final->SetMapper(solutionMapper);
    meas_icp_final->GetProperty()->EdgeVisibilityOn();
    meas_icp_final->GetProperty()->SetPointSize(6);
    meas_icp_final->GetProperty()->SetColor(0,255,0);
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    // actor->GetProperty()->SetOpacity(100.0);
    // renderer
    renderer->AddActor(meas_icp_final);
    renderer->ResetCameraClippingRange();
    GetRenderWindow()->Render();



    vtkSmartPointer<vtkSimplePointsWriter> meas_vtk_icp_writer =  vtkSmartPointer<vtkSimplePointsWriter>::New();
    //meas_cloud_source_writer->SetInputConnection(reader->GetOutputPort());
    meas_vtk_icp_writer->SetDecimalPrecision(16);
    meas_vtk_icp_writer->SetInputConnection(icpTransformFilter->GetOutputPort());
    meas_vtk_icp_writer->SetFileName("./data/meas/fengsim_meas_icp_final.vtk");
    meas_vtk_icp_writer->Write();















}




void VTKWidget::MeasureExportModel()
{

}

void VTKWidget::MeasureOpacity(double p)
{
    measure_op = p / 100.0;
    vtkActorCollection* acts = renderer->GetActors();
    int num = acts->GetNumberOfItems();
    acts->InitTraversal();
    for (int i = 0; i < num; ++i)
    {
        vtkActor* actor = acts->GetNextActor();
        if (actor==meas_cloud_source_actor) continue;
        if (actor==meas_cloud_target_actor) continue;
        if (actor==meas_cloud_icp_actor) continue;
        actor->GetProperty()->SetOpacity(measure_op);
        std::cout << "check vtkwidget opacity: " << measure_op << std::endl;
    }
    renderer->ResetCameraClippingRange();
    GetRenderWindow()->Render();
}


#include "vtkUnstructuredGridWriter.h"
#include "vtkPolyDataWriter.h"
#include "vtkPointCloudFilter.h"
#include "vtkVoxelGrid.h"
#include "vtkVertex.h"
#include "vtkIdTypeArray.h"

void VTKWidget::MeasureImportSource(std::string name, QString path)
{
    meas_reader->SetFileName(name.c_str());
    meas_reader->Update();




    meas_us->SetInputConnection(meas_reader->GetOutputPort());
    meas_us->SetConfigurationStyleToLeafSize();
    meas_us->SetLeafSize(0.5,0.5,0.5);
    meas_us->Update();
    std::cout << meas_us->GetOutput()->GetNumberOfCells() << std::endl;
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
    for (int i=0; i<meas_us->GetOutput()->GetNumberOfPoints(); i++)
    {
        vtkSmartPointer<vtkVertex> vertex = vtkSmartPointer<vtkVertex>::New();
        vertex->GetPointIds()->SetNumberOfIds(1);
        vertex->GetPointIds()->SetId(0,i);
        cells->InsertNextCell(vertex);
    }
    meas_us->GetOutput()->SetVerts(cells);




    vtkSmartPointer<vtkSimplePointsWriter> meas_scene_writer =  vtkSmartPointer<vtkSimplePointsWriter>::New();
    meas_scene_writer->SetDecimalPrecision(5);
    meas_scene_writer->SetInputConnection(meas_reader->GetOutputPort());
    meas_scene_writer->SetFileName((path + QString("/data/meas/fengsim_meas_source.vtk")).toStdString().c_str());
    meas_scene_writer->Write();


    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(meas_us->GetOutputPort());
    mapper->Update();

    meas_cloud_source_actor->SetMapper(mapper);
    meas_cloud_source_actor->GetProperty()->SetPointSize(6);
    meas_cloud_source_actor->GetProperty()->SetColor(255,0,0);

}

void VTKWidget::MeasureSTLTransform(QString path)
{
    // transform
    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    vtkSmartPointer<vtkMatrix4x4> T = vtkSmartPointer<vtkMatrix4x4>::New();
    ifstream is;
    is.open((path + QString("/data/meas/trans_matrix")).toStdString().c_str());
    double z[4];
    const int len = 256;
    char L[len];
    for (int i=0; i<4; i++)
    {
        is.getline(L,len);
        sscanf(L, "%lf %lf %lf %lf", z, z+1, z+2, z+3);
        for (int j=0; j<4; j++) {
            T->Element[i][j] = z[j];
        }
    }
    is.close();
    transform->SetMatrix(T);

    meas_un_transformFilter->SetInputConnection(meas_us->GetOutputPort());
    meas_un_transformFilter->SetTransform(transform);
    meas_un_transformFilter->Update();
}

#include "vtkPointDensityFilter.h"
void VTKWidget::MeasureSTLTransform2(QString path)
{
    // transform
    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    vtkSmartPointer<vtkMatrix4x4> T = vtkSmartPointer<vtkMatrix4x4>::New();
    ifstream is;
    is.open((path + QString("/data/meas/trans_matrix")).toStdString().c_str());
    double z[4];
    const int len = 256;
    char L[len];
    for (int i=0; i<4; i++)
    {
        is.getline(L,len);
        sscanf(L, "%lf %lf %lf %lf", z, z+1, z+2, z+3);
        for (int j=0; j<4; j++)
            T->Element[i][j] = z[j];
    }
    is.close();
    transform->SetMatrix(T);

    meas_transformFilter->SetInputConnection(meas_reader->GetOutputPort());
    meas_transformFilter->SetTransform(transform);
    meas_transformFilter->Update();


    unsigned seed;
    seed = time(0);
    srand(seed);
    double t = 0;
    for (int i=0; i<100; i++)
    {
        int n = rand()%meas_transformFilter->GetOutput()->GetNumberOfCells();
        t += meas_transformFilter->GetOutput()->GetCell(n)->GetLength2();
    }
    std::cout << t << std::endl;
    meas_source_density = floor(0.5/(t/100.0));
    std::cout << "meas_source_cell_size: " << meas_source_density << std::endl;
    //std::cout << "meas_source_cell_number: " << meas_source_cell_number << std::endl;


}

int VTKWidget::MeasureTransformCellsNumber ()
{
    return meas_transformFilter->GetOutput()->GetNumberOfCells();
}

void VTKWidget::MeasureTransformCell (int i, int j, double& x, double& y, double& z)
{
    vtkCell*  cell = meas_transformFilter->GetOutput()->GetCell(i);
    vtkPoints* ps = cell->GetPoints();
    x = ps->GetPoint(j)[0];
    y = ps->GetPoint(j)[1];
    z = ps->GetPoint(j)[2];
}

void VTKWidget::MeasureImportSource2(QString path)
{


    renderer->ResetCameraClippingRange();
    GetRenderWindow()->Render();
}

void VTKWidget::MeasuresPlotICPtrans(double r, double g, double b)
{
    // mapper

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(meas_un_transformFilter->GetOutputPort());
    //vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    //    lut->SetHueRange(0.666667,0.0);
    //    double * range = meas_un_transformFilter->GetOutput()->GetScalarRange();
    //    lut->Build();
    //    mapper->SetScalarRange(range);
    //    mapper->SetLookupTable(lut);
    //mapper->Update();
    // actor
    //renderer->RemoveActor(meas_cloud_target_actor);
    //meas_cloud_target_actor = vtkSmartPointer<vtkActor>::New();

    meas_cloud_icp_actor->SetMapper(mapper);
    meas_cloud_icp_actor->GetProperty()->SetRepresentationToPoints();
    meas_cloud_icp_actor->GetProperty()->SetPointSize(6);
    meas_cloud_icp_actor->GetProperty()->SetColor(r,g,b);









    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    // actor->GetProperty()->SetOpacity(100.0);
    // renderer



    renderer->AddActor(meas_cloud_icp_actor);



    //renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();

    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    GetRenderWindow()->Render();

}

// *******************************************************
// additive manufacturing

void VTKWidget::AMImportCAD (TopoDS_Shape S, bool t)
{
    IVtkOCC_Shape::Handle aShapeImpl = new IVtkOCC_Shape(S);
    ObjectId += 1;


    aShapeImpl->SetId(ObjectId);



    // vtkPolyDataAlgorithm
    vtkSmartPointer<IVtkTools_ShapeDataSource> aDS = vtkSmartPointer<IVtkTools_ShapeDataSource>::New();
    aDS->SetShape(aShapeImpl);

    // vtkAlgorithmOutput -> vtkAlgorithm
    vtkSmartPointer<vtkPolyDataMapper> Mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    Mapper->SetInputConnection(aDS->GetOutputPort());

    // actor
    renderer->RemoveActor(am_cad_actor);
    am_cad_actor = vtkSmartPointer<vtkActor>::New();
    am_cad_actor->SetMapper(Mapper);


    am_cad_actor->SetId(ObjectId);



    am_cad_actor->GetProperty()->SetColor(COLOR0);
    am_cad_actor->GetProperty()->SetOpacity(measure_op);
    am_cad_actor->SetSelected(false);

    // renderer
    renderer->AddActor(am_cad_actor);

    if (t)
    {
        renderer->ResetCamera();
    }

    switch (S.ShapeType()) {
    case 7:
        aPicker->SetSelectionMode(aShapeImpl, SM_Vertex);
        am_cad_actor->GetProperty()->SetPointSize(10);
        break;
    case 6:
        aPicker->SetSelectionMode(aShapeImpl, SM_Edge);
        am_cad_actor->GetProperty()->SetLineWidth(3);
        break;
    case 4:
        aPicker->SetSelectionMode(aShapeImpl, SM_Face);
        break;
    default:
        aPicker->SetSelectionMode(aShapeImpl, SM_Solid);
        break;
    }
    // qvtkopenglwidget update has some problems, it seems didn't use render again.
    // update();
    GetRenderWindow()->Render();
    return;
}

void VTKWidget::AMSetCADVisible(bool t)
{
    if (t) {
        am_cad_actor->VisibilityOn();
    }
    else {
        am_cad_actor->VisibilityOff();
    }
    GetRenderWindow()->Render();
}


#include <vtkSTLReader.h>
void VTKWidget::AMImportSTL ()
{
    fstream _file;
    _file.open("./data/am/am.stl", ios::in);
    if (!_file) return;
    // read a vtk file
    vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();
    reader->SetFileName("./data/am/am.stl");
    reader->Update();
    // mapper
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->Build();
    mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();
    // actor
    renderer->RemoveActor(am_stl_actor);
    am_stl_actor = vtkSmartPointer<vtkActor>::New();
    am_stl_actor->SetMapper(mapper);
    am_stl_actor->GetProperty()->EdgeVisibilityOn();
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    // actor->GetProperty()->SetOpacity(100.0);
    // renderer
    renderer->AddActor(am_stl_actor);
    //renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    GetRenderWindow()->Render();
}

void VTKWidget::AMSetSTLVisible(bool t)
{
    if (t) {
        am_stl_actor->VisibilityOn();
    }
    else {
        am_stl_actor->VisibilityOff();
    }
    GetRenderWindow()->Render();
}

void VTKWidget::AMImportSlices()
{
    fstream _file;
    _file.open("./data/am/slices.vtk", ios::in);
    if (!_file) return;
    // read a vtk file
    vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
    reader->SetFileName("./data/am/slices.vtk");
    reader->Update();
    // mapper
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->Build();
    mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();
    // actor
    renderer->RemoveActor(am_slices_actor);
    am_slices_actor = vtkSmartPointer<vtkActor>::New();
    am_slices_actor->SetMapper(mapper);
    am_slices_actor->GetProperty()->EdgeVisibilityOn();
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    // actor->GetProperty()->SetOpacity(100.0);
    // renderer
    renderer->AddActor(am_slices_actor);
    // renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    GetRenderWindow()->Render();
}

void VTKWidget::AMSetSlicesVisible(bool t)
{
    if (t) {
        am_slices_actor->VisibilityOn();
    }
    else {
        am_slices_actor->VisibilityOff();
    }
    GetRenderWindow()->Render();
}

void VTKWidget::AMImportPathPlanning()
{
    fstream _file;
    _file.open("./data/am/pathplanning.vtk", ios::in);
    if (!_file) return;
    // read a vtk file
    vtkSmartPointer<vtkUnstructuredGridReader> reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName("./data/am/pathplanning.vtk");
    reader->Update();
    // mapper
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->Build();
    mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();
    // actor
    renderer->RemoveActor(am_pathplanning_actor);
    am_pathplanning_actor = vtkSmartPointer<vtkActor>::New();
    am_pathplanning_actor->SetMapper(mapper);
    am_pathplanning_actor->GetProperty()->EdgeVisibilityOn();
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    // actor->GetProperty()->SetOpacity(100.0);
    am_pathplanning_actor->GetProperty()->SetColor(COLOR6);

    // renderer
    renderer->AddActor(am_pathplanning_actor);
    //renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    GetRenderWindow()->Render();
}

void VTKWidget::AMSetPathPlanningVisible (bool t)
{
    if (t) {
        am_pathplanning_actor->VisibilityOn();
    }
    else {
        am_pathplanning_actor->VisibilityOff();
    }
    GetRenderWindow()->Render();
}

void VTKWidget::AMImportMesh()
{
    fstream _file;
    _file.open("./data/am/mesh.vtk", ios::in);
    if (!_file) return;
    // read a vtk file
    vtkSmartPointer<vtkUnstructuredGridReader> reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName("./data/am/mesh.vtk");
    reader->Update();
    // mapper
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->Build();
    mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();
    // actor
    renderer->RemoveActor(am_mesh_actor);
    am_mesh_actor = vtkSmartPointer<vtkActor>::New();
    am_mesh_actor->SetMapper(mapper);
    am_mesh_actor->GetProperty()->EdgeVisibilityOn();
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    // actor->GetProperty()->SetOpacity(100.0);
    // renderer
    renderer->AddActor(am_mesh_actor);
    //renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    GetRenderWindow()->Render();
}

void VTKWidget::AMSetMeshVisible (bool t)
{
    if (t) {
        am_mesh_actor->VisibilityOn();
    }
    else {
        am_mesh_actor->VisibilityOff();
    }
    GetRenderWindow()->Render();
}

void VTKWidget::AMImportSimulation (int num)
{
    fstream _file;
    _file.open("./../../AM/data/vtk/am_mesh_" + std::to_string(num) + ".vtk", ios::in);
    if (!_file) return;
    // read a vtk file
    vtkSmartPointer<vtkUnstructuredGridReader> reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(std::string("./../../AM/data/vtk/am_mesh_" + std::to_string(num) + ".vtk").c_str());
    reader->Update();
    // mapper
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->SetVectorModeToMagnitude();
    lut->Build();
    mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();
    // actor
    renderer->RemoveActor(am_simulation_actor);
    am_simulation_actor = vtkSmartPointer<vtkActor>::New();
    am_simulation_actor->SetMapper(mapper);
    am_simulation_actor->GetProperty()->EdgeVisibilityOn();
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    am_simulation_actor->GetProperty()->SetLineWidth(1);
    // renderer
    renderer->AddActor(am_simulation_actor);
    //renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    //GetRenderWindow()->Render();
}


void VTKWidget::AMImportSource(int num)
{
    fstream _file;
    _file.open("./../../AM/data/vtk/am_current_pos_" + std::to_string(num) + ".vtk", ios::in);
    if (!_file) return;
    // read a vtk file
    vtkSmartPointer<vtkSimplePointsReader> reader = vtkSmartPointer<vtkSimplePointsReader>::New();
    reader->SetFileName(std::string("./../../AM/data/vtk/am_current_pos_" + std::to_string(num) + ".vtk").c_str());
    reader->Update();
    // mapper
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->Build();
    mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();
    // actor
    renderer->RemoveActor(am_source_actor);
    am_source_actor = vtkSmartPointer<vtkActor>::New();
    am_source_actor->SetMapper(mapper);
    am_source_actor->GetProperty()->EdgeVisibilityOn();
    am_source_actor->GetProperty()->SetPointSize(10);
    am_source_actor->GetProperty()->SetRenderPointsAsSpheres(true);
    am_source_actor->GetProperty()->SetColor(255,255,255);
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    // actor->GetProperty()->SetOpacity(100.0);
    // renderer
    renderer->AddActor(am_source_actor);
    //renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    // GetRenderWindow()->Render();




}





























// *******************************************************
// fem






void VTKWidget::FEMImportResults2(QString filename)
{
    fstream _file;
    _file.open(filename.toStdString(), ios::in);
    if (!_file) return;
    // read a vtk file
    vtkSmartPointer<vtkUnstructuredGridReader> reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(filename.toStdString().c_str());
    reader->Update();
    // mapper
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->Build();




    ifstream is;
    is.open(std::string("/home/jiping/M++/data/vtk/maxmin").c_str());
    double z[2];
    const int len = 256;
    char L[len];
    is.getline(L,len);
    sscanf(L, "%lf", z);
    is.getline(L,len);
    sscanf(L, "%lf", z+1);
    std::cout << range[0] << " " << range[1] << std::endl;
    std::cout << z[0] << " " << z[1] << std::endl;




    mapper->SetScalarRange(z[1], z[0]);
    //mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();
    // actor
    renderer->RemoveActor(fem_simulation_actor);
    fem_simulation_actor = vtkSmartPointer<vtkActor>::New();
    fem_simulation_actor->SetMapper(mapper);
    fem_simulation_actor->GetProperty()->EdgeVisibilityOn();
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    fem_simulation_actor->GetProperty()->SetLineWidth(1);
    // renderer
    renderer->AddActor(fem_simulation_actor);
    //renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    GetRenderWindow()->Render();
}

void VTKWidget::FEMImportResults(QString filename)
{
    fstream _file;
    _file.open(filename.toStdString(), ios::in);
    if (!_file) return;
    // read a vtk file
    vtkSmartPointer<vtkUnstructuredGridReader> reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(filename.toStdString().c_str());
    reader->Update();
    // mapper
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->Build();
    mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();
    // actor
    renderer->RemoveActor(fem_simulation_actor);
    fem_simulation_actor = vtkSmartPointer<vtkActor>::New();
    fem_simulation_actor->SetMapper(mapper);
    fem_simulation_actor->GetProperty()->EdgeVisibilityOn();
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    fem_simulation_actor->GetProperty()->SetLineWidth(1);
    // renderer
    renderer->AddActor(fem_simulation_actor);
    //renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    GetRenderWindow()->Render();
}

#include "vtkCubeSource.h"
#include "vtkCylinderSource.h"

void VTKWidget::mbdImportResults(int n, QString file_name)
{
    std::ifstream is(file_name.toStdString());
    const int len = 256;
    char L[len];
    for (int i=0; i<n*5; i++)
        is.getline(L,len);

    renderer->RemoveActor(mbd_simulation_actor_1);
    renderer->RemoveActor(mbd_simulation_actor_2);
    renderer->RemoveActor(mbd_simulation_actor_3);
    renderer->RemoveActor(mbd_simulation_actor_4);
    renderer->RemoveActor(mbd_simulation_actor_5);
    for (int i=0; i<5; i++) {
        is.getline(L,len);
        std::cout << L << std::endl;
        double z[13];
        sscanf(L,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf"
               , z, z+1, z+2, z+3, z+4, z+5, z+6, z+7, z+8, z+9, z+10, z+11, z+12);

        vtkNew<vtkTransform> transform;
        transform->Translate(z[1],z[2],z[3]);
        transform->RotateX(z[4]);
        transform->RotateY(z[5]);
        transform->RotateZ(z[6]);
        if (z[0]==1) {
            transformFilter1->SetInputConnection(reader1->GetOutputPort());
            transformFilter1->SetTransform(transform);
            mapper1->SetInputConnection(transformFilter1->GetOutputPort());
            mbd_simulation_actor_1 = vtkSmartPointer<vtkActor>::New();
            mbd_simulation_actor_1->SetMapper(mapper1);
            mbd_simulation_actor_1->GetProperty()->EdgeVisibilityOff();
            mbd_simulation_actor_1->GetProperty()->SetLineWidth(1);
        }
        if (z[0]==2) {
            transformFilter2->SetInputConnection(reader2->GetOutputPort());
            transformFilter2->SetTransform(transform);
            mapper2->SetInputConnection(transformFilter2->GetOutputPort());
            mbd_simulation_actor_2 = vtkSmartPointer<vtkActor>::New();
            mbd_simulation_actor_2->SetMapper(mapper2);
            mbd_simulation_actor_2->GetProperty()->EdgeVisibilityOff();
            mbd_simulation_actor_2->GetProperty()->SetLineWidth(1);
        }
        if (z[0]==3) {
            transformFilter3->SetInputConnection(reader3->GetOutputPort());
            transformFilter3->SetTransform(transform);
            mapper3->SetInputConnection(transformFilter3->GetOutputPort());
            mbd_simulation_actor_3 = vtkSmartPointer<vtkActor>::New();
            mbd_simulation_actor_3->SetMapper(mapper3);
            mbd_simulation_actor_3->GetProperty()->EdgeVisibilityOff();
            mbd_simulation_actor_3->GetProperty()->SetLineWidth(1);
        }
        if (z[0]==4) {
            transformFilter4->SetInputConnection(reader4->GetOutputPort());
            transformFilter4->SetTransform(transform);
            mapper4->SetInputConnection(transformFilter4->GetOutputPort());
            mbd_simulation_actor_4 = vtkSmartPointer<vtkActor>::New();
            mbd_simulation_actor_4->SetMapper(mapper4);
            mbd_simulation_actor_4->GetProperty()->EdgeVisibilityOff();
            mbd_simulation_actor_4->GetProperty()->SetLineWidth(1);
        }
        if (z[0]==5) {
            transformFilter5->SetInputConnection(reader5->GetOutputPort());
            transformFilter5->SetTransform(transform);
            mapper5->SetInputConnection(transformFilter5->GetOutputPort());
            mbd_simulation_actor_5 = vtkSmartPointer<vtkActor>::New();
            mbd_simulation_actor_5->SetMapper(mapper5);
            mbd_simulation_actor_5->GetProperty()->EdgeVisibilityOff();
            mbd_simulation_actor_5->GetProperty()->SetLineWidth(1);
        }
    }

    renderer->AddActor(mbd_simulation_actor_1);
    renderer->AddActor(mbd_simulation_actor_2);
    renderer->AddActor(mbd_simulation_actor_3);
    renderer->AddActor(mbd_simulation_actor_4);
    renderer->AddActor(mbd_simulation_actor_5);
    GetRenderWindow()->Render();

    is.close();
}


// *******************************************************
// machining


void VTKWidget::MachiningPartPlot (TopoDS_Shape S, bool t)
{
    IVtkOCC_Shape::Handle aShapeImpl = new IVtkOCC_Shape(S);
    ObjectId += 1;


    aShapeImpl->SetId(ObjectId);



    // vtkPolyDataAlgorithm
    vtkSmartPointer<IVtkTools_ShapeDataSource> aDS = vtkSmartPointer<IVtkTools_ShapeDataSource>::New();
    aDS->SetShape(aShapeImpl);

    // vtkAlgorithmOutput -> vtkAlgorithm
    vtkSmartPointer<vtkPolyDataMapper> Mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    Mapper->SetInputConnection(aDS->GetOutputPort());

    // actor
    Remove(machining_part_actor);
    machining_part_actor = vtkSmartPointer<vtkActor>::New();
    machining_part_actor->SetMapper(Mapper);


    machining_part_actor->SetId(ObjectId);



    machining_part_actor->GetProperty()->SetColor(COLOR0);
    machining_part_actor->SetSelected(false);

    // renderer
    renderer->AddActor(machining_part_actor);


    if (t)
    {
        renderer->ResetCamera();
    }

    switch (S.ShapeType()) {
    case 7:
        aPicker->SetSelectionMode(aShapeImpl, SM_Vertex);
        machining_part_actor->GetProperty()->SetPointSize(10);
        break;
    case 6:
        aPicker->SetSelectionMode(aShapeImpl, SM_Edge);
        machining_part_actor->GetProperty()->SetLineWidth(3);
        break;
    case 4:
        aPicker->SetSelectionMode(aShapeImpl, SM_Face);
        break;
    default:
        aPicker->SetSelectionMode(aShapeImpl, SM_Solid);
        break;
    }
    // qvtkopenglwidget update has some problems, it seems didn't use render again.
    // update();
    GetRenderWindow()->Render();
    return;
}

void VTKWidget::MachiningToolPlot (TopoDS_Shape S, bool t)
{
    IVtkOCC_Shape::Handle aShapeImpl = new IVtkOCC_Shape(S);
    ObjectId += 1;


    aShapeImpl->SetId(ObjectId);



    // vtkPolyDataAlgorithm
    vtkSmartPointer<IVtkTools_ShapeDataSource> aDS = vtkSmartPointer<IVtkTools_ShapeDataSource>::New();
    aDS->SetShape(aShapeImpl);

    // vtkAlgorithmOutput -> vtkAlgorithm
    vtkSmartPointer<vtkPolyDataMapper> Mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    Mapper->SetInputConnection(aDS->GetOutputPort());

    // actor
    Remove(machining_tool_actor);
    machining_tool_actor = vtkSmartPointer<vtkActor>::New();
    machining_tool_actor->SetMapper(Mapper);


    machining_tool_actor->SetId(ObjectId);



    machining_tool_actor->GetProperty()->SetColor(COLOR0);
    machining_tool_actor->SetSelected(false);
    machining_tool_actor->GetProperty()->SetRepresentationToWireframe();

    // renderer
    renderer->AddActor(machining_tool_actor);

    if (t)
    {
        renderer->ResetCamera();
    }

    switch (S.ShapeType()) {
    case 7:
        aPicker->SetSelectionMode(aShapeImpl, SM_Vertex);
        machining_tool_actor->GetProperty()->SetPointSize(10);
        break;
    case 6:
        aPicker->SetSelectionMode(aShapeImpl, SM_Edge);
        machining_tool_actor->GetProperty()->SetLineWidth(3);
        break;
    case 4:
        aPicker->SetSelectionMode(aShapeImpl, SM_Face);
        break;
    default:
        aPicker->SetSelectionMode(aShapeImpl, SM_Solid);
        break;
    }
    // qvtkopenglwidget update has some problems, it seems didn't use render again.
    // update();
    GetRenderWindow()->Render();
    return;
}

void VTKWidget::MachiningPlotPartBnds()
{
    for (int i = 0; i < machining_part_bnds->Size(); i++)
    {
        Plot(*((*machining_part_bnds)[i]->Value()), true);
    }
}

#include <vtkSTLWriter.h>

void VTKWidget::MachiningImportPart (int num)
{
    fstream _file;
    _file.open("/home/jiping/OpenDT/M++/data/vtk/linear_elasticity_deform_" + std::to_string(num) + ".vtk", ios::in);
    if (!_file) return;
    // read a vtk file
    vtkSmartPointer<vtkUnstructuredGridReader> reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(std::string("/home/jiping/OpenDT/M++/data/vtk/linear_elasticity_deform_" + std::to_string(num) + ".vtk").c_str());
    reader->Update();
    // mapper
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->Build();
    mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();
    // actor
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->EdgeVisibilityOn();
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    actor->GetProperty()->SetLineWidth(1);
    // renderer
    renderer->AddActor(actor);
    //renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    //GetRenderWindow()->Render();




    //        stlWriter = vtk.vtkSTLWriter();

    //        stlWriter.SetInputConnection(sphereSource.GetOutputPort());
    //        stlWriter.Write();

    if (num == 21) {
        vtkSmartPointer<vtkPolyDataReader> wbreader = vtkSmartPointer<vtkPolyDataReader>::New();
        wbreader->SetFileName("/home/jiping/webcad/Sample-of-WebGL-with-STL-loader-master/Models/bike_frame.vtk");
        wbreader->Update();
        vtkSmartPointer<vtkSTLWriter> stlWriter =  vtkSmartPointer<vtkSTLWriter>::New();
        stlWriter->SetFileName("/home/jiping/webcad/Sample-of-WebGL-with-STL-loader-master/Models/bike_frame.stl");
        stlWriter->SetInputConnection(wbreader->GetOutputPort());
        //meas_cloud_source_writer->SetInputData(reader->GetOutput());
        //meas_cloud_source_writer->SetInputData(transformFilter->GetOutput());
        //meas_cloud_source_writer->SetInputConnection(transformFilter->GetOutputPort());
        //        meas_cloud_source_writer->SetInputData(polygonPolyData);
        //meas_cloud_source_writer->SetFileName(name.c_str());
        stlWriter->Write();
    }
}

#include "vtkOutlineFilter.h"

void VTKWidget::MachiningImportTool (int num)
{
    fstream _file;
    _file.open("/home/jiping/OpenDT/M++/data/vtk/linear_elasticity_deform_tool_" + std::to_string(num) + ".vtk", ios::in);
    if (!_file) return;
    // read a vtk file
    vtkSmartPointer<vtkUnstructuredGridReader> reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(std::string("/home/jiping/OpenDT/M++/data/vtk/linear_elasticity_deform_tool_" + std::to_string(num) + ".vtk").c_str());
    reader->Update();
    // mapper
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->Build();
    mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();
    // actor
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    //actor->GetProperty()->EdgeVisibilityOn();
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    //actor->GetProperty()->SetLineWidth(1);
    // renderer



    actor->GetProperty()->SetColor(COLOR7);





    // Create the outline
    vtkSmartPointer<vtkOutlineFilter> outline =
            vtkSmartPointer<vtkOutlineFilter>::New();
    outline->SetInputData(reader->GetOutput());
    vtkSmartPointer<vtkPolyDataMapper> outlineMapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
    outlineMapper->SetInputConnection(outline->GetOutputPort());
    vtkSmartPointer<vtkActor> outlineActor =
            vtkSmartPointer<vtkActor>::New();
    outlineActor->SetMapper(outlineMapper);
    outlineActor->GetProperty()->SetColor(0,0,0);











    renderer->AddActor(actor);
    renderer->AddActor(outlineActor);
    //renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    //GetRenderWindow()->Render();
}


void VTKWidget::TransportPlot (TopoDS_Shape S, bool t, int color)
{
    IVtkOCC_Shape::Handle aShapeImpl = new IVtkOCC_Shape(S);
    ObjectId += 1;


    aShapeImpl->SetId(ObjectId);



    // vtkPolyDataAlgorithm
    vtkSmartPointer<IVtkTools_ShapeDataSource> aDS = vtkSmartPointer<IVtkTools_ShapeDataSource>::New();
    aDS->SetShape(aShapeImpl);

    // vtkAlgorithmOutput -> vtkAlgorithm
    vtkSmartPointer<vtkPolyDataMapper> Mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    Mapper->SetInputConnection(aDS->GetOutputPort());

    // actor
    vtkSmartPointer<vtkActor> Actor = vtkSmartPointer<vtkActor>::New();
    Actor->SetMapper(Mapper);


    Actor->SetId(ObjectId);


    if (color == 0)
        Actor->GetProperty()->SetColor(COLOR0);
    else if (color == 1)
        Actor->GetProperty()->SetColor(COLOR1);
    else if (color == 2)
        Actor->GetProperty()->SetColor(COLOR2);
    else if (color == 3)
        Actor->GetProperty()->SetColor(COLOR3);
    else if (color == 4)
        Actor->GetProperty()->SetColor(COLOR4);
    else if (color == 5)
        Actor->GetProperty()->SetColor(COLOR5);
    else if (color == 6)
        Actor->GetProperty()->SetColor(COLOR6);
    else if (color == 7)
        Actor->GetProperty()->SetColor(COLOR7);
    else if (color == 8)
        Actor->GetProperty()->SetColor(COLOR8);
    Actor->SetSelected(false);


    // renderer
    renderer->AddActor(Actor);

    if (t)
    {
        renderer->ResetCamera();
    }

    switch (S.ShapeType()) {
    case 7:
        aPicker->SetSelectionMode(aShapeImpl, SM_Vertex);
        Actor->GetProperty()->SetPointSize(10);
        break;
    case 6:
        aPicker->SetSelectionMode(aShapeImpl, SM_Edge);
        Actor->GetProperty()->SetLineWidth(3);
        break;
    case 4:
        aPicker->SetSelectionMode(aShapeImpl, SM_Face);
        break;
    default:
        aPicker->SetSelectionMode(aShapeImpl, SM_Solid);
        Actor->GetProperty()->SetOpacity(0.1);
        break;
    }
    // qvtkopenglwidget update has some problems, it seems didn't use render again.
    // update();
    GetRenderWindow()->Render();
    return;
}


void VTKWidget::TransportImportVTKFile(std::string name, int color)
{
    fstream _file;
    _file.open(name, ios::in);
    if (!_file) return;
    // read a vtk file
    vtkSmartPointer<vtkUnstructuredGridReader> reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(name.c_str());
    reader->Update();
    // mapper
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->Build();
    mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();
    // actor
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->EdgeVisibilityOn();
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    // actor->GetProperty()->SetOpacity(100.0);
    if (color == 0)
        actor->GetProperty()->SetColor(COLOR0);
    else if (color == 1)
        actor->GetProperty()->SetColor(COLOR1);
    else if (color == 2)
        actor->GetProperty()->SetColor(COLOR2);
    else if (color == 3)
        actor->GetProperty()->SetColor(COLOR3);
    else if (color == 4)
        actor->GetProperty()->SetColor(COLOR4);
    else if (color == 5)
        actor->GetProperty()->SetColor(COLOR5);
    else if (color == 6)
        actor->GetProperty()->SetColor(COLOR6);
    else if (color == 7)
        actor->GetProperty()->SetColor(COLOR7);
    else if (color == 8)
        actor->GetProperty()->SetColor(COLOR8);
    else if (color == 9)
        actor->GetProperty()->SetColor(COLOR9);
    // renderer
    renderer->AddActor(actor);
    renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    GetRenderWindow()->Render();
}

int VTKWidget::OCPoroImportVTKFile(std::string name, int n)
{
    fstream _file;
    _file.open(name, ios::in);
    if (!_file) return -1;
    // read a vtk file
    vtkSmartPointer<vtkUnstructuredGridReader> reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(name.c_str());
    reader->ReadAllScalarsOn();
    reader->Update();

    reader->GetOutput()->GetCellData()->SetActiveAttribute(n,0);
    //std::cout << reader->GetOutput()->GetCellData()->GetArrayName(1) << std::endl;
    //std::cout << reader->GetOutput()->GetCellData()->GetNumberOfArrays() << std::endl;
    int m = reader->GetOutput()->GetCellData()->GetNumberOfArrays();

    // mapper
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.666667,0.0);
    double * range = reader->GetOutput()->GetScalarRange();
    lut->SetVectorModeToMagnitude();
    lut->Build();
    mapper->SetScalarRange(range);
    mapper->SetLookupTable(lut);
    mapper->Update();





    // actor
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->EdgeVisibilityOn();
    actor->GetProperty()->SetAmbient(0.25);
    // actor->GetProperty()->SetFrontfaceCulling(1); // shit this is OK, check it for long time
    // actor->GetProperty()->SetOpacity(100.0);
    //actor->GetProperty()->SetEdgeColor(255.0/255.0,255.0/255.0,255.0/255.0);
    // renderer
    renderer->AddActor(actor);
    //renderer->ResetCamera();
    // Automatically set up the camera based on the visible actors.
    // The camera will reposition itself to view the center point of the actors,
    // and move along its initial view plane normal (i.e., vector defined from camera position to focal point)
    // so that all of the actors can be seen.
    renderer->ResetCameraClippingRange();
    // Reset the camera clipping range based on the bounds of the visible actors.
    // This ensures that no props are cut off
    // redraw
    GetRenderWindow()->Render();
    return m;
}
