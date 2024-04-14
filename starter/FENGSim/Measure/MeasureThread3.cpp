#include "MeasureThread3.h"

void MeasureThread3::run ()
{
    //TextOutput("Begin to change a CAD model to a point cloud model. Please wait...");

    if (meas_bnds->Size() == 0) {
        vtk_widget->MeasureClearCloudTarget();
        return;
    }

    //        if (vtk_widget->GetSelectedBnd() == NULL) return;

    //        int n = meas_bnds->Include(*(vtk_widget->GetSelectedBnd()->Value()));

    //        bool sel = true;
    //        for (int i = 0; i < meas_selected_bnds.size(); i++)
    //                if (n == meas_selected_bnds[i])
    //                        sel = false;
    //        if (!sel) return;

    //        meas_selected_bnds.push_back(n);
    //        std::cout << "faces num: " << meas_selected_bnds.size()  << std::endl;

    if (vtk_widget->selected_bnd_id.size() == 0) return;

    //vtk_widget->SetSelectable(false);

    BRep_Builder builder;
    TopoDS_Shell shell;
    builder.MakeShell(shell);
    for (int i = 0; i < vtk_widget->selected_bnd_id.size(); i++)
        builder.Add(shell,*((*meas_bnds)[vtk_widget->selected_bnd_id[i]]->Value()));
    MM.MeshGeneration(&shell, measure_dock->ui->doubleSpinBox_2->text().toDouble(), 0, path);
    //MM.MeasureModel(path);

    // the whole model
    /*BRep_Builder builder1;
    TopoDS_Shell shell1;
    builder.MakeShell(shell1);
    for (int i = 0; i < meas_bnds->Size(); i++)
        builder.Add(shell1,*((*meas_bnds)[i]->Value()));
    MM.MeshGeneration(&shell1, measure_dock->ui->doubleSpinBox_2->text().toDouble(), 0, path);
    MM.MeasureModel(path,QString("/data/meas/fengsim_meas_target_whole.vtk"));
*/
    //    ofstream out;
    //    out.open(std::string("/home/jiping/FENGSim/FENGSim/Measure/data/cloud_target.vtk").c_str(),ios::app);
    //    TopLoc_Location loc;
    //    TopoDS_Face aFace = TopoDS::Face(*((*bnds)[n]->Value()));
    //    Handle_Poly_Triangulation triFace = BRep_Tool::Triangulation(aFace, loc);
    //    //        Standard_Integer nTriangles = triFace->NbTriangles();
    //    gp_Pnt vertex1;
    //    //        gp_Pnt vertex2;
    //    //        gp_Pnt vertex3;
    //    //        Standard_Integer nVertexIndex1 = 0;
    //    //        Standard_Integer nVertexIndex2 = 0;
    //    //        Standard_Integer nVertexIndex3 = 0;
    //    TColgp_Array1OfPnt nodes(1, triFace->NbNodes());
    //    //            Poly_Array1OfTriangle triangles(1, triFace->NbTriangles());
    //    nodes = triFace->Nodes();
    //    //            triangles = triFace->Triangles();
    //    for (int i = 0; i < triFace->NbNodes(); i++) {
    //        vertex1 = nodes.Value(i+1);
    //        out << vertex1.X() << " " << vertex1.Y() << " "<< vertex1.Z() << endl;
    //    }


    exit();
    exec();




}
