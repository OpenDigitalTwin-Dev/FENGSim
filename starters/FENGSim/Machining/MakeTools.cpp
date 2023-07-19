#include "MakeTools.h"
#include <ShapeFix_Solid.hxx>
#include <ShapeFix_Face.hxx>
#include <BRepGProp.hxx>
#include <GProp_GProps.hxx>
#include <BRep_Tool.hxx>
#include <BRepAlgoAPI_Section.hxx>
#include <gp_Ax2.hxx>

TopoDS_Shape MakeTools(const Standard_Real toolLength, const Standard_Real drillRadius)
{

    // ************************************
    // ************************************

    // Profile : Define Support Points
    gp_Pnt p1(-toolLength / 2., 0, 0);
    gp_Pnt p2(toolLength / 2., 0, 0);
    gp_Pnt p3(toolLength/2. - drillRadius / sqrt(3), drillRadius, 0);
    gp_Pnt p4(-toolLength/2. + drillRadius/5., drillRadius, 0);
    gp_Pnt p5(-toolLength/2., drillRadius*4. / 5., 0);


    // Profile : Define the Topology
    TopoDS_Edge e1 = BRepBuilderAPI_MakeEdge(p1, p2);
    TopoDS_Edge e2 = BRepBuilderAPI_MakeEdge(p2, p3);
    TopoDS_Edge e3 = BRepBuilderAPI_MakeEdge(p3, p4);
    TopoDS_Edge e4 = BRepBuilderAPI_MakeEdge(p4, p5);
    TopoDS_Edge e5 = BRepBuilderAPI_MakeEdge(p5, p1);
    TopoDS_Wire w = BRepBuilderAPI_MakeWire(e1, e2, e3, e4);
    TopoDS_Wire aWire = BRepBuilderAPI_MakeWire(w, e5);

    // Body : Prism the Profile
    TopoDS_Face aFace = BRepBuilderAPI_MakeFace(aWire);
    gp_Ax1 axe = gp_Ax1(gp_Pnt(0., 0., 0.), gp_Dir(1., 0., 0.));
    TopoDS_Shape toolInit = BRepPrimAPI_MakeRevol(aFace, axe, false).Shape();


    // ************************************
    // ************************************

    Standard_Real aRadius = 1.5;
    Standard_Real aPitch = 2.;
    // the pcurve is a 2d line in the parametric space.
    gp_Lin2d aLine2d(gp_Pnt2d(0.0, 0.0), gp_Dir2d(aRadius, aPitch));
    Handle(Geom2d_TrimmedCurve) aSegment = GCE2d_MakeSegment(aLine2d, 0.0, M_PI * 2.0).Value();
    Handle(Geom_CylindricalSurface) aCylinder = new Geom_CylindricalSurface(gp::XOY(), aRadius);
    TopoDS_Edge aHelixEdge = BRepBuilderAPI_MakeEdge(aSegment, aCylinder, 0.0, 6.0 * M_PI).Edge();
    BRepLib::BuildCurve3d(aHelixEdge);

    gp_Ax2 anAxis;
    anAxis.SetDirection(gp_Dir(0.0, 4.0, 1.0));
    anAxis.SetLocation(gp_Pnt(aRadius, 0.0, 0.0));
    gp_Circ aProfileCircle(anAxis, 1.2);
    TopoDS_Edge aProfileEdge = BRepBuilderAPI_MakeEdge(aProfileCircle).Edge();
    TopoDS_Wire aProfileWire = BRepBuilderAPI_MakeWire(aProfileEdge).Wire();
    TopoDS_Face aProfileFace = BRepBuilderAPI_MakeFace(aProfileWire).Face();
    TopoDS_Wire aHelixWire = BRepBuilderAPI_MakeWire(aHelixEdge).Wire();
    BRepOffsetAPI_MakePipe aPipeMaker(aHelixWire, aProfileFace);


    aPipeMaker.Build();

    //return aPipeMaker.Shape();



    // ************************************
    // ************************************


    Standard_Real rotate = M_PI / 2.;
    gp_Trsf aTrsf;
    aTrsf.SetRotation(gp::OY(), rotate);
    BRepBuilderAPI_Transform toolsTransform(aPipeMaker.Shape(), aTrsf);
    // aTrsf.SetTranslation(gp_Vec(-6.6, 0., 0.));
    aTrsf.SetTranslation(gp_Vec(-5, 0., 0.));
    BRepBuilderAPI_Transform pan(toolsTransform.Shape(), aTrsf);
    TopoDS_Shape tools;
    tools = pan.Shape();


    //TopoDS_Shape uptool = BRepAlgoAPI_Fuse(toolInit, tools);
    gp_Trsf mirr;
    mirr.SetMirror(gp::OX());
    BRepBuilderAPI_Transform m(tools, mirr);
    TopoDS_Shape mirTool = m.Shape();

    TopoDS_Shape cy(BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0,0,0),gp_Dir(1,0,0)),drillRadius,toolLength).Shape());

    TopoDS_Shape downtool = BRepAlgoAPI_Cut(toolInit, tools);
    downtool = BRepAlgoAPI_Cut(downtool, mirTool);


    return downtool;


    TopExp_Explorer faceExplorer;
    int num_face = 0;
    for (faceExplorer.Init(downtool, TopAbs_FACE); faceExplorer.More(); faceExplorer.Next())
    {
        num_face++;
    }
    std::cout << num_face << std::endl;
    BRepBuilderAPI_MakeWire mw;
    for (faceExplorer.Init(downtool, TopAbs_FACE); faceExplorer.More(); faceExplorer.Next())
    {
        faceExplorer.Next();
        //        faceExplorer.Next();
        //        faceExplorer.Next();
        //        faceExplorer.Next();
        //        faceExplorer.Next();
        //        faceExplorer.Next();
        //        faceExplorer.Next();
        //        faceExplorer.Next();
        //        faceExplorer.Next();
        //        faceExplorer.Next();
        //        faceExplorer.Next();
        //        faceExplorer.Next();

        //return faceExplorer.Current();


        TopExp_Explorer edgeExplorer1;
        for (edgeExplorer1.Init(faceExplorer.Current(), TopAbs_EDGE); edgeExplorer1.More(); edgeExplorer1.Next())
        {
            mw.Add(TopoDS::Edge(edgeExplorer1.Current()));
        }

        //return mw.Shape();

        BRepBuilderAPI_MakeFace mf;
        mf.Add(mw.Wire());
        return mf.Shape();
        //        return mw.Shape();
        //        return faceExplorer.Current();

    }

    //return mw.Shape();




    //gp_Trsf tr;
    //tr.SetTranslation(gp_Vec(0, 0, 5.));
    //BRepBuilderAPI_Transform uptr(uptool, tr);

    //downtool = BRepAlgoAPI_Cut(downtool, mirTool);

    //downtool = BRepAlgoAPI_Common(downtool, mirTool);

}








