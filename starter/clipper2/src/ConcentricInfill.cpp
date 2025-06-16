#include <cstdlib>
#include <string>

#include <chrono> 
#include <random>


#include "clipper2/clipper.h"
#include "Utils/clipper.svg.utils.h"
#include "Utils/ClipFileLoad.h"
//#include "../../../toolkit/Geometry/clipper2/CPP/Utils/clipper.svg.utils.h"
//#include "Utils/clipper.svg.utils.h"

#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkPolygon.h>


using namespace Clipper2Lib;

void DoSimpleTest(bool show_solution_coords = false);
Path64 MakeRandomPoly(int width, int height, unsigned vertCnt);
void System(const std::string &filename);
Paths64 readInPolygon();
void ExportPathsToVTK(const Clipper2Lib::Paths64& paths, const std::string& filename);

int main()
{  
  //readInPolygon();
  DoSimpleTest();
}

inline Path64 MakeStar(const Point64& center, int radius, int points)
{
  if (!(points % 2)) --points;
  if (points < 5) points = 5;
  Path64 tmp = Ellipse<int64_t>(center, radius, radius, points);
  Path64 result;
  result.reserve(points);
  result.push_back(tmp[0]);
  for (int i = points - 1, j = i / 2; j;)
  {
    result.push_back(tmp[j--]);
    result.push_back(tmp[i--]);
  }
  return result;
}


Paths64 readInPolygon()
{
  std::ifstream ifs("/home/maning/MANING/FENGSim/starter/clipper2/testfile/Offsets.txt");

  if (!ifs.is_open()) {
    std::cerr << "Failed to open file: /testfile/Offsets.txt" << std::endl;
    return Paths64(); // 返回空路径
  }


  Paths64 subject, subject_open, clip;
  Paths64 solution, solution_open;
  ClipType ct = ClipType::NoClip;
  FillRule fr = FillRule::NonZero;
  int64_t stored_area = 0, stored_count = 0;
  LoadTestNum(ifs, 2, subject, subject_open, clip, stored_area, stored_count, ct, fr);



 /*  SvgWriter svg;
  SvgAddSubject(svg, subject, fr);
  SvgSaveToFile(svg, "subject.svg", 450, 450, 10);
  System("subject.svg"); */

  return subject;
}




void DoSimpleTest(bool show_solution_coords)
{
  Paths64 tmp, solution;
  FillRule fr = FillRule::NonZero;




  Paths64 subject= readInPolygon();


  //Paths64 subject;
  
  //subject.push_back(MakeStar(Point64(225, 225), 220, 9));

  //subject.push_back(MakeStar(Point64(225, 225), 220, 9));
  //clip.push_back(Ellipse<int64_t>(Point64(225,225), 150, 150));  

  solution.push_back(subject[0]);

  ClipperOffset co;
  Paths64 result;
  do
  {
    co.AddPaths(subject, JoinType::Round, EndType::Polygon);
    
    co.Execute(-5, result);
    co.Clear();


    for(int i = 0; i < result.size(); i++)
    {
      solution.push_back(result[i]);
    }

    subject= result;
    
  } while (result.empty()!= true);
  
  //Intersect both shapes and then 'inflate' result -10 (ie deflate)
  //solution = Intersect(subject, clip, fr);
  //solution = InflatePaths(solution, -10, JoinType::Round, EndType::Polygon);

  SvgWriter svg;
  SvgAddSubject(svg, subject, fr);

  //SvgAddClip(svg, clip, fr);
  SvgAddSolution(svg, solution, fr, false);
  SvgSaveToFile(svg, "solution.svg", 450, 450, 10);
  //System("solution.svg");

  ExportPathsToVTK(solution, "solution.vtk");

}


Paths64 ConcentricInfill(const Paths64& subject)
{
  Paths64 solution;

  Paths64 offset;


  ClipperOffset co;
  co.AddPaths(subject, JoinType::Round, EndType::Polygon);
  co.Execute(-10, offset);
  co.Clear();  



  Clipper64 clip;
  clip.AddSubject(offset);

  co.AddPaths(subject, JoinType::Round, EndType::Polygon);
  co.Execute(-10, offset);
  co.Clear();  


  co.AddPaths(subject, JoinType::Round, EndType::Polygon);
  co.Execute(-20, offset);
  co.Clear();  

  clip.AddSubject(offset);

  clip.Execute(ClipType::Union, FillRule::NonZero, solution);
  return solution;
}


void ExportPathsToVTK(const Clipper2Lib::Paths64& paths, const std::string& filename) {
    // 创建VTK数据结构
    auto points = vtkSmartPointer<vtkPoints>::New();
    auto polygons = vtkSmartPointer<vtkCellArray>::New();
    
    // 填充点数clipper2 生成的Paths64 导入vtk库 ,输出为vtk文件据
    /* vtkIdType pointId = 0;
    for (const auto& path : paths) {
        auto polygon = vtkSmartPointer<vtkPolygon>::New();
        polygon->GetPointIds()->SetNumberOfIds(path.size());
        
        for (size_t i = 0; i < path.size(); ++i) {
            points->InsertNextPoint(path[i].x, path[i].y, 0.0);
            polygon->GetPointIds()->SetId(i, pointId++);
        }
        
        polygons->InsertNextCell(polygon);
    } */
    
    // 填充点和多边形
    for (const auto& path : paths) {
        vtkIdType npts = static_cast<vtkIdType>(path.size());
        polygons->InsertNextCell(npts);

        for (size_t i = 0; i < path.size(); ++i) {
            double point[3] = {static_cast<double>(path[i].x),
                               static_cast<double>(path[i].y), 0.0};
            vtkIdType pointId = points->InsertNextPoint(point);
            polygons->InsertCellPoint(pointId);
        }
    }

    // 创建PolyData
    auto polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->SetPoints(points);
    polyData->SetPolys(polygons);
    
    // 写入文件
    auto writer = vtkSmartPointer<vtkPolyDataWriter>::New();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(polyData);
    writer->Write();
}



void System(const std::string &filename)
{
#ifdef _WIN32
  system(filename.c_str());
#else
  system(("firefox " + filename).c_str());
#endif
}
