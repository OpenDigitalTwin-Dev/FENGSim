// Gmsh - Copyright (C) 1997-2017 C. Geuzaine, J.-F. Remacle
//
// See the LICENSE.txt file for license information. Please report all
// bugs and problems to the public mailing list <gmsh@onelab.info>.

#ifndef _GFACE_H_
#define _GFACE_H_

#include <list>
#include <string>
#include <vector>
#include <map>
#include "GEntity.h"
#include "GPoint.h"
#include "GEdgeLoop.h"
#include "SPoint2.h"
#include "SVector3.h"
#include "Pair.h"
#include "Numeric.h"
#include "boundaryLayersData.h"

class MElement;
class MTriangle;
class MQuadrangle;
class MPolygon;
class ExtrudeParams;
class GFaceCompound;

class GRegion;

// A model face.
class GFace : public GEntity {
 protected:
  // edge loops might replace what follows (list of all the edges of
  // the face + directions)
  std::list<GEdge *> l_edges;
  std::list<int> l_dirs;
  GRegion *r1, *r2;
  mean_plane meanPlane;
  std::list<GEdge *> embedded_edges;
  std::list<GVertex *> embedded_vertices;
  GFaceCompound *compound; // this model face belongs to a compound

  BoundaryLayerColumns _columns;

 public: // this will become protected or private
  std::list<GEdgeLoop> edgeLoops;

  // periodic counterparts of edges
  std::map<GEdge*,std::pair<GEdge*,int> > edgeCounterparts;

  // specify mesh master with transformation, deduce edgeCounterparts
  void setMeshMaster(GFace* master,const std::vector<double>&);

  // specify mesh master and edgeCounterparts, deduce transformation
  void setMeshMaster(GFace* master,const std::map<int,int>&);

  // an array with additional vertices that are supposed to exist in
  // the final mesh of the model face. This can be used for boundary
  // layer meshes or when using Lloyd-like smoothing algorithms those
  // vertices are classifed on this GFace, their type is MFaceVertex.
  // After mesh generation, those are moved to the mesh_vertices array
  std::vector<MVertex*> additionalVertices;

 public:
  GFace(GModel *model, int tag);
  virtual ~GFace();


  std::set<MVertex*> constr_vertices;

  // delete mesh data
  virtual void deleteMesh();

  // add/delete regions that are bounded by the face
  void addRegion(GRegion *r)
  {
    if(r == r1 || r == r2) return;
    r1 ? r2 = r : r1 = r;
  }
  void delRegion(GRegion *r){ if(r1 == r) r1 = r2; r2 = 0; }
  GRegion* getRegion(int num) const{ if(num==0) return r1; else return r2; }

  // get number of regions
  int numRegions() const { int num=0; if(r1) num++; if(r2) num++; return num; }
  std::list<GRegion*> regions() const
  {
    std::list<GRegion*> r;
    for (int i = 0; i <numRegions(); i++) r.push_back(getRegion(i));
    return r;
  }

  // add embedded vertices/edges
  void addEmbeddedVertex(GVertex *v){ embedded_vertices.push_back(v); }
  void addEmbeddedEdge(GEdge *e){ embedded_edges.push_back(e); }

  // delete the edge from the face (the edge is supposed to be a free
  // edge in the face, not part of any edge loops--use with caution!)
  void delFreeEdge(GEdge *e);

  //find the edge 1 from the list of edges and replace it by edge 2
  //dont change the edge loops, and is susceptible to break some functionalities
  void replaceEdge(GEdge *e1, GEdge *e2);

  // edge orientations
  virtual std::list<int> orientations() const { return l_dirs; }

  // edges that bound the face
  virtual std::list<GEdge*> edges() const { return l_edges; }
  inline void set(const std::list<GEdge*> f) { l_edges= f; }
  virtual std::list<int> edgeOrientations() const { return l_dirs; }
  inline bool containsEdge (int iEdge) const
  {
    for (std::list<GEdge*>::const_iterator it = l_edges.begin(); it !=l_edges.end(); ++it)
      if ((*it)->tag() == iEdge) return true;
    return false;
  }

  // edges that are embedded in the face
  virtual std::list<GEdge*> embeddedEdges() const { return embedded_edges; }

  // edges that are embedded in the face
  virtual std::list<GVertex*> embeddedVertices() const { return embedded_vertices; }

  std::vector<MVertex*> getEmbeddedMeshVertices() const;

  // vertices that bound the face
  virtual std::list<GVertex*> vertices() const;

  // dimension of the face (2)
  virtual int dim() const { return 2; }

  // set visibility flag
  virtual void setVisibility(char val, bool recursive=false);

  // set color
  virtual void setColor(unsigned int val, bool recursive=false);

  // compute the parameters UV from a point XYZ
  void XYZtoUV(double X, double Y, double Z, double &U, double &V,
               double relax, bool onSurface=true) const;

  // get the bounding box
  virtual SBoundingBox3d bounds() const;

  // get the oriented bounding box
  virtual SOrientedBoundingBox getOBB();

  // compute the genus G of the surface
  virtual int genusGeom() const;
  virtual bool checkTopology() const { return true; }

  // return the point on the face corresponding to the given parameter
  virtual GPoint point(double par1, double par2) const = 0;
  virtual GPoint point(const SPoint2 &pt) const { return point(pt.x(), pt.y()); }

  // if the mapping is a conforming mapping, i.e. a mapping that
  // conserves angles, this function returns the eigenvalue of the
  // metric at a given point this is a special feature for
  // stereographic mappings of the sphere that is used in 2D mesh
  // generation !
  virtual double getMetricEigenvalue(const SPoint2 &);

  // eigen values are absolute values and sorted from min to max of absolute values
  // eigen vectors are the COLUMNS of eigVec
  virtual void getMetricEigenVectors(const SPoint2 &param,
                                     double eigVal[2], double eigVec[4]) const;

  // return the parmater location on the face given a point in space
  // that is on the face
  virtual SPoint2 parFromPoint(const SPoint3 &, bool onSurface=true) const;

  // true if the parameter value is interior to the face
  virtual bool containsParam(const SPoint2 &pt);

  // return the point on the face closest to the given point
  virtual GPoint closestPoint(const SPoint3 & queryPoint,
                              const double initialGuess[2]) const;

  // return the normal to the face at the given parameter location
  virtual SVector3 normal(const SPoint2 &param) const;

  // return the first derivate of the face at the parameter location
  virtual Pair<SVector3, SVector3> firstDer(const SPoint2 &param) const = 0;

  // compute the second derivates of the face at the parameter location
  // the derivates have to be allocated before calling this function
  virtual void secondDer(const SPoint2 &param,
                         SVector3 *dudu, SVector3 *dvdv, SVector3 *dudv) const = 0;

  // return the curvature computed as the divergence of the normal
  inline double curvature(const SPoint2 &param) const { return curvatureMax(param); }
  virtual double curvatureDiv(const SPoint2 &param) const;

  // return the maximum curvature at a point
  virtual double curvatureMax(const SPoint2 &param) const;

  // compute the min and max curvatures and the corresponding directions
  // return the max curvature
  // outputs have to be allocated before calling this function
  virtual double curvatures(const SPoint2 &param, SVector3 *dirMax, SVector3 *dirMin,
                            double *curvMax, double *curvMin) const;

  // return a type-specific additional information string
  virtual std::string getAdditionalInfoString();

  // export in GEO format
  virtual void writeGEO(FILE *fp);

  // fill the crude representation cross
  virtual bool buildRepresentationCross(bool force=false);

  // build an STL triangulation (or do nothing if it already exists,
  // unless force=true)
  virtual bool buildSTLTriangulation(bool force=false);

  // fill the vertex array using an STL triangulation
  bool fillVertexArray(bool force=false);

  // recompute the mean plane of the surface from a list of points
  void computeMeanPlane(const std::vector<MVertex*> &points);
  void computeMeanPlane(const std::vector<SPoint3> &points);

  // recompute the mean plane of the surface from its bounding vertices
  void computeMeanPlane();

  // get the mean plane info
  void getMeanPlaneData(double VX[3], double VY[3],
                        double &x, double &y, double &z) const;
  void getMeanPlaneData(double plan[3][3]) const;

  // number of types of elements
  int getNumElementTypes() const { return 3; }

  // get total/by-type number of elements in the mesh
  unsigned int getNumMeshElements();
  unsigned int getNumMeshParentElements();
  void getNumMeshElements(unsigned *const c) const;

  // get the start of the array of a type of element
  MElement *const *getStartElementType(int type) const;

  // get the element at the given index
  MElement *getMeshElement(unsigned int index);

  // reset the mesh attributes to default values
  virtual void resetMeshAttributes();

  // for periodic faces, move parameters into the range chosen
  // for that face
  void moveToValidRange(SPoint2 &pt) const;

  //compute mesh statistics
  void computeMeshSizeFieldAccuracy(double &avg,double &max_e, double &min_e,
				    int &nE, int &GS);

  // compound
  void setCompound(GFaceCompound *gfc) { compound = gfc; }
  GFaceCompound *getCompound() const { return compound; }

  // add points (and optionally normals) in vectors so that two
  // points are at most maxDist apart
  bool fillPointCloud(double maxDist,
		      std::vector<SPoint3> *points,
		      std::vector<SPoint2> *uvpoints=0,
                      std::vector<SVector3> *normals=0);

  // apply Lloyd's algorithm to the mesh
  void lloyd(int nIter, int infNorm = 0);

  // replace edges (gor gluing)
  void replaceEdges(std::list<GEdge*> &);

  // tells if it's a sphere, and if it is, returns parameters
  virtual bool isSphere(double &radius, SPoint3 &center) const { return false; }

  // new interface for meshing
  virtual void mesh(bool verbose);

  struct {
    // do we recombine the triangles of the mesh?
    int recombine;
    // what is the treshold angle for recombination
    double recombineAngle;
    // is this surface meshed using a transfinite interpolation
    char method;
    // corners of the transfinite interpolation
    std::vector<GVertex*> corners;
    // all diagonals of the triangulation are left (-1), right (1) or
    // alternated starting at right (2) or left (-2)
    int transfiniteArrangement;
    // do we smooth (transfinite) mesh? (<0 to use default smoothing)
    int transfiniteSmoothing;
    // the extrusion parameters (if any)
    ExtrudeParams *extrude;
    // reverse mesh orientation
    bool reverseMesh;
    // global mesh size constraint for the surface
    double meshSize;
  } meshAttributes ;

  int getMeshingAlgo() const;
  void setMeshingAlgo(int);
  int getCurvatureControlParameter () const;
  void setCurvatureControlParameter(int);
  virtual double getMeshSize() const { return meshAttributes.meshSize; }

  struct {
    mutable GEntity::MeshGenerationStatus status;
    double worst_element_shape, best_element_shape, average_element_shape;
    double smallest_edge_length, longest_edge_length, efficiency_index;
    int nbEdge, nbTriangle;
    int nbGoodQuality, nbGoodLength;
  } meshStatistics;

  // a crude graphical representation using a "cross" defined by pairs
  // of start/end points
  std::vector<SPoint3> cross;

  // the STL mesh
  std::vector<SPoint2> stl_vertices;
  std::vector<int> stl_triangles;

  // a vertex array containing a geometrical representation of the
  // surface
  VertexArray *va_geom_triangles;

  // a array for accessing the transfinite vertices using a pair of
  // indices
  std::vector<std::vector<MVertex*> > transfinite_vertices;

  // relocate mesh vertices using parametric coordinates
  void relocateMeshVertices();




  





  
  std::vector<MTriangle*> triangles;
  std::vector<MQuadrangle*> quadrangles;
  std::vector<MPolygon*> polygons;


  









  
  void addTriangle(MTriangle *t){ triangles.push_back(t); }
  void addQuadrangle(MQuadrangle *q){ quadrangles.push_back(q); }
  void addPolygon(MPolygon *p){ polygons.push_back(p); }

  // get the boundary layer columns
  BoundaryLayerColumns *getColumns () {return &_columns;}

  std::vector<SPoint3> storage1; //sizes and directions storage
  std::vector<SVector3> storage2; //sizes and directions storage
  std::vector<SVector3> storage3; //sizes and directions storage
  std::vector<double> storage4; //sizes and directions storage
};

#endif
