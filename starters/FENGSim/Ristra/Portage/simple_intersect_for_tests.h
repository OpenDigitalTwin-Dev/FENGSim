/*
This file is part of the Ristra portage project.
Please see the license file at the root of this repository, or at:
    https://github.com/laristra/portage/blob/master/LICENSE
*/

#include <vector>

#include "wonton/support/wonton.h"
#include "wonton/support/Point.h"
#include "wonton/support/CoordinateSystem.h"

#include "portage/support/portage.h"


namespace BOX_INTERSECT {

template <int D>
void bounding_box(std::vector<Wonton::Point<D>> coords,
                  Wonton::Point<D> *pmin, Wonton::Point<D> *pmax) {
  *pmin = coords[0];
  *pmax = coords[0];
  for (auto pcoord : coords) {
    for (int d = 0; d < D; d++) {
      if (pcoord[d] < (*pmin)[d])
        (*pmin)[d] = pcoord[d];
      if (pcoord[d] > (*pmax)[d])
        (*pmax)[d] = pcoord[d];
    }
  }
}

template <int D, class CoordSys = Wonton::CartesianCoordinates>
bool intersect_boxes(Wonton::Point<D> min1, Wonton::Point<D> max1,
                     Wonton::Point<D> min2, Wonton::Point<D> max2,
                     std::vector<double> *xsect_moments) {

  Wonton::Point<D> intmin, intmax;

  for (int d = 0; d < D; ++d) {
    // check for non-intersection in this dimension

    if (min1[d] > max2[d]) return false;
    if (min2[d] > max1[d]) return false;

    // sort the min max vals in this dimension
    double val[4];
	
    val[0] = min1[d]; val[1] = max1[d]; val[2] = min2[d]; val[3] = max2[d];

	
    for (int i = 0; i < 3; i++)
      for (int j = i+1; j < 4; j++)
        if (val[i] > val[j]) {
          double tmp = val[i];
          val[i] = val[j];
          val[j] = tmp;
        }

    // pick the middle two as the min max coordinates of intersection
    // box in this dimension

	
    intmin[d] = val[1]; intmax[d] = val[2];
  }

  // Calculate the volume

  double vol = 1.0;
  for (int d = 0; d < D; d++)
    vol *= intmax[d]-intmin[d];

  // Sanity check

  assert(vol >= 0.0);


  
  if (vol == 0.0) return false;

  // Calculate the centroid

  Wonton::Point<D> centroid = (intmin+intmax)/2.0;

  // moments

  auto moments = centroid * vol;


  CoordSys::modify_volume(vol, intmin, intmax);
  CoordSys::modify_first_moments(moments, intmin, intmax);

  xsect_moments->clear();
  xsect_moments->push_back(vol);
  for (int d = 0; d < D; d++)
    xsect_moments->push_back(moments[d]);

  return true;
}

template <int D, class CoordSys = Wonton::CartesianCoordinates>
void intersection_moments(std::vector<Wonton::Point<D>> cell_xyz,
                           std::vector<std::vector<Wonton::Point<D>>> candidate_cells_xyz,
                           std::vector<int> *xcells,
                           std::vector<std::vector<double>> *xwts) {

  int num_candidates = candidate_cells_xyz.size();
  
  xwts->clear();

  Wonton::Point<D> cmin, cmax;
  bounding_box<D>(cell_xyz, &cmin, &cmax);

 

  for (int c = 0; c < num_candidates; ++c) {
    Wonton::Point<D> cmin2, cmax2;
    bounding_box<D>(candidate_cells_xyz[c], &cmin2, &cmax2);

    std::vector<double> xsect_moments;
    if (intersect_boxes<D,CoordSys>(cmin, cmax, cmin2, cmax2, &xsect_moments)) {
      xwts->push_back(xsect_moments);
      xcells->push_back(c);
    }
  }
}


}  // namespace BOX_INTERSECT

