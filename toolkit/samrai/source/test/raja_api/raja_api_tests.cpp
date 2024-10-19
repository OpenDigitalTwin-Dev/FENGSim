#include "SAMRAI/pdat/ForAll.h"
#include "SAMRAI/hier/Index.h"
#include "SAMRAI/hier/Box.h"
#include <cstdio>

using namespace SAMRAI;

int main(int argc, char* argv[]) {

  const tbox::Dimension dim((unsigned short) 2 );
  hier::Index box_lower(dim, 0);
  hier::Index box_upper(dim);

  for (int d = 0; d < dim.getValue(); ++d) {
     //box_upper(d) = (d + 4) * 3;
     box_upper(d) = d;
  }

  hier::Box box(box_lower, box_upper, hier::BlockId(0));

  hier::Box::iterator bend(box.end());

  for ( hier::Box::iterator bi(box.begin()); bi != bend; ++bi) {
    std::cout << "bi = " << *bi << std::endl;
  }

  pdat::parallel_for_all(box, [=] __host__ __device__ (int k, int j) {
      //std::cout << "(" << j << "," << k << ")" << std::endl;
      printf("[j,k] = [%d,%d]\n",j,k);
  });

  return 0;
}
