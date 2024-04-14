#include "test.h"
#include "mpi.h"

int main(int argc, char** argv) {
  	MPI_Init(&argc, &argv);

	std::cout << "************************************" << std::endl;
	std::cout << "************************************" << std::endl << std::endl;
	std::cout << "test search" << std::endl << std::endl;
	std::cout << "************************************" << std::endl;
	std::cout << "************************************" << std::endl;
	test_search();
	std::cout << "************************************" << std::endl;
	std::cout << "************************************" << std::endl << std::endl;
	std::cout << "test intersection" << std::endl << std::endl;
	std::cout << "************************************" << std::endl;
	std::cout << "************************************" << std::endl;
	test_intersection();
	std::cout << "************************************" << std::endl;
	std::cout << "************************************" << std::endl << std::endl;
	std::cout << "test interpolation 1 with" << std::endl;
	std::cout << "wonton::simple mesh, box intersect" << std::endl;
	std::cout << "and 1st order interpolation" << std::endl << std::endl;
	std::cout << "************************************" << std::endl;
	std::cout << "************************************" << std::endl;
	test_interpolation();
	std::cout << "************************************" << std::endl;
	std::cout << "************************************" << std::endl << std::endl;
	std::cout << "test remapping with Jali mesh and MMDriver" << std::endl << std::endl;
	std::cout << "************************************" << std::endl;
	std::cout << "************************************" << std::endl;
	test_remapping();


	MPI_Finalize();
}


