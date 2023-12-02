#include "odt_pcl.h"
#include <stdlib.h>
#include <iostream>

int main(int argc, char** argv) {
    std::cout << "####################################" << std::endl;
    std::cout << "test uniform sampling" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    test_uniform_sampling();
    std::cout << std::endl;
    std::cout << "####################################" << std::endl;
    std::cout << "test kdtree_search" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    test_kdtree_search();
    std::cout << std::endl;
    std::cout << "####################################" << std::endl;
    std::cout << "test normal estimation" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    test_normal_estimation();
    std::cout << std::endl;
    std::cout << "####################################" << std::endl;
    std::cout << "test pfh estimation" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    test_pfh_estimation();
    std::cout << "####################################" << std::endl;
    std::cout << "test sacia" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    test_sacia();
    return 0;
}

