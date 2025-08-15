// Copyright  (C)  2007  Francois Cauwe <francois at cauwe dot org>
 
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
 
#include <kdl/chain.hpp>
#include <kdl/chainfksolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/frames_io.hpp>
#include <stdio.h>
#include <iostream>

#include <kdl/chainiksolver.hpp>
#include <kdl/chainiksolverpos_lma.hpp>

#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainiksolverpos_nr.hpp>

#include <fstream>
#include <vector>
#include <string>

using namespace KDL;
 

int main( int argc, char** argv )
{
    //Definition of a kinematic chain & add segments to the chain
    KDL::Chain chain;
    chain.addSegment(Segment(Joint(Joint::RotZ),Frame(Vector(0.0,0.0,0.1518))));
    chain.addSegment(Segment(Joint(Joint::RotX),Frame(Vector(0.0,0.0,0.2435))));
    chain.addSegment(Segment(Joint(Joint::RotX),Frame(Vector(0.0,0.0,0.2132))));
    chain.addSegment(Segment(Joint(Joint::RotX),Frame(Vector(0.11040,0.0,0.0))));
    chain.addSegment(Segment(Joint(Joint::RotZ),Frame(Vector(0.0,0.0,0.08535))));
    chain.addSegment(Segment(Joint(Joint::RotX),Frame(Vector(0.08246,0.0,0.0))));






			 
    
    // Create solver based on inverse kinematic chain
    ChainIkSolverPos_LMA iksolver(chain,1E-5,1000,1E-15);

    KDL::ChainFkSolverPos_recursive fk_solver(chain);
    KDL::ChainIkSolverVel_pinv vel_ik_solver(chain);
    //KDL::ChainIkSolverPos_NR iksolver(chain, fk_solver, vel_ik_solver, 100, 1e-6);
    
    KDL::Frame desired_pose(
	//KDL::Rotation::RPY(0.1, 0, 0), 
        KDL::Vector(0.3, 0.3, 0.3)    // Position (x, y, z)
	);
    
    KDL::JntArray joint_init(chain.getNrOfJoints());
    for (unsigned int i = 0; i < joint_init.rows(); ++i) {
        joint_init(i) = 0.0;  // Initialize to zero or another reasonable guess
    }
    
    unsigned int nj = chain.getNrOfJoints();
    KDL::JntArray joint_result = JntArray(nj);
    
    // Solve IK
    int status = iksolver.CartToJnt(joint_init, desired_pose, joint_result);
    if (status >= 0) {
	//std::cout << "IK Solution: " << std::endl << joint_result.data << std::endl;
	std::cout << "z: " << joint_result(0) << std::endl;
	std::cout << "x: " << joint_result(1) << std::endl;
	std::cout << "x: " << joint_result(2) << std::endl;
	std::cout << "x: " << joint_result(3) << std::endl;
	std::cout << "z: " << joint_result(4) << std::endl;
	std::cout << "x: " << joint_result(5) << std::endl << std::endl;
	

	std::cout << "z: " <<
	    0.1518+
	    0.2435*cos(joint_result(1))+
	    0.2132*cos(joint_result(1)+joint_result(2))+
	    0.08535*cos(joint_result(1)+joint_result(2)+joint_result(3)) << std::endl;
    } else {
	std::cerr << "IK Failed!" << std::endl;
    }

    /*!
      check by fk
    */
    
    ChainFkSolverPos_recursive fksolver = ChainFkSolverPos_recursive(chain);
 
    KDL::JntArray jointpositions = JntArray(nj);
 
    for(unsigned int i=0;i<nj;i++){
        jointpositions(i)=joint_result(i);
    }
 
    KDL::Frame cartpos;    
 
    bool kinematics_status;
    kinematics_status = fksolver.JntToCart(jointpositions,cartpos);
    if(kinematics_status>=0){
        std::cout << cartpos <<std::endl;
        printf("%s \n","Succes, thanks KDL!");
    }else{
        printf("%s \n","Error: could not calculate forward kinematics :(");
    }


    /*!
      output
    */
    std::ifstream inFile("../../../mbdyn/robot/robot_arm.mbd");

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(inFile, line)) {
        lines.push_back(line);
    }
    inFile.close();

    lines[60-1] = "      3.0, " + std::to_string(joint_result(0)) + ";";
    lines[67-1] = "      3.0, " + std::to_string(joint_result(1)) + ";";
    lines[74-1] = "      3.0, " + std::to_string(joint_result(2)) + ";";
    lines[81-1] = "      3.0, " + std::to_string(joint_result(3)) + ";";
    lines[88-1] = "      3.0, " + std::to_string(joint_result(4)) + ";";

    // 写回文件
    std::ofstream outFile("../../../mbdyn/robot/robot_arm.mbd");
    for (const auto& l : lines) {
        outFile << l << '\n';
    }
    outFile.close();
}
