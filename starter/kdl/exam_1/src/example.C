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

double* Angle (double* pos) {
    KDL::Chain chain;
    
    chain.addSegment(Segment(Joint(Joint::RotZ),Frame(Vector(0.0,0.0,0.1518))));
    chain.addSegment(Segment(Joint(Joint::RotY),Frame(Vector(0.0,0.0,0.2435))));
    chain.addSegment(Segment(Joint(Joint::RotY),Frame(Vector(0.0,0.0,0.2132))));
    chain.addSegment(Segment(Joint(Joint::RotY),Frame(Vector(0.0,-0.11040,0.08535))));
    chain.addSegment(Segment(Joint(Joint::RotZ),Frame(Vector(0.0,0.0,0.0))));
    chain.addSegment(Segment(Joint(Joint::RotY),Frame(Vector(0.0,-0.08246,0.0))));
    
    // Create solver based on inverse kinematic chain
    ChainIkSolverPos_LMA iksolver(chain,1E-5,1000,1E-15);
    
    KDL::Frame desired_pose(
	KDL::Rotation::RPY(M_PI/2, 0, 0), 
        KDL::Vector(pos[0], pos[1], pos[2])    // Position (x, y, z)
	);
    
    KDL::JntArray joint_init(chain.getNrOfJoints());
    for (unsigned int i = 0; i < joint_init.rows(); ++i) {
        joint_init(i) = 0.0;  // Initialize to zero or another reasonable guess
    }
    
    unsigned int nj = chain.getNrOfJoints();
    KDL::JntArray joint_result = JntArray(nj);
    
    // Solve IK
    int status = iksolver.CartToJnt(joint_init, desired_pose, joint_result);

    double* angle = new double[6];
    for (int i=0; i<6; i++) angle[i] = joint_result(i);
    return angle;
    
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
	    0.08535*cos(joint_result(1)+joint_result(2)+joint_result(3)) << std::endl << std::endl;
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
    if(kinematics_status>=0) {
        std::cout << cartpos <<std::endl;
        printf("%s \n","Succes, thanks KDL!");
    }
    else {
        printf("%s \n","Error: could not calculate forward kinematics :(");
    }
}


int main( int argc, char** argv ) {
    std::ifstream inFile("../../../mbdyn/robot/ur3.pnts");
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(inFile, line)) {
        lines.push_back(line);
    }
    inFile.close();

    double Time = 1;
    std::vector<double*> angles;
    for (int i=0; i<lines.size(); i++) {
	double z[4];
	sscanf(lines[i].data(),"%lf %lf %lf %lf",z,z+1,z+2,z+3);
	double t = z[0];
	double* pos = new double[3];
	pos[0] = z[1]; 	pos[1] = z[2]; 	pos[2] = z[3];
	double* angle = new double[6];
	angle = Angle(pos);
	double* angle_t = new double[7];
	angle_t[0] = t;
	for (int j=0; j<6; j++) angle_t[j+1] = angle[j];
	angles.push_back(angle_t);
	Time = t;
    }
    std::cout << "Time: " << Time << std::endl;

    for (int i=0; i<angles.size(); i++) {
	for (int j=0; j<7; j++) {
	    std::cout << angles[i][j] << " ";
	}
	std::cout << std::endl;
    }

    /*!
      output
    */
    inFile.open("../../../mbdyn/robot/ur3.mbd");
    lines.clear();
    while (std::getline(inFile, line)) {
        lines.push_back(line);
    }
    inFile.close();

    lines[14] = "   final time:     "+std::to_string(Time)+";";
    
    int line_id[5];
    line_id[0] = 67;
    line_id[1] = 64;
    line_id[2] = 61;
    line_id[3] = 58;
    line_id[4] = 55;
    for (int k=0; k<5; k++) {
	for (int i=0; i<angles.size(); i++) {
	    double* s = angles[angles.size()-(i+1)];
	    if (i==0) {
		lines.insert(lines.begin()+line_id[k],"      "+std::to_string(s[0])+", "+std::to_string(s[5-k])+";");
	    }
	    else {
		lines.insert(lines.begin()+line_id[k],"      "+std::to_string(s[0])+", "+std::to_string(s[5-k])+",");
	    }
	}
	lines.insert(lines.begin()+line_id[k],"      0.1, 0.0,");
	lines.insert(lines.begin()+line_id[k],"      0.0, 0.0,");
    }
    
    std::ofstream outFile("../../../mbdyn/robot/robot_arm.mbd");
    for (const auto& l : lines) {
        outFile << l << '\n';
    }
    outFile.close();
}
