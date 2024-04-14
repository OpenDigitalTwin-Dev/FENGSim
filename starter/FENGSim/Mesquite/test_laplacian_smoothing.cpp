/* ***************************************************************** 
    MESQUITE -- The Mesh Quality Improvement Toolkit

    Copyright 2004 Sandia Corporation and Argonne National
    Laboratory.  Under the terms of Contract DE-AC04-94AL85000 
    with Sandia Corporation, the U.S. Government retains certain 
    rights in this software.

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License 
    (lgpl.txt) along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 
    diachin2@llnl.gov, djmelan@sandia.gov, mbrewer@sandia.gov, 
    pknupp@sandia.gov, tleurent@mcs.anl.gov, tmunson@mcs.anl.gov      
   
  ***************************************************************** */
// -*- Mode : c++; tab-width: 2; c-tab-always-indent: t; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//
//   SUMMARY: 
//     USAGE:
//
// ORIG-DATE: 19-Feb-02 at 10:57:52
//  LAST-MOD: 23-Jul-03 at 18:04:37 by Thomas Leurent
//
//
// DESCRIPTION:
// ============
/*! \file main.cpp

describe main.cpp here

 */
// DESCRIP-END.
//

//#include "meshfiles.h"

#include <iostream>
using std::cout;
using std::endl;
#include <cstdlib>

#include "Mesquite.hpp"
#include "MsqError.hpp"
#include "MeshImpl.hpp"
#include "Vector3D.hpp"
#include "InstructionQueue.hpp"
#include "PatchData.hpp"
#include "TerminationCriterion.hpp"
#include "QualityAssessor.hpp"
#include "PlanarDomain.hpp"
#include "MsqTimer.hpp"
#include "MeshImplData.hpp"

// algorythms
#include "ConditionNumberQualityMetric.hpp"
#include "LInfTemplate.hpp"
#include "SteepestDescent.hpp"
#include "LaplacianSmoother.hpp"
#include "EdgeLengthQualityMetric.hpp"
using namespace Mesquite;

int main(int argc, char* argv[]) {
  	MsqError err;
	if (argc != 2) {
		std::cerr << "Expected mesh file names as single argument." << std::endl;
		exit (EXIT_FAILURE);
	}
	Mesquite::MeshImpl mesh;
	mesh.read_vtk(argv[1], err);
	if (err)
		{
			std::cout << err << std::endl;
			return 1;
		}
	mesh.write_vtk("original_mesh.vtk",err);
	
	// creates an intruction queue
	InstructionQueue queue1;
  
    // creates a mean ratio quality metric ...
	//ConditionNumberQualityMetric shape_metric;
	//EdgeLengthQualityMetric lapl_met;
	//lapl_met.set_averaging_method(QualityMetric::RMS);
 
    // creates the laplacian smoother  procedures
	LaplacianSmoother lapl1;
	//QualityAssessor stop_qa=QualityAssessor(&shape_metric);
	//stop_qa.add_quality_assessment(&lapl_met);
  
    //**************Set stopping criterion****************
	TerminationCriterion sc2;
	sc2.add_iteration_limit( 1 );
	if (err) return 1;
	lapl1.set_outer_termination_criterion(&sc2);
  
    // adds 1 pass of pass1 to mesh_set1
	//queue1.add_quality_assessor(&stop_qa,err); 
	//if (err) return 1;
	queue1.set_master_quality_improver(&lapl1, err); 
	//if (err) return 1;
	//queue1.add_quality_assessor(&stop_qa,err); 
	//if (err) return 1;
    
	PlanarDomain plane(Vector3D(0,0,1),
					   Vector3D(0,0,0));
  
    // launches optimization on mesh_set1
	MeshDomainAssoc mesh_and_domain = MeshDomainAssoc(&mesh, &plane);
	queue1.run_instructions(&mesh_and_domain, err);
	mesh.write_vtk("smoothed_mesh.vtk", err);
	if (err) return 1;
	
	return 0;
}
