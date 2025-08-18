#include <iostream>
#include <iomanip>

#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

using namespace KDL;

int main(int argc, char *argv[])
{
	//
	// Create a KDL kinematic Chain.
	//
	// A Chain is made up of Segments. Each Segment consists of a Joint and a Frame.
	// The Joint indicates how the Frame moves - rotation or translation about / along an axis.
	//

	Chain kdlChain = Chain();

	Joint joint1(Joint::None);
	Frame frame1 = Frame(Vector(0.0, 1.0, 0.0));
	kdlChain.addSegment(Segment(joint1, frame1));

	Joint joint2(Joint::RotZ);
	Frame frame2 = Frame(Vector(0.0, 2.0, 0.0));
	kdlChain.addSegment(Segment(joint2, frame2));

	Joint joint3(Joint::RotZ);
	Frame frame3 = Frame(Rotation::EulerZYX(0.0, 0.0, -M_PI / 2)) * Frame(Vector(0.0, 0.0, 2.0));
	kdlChain.addSegment(Segment(joint3, frame3));

	Joint joint4(Joint::RotZ);
	Frame frame4 = Frame(Rotation::EulerZYX(0.0, 0.0, M_PI / 2)) * Frame(Vector(1.0, 1.0, 0.0));
	kdlChain.addSegment(Segment(joint4, frame4));

	//
	// Joint Angles
	//

	JntArray jointAngles = JntArray(3);
	jointAngles(0) = -M_PI / 4.;       // Joint 1
	jointAngles(1) = M_PI / 2.;        // Joint 2
	jointAngles(2) = M_PI;             // Joint 3

	//
	// Perform Forward Kinematics
	//

	ChainFkSolverPos_recursive FKSolver = ChainFkSolverPos_recursive(kdlChain);
	Frame eeFrame;
	FKSolver.JntToCart(jointAngles, eeFrame);

	// Print the frame
	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++) {
			double a = eeFrame(i, j);
			if (a < 0.0001 && a > -0.001) {
				a = 0.0;
			}
			std::cout << std::setprecision(4) << a << "\t\t";
		}
		std::cout << std::endl;
	}

	return 0;
}

