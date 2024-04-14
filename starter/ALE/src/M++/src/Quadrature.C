// file:     Quadrature.C
// author:   Christian Wieners
// $Header: /public/M++/src/Quadrature.C,v 1.4 2009-03-25 09:39:53 wieners Exp $

#include "Debug.h"
#include "Quadrature.h"

const double X = 0.2113248654051871177454256097490212721761991243649365619906;
const double Y = 0.7886751345948128822545743902509787278238008756350634380093;

const double A_1 = 0.112701665379258;
const double A_2 = 0.887298334620742;
const double W_1 = 0.277777777777777;
const double W_2 = 0.444444444444444;

const double C_0 = 0.5*(1-sqrt(525+70*sqrt(30.0))/35);
const double C_1 = 0.5*(1-sqrt(525-70*sqrt(30.0))/35);
const double C_2 = 1 - C_1;
const double C_3 = 1 - C_0;
const double G_0 = 0.25 - sqrt(30.0)/72;
const double G_1 = 0.25 + sqrt(30.0)/72;

const double p = (5-sqrt(5.0)) / 20;
const double q = (5+3*sqrt(5.0)) / 20;

const Point Qint1_z[] = { Point(0.5,0.0,0.0) };
const double Qint1_w[] = { 1.0 };
const Quadrature Qint1(1,Qint1_z,Qint1_w);

const Point Qint2_z[] = { Point(X,0.0,0.0), Point(Y,0.0,0.0) };
const double Qint2_w[] = { 0.5, 0.5 };
const Quadrature Qint2(2,Qint2_z,Qint2_w);   

const Point Qint3_z[] = { Point(A_1,0.0,0.0), 
			  Point(0.5,0.0,0.0), 
			  Point(A_2,0.0,0.0) };
const double Qint3_w[] = { W_1, W_2, W_1 };
const Quadrature Qint3(3,Qint3_z,Qint3_w);

const Point Qint5_z[] = { Point(0.04691,0.0,0.0),
			  Point(0.2307655,0.0,0.0),
			  Point(0.5,0.0,0.0),
			  Point(0.7692345,0.0,0.0),
			  Point(0.95309,0.0,0.0)};
const double Qint5_w[] = { 0.236927/2.0, 0.478629/2.0, 0.568889/2.0, 0.478629/2.0, 0.236927/2.0 };
const Quadrature Qint5(5,Qint5_z,Qint5_w);

const Point Qtri1_z[] = { Point(1.0/3.0, 1.0/3.0) };
const double Qtri1_w[] = { 0.5 };
const Quadrature Qtri1(1,Qtri1_z,Qtri1_w);   

const Point Qtri3_z[] = { Point(0.5,0.0),
			  Point(0.5,0.5),
			  Point(0.0,0.5) };
const double Qtri3_w[] = { 1.0/6.0, 1.0/6.0, 1.0/6.0 };
const Quadrature Qtri3(3,Qtri3_z,Qtri3_w);   

const Point Qtri4_z[] = { Point(1.0/3.0, 1.0/3.0),
			  Point(0.6,0.2),
			  Point(0.2,0.2),
			  Point(0.2,0.6) };
const double Qtri4_w[] = { -0.5625 * 0.5,	   
			   0.520833333333333 * 0.5,	      
			   0.520833333333333 * 0.5,	   
			   0.520833333333333 * 0.5 };
const Quadrature Qtri4(4,Qtri4_z,Qtri4_w);   

const Point Qtri7_z[] = { Point(0.333333333333333, 0.333333333333333),
			  Point(0.059715871789770, 0.470142064105115),
			  Point(0.470142064105115, 0.059715871789770),
			  Point(0.470142064105115, 0.470142064105115),
			  Point(0.797426985368435, 0.101286507323456),
			  Point(0.101286507323456, 0.797426985368435),
			  Point(0.101286507323456, 0.101286507323456) };
const double Qtri7_w[] = { 0.5 * 0.225,
			   0.5 * 0.132394152788506,
			   0.5 * 0.132394152788506,
			   0.5 * 0.132394152788506,
			   0.5 * 0.125939180544827,
			   0.5 * 0.125939180544827,
			   0.5 * 0.125939180544827 };
const Quadrature Qtri7(7,Qtri7_z,Qtri7_w);   

const Point Qtri16_z[] = { Point(0.0571041961000000,0.0654669926674269), 
			   Point(0.2768430136000000,0.0502101217655523),  
			   Point(0.5835904324000000,0.0289120833881734),  
			   Point(0.8602401357000000,0.0097037848439710),  
			   Point(0.0571041961000000,0.3111645522420085),  
			   Point(0.2768430136000000,0.2386486597385485),  
			   Point(0.5835904324000000,0.1374191041211636),  
			   Point(0.8602401357000000,0.0461220798909458),  
			   Point(0.0571041961000000,0.6317312516579915),  
			   Point(0.2768430136000000,0.4845083266614514),  
			   Point(0.5835904324000000,0.2789904634788364),  
			   Point(0.8602401357000000,0.0936377844090542),  
			   Point(0.0571041961000000,0.8774288093467815),  
			   Point(0.2768430136000000,0.6729468631881337),  
			   Point(0.5835904324000000,0.3874974833790075),  
			   Point(0.8602401357000000,0.1300560791765092) };
const double Qtri16_w[] = { 0.0235683681921434,	
			    0.03538806790266244,	
			    0.02258404928499881,	
			    0.005423225902802602,
			    0.0441850885078566,	
			    0.06634421609733757,	
			    0.04233972451500118,	
			    0.0101672595471974,	
			    0.0441850885078566,	
			    0.06634421609733757,	
			    0.04233972451500118,	
			    0.0101672595471974,	
			    0.0235683681921434,	
			    0.03538806790266244,	
			    0.02258404928499881,	
			    0.005423225902802602 };
const Quadrature Qtri16(16,Qtri16_z,Qtri16_w);

const Point Qquad1_z[] = { Point(0.5,0.5) };
const double Qquad1_w[] = { 1.0 };
const Quadrature Qquad1(1,Qquad1_z,Qquad1_w);

const Point Qquad4_z[] = { Point(X,X), Point(Y,X), Point(X,Y), Point(Y,Y) };
const double Qquad4_w[] = { 0.25, 0.25, 0.25, 0.25 };
const Quadrature Qquad4(4,Qquad4_z,Qquad4_w);

const Point Qquad9_z[] = { Point(A_1,A_1),Point(0.5,A_1),Point(A_2,A_1),
			   Point(A_1,0.5),Point(0.5,0.5),Point(A_2,0.5),
			   Point(A_1,A_2),Point(0.5,A_2),Point(A_2,A_2) };
const double Qquad9_w[] = { W_1*W_1, W_2*W_1, W_1*W_1,
			    W_1*W_2, W_2*W_2, W_1*W_2,
			    W_1*W_1, W_2*W_1, W_1*W_1 };
const Quadrature Qquad9(9,Qquad9_z,Qquad9_w);

const Point Qquad16_z[] = { 
    Point(C_0,C_0),Point(C_0,C_1),Point(C_0,C_2),Point(C_0,C_3),
    Point(C_1,C_0),Point(C_1,C_1),Point(C_1,C_2),Point(C_1,C_3),
    Point(C_2,C_0),Point(C_2,C_1),Point(C_2,C_2),Point(C_2,C_3),
    Point(C_3,C_0),Point(C_3,C_1),Point(C_3,C_2),Point(C_3,C_3)};
const double Qquad16_w[] = { G_0*G_0, G_0*G_1, G_0*G_1, G_0*G_0,
			     G_1*G_0, G_1*G_1, G_1*G_1, G_1*G_0,
			     G_1*G_0, G_1*G_1, G_1*G_1, G_1*G_0,
			     G_0*G_0, G_0*G_1, G_0*G_1, G_0*G_0 };
const Quadrature Qquad16(16,Qquad16_z,Qquad16_w);

const Point Qtet1_z[] = { Point(0.25,0.25,0.25) };
const double Qtet1_w[] = { 1.0/6.0 };
const Quadrature Qtet1(1,Qtet1_z,Qtet1_w);   

const Point Qtet4_z[] = {Point(p,p,p),Point(q,p,p),Point(p,q,p),Point(p,p,q)};
const double Qtet4_w[] = { 1.0/24.0, 1.0/24.0, 1.0/24.0, 1.0/24.0 };
const Quadrature Qtet4(4,Qtet4_z,Qtet4_w);   

const Point Qtet11_z[] = { 
    Point(0.2500000000000000,0.2500000000000000,0.2500000000000000),
    Point(0.0714285715000000,0.0714285715000000,0.0714285715000000),
    Point(0.0714285715000000,0.0714285715000000,0.7857142855000000),
    Point(0.0714285715000000,0.7857142855000000,0.0714285715000000),
    Point(0.7857142855000000,0.0714285715000000,0.0714285715000000),
    Point(0.3994035760000000,0.3994035760000000,0.1005964240000000),
    Point(0.3994035760000000,0.1005964240000000,0.3994035760000000),
    Point(0.1005964240000000,0.3994035760000000,0.3994035760000000),
    Point(0.3994035760000000,0.1005964240000000,0.1005964240000000),
    Point(0.1005964240000000,0.3994035760000000,0.1005964240000000),
    Point( 0.100596424000000,0.1005964240000000,0.3994035760000000) };
double Qtet11_w[] = { -0.0131555555,
		      0.00762222225,
		      0.00762222225,
		      0.00762222225,
		      0.00762222225,
		      0.024888888875,
		      0.024888888875,
		      0.024888888875,
		      0.024888888875,
		      0.024888888875,
		      0.024888888875 };
const Quadrature Qtet11(11,Qtet11_z,Qtet11_w);

const Point Qpri6_z[] = { 
    Point(0.66666666666666666, 0.16666666666666666, 0.211324865405187),
    Point(0.16666666666666666, 0.66666666666666666, 0.211324865405187),
    Point(0.16666666666666666, 0.16666666666666666, 0.211324865405187),
    Point(0.66666666666666666, 0.16666666666666666, 0.788675134594813),
    Point(0.16666666666666666, 0.66666666666666666, 0.788675134594813),
    Point(0.16666666666666666, 0.16666666666666666, 0.788675134594813) };
const double Qpri6_w[] ={ 0.16666666666666666,
			  0.16666666666666666,
			  0.16666666666666666,
			  0.16666666666666666,
			  0.16666666666666666,
			  0.16666666666666666 };
const Quadrature Qpri6(6,Qpri6_z,Qpri6_w);   

const Point Qpri8_z[] = { 
    Point(0.333333333333333, 0.333333333333333, 0.211324865405187),
    Point(0.6, 0.2, 0.211324865405187),
    Point(0.2, 0.6, 0.211324865405187),
    Point(0.2, 0.2, 0.211324865405187),
    Point(0.333333333333333, 0.333333333333333, 0.788675134594813),
    Point(0.6, 0.2, 0.788675134594813),
    Point(0.2, 0.6, 0.788675134594813),
    Point(0.2, 0.2, 0.788675134594813) };
const double Qpri8_w[] ={ -0.28125,
			  0.2604166666666666,
			  0.2604166666666666,
			  0.2604166666666666,
			  -0.28125,
			  0.2604166666666666,
			  0.2604166666666666,
			  0.2604166666666666 };
const Quadrature Qpri8(8,Qpri8_z,Qpri8_w);   

const Point Qhex1_z[] = { 
    Point(X,X,X) };
const double Qhex1_w[] = { 0.5 };
const Quadrature Qhex1(1,Qhex1_z,Qhex1_w);   

const Point Qhex8_z[] = { 
    Point(X,X,X), Point(X,Y,X), Point(Y,X,X), Point(Y,Y,X), 
    Point(X,X,Y), Point(X,Y,Y), Point(Y,X,Y), Point(Y,Y,Y) };
const double Qhex8_w[] = { 0.125, 0.125, 0.125, 0.125,
			   0.125, 0.125, 0.125, 0.125 };
const Quadrature Qhex8(8,Qhex8_z,Qhex8_w);   

const Point Qhex27_z[] = { 
    Point(A_1,A_1,A_1), Point(0.5,A_1,A_1), Point(A_2,A_1,A_1),
    Point(A_1,0.5,A_1), Point(0.5,0.5,A_1), Point(A_2,0.5,A_1),
    Point(A_1,A_2,A_1), Point(0.5,A_2,A_1), Point(A_2,A_2,A_1),
    Point(A_1,A_1,0.5), Point(0.5,A_1,0.5), Point(A_2,A_1,0.5),
    Point(A_1,0.5,0.5), Point(0.5,0.5,0.5), Point(A_2,0.5,0.5),
    Point(A_1,A_2,0.5), Point(0.5,A_2,0.5), Point(A_2,A_2,0.5),
    Point(A_1,A_1,A_2), Point(0.5,A_1,A_2), Point(A_2,A_1,A_2),
    Point(A_1,0.5,A_2), Point(0.5,0.5,A_2), Point(A_2,0.5,A_2),
    Point(A_1,A_2,A_2), Point(0.5,A_2,A_2), Point(A_2,A_2,A_2) };
const double Qhex27_w[] = { 
    W_1*W_1*W_1, W_2*W_1*W_1, W_1*W_1*W_1,
    W_1*W_2*W_1, W_2*W_2*W_1, W_1*W_2*W_1,
    W_1*W_1*W_1, W_2*W_1*W_1, W_1*W_1*W_1,
    W_1*W_1*W_2, W_2*W_1*W_2, W_1*W_1*W_2,
    W_1*W_2*W_2, W_2*W_2*W_2, W_1*W_2*W_2,
    W_1*W_1*W_2, W_2*W_1*W_2, W_1*W_1*W_2,
    W_1*W_1*W_1, W_2*W_1*W_1, W_1*W_1*W_1,
    W_1*W_2*W_1, W_2*W_2*W_1, W_1*W_2*W_1,
    W_1*W_1*W_1, W_2*W_1*W_1, W_1*W_1*W_1 };
const Quadrature Qhex27(27,Qhex27_z,Qhex27_w);

const Quadrature& GetQuadrature (const string& name) {
    if (name == "Qint1")   return Qint1;
    if (name == "Qint2")   return Qint2;  
    if (name == "Qint3")   return Qint3;
    if (name == "Qint5")   return Qint5; 
    if (name == "Qtri1")   return Qtri1; 
    if (name == "Qtri3")   return Qtri3;
    if (name == "Qtri4")   return Qtri4; 
    if (name == "Qtri7")   return Qtri7; 
    if (name == "Qtri16")  return Qtri16; 
    if (name == "Qquad1")  return Qquad1; 
    if (name == "Qquad4")  return Qquad4; 
    if (name == "Qquad9")  return Qquad9; 
    if (name == "Qquad16") return Qquad16; 
    if (name == "Qtet1")   return Qtet1; 
    if (name == "Qtet4")   return Qtet4; 
    if (name == "Qtet11")  return Qtet11; 
    if (name == "Qpri6")   return Qpri6; 
    if (name == "Qpri8")   return Qpri8; 
    if (name == "Qhex1")   return Qhex1; 
    if (name == "Qhex8")   return Qhex8; 
    if (name == "Qhex27")  return Qhex27; 
    Exit("unknown quadrature " + name);
}
