/*
 *
 *		utest.h 
 *		
 *		Declarations for a simple unit testing framework.
 *		
 *		Devon Powell
 *		31 August 2015
 *
 *		This program was prepared by Los Alamos National Security, LLC at Los Alamos National
 *		Laboratory (LANL) under contract No. DE-AC52-06NA25396 with the U.S. Department of Energy (DOE). 
 *		All rights in the program are reserved by the DOE and Los Alamos National Security, LLC.  
 *		Permission is granted to the public to copy and use this software without charge, provided that 
 *		this Notice and any statement of authorship are reproduced on all copies.  Neither the U.S. 
 *		Government nor LANS makes any warranty, express or implied, or assumes any liability 
 *		or responsibility for the use of this software.
 *
 */

#ifndef _UTEST_H_
#define _UTEST_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>



// -- functions to be defined by the user -- //

// setup() will run prior to the tests, 
// and can be left as an empty function if desired.
void setup();

// register_all_tests() should be defined and filled with calls
// to register_test(), one for each defined unit test.
void register_all_tests();



// -- assertion and expectation macros to be used inside user code -- //

// pass/fail/warn state IDs
#define PASS 2
#define WARN 1
#define FAIL 0

// fails if the argument is false
#define ASSERT_TRUE(x) {	\
	if(!(x)) {	\
		test_state = FAIL;	\
		printf("\x1b[31mASSERT_TRUE( %d ) failed:\x1b[0m\n  %s, line %d.\n",	\
				x, __FILE__, __LINE__);	\
		return; \
	}	\
}	\

// fails if the argument is true
#define ASSERT_FALSE(x) {	\
	if((x)) {	\
		test_state = FAIL;	\
		printf("\x1b[31mASSERT_FALSE( %d ) failed:\x1b[0m\n  %s, line %d.\n", 	\
				x, __FILE__, __LINE__);	\
		return; \
	}	\
}	\

// warns if the argument is false
#define EXPECT_TRUE(x) {	\
	if(!(x)) {	\
		test_state = WARN;	\
		printf("\x1b[33mEXPECT_TRUE( %d ) failed:\x1b[0m\n  %s, line %d.\n",	\
				x, __FILE__, __LINE__);	\
	}	\
}	\

// warns if the argument is true
#define EXPECT_FALSE(x) {	\
	if((x)) {	\
		test_state = WARN;	\
		printf("\x1b[33mEXPECT_FALSE( %d ) failed:\x1b[0m\n  %s, line %d.\n",	\
			   x, __FILE__, __LINE__);	\
	}	\
}	\

// fails if two floating-point numbers are not within some fractional tolerance
#define ASSERT_EQ(x, y, tol) {	\
	double err = fabs(1.0-(x)/(y));	\
	if(err > (tol)) {	\
		test_state = FAIL;	\
		printf("\x1b[31mASSERT_EQ( %.3e , %.3e , %.3e ) failed ( err = %.3e ):\x1b[0m\n  %s, line %d.\n",	\
				x, y, tol, err, __FILE__, __LINE__);	\
		return; \
	}	\
}	\

// warns if two floating-point numbers are not within some fractional tolerance
#define EXPECT_EQ(x, y, tol) {	\
	double err = fabs(1.0-(x)/(y));	\
	if(err > (tol)) {	\
		test_state = WARN;	\
		printf("\x1b[33mEXPECT_EQ( %.3e , %.3e , %.3e ) failed ( err = %.3e ):\x1b[0m\n  %s, line %d.\n",	\
				x, y, tol, err, __FILE__, __LINE__);	\
	}	\
}	\


// fails if x >= y 
#define ASSERT_LT(x, y) {	\
	if((x) >= (y)) {	\
		test_state = FAIL;	\
		printf("\x1b[31mASSERT_LT( %.3e , %.3e ) failed:\x1b[0m\n  %s, line %d.\n",	\
				x, y, __FILE__, __LINE__);	\
		return; \
	}	\
}	\

// warns if x >= y 
#define EXPECT_LT(x, y) {	\
	if((x) >= (y)) {	\
		test_state = WARN;	\
		printf("\x1b[33mEXPECT_LT( %.3e , %.3e ) failed:\x1b[0m\n  %s, line %d.\n",	\
				x, y, __FILE__, __LINE__);	\
	}	\
}	\





// -- other -- //

// use this to register a test by passing
// a function pointer and a test name
void register_test(void (*testfn)(void), char* name);

// the state of the current test
extern int test_state;


#endif // _UTEST_H_
