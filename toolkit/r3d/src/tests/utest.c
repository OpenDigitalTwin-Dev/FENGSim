/*
 *
 *		utest.c 
 *		
 *		A simple unit testing framework.
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

#include "utest.h"

// list of unit test function pointers and names
static int num_tests;
static struct {
	void (*testfn)(void);
	char name[128];
} all_tests[256];

// global for the current test state
int test_state;

// registers a unit test prior to running
void register_test(void (*testfn)(void), char* name) {
	all_tests[num_tests].testfn = testfn;
	strcpy(all_tests[num_tests].name, name);
	++num_tests;
}

int main() {

	// call the user-implemented explicit 
	// registration of unit test functions
	num_tests = 0;
	register_all_tests();

	// call the user-implemented setup function
	// (which can be empty)
	setup();

	// run tests
	printf("===========================\n");
	printf("\x1b[1mRunning %d unit tests...\x1b[0m\n", num_tests);
	printf("===========================\n");
		
	int t, npass, nwarn, nfail;
	clock_t start, stop;
	double millis;
	npass = 0; nwarn = 0; nfail = 0;
	for(t = 0; t < num_tests; ++t) {
		printf("\x1b[1m%s\x1b[0m\n", all_tests[t].name);
		start = clock();
		test_state = PASS;
		all_tests[t].testfn();
		stop = clock();
		millis = 1000.0*((double)(stop - start))/CLOCKS_PER_SEC;
		switch(test_state) {
			case PASS:
				printf("\x1b[1;32m[ PASS - %.1lf ms ]\x1b[0m\n", millis);
				++npass;
				break;
			case WARN:
				printf("\x1b[1;33m[ WARN - %.1lf ms ]\x1b[0m\n", millis);
				++nwarn;
				break;
			case FAIL:
				printf("\x1b[1;31m[ FAIL - %.1lf ms ]\x1b[0m\n", millis);
				++nfail;
				break;
			default:
				break;
		}

		printf("---------------------------\n");

	}
	printf("\x1b[1m...done.\n");
	printf("\x1b[1;32m%d x [ PASS ]\x1b[0m\n", npass);
	printf("\x1b[1;33m%d x [ WARN ]\x1b[0m\n", nwarn);
	printf("\x1b[1;31m%d x [ FAIL ]\x1b[0m\n", nfail);
	printf("===========================\n");
	return 0;
}

