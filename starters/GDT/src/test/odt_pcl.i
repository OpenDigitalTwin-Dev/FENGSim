/* example.i */
%module odt_pcl
%{
  /* Put header files here or function declarations like below */
  //extern void test_uniform_sampling ();
#include "odt_pcl.h"
  %}

//extern void test_uniform_sampling ();
%include "odt_pcl.h"
