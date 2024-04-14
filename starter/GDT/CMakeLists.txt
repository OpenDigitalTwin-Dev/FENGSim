cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(gdt_api)

add_subdirectory(src)

set_target_properties(fengsim_meas PROPERTIES RUNTIME_OUTPUT_DIRECTORY "./../")
