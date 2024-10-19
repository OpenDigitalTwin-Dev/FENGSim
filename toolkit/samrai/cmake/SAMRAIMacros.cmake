macro (samrai_add_tests)

  set(singleValueArgs NAME EXECUTABLE PARALLEL PERFORMANCE EXTRA_ARG)
  set(multiValueArgs INPUTS)
  set(base_name ${arg_NAME})

  cmake_parse_arguments(arg
      "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

  if (${arg_PERFORMANCE})
    set(do_small_tests FALSE)
    set(test_label "perf_test")
  else()
    set(do_small_tests TRUE)
    set(test_label "test")
  endif()  


  foreach (test_file ${arg_INPUTS})

    set(command_args ${test_file} ${arg_EXTRA_ARG})

    get_filename_component(short_test_file ${test_file} NAME)
    set(test_name "${base_name}_${test_label}_${short_test_file}")
    message(STATUS "Test: ${test_name} with input ${command_args}")

    if (${do_small_tests})
      if(ENABLE_MPI)
        if (${MIN_TEST_PROCS} LESS_EQUAL 1)
          blt_add_test(NAME ${test_name}
            COMMAND ${arg_EXECUTABLE} ${command_args}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            NUM_MPI_TASKS 1)
          list(APPEND active_tests ${test_name})
        endif()
      else()
        blt_add_test(NAME ${test_name}
          COMMAND ${arg_EXECUTABLE} ${command_args}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        list(APPEND active_tests ${test_name})
      endif()  
    endif()  
 
    if(${arg_PARALLEL})

      if (${do_small_tests})
        if (${MIN_TEST_PROCS} LESS_EQUAL 2)
          set(test_name "${base_name}_${test_label}_${short_test_file}_2")
          blt_add_test(NAME ${test_name}
            COMMAND ${arg_EXECUTABLE} ${command_args}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            NUM_MPI_TASKS 2)
          list(APPEND active_tests ${test_name})
        endif()
        if (${MIN_TEST_PROCS} LESS_EQUAL 4)
          set(test_name "${base_name}_${test_label}_${short_test_file}_4")
          blt_add_test(NAME ${test_name}
            COMMAND ${arg_EXECUTABLE} ${command_args}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            NUM_MPI_TASKS 4)
          list(APPEND active_tests ${test_name})
        endif()
      else()
        set(num_perf_procs ${NUM_PERF_PROCS})
        set(test_name "${base_name}_${test_label}_${short_test_file}_${num_perf_procs}")
        blt_add_test(NAME ${test_name}
          COMMAND ${arg_EXECUTABLE} ${command_args}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          NUM_MPI_TASKS ${num_perf_procs})
        list(APPEND active_tests ${test_name})
      endif()
    endif()

    set_tests_properties(${active_tests} PROPERTIES PASS_REGULAR_EXPRESSION "PASSED")

  endforeach ()
endmacro ()

macro (samrai_add_perf_tests)

  set(singleValueArgs NAME EXECUTABLE PARALLEL EXTRA_ARG)
  set(multiValueArgs INPUTS)

  samrai_add_tests(NAME ${arg_NAME}
                   EXECUTABLE ${arg_EXECUTABLE} 
                   PARALLEL ${arg_PARALLEL} 
                   EXTRA_ARG ${arg_EXTRA_ARG}
                   INPUTS ${arg_INPUTS} )

endmacro()
