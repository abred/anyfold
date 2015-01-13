INCLUDE_DIRECTORIES(.)

SET(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE (Boost 1.42 QUIET COMPONENTS system filesystem unit_test_framework REQUIRED)
IF(Boost_FOUND)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS}) 
ENDIF()

add_executable(test_cpu_convolve test_cpu_convolve.cpp)
target_link_libraries(test_cpu_convolve boost_system boost_filesystem boost_unit_test_framework anyfold)
set_target_properties(test_cpu_convolve PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/include")

if(BUILD_OPENCL_ANYFOLD)
  add_executable(test_opencl_convolve test_opencl_convolve.cpp)
  target_link_libraries(test_opencl_convolve boost_system boost_filesystem boost_unit_test_framework ${OpenCL_LIBRARIES} anyfold)
  set_target_properties(test_opencl_convolve PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/include")
ENDIF()