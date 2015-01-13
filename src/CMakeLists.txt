add_library(anyfold opencl/convolution3DCL.cpp)
target_link_libraries(anyfold ${OpenCL_LIBRARIES})
set_target_properties(anyfold PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/include")
