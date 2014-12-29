add_library(anyfold gpu/convolution3DCL.cpp)
target_link_libraries(anyfold OpenCL)
set_target_properties(anyfold PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/include -I/opt/AMDAPPSDK-2.9-1/include/")
