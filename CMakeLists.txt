# cmake compatibility issues
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# project name
PROJECT(anyfold CXX C)
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
 
# version number
SET (ANYFOLD_NAME "anyfold")
SET (ANYFOLD_CODENAME "${PROJECT_NAME}")
SET (ANYFOLD_COPYRIGHT_YEARS "2014")
SET (ANYFOLD_VERSION_MAJOR 0)
SET (ANYFOLD_VERSION_MINOR 0)
SET (ANYFOLD_VERSION_PATCH 0)
SET (ANYFOLD_VERSION "${ANYFOLD_VERSION_MAJOR}.${ANYFOLD_VERSION_MINOR}.${ANYFOLD_VERSION_PATCH}")
SET (ANYFOLD_VENDOR_ID "mpi cbg")
SET (ANYFOLD_VENDOR_NAME "Max Planck Institute of Molecular Cell Biology and Genetics ")
SET (ANYFOLD_VENDOR_URL "www.mpi-cbg.de")
SET (ANYFOLD_ID "${ANYFOLD_VENDOR_ID}.${PROJECT_NAME}")

# trying to setup paths so this package can be picked up
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")

# shared path is architecture independent for now, TODO extend this to lib/bin/include
IF(UNIX)
  IF(APPLE)
    set(INSTALL_SHARE_DIR ${PROJECT_NAME}.app/Contents/Resources/ CACHE PATH "Installation directory for shared files")
  ELSE(APPLE)
    set(INSTALL_SHARE_DIR lib/CMake/${PROJECT_NAME} CACHE PATH "Installation directory for shared files")
  ENDIF(APPLE)
    #the following was tested with OSX 10.8.5 and Xcode 5.0.2
    #seems to me that under apple the rpath is not stripped automatically when doing the install
    #under linux it is
    SET(CMAKE_SKIP_RPATH ON)
ELSE(UNIX)
  IF(WIN32 AND NOT CYGWIN)
    set(INSTALL_SHARE_DIR CMake CACHE PATH "Installation directory for shared files")
  ELSE(WIN32)
    MESSAGE(FATAL_ERROR ">> UNKNOWN ARCHITECTURE .. unable to set share dir")
  ENDIF(WIN32)
ENDIF(UNIX)
 
# Make relative paths absolute (needed later on)
foreach(p LIB INCLUDE SHARE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

include_directories("${PROJECT_SOURCE_DIR}/include")

# set project build type
IF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Release)
ENDIF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
MESSAGE(">> Setting up ${CMAKE_BUILD_TYPE} build")

#C/C++ specific flags
IF(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_GNUCC)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -std=c++11 ")
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g -ggdb -std=c++11 ")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -g -ggdb ")
ENDIF(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_GNUCC)
 
IF("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g -ggdb ")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -g -ggdb ")
ENDIF()

check_cxx_compiler_flag(-fvectorize HAS_CLANG_VECTORIZE_COMPILERFLAG)
check_cxx_compiler_flag(-ftree-vectorize HAS_GCC_TREE_VECTORIZE_COMPILERFLAG)
check_cxx_compiler_flag(-march=native HAS_MARCH_COMPILERFLAG)
check_cxx_compiler_flag(-ffast-math HAS_FAST_MATH_COMPILERFLAG)

SET(HOST_COMPILER_RELEASE_FLAGS "-Wall -O3")
IF(HAS_CLANG_VECTORIZE_COMPILERFLAG)
  SET(HOST_COMPILER_RELEASE_FLAGS "${HOST_COMPILER_RELEASE_FLAGS} -fvectorize")
ENDIF()

IF(HAS_GCC_TREE_VECTORIZE_COMPILERFLAG)
  SET(HOST_COMPILER_RELEASE_FLAGS "${HOST_COMPILER_RELEASE_FLAGS} -ftree-vectorize")
ENDIF()

IF(HAS_MARCH_COMPILERFLAG)
  SET(HOST_COMPILER_RELEASE_FLAGS "${HOST_COMPILER_RELEASE_FLAGS} -march=native")
ENDIF()

IF(HAS_FAST_MATH_COMPILERFLAG)
  SET(HOST_COMPILER_RELEASE_FLAGS "${HOST_COMPILER_RELEASE_FLAGS} -ffast-math")
ENDIF()

SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${HOST_COMPILER_RELEASE_FLAGS} ")
SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${HOST_COMPILER_RELEASE_FLAGS} ")
MESSAGE(">> using release build flags: ${HOST_COMPILER_RELEASE_FLAGS}")

OPTION(BUILD_OPENCL_ANYFOLD "build anyfold with opencl" true)
IF(BUILD_OPENCL_ANYFOLD)
  LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
  FIND_PACKAGE (OpenCL REQUIRED)
  IF(OpenCL_FOUND)
    INCLUDE_DIRECTORIES(${OpenCL_INCLUDE_DIR})
    ADD_DEFINITIONS(-DHAS_OPENCL -DPROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")
  ENDIF()
ENDIF()

ADD_SUBDIRECTORY(src)

FIND_PACKAGE (Boost 1.42 COMPONENTS system filesystem unit_test_framework thread REQUIRED)
IF(Boost_FOUND)
ADD_SUBDIRECTORY(tests)
enable_testing()
include("CTestLists.txt")
ELSE()
MESSAGE(">> Boost libraries not found, unable to compile tests (skipping)")
ENDIF()

export(PACKAGE ${PROJECT_NAME})
# Create the anyfold-config.cmake and anyfold-config-version files
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_SHARE_DIR}"
   "${INSTALL_INCLUDE_DIR}")
# ... for the build tree
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
configure_file(anyfold-config.cmake.in
  "${PROJECT_BINARY_DIR}/anyfold-config.cmake" @ONLY)
# ... for the install tree
set(CONF_INCLUDE_DIRS "\${ANYFOLD}/${REL_INCLUDE_DIR}")
configure_file(anyfold-config.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/anyfold-config.cmake" @ONLY)
# ... for both
configure_file(anyfold-config-version.cmake.in
  "${PROJECT_BINARY_DIR}/anyfold-config-version.cmake" @ONLY)

# Install the anyfold-config.cmake and anyfold-config-version.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/anyfold-config.cmake"
  "${PROJECT_BINARY_DIR}/anyfold-config-version.cmake"
  DESTINATION "${INSTALL_SHARE_DIR}" COMPONENT dev)




