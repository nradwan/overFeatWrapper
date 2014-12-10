cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#find_package(cmake_modules REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})
add_definitions(${EIGEN_DEFINITIONS})

#Find PCL 1.7
find_package(PCL 1.5 REQUIRED COMPONENTS segmentation filters visualization)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
include_directories("/usr/include/ni")

#check for OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  if(MSVC90 OR MSVC10)
    if(MSVC90)
      set(OPENMP_DLL VCOMP90)
    elseif(MSVC10)
      set(OPENMP_DLL VCOMP100)
    endif(MSVC90)
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /DELAYLOAD:${OPENMP_DLL}D.dll")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DELAYLOAD:${OPENMP_DLL}.dll")
  endif(MSVC90 OR MSVC10)
else(OPENMP_FOUND)
  message (STATUS "OpenMP not found")
endif()

if (MSVC)
  Set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /O2 ${SSE_FLAGS}")
else (MSVC)
  set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS} ${SSE_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g ${SSE_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE " ${CMAKE_CXX_FLAGS} -O3 ${SSE_FLAGS}")
endif (MSVC)

#uncomment if you have defined messages
rosbuild_genmsg()
#uncomment if you have defined services
rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
#rosbuild_add_executable(example examples/example.cpp)
#target_link_libraries(example ${PROJECT_NAME})
rosbuild_add_executable(Wrapper src/wrapper.cpp)
target_link_libraries(Wrapper ${OpenCV_LIBRARIES} ${PCL_LIBRARIES}
						/home/radwann/overfeat/src/liboverfeat.a
						/home/radwann/overfeat/src/libTH.a)
