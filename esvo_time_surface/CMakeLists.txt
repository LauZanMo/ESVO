cmake_minimum_required(VERSION 2.8.3)
project(esvo_time_surface)

# To be consistent with the configuration in esvo_core
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
set(CMAKE_CXX_FLAGS "-O3")

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4/")
find_package(OpenCV REQUIRED)

# set(ESVIO_INCLUDE_DIRS "/home/ubuntu/Code/ESVIO/src/")

find_path(ESVIO_INCLUDE_DIRS
  NAMES ESVIO.h
  PATHS
    /home/ubuntu/Code/ESVIO/src/
)

find_library(ESVIO_LIBRARIES
  NAMES ESVIO
  PATHS 
    /home/ubuntu/Code/ESVIO/lib/
)

# make the executable
cs_add_executable(${PROJECT_NAME}
    src/TimeSurface.cpp
    src/TimeSurface_node.cpp
    src/SedtsWrapper.cpp
)

# include_directories(include ${catkin_INCLUDE_DIRS} ${ESVIO_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC
    include
    ${catkin_INCLUDE_DIRS}
    ${ESVIO_INCLUDE_DIRS}
)

# link the executable to the necesarry libs
target_link_libraries(${PROJECT_NAME}
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
   ${ESVIO_LIBRARIES}
)

cs_install()