cmake_minimum_required(VERSION 3.0)

project(pointcloud_preprocessing_cuda)
find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        sensor_msgs
        pcl_conversions
        pcl_ros
        tf2_eigen
        tf2_ros
        rosfmt
        ouster_ros
        swri_profiler
        message_filters
)

find_package(OpenMP REQUIRED)


# Default Flags settings for code optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STRICT_ANSI__")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DOPTIMIZE_AGGRESSIVELY")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -ggdb3")

# Flags for C++17 support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDA 11.8 REQUIRED)
find_package(PCL 1.10 REQUIRED)

catkin_package(
        INCLUDE_DIRS include
        LIBRARIES ${PROJECT_NAME}
        #  CATKIN_DEPENDS cv_bridge image_transport roscpp sensor_msgs std_msgs
        #  DEPENDS system_lib
)

set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )


include_directories(include
        ${catkin_INCLUDE_DIRS}
)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_86,code=sm_86)
set(CUDA_NVCC_FLAGS CACHE STRING "nvcc flags" FORCE)
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
set(LIB_TYPE STATIC)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-use_fast_math)

cuda_add_executable(
        ${PROJECT_NAME}

        src/pointcloud_preprocessing_cuda/cuda_prep_pipeline.cpp
        src/pointcloud_preprocessing_cuda/main.cpp
        src/pointcloud_preprocessing_cuda/point_ops.cpp
        src/pointcloud_preprocessing_cuda/kernels/crop_box.cu
        src/pointcloud_preprocessing_cuda/kernels/transform.cu
        src/pointcloud_preprocessing_cuda/kernels/common.cu
        src/pointcloud_preprocessing_cuda/kernels/motion_undistort.cu
        src/pointcloud_preprocessing_cuda/kernels/concatenate.cu
        src/pointcloud_preprocessing_cuda/ros/pointcloud_msg.cu
)

target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        OpenMP::OpenMP_CXX
)

target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${catkin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
