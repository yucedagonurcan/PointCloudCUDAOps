//
// Created by yucedagonurcan on 09.03.2024.
//

#ifndef CU_POINTCLOUD_OPS_WRAPPER_H_
#define CU_POINTCLOUD_OPS_WRAPPER_H_
#include "pointcloud_preprocessing_cuda/ros/pointcloud_msg.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

cudaError_t cudaConcatenatePointClouds( const std::vector<ppc::ros::OperatedCloud> &in_pc_vec,
                                        sensor_msgs::PointCloud2& h_point_cloud_out, cudaStream_t stream  );

#endif// CU_POINTCLOUD_OPS_WRAPPER_H_
