//
// Created by yucedagonurcan on 09.03.2024.
//

#ifndef CUDA_LEARN_INCLUDE_LESSON_0_CUDAWRAPPER_H_
#define CUDA_LEARN_INCLUDE_LESSON_0_CUDAWRAPPER_H_

#include "cu_point_ops_wrapper.h"
#include "cu_pointcloud_ops_wrapper.h"
#include "pointcloud_preprocessing_cuda/ros/pointcloud_msg.hpp"
#include <optional>
#include <pcl/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>



class CuPointOps {
 public:
  CuPointOps();
  ~CuPointOps();

  void setPointCloud( const sensor_msgs::PointCloud2ConstPtr &point_cloud, const std::string &timestamp_field_name = "t");
  void setPointTimestampOffset( float point_timestamp_offset );

  void warmUpGPU();
  void reset();

  //  bool rotate(double degree);
  cudaError_t transform( const Eigen::Affine3f &matrix );
  cudaError_t cropPoints( CropBox crop_box, bool negative = false );
  //  bool filterRadius( float radius, bool negative=false );
  cudaError_t motionUndistort( const Eigen::Isometry3f& consecutive_transform_inv);

  ppc::ros::OperatedCloud getDevicePointCloud();
  size_t getDevicePointCloudSize() const;

  sensor_msgs::PointCloud2 getPointCloud( bool filter_invalid = false );
  void setStream( cudaStream_t& stream );

 private:
  PointT *d_point_cloud_;
  uint32_t *d_point_cloud_t_;

  std::vector<sensor_msgs::PointField> fields_;

  float point_timestamp_offset_{0.0};

  cudaStream_t stream_;
  size_t cur_pointcloud_size_;
  ppc::ros::CUDAConverter converter;
};

#endif// CUDA_LEARN_INCLUDE_LESSON_0_CUDAWRAPPER_H_
