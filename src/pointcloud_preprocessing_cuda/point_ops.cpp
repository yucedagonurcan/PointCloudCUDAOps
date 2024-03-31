//
// Created by yucedagonurcan on 09.03.2024.
//
#include "pointcloud_preprocessing_cuda/points_ops.h"
#include <sensor_msgs/PointCloud2.h>

CuPointOps::CuPointOps() {
  cudaError_t err = ::cudaSuccess;

  ROS_INFO( "Setting CUDA device to 0." );
  err = cudaSetDevice( 0 );
  if ( err != ::cudaSuccess ) { std::cerr << "Error: " << cudaGetErrorString( err ) << std::endl; }

  warmUpGPU();
}

CuPointOps::~CuPointOps() {
  cudaError_t err = ::cudaSuccess;
  err = cudaDeviceReset();
  if ( err != ::cudaSuccess ) { std::cerr << "Error: " << cudaGetErrorString( err ) << std::endl; }
}

void CuPointOps::warmUpGPU() {
  cudaError_t err = ::cudaSuccess;
  err = cudaSetDevice( 0 );
  if ( err != ::cudaSuccess ) { std::cerr << "Error: " << cudaGetErrorString( err ) << std::endl; }

  err = cudaWarmUpGPU();
  if ( err != ::cudaSuccess ) { std::cerr << "Error: " << cudaGetErrorString( err ) << std::endl; }
}

void CuPointOps::setStream( cudaStream_t &stream ) { stream_ = stream; }

void CuPointOps::setPointCloud( const sensor_msgs::PointCloud2ConstPtr &point_cloud,
                                const std::string &timestamp_field_name ) {
  cudaError_t err = ::cudaSuccess;

  cur_pointcloud_size_ = cur_pointcloud_size_ = point_cloud->width * point_cloud->height;

  fields_.resize(3);
  for (size_t i = 0; i < fields_.size(); ++i) {
    fields_[i] = point_cloud->fields[i];
  }

  err = cudaMallocAsync( (void **)&d_point_cloud_, cur_pointcloud_size_ * sizeof( PointT ), stream_ );
  if ( err != ::cudaSuccess ) { std::cerr << "Error: " << cudaGetErrorString( err ) << std::endl; }

  err = cudaMallocAsync( (void **)&d_point_cloud_t_, cur_pointcloud_size_ * sizeof( uint32_t ), stream_ );
  if ( err != ::cudaSuccess ) { std::cerr << "Error: " << cudaGetErrorString( err ) << std::endl; }

  converter.convertFromPointCloudMsg( point_cloud, d_point_cloud_, d_point_cloud_t_, timestamp_field_name, stream_ );
}

void CuPointOps::setPointTimestampOffset( const float point_timestamp_offset ) {
  point_timestamp_offset_ = point_timestamp_offset;
}

cudaError_t CuPointOps::cropPoints( CropBox crop_box, bool negative ) {

  cudaError_t err = ::cudaSuccess;
  err = cudaCropPoints( d_point_cloud_, cur_pointcloud_size_, crop_box, negative, stream_ );
  return err;
}
cudaError_t CuPointOps::motionUndistort( const Eigen::Isometry3f& consecutive_transform_inv ) {
  cudaError_t err = ::cudaSuccess;

  // Get rotation angle axis & translation vector of the full transform
  const Eigen::AngleAxisf full_rotation_angle_axis( consecutive_transform_inv.linear() );
  const auto &full_rotation_angle = full_rotation_angle_axis.angle();
  const auto &full_rotation_axis = full_rotation_angle_axis.axis();
  const Eigen::Vector3f full_translation_vector = consecutive_transform_inv.translation();

  err = cudaMotionUndistort( d_point_cloud_, d_point_cloud_t_, cur_pointcloud_size_, point_timestamp_offset_,
                             full_rotation_axis, full_rotation_angle, full_translation_vector, stream_ );
  return err;
}

cudaError_t CuPointOps::transform( const Eigen::Affine3f &matrix ) {
  cudaError_t err = ::cudaSuccess;
  err = cudaTransformPoints( d_point_cloud_, cur_pointcloud_size_, matrix, stream_ );
  return err;
}

ppc::ros::OperatedCloud CuPointOps::getDevicePointCloud() { return {d_point_cloud_, cur_pointcloud_size_, fields_}; }

size_t CuPointOps::getDevicePointCloudSize() const { return cur_pointcloud_size_; }

sensor_msgs::PointCloud2 CuPointOps::getPointCloud( bool filter_invalid ) {
  cudaError_t err = ::cudaSuccess;

  sensor_msgs::PointCloud2 point_cloud;
  err = converter.converToPointCloudMsg( getDevicePointCloud(), point_cloud, stream_ );
  if ( err != ::cudaSuccess ) {
    std::cerr << "Error: " << cudaGetErrorString( err ) << std::endl;
    return point_cloud;
  }

  return point_cloud;
}

void CuPointOps::reset() {
  cudaStreamSynchronize( stream_ );
  cudaFreeAsync( d_point_cloud_, stream_ );
  cudaFreeAsync( d_point_cloud_t_, stream_ );
  cur_pointcloud_size_ = 0;
}