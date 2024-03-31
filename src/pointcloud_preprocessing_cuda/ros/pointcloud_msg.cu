#include "../kernels/common.cuh"
#include "pointcloud_preprocessing_cuda/ros/pointcloud_msg.hpp"
#include <pcl_conversions/pcl_conversions.h>

struct ConvertFromPointCloudMsg {

  ConvertFromPointCloudMsg( const uint8_t *data, PointT *output_xyz, uint32_t *output_timestamp,
                            uint32_t point_x_offset, uint32_t point_y_offset, uint32_t point_z_offset,
                            uint32_t point_timestamp_offset, uint32_t point_step )
      : data_( data ), output_xyz_( output_xyz ), output_timestamp_( output_timestamp ),
        point_x_offset_( point_x_offset ), point_y_offset_( point_y_offset ), point_z_offset_( point_z_offset ),
        point_timestamp_offset_( point_timestamp_offset ), point_step_( point_step ) {}

  __device__ PointT operator()( size_t idx ) const {

    const float x = *reinterpret_cast<const float *>( data_ + idx * point_step_ + point_x_offset_ );
    const float y = *reinterpret_cast<const float *>( data_ + idx * point_step_ + point_y_offset_ );
    const float z = *reinterpret_cast<const float *>( data_ + idx * point_step_ + point_z_offset_ );

    const uint32_t t = *reinterpret_cast<const uint32_t *>( data_ + idx * point_step_ + point_timestamp_offset_ );

    if ( isfinite( x ) && isfinite( y ) && isfinite( z ) ) {
      //      printf( "PointTimestampOffset: %d, PointStep: %d\n", point_timestamp_offset_, point_step_ );
      //      printf( "Point: %f, %f, %f, %d\n", x, y, z, t );
      output_xyz_[idx].x = x;
      output_xyz_[idx].y = y;
      output_xyz_[idx].z = z;
      output_timestamp_[idx] = t;

    } else {
      output_xyz_[idx].x = 0.0f;
      output_xyz_[idx].y = 0.0f;
      output_xyz_[idx].z = 0.0f;
      output_timestamp_[idx] = 0;
    }
  }

  const uint8_t *data_;
  PointT *output_xyz_;
  uint32_t *output_timestamp_;
  uint32_t point_x_offset_;
  uint32_t point_y_offset_;
  uint32_t point_z_offset_;
  uint32_t point_timestamp_offset_;
  uint32_t point_step_;
};

struct ConvertToPointCloudMsg {

  ConvertToPointCloudMsg( const PointT *input_xyz, uint8_t *output_xyz, uint32_t point_step )
      : input_xyz_( input_xyz ), output_xyz_( output_xyz ), point_step_( point_step ) {}

  __device__ void operator()( size_t idx ) const {

    PointT point = input_xyz_[idx];
    memcpy( output_xyz_ + idx * point_step_, point.data, 3 * sizeof( float ) );
  }

  const PointT *input_xyz_;
  uint32_t point_step_;
  uint8_t *output_xyz_;
};

namespace ppc {
namespace ros {

cudaError CUDAConverter::convertFromPointCloudMsg( const sensor_msgs::PointCloud2ConstPtr &msg, PointT *d_point_cloud,
                                                   uint32_t *d_point_cloud_t, std::string timestamp_field_name,
                                                   cudaStream_t stream ) {

  const auto fields = msg->fields;
  const auto &num_points = msg->width * msg->height;
  const auto &size = msg->row_step * msg->height;

  const auto &point_step = msg->point_step;
  const auto &row_step = msg->row_step;

  auto findFieldOffset = [this, fields]( const std::string &ref ) -> unsigned int {
    for ( auto &field : fields ) {
      if ( field.name == ref ) { return field.offset; }
    }
    return -1;
  };

  uint32_t point_x_offset = findFieldOffset( "x" );
  uint32_t point_y_offset = findFieldOffset( "y" );
  uint32_t point_z_offset = findFieldOffset( "z" );
  uint32_t point_timestamp_offset = findFieldOffset( timestamp_field_name );

  uint8_t *d_dev_data;
  ppc::cuda::cudaSafeCall( cudaMallocAsync( (void **)&d_dev_data, size, stream ) );
  ppc::cuda::cudaSafeCall( cudaMemcpyAsync( d_dev_data, msg->data.data(), size, cudaMemcpyHostToDevice, stream ) );

  ConvertFromPointCloudMsg convert( d_dev_data, d_point_cloud, d_point_cloud_t, point_x_offset, point_y_offset,
                                    point_z_offset, point_timestamp_offset, point_step );

  thrust::for_each( thrust::cuda::par.on( stream ), thrust::make_counting_iterator<uint32_t>( 0 ),
                    thrust::make_counting_iterator<uint32_t>( num_points ), convert );

  ppc::cuda::cudaSafeCall( cudaFreeAsync( d_dev_data, stream ) );
  return cudaGetLastError();
}

cudaError CUDAConverter::converToPointCloudMsg( const OperatedCloud &in_pc, sensor_msgs::PointCloud2 &out_pc,
                                                cudaStream_t stream ) {

  uint32_t point_step = sizeof( PointT );
  const auto &size = in_pc.numPoints * point_step;

  uint8_t *d_dev_data;
  ppc::cuda::cudaSafeCall( cudaMallocAsync( (void **)&d_dev_data, size, stream ) );
  ConvertToPointCloudMsg convert( in_pc.devicePtr, d_dev_data, point_step );
  thrust::for_each( thrust::cuda::par.on( stream ), thrust::make_counting_iterator<size_t>( 0 ),
                    thrust::make_counting_iterator<size_t>( in_pc.numPoints ), convert );

  out_pc.data.resize( size );
  ppc::cuda::cudaSafeCall( cudaMemcpyAsync( reinterpret_cast<uint8_t *>( out_pc.data.data() ), d_dev_data, size,
                                            cudaMemcpyDeviceToHost, stream ) );
  out_pc.width = in_pc.numPoints;
  out_pc.height = 1;
  out_pc.row_step = in_pc.numPoints * point_step;
  out_pc.is_bigendian = false;
  out_pc.is_dense = false;
  out_pc.point_step = point_step;
  out_pc.fields = in_pc.fields;

  ppc::cuda::cudaSafeCall( cudaFreeAsync( d_dev_data, stream ) );

  return cudaGetLastError();
}

}// namespace ros
}// namespace ppc