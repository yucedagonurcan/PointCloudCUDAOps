#pragma once
#include "common.cuh"
#include "pointcloud_preprocessing_cuda/ros/pointcloud_msg.hpp"
#include <numeric>

cudaError_t cudaConcatenatePointClouds( const std::vector<ppc::ros::OperatedCloud> &in_pc_vec,
                                        sensor_msgs::PointCloud2 &h_point_cloud_out, cudaStream_t stream ) {
  size_t number_of_points =
      std::accumulate( in_pc_vec.begin(), in_pc_vec.end(), 0,
                       []( size_t sum, const ppc::ros::OperatedCloud &oc ) { return sum + oc.numPoints; } );

  thrust::device_vector<PointT> d_point_cloud_out( number_of_points );

  size_t offset = 0;
  for ( const auto &pointCloud : in_pc_vec ) {
    thrust::device_ptr<PointT> dev_ptr = thrust::device_pointer_cast( pointCloud.devicePtr );

    thrust::copy( thrust::cuda::par.on( stream ), dev_ptr, dev_ptr + static_cast<long>( pointCloud.numPoints ),
                  d_point_cloud_out.begin() + static_cast<long>( offset ) );
    offset += pointCloud.numPoints;
  }

  h_point_cloud_out.data.resize( number_of_points * sizeof( PointT ) );

  ppc::ros::OperatedCloud out_pc( thrust::raw_pointer_cast( d_point_cloud_out.data() ), number_of_points,
                                  in_pc_vec.front().fields );
  ppc::ros::CUDAConverter::converToPointCloudMsg( out_pc, h_point_cloud_out, stream );
  return cudaGetLastError();
}