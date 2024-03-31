#pragma once

#include <Eigen/Core>
#include <sensor_msgs/PointCloud2.h>
#include <thrust/device_vector.h>
#include <utility>

namespace ppc {
namespace ros {

struct OperatedCloud {
  PointT *devicePtr;
  size_t numPoints;
  std::vector<sensor_msgs::PointField> fields;
  OperatedCloud( PointT *devicePtr, size_t numPoints, std::vector<sensor_msgs::PointField> fields )
      : devicePtr( devicePtr ), numPoints( numPoints ), fields( std::move( fields ) ) {}
};

class CUDAConverter {
 public:
  CUDAConverter(){};
  cudaError convertFromPointCloudMsg( const sensor_msgs::PointCloud2ConstPtr &msg, PointT *d_point_cloud,
                                      uint32_t *d_point_cloud_t, std::string timestamp_field_name,
                                      cudaStream_t stream );
  static cudaError converToPointCloudMsg( const OperatedCloud &in_pc, sensor_msgs::PointCloud2 &out_pc, cudaStream_t stream );
};

}// namespace ros
}// namespace ppc