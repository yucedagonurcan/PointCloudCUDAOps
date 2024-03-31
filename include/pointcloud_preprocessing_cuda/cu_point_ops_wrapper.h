//
// Created by yucedagonurcan on 09.03.2024.
//

#ifndef CU_POINT_OPS_WRAPPER_H_
#define CU_POINT_OPS_WRAPPER_H_
#include <cuda.h>
#include <cuda_runtime.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

struct CropBox {
  float min_x;
  float max_x;
  float min_y;
  float max_y;
  float min_z;
  float max_z;
};

cudaError_t cudaWarmUpGPU();
cudaError_t cudaTransformPoints( PointT *d_point_cloud, int number_of_points, const Eigen::Affine3f &h_matrix, cudaStream_t stream );
cudaError_t cudaCropPoints( PointT *d_point_cloud, size_t number_of_points, CropBox crop_box, bool negative,
                            cudaStream_t stream );
cudaError_t cudaMotionUndistort( PointT *d_point_cloud, uint32_t *d_point_cloud_t, size_t h_number_of_points,
                                 float point_timestamp_offset, Eigen::Vector3f h_full_rotation_axis,
                                 float h_full_rotation_angle, Eigen::Vector3f h_full_translation_vector,
                                 cudaStream_t stream );
//cudaError_t cudaRadiusFilter( PointT *d_point_cloud, int number_of_points, float radius_th, bool negative = false );
//cudaError_t cudaFilterInvalidPoints( PointT *d_point_cloud, size_t point_cloud_size, bool *h_filtered_points_marker );

#endif// CU_POINT_OPS_WRAPPER_H_
