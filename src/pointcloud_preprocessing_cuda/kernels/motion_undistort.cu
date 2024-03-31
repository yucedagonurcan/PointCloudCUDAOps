#pragma once
#include "common.cuh"

__device__ void rotMatFromAxisAngle( float *axis, float angle, float *rotation_matrix ) {

  const float ca = cosf( angle );
  const float sa = sinf( angle );
  const float C = 1.0f - ca;

  const float x = axis[0];
  const float y = axis[1];
  const float z = axis[2];

  const float xs = x * sa;
  const float ys = y * sa;
  const float zs = z * sa;

  const float xC = x * C;
  const float yC = y * C;
  const float zC = z * C;

  const float xyC = x * yC;
  const float yzC = y * zC;
  const float zxC = z * xC;

  rotation_matrix[0] = x * xC + ca;
  rotation_matrix[1] = xyC - zs;
  rotation_matrix[2] = zxC + ys;
  rotation_matrix[3] = xyC + zs;
  rotation_matrix[4] = y * yC + ca;
  rotation_matrix[5] = yzC - xs;
  rotation_matrix[6] = zxC - ys;
  rotation_matrix[7] = yzC + xs;
  rotation_matrix[8] = z * zC + ca;
}

__global__ void k_cudaMotionUndistort( PointT *d_point_cloud, uint32_t *d_point_cloud_t, size_t *d_number_of_points,
                                       uint32_t *d_max_point_timestamp, float *d_point_timestamp_offset,
                                       float *d_full_rotation_angle_axis, float *d_full_rotation_angle,
                                       float *d_full_translation_vector ) {

  int ind = blockIdx.x * blockDim.x + threadIdx.x;

  if ( ind < *d_number_of_points ) {

    const Eigen::Vector3f point( d_point_cloud[ind].x, d_point_cloud[ind].y, d_point_cloud[ind].z );
    const uint32_t &point_t = d_point_cloud_t[ind];

    if ( point.x() == 0 && point.y() == 0 && point.z() == 0 ) { return; }

    // Normalized timestamp of the point inside the pointcloud, used for interpolation
    const auto &point_normalized_timepoint =
        static_cast<double>( point_t ) / static_cast<double>( *d_max_point_timestamp );

    // Timestamp difference between the point and the phase lock degree time, used for interpolation
    const auto &point_transform_scalar = *d_point_timestamp_offset - point_normalized_timepoint;

    float point_scaled_rot_mat[9];
    rotMatFromAxisAngle( d_full_rotation_angle_axis, point_transform_scalar * (*d_full_rotation_angle),
                         point_scaled_rot_mat );

    float point_scaled_translation[3];
    for ( int i = 0; i < 3; i++ ) {
      point_scaled_translation[i] = point_transform_scalar * d_full_translation_vector[i];
    }

    d_point_cloud[ind].x = point_scaled_rot_mat[0] * point.x() + point_scaled_rot_mat[1] * point.y()
        + point_scaled_rot_mat[2] * point.z() + point_scaled_translation[0];
    d_point_cloud[ind].y = point_scaled_rot_mat[3] * point.x() + point_scaled_rot_mat[4] * point.y()
        + point_scaled_rot_mat[5] * point.z() + point_scaled_translation[1];
    d_point_cloud[ind].z = point_scaled_rot_mat[6] * point.x() + point_scaled_rot_mat[7] * point.y()
        + point_scaled_rot_mat[8] * point.z() + point_scaled_translation[2];
  }
}

cudaError_t cudaMotionUndistort( PointT *d_point_cloud, uint32_t *d_point_cloud_t, size_t h_number_of_points,
                                 float point_timestamp_offset, Eigen::Vector3f h_full_rotation_axis,
                                 float h_full_rotation_angle, Eigen::Vector3f h_full_translation_vector,
                                 cudaStream_t stream ) {

  int blk = static_cast<int>(floor( h_number_of_points / THREADS_PER_BLOCK )) + 1;
  int thr = THREADS_PER_BLOCK;

  // Get maximum timestamp
  thrust::device_ptr<uint32_t> dev_t_ptr = thrust::device_pointer_cast( d_point_cloud_t );
  uint32_t d_max_t = *thrust::max_element( thrust::cuda::par.on( stream ), dev_t_ptr, dev_t_ptr + static_cast<long>(h_number_of_points) );

  uint32_t *d_max_point_timestamp;
  ppc::cuda::cudaSafeCall( cudaMallocAsync( (void **)&d_max_point_timestamp, sizeof( uint32_t ), stream ) );
  ppc::cuda::cudaSafeCall(
      cudaMemcpyAsync( d_max_point_timestamp, &d_max_t, sizeof( uint32_t ), cudaMemcpyHostToDevice, stream ) );

  float *d_full_translation_vector;
  ppc::cuda::cudaSafeCall( cudaMallocAsync( (void **)&d_full_translation_vector, 3 * sizeof( float ), stream ) );
  ppc::cuda::cudaSafeCall( cudaMemcpyAsync( d_full_translation_vector, h_full_translation_vector.data(),
                                            3 * sizeof( float ), cudaMemcpyHostToDevice, stream ) );
  float *d_full_rotation_angle_axis;
  ppc::cuda::cudaSafeCall( cudaMallocAsync( (void **)&d_full_rotation_angle_axis, 3 * sizeof( float ), stream ) );
  ppc::cuda::cudaSafeCall( cudaMemcpyAsync( d_full_rotation_angle_axis, h_full_rotation_axis.data(),
                                            3 * sizeof( float ), cudaMemcpyHostToDevice, stream ) );
  float *d_full_rotation_angle;
  ppc::cuda::cudaSafeCall( cudaMallocAsync( (void **)&d_full_rotation_angle, sizeof( float ), stream ) );
  ppc::cuda::cudaSafeCall( cudaMemcpyAsync( d_full_rotation_angle, &h_full_rotation_angle, sizeof( float ),
                                            cudaMemcpyHostToDevice, stream ) );
  size_t *d_number_of_points;
  ppc::cuda::cudaSafeCall( cudaMallocAsync( (void **)&d_number_of_points, sizeof( size_t ), stream ) );
  ppc::cuda::cudaSafeCall(
      cudaMemcpyAsync( d_number_of_points, &h_number_of_points, sizeof( size_t ), cudaMemcpyHostToDevice, stream ) );

  float *d_point_timestamp_offset;
  ppc::cuda::cudaSafeCall( cudaMallocAsync( (void **)&d_point_timestamp_offset, sizeof( float ), stream ) );
  ppc::cuda::cudaSafeCall( cudaMemcpyAsync( d_point_timestamp_offset, &point_timestamp_offset, sizeof( float ),
                                            cudaMemcpyHostToDevice, stream ) );

  k_cudaMotionUndistort<<<blk, thr, 0, stream>>>(
      d_point_cloud, d_point_cloud_t, d_number_of_points, d_max_point_timestamp, d_point_timestamp_offset,
      d_full_rotation_angle_axis, d_full_rotation_angle, d_full_translation_vector );

  ppc::cuda::cudaSafeCall( cudaFreeAsync( d_full_translation_vector, stream ) );
  ppc::cuda::cudaSafeCall( cudaFreeAsync( d_full_rotation_angle_axis, stream ) );
  ppc::cuda::cudaSafeCall( cudaFreeAsync( d_full_rotation_angle, stream ) );
  ppc::cuda::cudaSafeCall( cudaFreeAsync( d_number_of_points, stream ) );
  ppc::cuda::cudaSafeCall( cudaFreeAsync( d_max_point_timestamp, stream ) );
  ppc::cuda::cudaSafeCall( cudaFreeAsync( d_point_timestamp_offset, stream ) );

  return cudaGetLastError();
}