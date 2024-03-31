#pragma once
#include "common.cuh"

__global__ void k_cudaTransformPoints( PointT *d_point_cloud, const size_t *number_of_points, const float *d_matrix ) {

  int ind = blockIdx.x * blockDim.x + threadIdx.x;

  if ( ind < *number_of_points ) {
    float vSrcVector[3] = { d_point_cloud[ind].x, d_point_cloud[ind].y, d_point_cloud[ind].z };
    float vOut[3];
    vOut[0] = d_matrix[0] * vSrcVector[0] + d_matrix[4] * vSrcVector[1] + d_matrix[8] * vSrcVector[2] + d_matrix[12];
    vOut[1] = d_matrix[1] * vSrcVector[0] + d_matrix[5] * vSrcVector[1] + d_matrix[9] * vSrcVector[2] + d_matrix[13];
    vOut[2] = d_matrix[2] * vSrcVector[0] + d_matrix[6] * vSrcVector[1] + d_matrix[10] * vSrcVector[2] + d_matrix[14];

    d_point_cloud[ind].x = vOut[0];
    d_point_cloud[ind].y = vOut[1];
    d_point_cloud[ind].z = vOut[2];
  }
}

cudaError_t cudaTransformPoints( PointT *d_point_cloud, int number_of_points, const Eigen::Affine3f &h_matrix,
                                 cudaStream_t stream ) {

  int blk = floor( number_of_points / THREADS_PER_BLOCK ) + 1;
  int thr = THREADS_PER_BLOCK;

  float h_m[16];
  cudaError_t err = ::cudaSuccess;

  h_m[0] = h_matrix.matrix()( 0, 0 );
  h_m[1] = h_matrix.matrix()( 1, 0 );
  h_m[2] = h_matrix.matrix()( 2, 0 );
  h_m[3] = h_matrix.matrix()( 3, 0 );

  h_m[4] = h_matrix.matrix()( 0, 1 );
  h_m[5] = h_matrix.matrix()( 1, 1 );
  h_m[6] = h_matrix.matrix()( 2, 1 );
  h_m[7] = h_matrix.matrix()( 3, 1 );

  h_m[8] = h_matrix.matrix()( 0, 2 );
  h_m[9] = h_matrix.matrix()( 1, 2 );
  h_m[10] = h_matrix.matrix()( 2, 2 );
  h_m[11] = h_matrix.matrix()( 3, 2 );

  h_m[12] = h_matrix.matrix()( 0, 3 );
  h_m[13] = h_matrix.matrix()( 1, 3 );
  h_m[14] = h_matrix.matrix()( 2, 3 );
  h_m[15] = h_matrix.matrix()( 3, 3 );

  size_t *d_number_of_points;
  ppc::cuda::cudaSafeCall( cudaMallocAsync( (void **)&d_number_of_points, sizeof( size_t ), stream ) );
  ppc::cuda::cudaSafeCall(
      cudaMemcpyAsync( d_number_of_points, &number_of_points, sizeof( size_t ), cudaMemcpyHostToDevice, stream ) );


  float *d_m;
  ppc::cuda::cudaSafeCall( cudaMallocAsync( (void **)&d_m, 16 * sizeof( float ), stream ) );
  ppc::cuda::cudaSafeCall( cudaMemcpyAsync( d_m, h_m, 16 * sizeof( float ), cudaMemcpyHostToDevice, stream ) );


  k_cudaTransformPoints<<<blk, thr, 0, stream>>>( d_point_cloud, d_number_of_points, d_m );

  ppc::cuda::cudaSafeCall( cudaFreeAsync( d_m, stream ) );
  ppc::cuda::cudaSafeCall( cudaFreeAsync( d_number_of_points, stream ) );

  return cudaGetLastError();
}