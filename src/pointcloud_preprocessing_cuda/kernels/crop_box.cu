#pragma once
#include "common.cuh"

__global__ void k_cudaCropPoints( PointT *d_point_cloud, size_t number_of_points, const CropBox crop_box, bool negative ) {
  const auto ind = blockIdx.x * blockDim.x + threadIdx.x;
  if ( ind < number_of_points ) {
    if ( ( negative
           && ( d_point_cloud[ind].x > crop_box.min_x && d_point_cloud[ind].x < crop_box.max_x
                && d_point_cloud[ind].y > crop_box.min_y && d_point_cloud[ind].y < crop_box.max_y
                && d_point_cloud[ind].z > crop_box.min_z && d_point_cloud[ind].z < crop_box.max_z ) )
         || ( !negative
              && ( d_point_cloud[ind].x < crop_box.min_x || d_point_cloud[ind].x > crop_box.max_x
                   || d_point_cloud[ind].y < crop_box.min_y || d_point_cloud[ind].y > crop_box.max_y
                   || d_point_cloud[ind].z < crop_box.min_z || d_point_cloud[ind].z > crop_box.max_z ) ) ) {
      kk_setCoordinatesToZero( &d_point_cloud[ind] );
    }
  }
}

cudaError_t cudaCropPoints( PointT *d_point_cloud, size_t number_of_points, CropBox crop_box, bool negative, cudaStream_t stream){

  int blk = static_cast<int>(floor( number_of_points / THREADS_PER_BLOCK )) + 1;
  int thr = THREADS_PER_BLOCK;

  k_cudaCropPoints<<<blk, thr, 0, stream>>>( d_point_cloud, number_of_points, crop_box, negative );

  return cudaGetLastError();

}
