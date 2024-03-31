#ifndef CUDA_LEARNING_MATERIALS_SRC_LESSON_0_KERNELS_COMMON_CUH_
#define CUDA_LEARNING_MATERIALS_SRC_LESSON_0_KERNELS_COMMON_CUH_
#include "pointcloud_preprocessing_cuda/cu_point_ops_wrapper.h"
#include "pointcloud_preprocessing_cuda/cu_pointcloud_ops_wrapper.h"
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#define THREADS_PER_BLOCK 1024

namespace ppc {
namespace cuda {

// ERROR HANDLING
void Error( const char *error_string, const char *file, const int line, const char *func );
static inline void ___cudaSafeCall(cudaError_t err,
                                    const char *file,
                                    const int line,
                                    const char *func = "") {
  if (cudaSuccess != err)
    Error(cudaGetErrorString(err), file, line, func);
}



#define __FILENAME__ ( strrchr( __FILE__, '/' ) ? strrchr( __FILE__, '/' ) + 1 : __FILE__ )

#if defined( __GNUC__ )
#define cudaSafeCall( expr ) ___cudaSafeCall( expr, __FILENAME__, __LINE__, __func__ )
#else /* defined(__CUDACC__) || defined(__MSVC__) */
#define cudaSafeCall( expr ) ___cudaSafeCall( expr, __FILENAME__, __LINE__ )
#endif
// END ERROR HANDLING

}
}

__global__ static void k_cudaWarmupGPU() {
  int ind = blockIdx.x * blockDim.x + threadIdx.x;
  ind = ind + 1;
}

__device__ static void kk_calculateRange( PointT *d_point, float *d_range ) {
  *d_range = sqrt( d_point->x * d_point->x + d_point->y * d_point->y + d_point->z * d_point->z );
}

__device__ static void kk_setCoordinatesToZero( PointT *d_point ) {
  d_point->x = 0;
  d_point->y = 0;
  d_point->z = 0;
}

struct compareX {
  __host__ __device__ bool operator()( const PointT &a, const PointT &b ) {
    return a.x < b.x;
  }
};

struct compareY {
  __host__ __device__ bool operator()( const PointT &a, const PointT &b ) {
    return a.y < b.y;
  }
};

struct compareZ {
  __host__ __device__ bool operator()( const PointT &a, const PointT &b ) {
    return a.z < b.z;
  }
};

struct compareT {
  __host__ __device__ bool operator()( const PointT &a, const PointT &b ) {
    return a.z < b.z;
  }
};

#endif//CUDA_LEARNING_MATERIALS_SRC_LESSON_0_KERNELS_COMMON_CUH_
