#include "common.cuh"

cudaError_t cudaWarmUpGPU() {
  k_cudaWarmupGPU<<<1, 1>>>();
  cudaDeviceSynchronize();
  return cudaGetLastError();
}


namespace ppc {
namespace cuda {

void Error( const char *error_string, const char *file, const int line, const char *func ) {
  std::cout << "Error: " << error_string << "\t" << file << ":" << line << std::endl;
  exit( 0 );
}
}
}

