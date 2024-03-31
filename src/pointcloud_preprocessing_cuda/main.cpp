//
// Created by yucedagonurcan on 09.03.2024.
//
#include <iostream>
#include <ros/ros.h>
#include "pointcloud_preprocessing_cuda/cuda_prep_pipeline.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "PCPrepCUDA");

  auto pc_transformer_cuda = PCPrepCUDA();

  ros::spin();
  return 0;
}