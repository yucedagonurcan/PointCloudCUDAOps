//
// Created by yucedagonurcan on 09.03.2024.
//

#ifndef POINTCLOUD_PREPROCESSING_INCLUDE_PC_TRANSFORMER_CUDA_HPP_
#define POINTCLOUD_PREPROCESSING_INCLUDE_PC_TRANSFORMER_CUDA_HPP_
#if ( defined __GNUC__ ) && ( __GNUC__ > 4 || __GNUC_MINOR__ >= 7 )
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128
#endif
#include "pointcloud_preprocessing_cuda/points_ops.h"
#include <optional>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32.h>
#include <std_msgs/String.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>

struct CropBoxParameters {
  CropBox crop_box;
  bool negative;
};

struct MotionUndistortionParameters {
  std::string timestamp_field_name;
  float velocity_threshold;
};

typedef message_filters::Subscriber<sensor_msgs::PointCloud2> PCSub;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2,
                                                        sensor_msgs::PointCloud2, sensor_msgs::PointCloud2,
                                                        sensor_msgs::PointCloud2>
    SyncPolicyT;
typedef message_filters::Synchronizer<SyncPolicyT> SyncT;

class PCPrepCUDA {
 public:
  PCPrepCUDA();
  ~PCPrepCUDA();

 private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Publisher final_pub_;
  std::vector<ros::Publisher> undistorted_pc_pub_list_;

  int maximum_queue_size_{ 10 };

  CropBoxParameters crop_box_params_;
  MotionUndistortionParameters motion_undistortion_params_;

  std::string odom_frame_;
  std::string base_frame_;


  std::vector<sensor_msgs::PointField> xyz_fields_;

  bool metadata_acquired_{ false };
  bool timestamp_field_exists_{ false };

  std::vector<std::string> lidars_;
  std::vector<std::string> input_topics_;
  std::vector<std::string> input_metadata_topics_;

  std::vector<std::shared_ptr<PCSub>> pc_subscriber_list_;
  std::vector<ros::Subscriber> metadata_subscriber_list_;


  std::vector<float> point_timestamp_offsets_;
  std::vector<cudaStream_t> stream_list_;
  std::vector<CuPointOps> point_ops_list_;
  std::vector<Eigen::Isometry3d> last_odom2cloud_list_;
  std::vector<float> spinning_rate_list_;

  void subscribe();
  void syncCallbackClouds( const sensor_msgs::PointCloud2ConstPtr &cloud1,
                           const sensor_msgs::PointCloud2ConstPtr &cloud2,
                           const sensor_msgs::PointCloud2ConstPtr &cloud3,
                           const sensor_msgs::PointCloud2ConstPtr &cloud4,
                           const sensor_msgs::PointCloud2ConstPtr &cloud5 );
  void checkTimeStampField( const sensor_msgs::PointCloud2ConstPtr &cloud1,
                            const sensor_msgs::PointCloud2ConstPtr &cloud2,
                            const sensor_msgs::PointCloud2ConstPtr &cloud3,
                            const sensor_msgs::PointCloud2ConstPtr &cloud4,
                            const sensor_msgs::PointCloud2ConstPtr &cloud5 );

//  void pointCloudCallback( const sensor_msgs::PointCloud2ConstPtr &msg );

  void onMetadata( const std_msgs::String::ConstPtr &msg, const int pc_index );
  std::optional<Eigen::Isometry3d> getTransform( const std::string &target_frame, const std::string &source_frame,
                                                 const ros::Time &time = ros::Time( 0 ) );

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::shared_ptr<SyncT> clouds_synchronizer;
};

#endif//POINTCLOUD_PREPROCESSING_INCLUDE_PC_TRANSFORMER_CUDA_HPP_
