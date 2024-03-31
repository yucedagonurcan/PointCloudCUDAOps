//
// Created by yucedagonurcan on 09.03.2024.
//
#include "pointcloud_preprocessing_cuda/cuda_prep_pipeline.hpp"
#include "pointcloud_preprocessing_cuda/points_ops.h"
#include "pointcloud_preprocessing_cuda/ros/pointcloud_msg.hpp"
#include <fmt/format.h>
#include <omp.h>
#include <ouster/types.h>
#include <rosfmt/rosfmt.h>
#include <swri_profiler/profiler.h>

PCPrepCUDA::PCPrepCUDA() : nh_( "" ), pnh_( "~" ), tf_listener_( tf_buffer_ ) {

  pnh_.param( "crop_box/min_x", crop_box_params_.crop_box.min_x, -2.0f );
  pnh_.param( "crop_box/max_x", crop_box_params_.crop_box.max_x, 7.0f );
  pnh_.param( "crop_box/min_y", crop_box_params_.crop_box.min_y, -3.0f );
  pnh_.param( "crop_box/max_y", crop_box_params_.crop_box.max_y, 3.0f );
  pnh_.param( "crop_box/min_z", crop_box_params_.crop_box.min_z, -400.0f );
  pnh_.param( "crop_box/max_z", crop_box_params_.crop_box.max_z, 500.0f );
  pnh_.param( "crop_box/negative", crop_box_params_.negative, true );

  pnh_.param( "odom_frame", odom_frame_, std::string( "map" ) );
  pnh_.param( "base_frame", base_frame_, std::string( "base_link" ) );

  pnh_.param( "motion_undistortion/timestamp_field_name", motion_undistortion_params_.timestamp_field_name,
              std::string( "t" ) );
  pnh_.param( "motion_undistortion/velocity_threshold", motion_undistortion_params_.velocity_threshold, 0.0f );

  if ( !pnh_.getParam( "input/lidars", lidars_ ) ) {
    ROS_ERROR( "Error: Could not get input topics" );
    return;
  }

  if ( !pnh_.getParam( "input/metadata_topics", input_metadata_topics_ ) ) {
    ROS_ERROR( "Error: Could not get input metadata topics" );
    return;
  }

  fmt::print( "Input LiDARs: \n" );
  for ( const auto &lidar_name : lidars_ ) {
    input_topics_.push_back( fmt::format( "/lidar/{}/points_raw", lidar_name ) );
    input_metadata_topics_.push_back( fmt::format( "/lidar/{}/metadata", lidar_name ) );
    fmt::print( "\t> {}\n", lidar_name );
  }

  point_ops_list_.resize( input_topics_.size() );
  point_timestamp_offsets_.resize( input_topics_.size() );
  pc_subscriber_list_.resize( input_topics_.size() );
  metadata_subscriber_list_.resize( input_topics_.size() );
  undistorted_pc_pub_list_.resize( input_topics_.size() );
  stream_list_.resize( input_topics_.size() );
  last_odom2cloud_list_.resize( input_topics_.size() );
  spinning_rate_list_.resize( input_topics_.size() );

  subscribe();
}

void PCPrepCUDA::subscribe() {

  final_pub_ = nh_.advertise<sensor_msgs::PointCloud2>( "out/cloud", 1 );

  for ( int i = 0; i < input_topics_.size(); i++ ) {
    pc_subscriber_list_[i] = std::make_shared<PCSub>( nh_, input_topics_.at( i ), 1 );
    metadata_subscriber_list_[i] = nh_.subscribe<std_msgs::String>(
        input_metadata_topics_.at( i ), 1, boost::bind( &PCPrepCUDA::onMetadata, this, _1, i ) );

    cudaStreamCreate( &stream_list_[i] );
    point_ops_list_[i].setStream( stream_list_[i] );

    undistorted_pc_pub_list_[i] = nh_.advertise<sensor_msgs::PointCloud2>( input_topics_.at( i ) + "/undistorted", 1 );
  }

  SyncPolicyT sync_policy( maximum_queue_size_ );
  clouds_synchronizer = std::make_shared<SyncT>(
      static_cast<const SyncPolicyT &>( sync_policy ), *pc_subscriber_list_.at( 0 ), *pc_subscriber_list_.at( 1 ),
      *pc_subscriber_list_.at( 2 ), *pc_subscriber_list_.at( 3 ), *pc_subscriber_list_.at( 4 ) );
  clouds_synchronizer->registerCallback( boost::bind( &PCPrepCUDA::syncCallbackClouds, this, _1, _2, _3, _4, _5 ) );
}

void PCPrepCUDA::checkTimeStampField( const sensor_msgs::PointCloud2ConstPtr &cloud1,
                                      const sensor_msgs::PointCloud2ConstPtr &cloud2,
                                      const sensor_msgs::PointCloud2ConstPtr &cloud3,
                                      const sensor_msgs::PointCloud2ConstPtr &cloud4,
                                      const sensor_msgs::PointCloud2ConstPtr &cloud5 ) {

  auto msg_vec = { cloud1, cloud2, cloud3, cloud4, cloud5 };

  for ( auto &msg : msg_vec ) {

    const auto &fields = msg->fields;
    auto findFieldOffset = [fields]( const std::string &ref ) -> unsigned int {
      for ( auto &field : fields ) {
        if ( field.name == ref ) { return field.offset; }
      }
      return -1;
    };

    const auto timestamp_offset = findFieldOffset( motion_undistortion_params_.timestamp_field_name );
    if ( timestamp_offset == -1 ) {
      fmt::print( "Error: Could not find timestamp field ({}) in the point cloud message!\n",
                  motion_undistortion_params_.timestamp_field_name );
      return;
    }
  }
  timestamp_field_exists_ = true;
}

void PCPrepCUDA::syncCallbackClouds( const sensor_msgs::PointCloud2ConstPtr &cloud1,
                                     const sensor_msgs::PointCloud2ConstPtr &cloud2,
                                     const sensor_msgs::PointCloud2ConstPtr &cloud3,
                                     const sensor_msgs::PointCloud2ConstPtr &cloud4,
                                     const sensor_msgs::PointCloud2ConstPtr &cloud5 ) {

  SWRI_PROFILE( "syncCallbackClouds" );

  if ( !metadata_acquired_ ) {
    ROSFMT_ERROR( "One or many metadata is not acquired yet!\n" );
    return;
  }

  if ( !timestamp_field_exists_ ) {
    checkTimeStampField( cloud1, cloud2, cloud3, cloud4, cloud5 );

    xyz_fields_.resize( 3 );
    for ( size_t i = 0; i < xyz_fields_.size(); ++i ) { xyz_fields_[i] = cloud1->fields[i]; }
  }

  // Reset point ops
  for ( int i = 0; i < lidars_.size(); i++ ) { point_ops_list_.at( i ).reset(); }

  const std::vector<sensor_msgs::PointCloud2ConstPtr> msg_vec = { cloud1, cloud2, cloud3, cloud4, cloud5 };

  std::vector<Eigen::Isometry3f> consecutive_transform_inv_list;
  consecutive_transform_inv_list.resize( msg_vec.size() );

  std::vector<sensor_msgs::PointCloud2ConstPtr> out_msg_vec;
  out_msg_vec.resize( msg_vec.size() );

  std::vector<std::optional<Eigen::Isometry3d>> lidar2base_list;
  lidar2base_list.resize( msg_vec.size() );

  {
    SWRI_PROFILE( "Transforms" );

#pragma omp parallel for num_threads( msg_vec.size() ) shared( consecutive_transform_inv_list, lidar2base_list )
    for ( int i = 0; i < msg_vec.size(); i++ ) {

      const auto &cur_cloud_msg = msg_vec.at( i );
      const auto &base2map = getTransform( odom_frame_, cur_cloud_msg->header.frame_id, cur_cloud_msg->header.stamp );
      if ( !base2map ) {
        ROSFMT_ERROR( "[ERROR] [{}]: No Transform, {} to {}\n", lidars_[i], odom_frame_,
                      cur_cloud_msg->header.frame_id );
      } else {

        lidar2base_list[i] = getTransform( base_frame_, cur_cloud_msg->header.frame_id );
        consecutive_transform_inv_list[i] = ( base2map->inverse() * last_odom2cloud_list_[i] ).cast<float>();
        last_odom2cloud_list_[i] = *base2map;
      }
    }

    if ( std::any_of( lidar2base_list.begin(), lidar2base_list.end(),
                      []( const std::optional<Eigen::Isometry3d> &i ) { return !i; } ) ) {
      ROSFMT_ERROR( "[ERROR]: Could not get transform, for one or multiple LiDARs" );
      return;
    }
  }

  {
    SWRI_PROFILE( "PointOps" );

    for ( int i = 0; i < msg_vec.size(); i++ ) {
      SWRI_PROFILE( fmt::format( "SetPointCloud_{}", i ) );

      const auto &cur_cloud_msg = msg_vec.at( i );
      auto &cur_point_ops = point_ops_list_.at( i );

      cur_point_ops.setPointCloud( cur_cloud_msg, motion_undistortion_params_.timestamp_field_name );
      cudaStreamSynchronize( stream_list_.at( i ) );
    }

    // Motion undistort points
    const auto &cur_velocity =
        consecutive_transform_inv_list.front().translation().norm() / spinning_rate_list_.front();

    if ( cur_velocity > motion_undistortion_params_.velocity_threshold ) {

      for ( int i = 0; i < msg_vec.size(); i++ ) {
        SWRI_PROFILE( fmt::format( "MotionUndistort_{}", lidars_[i] ) );

        auto &cur_point_ops = point_ops_list_.at( i );
        const auto &cur_cloud_msg = msg_vec.at( i );
        const auto &cur_consecutive_transform_inv = consecutive_transform_inv_list.at( i );

        auto success = cur_point_ops.motionUndistort( cur_consecutive_transform_inv );
        if ( success != cudaSuccess ) {
          ROSFMT_ERROR( "[ERROR] [{}]: Could not motion undistort, %s", lidars_[i], cudaGetErrorString( success ) );
          out_msg_vec.at( i ) = cur_cloud_msg;
          continue;
        }
      }
    }
#ifdef DEBUGIT
    else {
      ROSFMT_WARN( "Velocity is below the threshold, {} < {}", cur_velocity,
                   motion_undistortion_params_.velocity_threshold );
    }
#endif

    // Transform points
    for ( int i = 0; i < msg_vec.size(); i++ ) {
      SWRI_PROFILE( fmt::format( "Transform_{}", i ) );
      const auto &cur_cloud_msg = msg_vec.at( i );
      const auto &cur_lidar2base = lidar2base_list.at( i );

      auto &cur_point_ops = point_ops_list_.at( i );

      auto success = cur_point_ops.transform( ( *cur_lidar2base ).cast<float>() );
      if ( success != cudaSuccess ) {
        ROSFMT_ERROR( "[ERROR] [{}]: Could not transform points, %s", lidars_[i], cudaGetErrorString( success ) );
        out_msg_vec.at( i ) = cur_cloud_msg;
        continue;
      }
    }

    // Crop points
    for ( int i = 0; i < msg_vec.size(); i++ ) {
      SWRI_PROFILE( fmt::format( "Crop_{}", lidars_[i] ) );
      const auto &cur_cloud_msg = msg_vec.at( i );
      auto &cur_point_ops = point_ops_list_.at( i );

      auto success = cur_point_ops.cropPoints( crop_box_params_.crop_box, crop_box_params_.negative );
      if ( success != cudaSuccess ) {
        ROSFMT_ERROR( "[ERROR] [{}]: Could not crop points, %s", lidars_[i], cudaGetErrorString( success ) );
        out_msg_vec.at( i ) = cur_cloud_msg;
        continue;
      }
    }

    for ( int i = 0; i < msg_vec.size(); i++ ) {
      SWRI_PROFILE( fmt::format( "GetOutput_{}", lidars_[i] ) );

      auto &cur_point_ops = point_ops_list_.at( i );
      const auto &cur_cloud_msg = msg_vec.at( i );

      sensor_msgs::PointCloud2 final_pc;
      {
        SWRI_PROFILE( fmt::format( "GetPointCloud_{}", lidars_[i] ) );
        final_pc = cur_point_ops.getPointCloud( true );
      }

      {
        SWRI_PROFILE( fmt::format( "StoreOutput_{}", lidars_[i] ) );
        final_pc.header.stamp = cur_cloud_msg->header.stamp;
        final_pc.header.frame_id = base_frame_;

        out_msg_vec.at( i ) = boost::make_shared<sensor_msgs::PointCloud2>( final_pc );
      }
      cudaStreamSynchronize( stream_list_.at( i ) );
    }

    {
      SWRI_PROFILE( "Publish" );

#pragma omp parallel for num_threads( msg_vec.size() )                                                                 \
    shared( undistorted_pc_pub_list_, msg_vec, out_msg_vec ) default( none )
      for ( int i = 0; i < msg_vec.size(); i++ ) { undistorted_pc_pub_list_.at( i ).publish( out_msg_vec.at( i ) ); }
    }

    {
      SWRI_PROFILE( "Concatenate" );

      sensor_msgs::PointCloud2 final_cloud;

      std::vector<ppc::ros::OperatedCloud> operated_clouds;
      operated_clouds.reserve( msg_vec.size() );

      for ( int i = 0; i < msg_vec.size(); i++ ) {
        operated_clouds.push_back( point_ops_list_.at( i ).getDevicePointCloud() );
      }

      auto success = cudaConcatenatePointClouds( operated_clouds, final_cloud, stream_list_.at( 0 ) );
      if ( success != cudaSuccess ) {
        ROSFMT_ERROR( "[ERROR]: Could not concatenate point clouds, %s", cudaGetErrorString( success ) );
        return;
      }

      final_cloud.header.stamp = cloud1->header.stamp;
      final_cloud.header.frame_id = base_frame_;
      final_pub_.publish( final_cloud );
    }
  }
}

PCPrepCUDA::~PCPrepCUDA() { ROS_INFO( "PCPrepCUDA is shutting down" ); }

void PCPrepCUDA::onMetadata( const std_msgs::String::ConstPtr &msg, const int pc_index ) {

  const auto sensor_info = ouster::sensor::parse_metadata( msg->data );
  if ( sensor_info.config.phase_lock_enable ) {
    const auto lidar_frequency = static_cast<float>( frequency_of_lidar_mode( *sensor_info.config.ld_mode ) );

    const auto full_rotation_time = 1.0f / lidar_frequency;
    spinning_rate_list_.at( pc_index ) = full_rotation_time;

    const auto time_per_degree_increment = full_rotation_time / 360.0f;
    const auto phase_lock_offset_degree = static_cast<float>( *sensor_info.config.phase_lock_offset ) / 1000.0f;
    const auto degree_travelled_for_phase_locking = 360.0f - phase_lock_offset_degree;

    const auto delta_time_due_phase_lock = time_per_degree_increment * degree_travelled_for_phase_locking;
    point_timestamp_offsets_.at( pc_index ) = delta_time_due_phase_lock * lidar_frequency;
    point_ops_list_.at( pc_index ).setPointTimestampOffset( point_timestamp_offsets_.at( pc_index ) );

    if ( std::all_of( point_timestamp_offsets_.begin(), point_timestamp_offsets_.end(),
                      []( float i ) { return i != 0.0f; } ) ) {
      metadata_acquired_ = true;
    }

  } else {
    ROSFMT_ERROR( "[ERROR] [{}]: Phase lock is not enabled!", lidars_[pc_index] );
  }

  // Shutdown
  metadata_subscriber_list_[pc_index].shutdown();
}

std::optional<Eigen::Isometry3d> PCPrepCUDA::getTransform( const std::string &target_frame,
                                                           const std::string &source_frame, const ros::Time &time ) {
  geometry_msgs::TransformStamped transform;
  try {
    transform = tf_buffer_.lookupTransform( target_frame, source_frame, time, ros::Duration( 0.05f ) );
  } catch ( tf2::TransformException &ex ) {
    ROS_WARN( "%s", ex.what() );
    return std::nullopt;
  }

  Eigen::Isometry3d eigen_transform;
  eigen_transform = tf2::transformToEigen( transform.transform );
  return eigen_transform;
}