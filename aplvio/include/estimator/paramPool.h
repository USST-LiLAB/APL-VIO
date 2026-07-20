#ifndef __PARAM_POOL_H
#define __PARAM_POOL_H

#include "estimator/customType.h"


struct ParamPool
{
    void setParameters();

    std::string config_path;
    std::string output_path;
    
    int imu_num;
    std::string imu_topic;
    double acc_n, acc_w, gyr_n, gyr_w;
    std::shared_ptr<IntegrationBase> pre_integration;

    int cam_num;
    std::string cam0_topic, cam1_topic;
    std::vector<camodocal::CameraPtr> m_camera;
    cv::Mat fisheye_mask;
    double pixel_n;
    int image_width, image_height;
    double focal_length, init_depth;
    double keyframe_parallax, min_parallax;
    int max_feature_count, min_feature_distance;
    int min_line_feature_count;
    size_t reliable_track_count;
    size_t reliable_line_track_count;
    int down_sampling_raw_image, down_sampling_track_image;


    Extrinsics extrinsics;

    double time_delay;


    bool is_use_imu, is_use_stereo;
    bool is_fisheye;
    bool is_flow_back, is_lineflow_back, is_show_track, is_equalize;
    bool is_estimate_extrinsic, is_estimate_timedelay;


    double max_solver_time, max_num_iterations;

    
    SolverFlag solver_flag;
    MarginalizationFlag marginalization_flag;
    
    std::vector<Eigen::Vector3d> key_poses;

    BodyState latest_state;
    std::mutex propagate_mutex;
};

#endif
