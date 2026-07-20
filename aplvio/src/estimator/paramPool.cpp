#include "factor/projectionOneFrameTwoCamFactor.h"
#include "factor/projectionTwoFrameOneCamFactor.h"
#include "factor/projectionTwoFrameTwoCamFactor.h"
#include "factor/projectionLineFactor.h"

#include "estimator/paramPool.h"


void ParamPool::setParameters()
{
    config_path = Parameters::config_path;
    output_path = Parameters::output_path;


    imu_num = Parameters::imu_num;
    imu_topic = Parameters::imu_topic;
    acc_n = Parameters::acc_n;
    acc_w = Parameters::acc_w;
    gyr_n = Parameters::gyr_n;
    gyr_w = Parameters::gyr_w;
    pre_integration = nullptr;

    cam_num = Parameters::cam_num;
    cam0_topic = Parameters::cam0_topic;
    cam1_topic = Parameters::cam1_topic;
    m_camera = Parameters::m_camera;
    fisheye_mask = Parameters::fisheye_mask;
    pixel_n = Parameters::pixel_n;
    image_width = Parameters::image_width;
    image_height = Parameters::image_height;
    focal_length = Parameters::focal_length;
    init_depth = Parameters::init_depth;
    min_parallax = Parameters::min_parallax;
    max_feature_count = Parameters::max_feature_count;
    min_line_feature_count = Parameters::min_line_feature_count;
    min_feature_distance = Parameters::min_feature_distance;
    reliable_track_count = Parameters::reliable_track_count;
    reliable_line_track_count = Parameters::reliable_line_track_count;
    down_sampling_raw_image = Parameters::down_sampling_raw_image;
    down_sampling_track_image = Parameters::down_sampling_track_image;

    extrinsics = Parameters::extrinsics;

    /* time delay between instant0 when sensor sampling and instant1 when data received by estimator (before tracking) */
    time_delay = Parameters::time_delay; // instant0 = instant1 + time_delay


    is_use_imu = Parameters::is_use_imu;
    is_use_stereo = Parameters::is_use_stereo;
    is_fisheye = Parameters::is_fisheye;
    is_flow_back = Parameters::is_flow_back;
    is_lineflow_back = Parameters::is_lineflow_back;
    is_show_track = Parameters::is_show_track;
    is_equalize = Parameters::is_equalize;
    is_estimate_extrinsic = Parameters::is_estimate_extrinsic;
    is_estimate_timedelay = Parameters::is_estimate_timedelay;


    max_solver_time = Parameters::max_solver_time;
    max_num_iterations = Parameters::max_num_iterations;


    solver_flag = INITIAL;
    marginalization_flag = MARGIN_OLD;

    key_poses.clear();
    latest_state.clearState();

    ProjectionTwoFrameOneCamFactor::sqrt_info = focal_length / pixel_n * Eigen::Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = focal_length / pixel_n * Eigen::Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = focal_length / pixel_n * Eigen::Matrix2d::Identity();
    ProjectionLineFactor::sqrt_info = focal_length / pixel_n * Eigen::Matrix2d::Identity();
    // ProjectionLineFactor::sqrt_info = focal_length / 0.5 * Eigen::Matrix2d::Identity();
    // ProjectionLineFactor::sqrt_info = Eigen::Matrix2d::Identity();
}