#include "catkin_settings.h"
#include "estimator/customType.h"

#include "estimator/parameters.h"


Eigen::Vector3d G(0.0, 0.0, 9.8);

std::string Parameters::config_path, Parameters::output_path;

int Parameters::imu_num, Parameters::cam_num;
std::string Parameters::imu_topic, Parameters::cam0_topic, Parameters::cam1_topic;

double Parameters::acc_n, Parameters::acc_w, Parameters::gyr_n, Parameters::gyr_w, Parameters::pixel_n;

std::vector<camodocal::CameraPtr> Parameters::m_camera;
cv::Mat Parameters::fisheye_mask;
int Parameters::image_width, Parameters::image_height;
double Parameters::keyframe_parallax, Parameters::min_parallax;;
int Parameters::max_feature_count, Parameters::min_feature_distance;
int Parameters::min_line_feature_count;
double Parameters::focal_length, Parameters::init_depth;
size_t Parameters::reliable_track_count = 4;
size_t Parameters::reliable_line_track_count = 3;
int Parameters::down_sampling_raw_image, Parameters::down_sampling_track_image;


bool Parameters::is_use_imu, Parameters::is_use_stereo;
bool Parameters::is_fisheye;
bool Parameters::is_flow_back;
bool Parameters::is_lineflow_back;
bool Parameters::is_show_track;
bool Parameters::is_equalize;
bool Parameters::is_estimate_extrinsic;
bool Parameters::is_estimate_timedelay;

double Parameters::time_delay;

Extrinsics Parameters::extrinsics;

double Parameters::max_solver_time, Parameters::max_num_iterations;


void Parameters::param_from_ros_handle(const ros::NodeHandle &nh)
{
    std::cout << std::endl;
    read_essential_param(nh, "config_path", config_path);
    read_essential_param(nh, "output_path", output_path);
    ROS_INFO_STREAM("config_path: \n" << config_path);
    ROS_INFO_STREAM("output_path: \n" << output_path << "\n");

    read_essential_param(nh, "show_track", is_show_track);
    read_essential_param(nh, "flow_back", is_flow_back);
    read_essential_param(nh, "lineflow_back", is_lineflow_back);
    read_essential_param(nh, "equalize", is_equalize);
    read_essential_param(nh, "estimate_extrinsic", is_estimate_extrinsic);
    read_essential_param(nh, "estimate_timedelay", is_estimate_timedelay);

    ROS_INFO_STREAM("GPU acceleration     ==> " << (ENABLE_CUDA ? "\033[32mYES" : "\033[33mNO"));
    ROS_INFO_STREAM("Multiple threads     ==> " << (ENABLE_MULTITHREAD ? "\033[32mYES" : "\033[33mNO"));
    ROS_INFO_STREAM("Show track image     ==> " << (is_show_track ? "\033[32mYES" : "\033[33mNO"));
    ROS_INFO_STREAM("Optical flow back    ==> " << (is_flow_back ? "\033[32mYES" : "\033[33mNO"));
    ROS_INFO_STREAM("Line optical flow back    ==> " << (is_lineflow_back ? "\033[32mYES" : "\033[33mNO"));
    ROS_INFO_STREAM("Extrinsics estimated ==> " << (is_estimate_extrinsic ? "\033[32mYES" : "\033[33mNO"));
    ROS_INFO_STREAM("Time delay estimated ==> " << (is_estimate_timedelay ? "\033[32mYES" : "\033[33mNO") << "\n");

    read_essential_param(nh, "imu_num", imu_num);
    if (imu_num == 1)
    {
        is_use_imu = true;
        read_essential_param(nh, "imu_topic", imu_topic);
        read_essential_param(nh, "acc_n", acc_n);
        read_essential_param(nh, "acc_w", acc_w);
        read_essential_param(nh, "gyr_n", gyr_n);
        read_essential_param(nh, "gyr_w", gyr_w);
        read_essential_param(nh, "g_norm", G.z());
    }

    read_essential_param(nh, "cam_num", cam_num);
    read_essential_param(nh, "focal_length", focal_length);


    /* intrinsic parameters */
    read_essential_param(nh, "cam0_topic", cam0_topic);

    std::string cam0_calib;
    read_essential_param(nh, "cam0_calib", cam0_calib);
    std::string cam0_path = config_path.substr(0, config_path.find_last_of('/')) + "/" + cam0_calib;
    m_camera.push_back(camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam0_path));
    image_width = m_camera[0]->imageWidth();
    image_height = m_camera[0]->imageHeight();
    read_essential_param(nh, "fisheye", is_fisheye);
    if (is_fisheye)
    {
        std::string fisheye_mask_path;
        read_essential_param(nh, "fisheye_mask", fisheye_mask_path);
        fisheye_mask_path = config_path.substr(0, config_path.find_last_of('/')) + "/" + fisheye_mask_path;
        fisheye_mask = cv::imread(fisheye_mask_path, false);
    }

    
    /* extrinsic parameters */
    std::vector<double> T_imu_cam0;
    read_essential_param(nh, "T_imu_cam0", T_imu_cam0);
    cv::Mat cv_transform0 = cv::Mat(T_imu_cam0).clone().reshape(1, 4);
    cv::cv2eigen(cv_transform0, extrinsics.T_imu_cam[0]);
    extrinsics.R_imu_cam[0] = extrinsics.T_imu_cam[0].block<3, 3>(0, 0);
    extrinsics.t_imu_cam[0] = extrinsics.T_imu_cam[0].block<3, 1>(0, 3);
    ROS_INFO_STREAM("T_imu_cam0 = \n" << std::fixed << std::setprecision(6) 
        << extrinsics.T_imu_cam[0].topRows(3) << "\n");


    if (cam_num == 2)
    {
        is_use_stereo = true;

        /* intrinsic parameters */
        read_essential_param(nh, "cam1_topic", cam1_topic);

        std::string cam1_calib;
        read_essential_param(nh, "cam1_calib", cam1_calib);
        std::string cam1_path = config_path.substr(0, config_path.find_last_of('/')) + "/" + cam1_calib;
        m_camera.push_back(camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam1_path));

        /* extrinsic parameters */
        std::vector<double> T_imu_cam1;
        read_essential_param(nh, "T_imu_cam1", T_imu_cam1);
        cv::Mat cv_transform1 = cv::Mat(T_imu_cam1).clone().reshape(1, 4);
        cv::cv2eigen(cv_transform1, extrinsics.T_imu_cam[1]);
        extrinsics.R_imu_cam[1] = extrinsics.T_imu_cam[1].block<3, 3>(0, 0);
        extrinsics.t_imu_cam[1] = extrinsics.T_imu_cam[1].block<3, 1>(0, 3);
        ROS_INFO_STREAM("T_imu_cam1 = \n" << std::fixed << std::setprecision(6) 
            << extrinsics.T_imu_cam[1].topRows(3) << std::endl);
    }
    
    read_essential_param(nh, "pixel_n", pixel_n);
    read_essential_param(nh, "init_depth", init_depth);
    read_essential_param(nh, "max_point_cnt", max_feature_count);
    read_essential_param(nh, "min_dist", min_feature_distance);
    read_essential_param(nh, "min_line_cnt", min_line_feature_count);
    read_essential_param(nh, "down_sampling_raw_image", down_sampling_raw_image);
    read_essential_param(nh, "down_sampling_track_image", down_sampling_track_image);


    read_essential_param(nh, "time_delay", time_delay);
    read_essential_param(nh, "max_solver_time", max_solver_time);
    read_essential_param(nh, "max_num_iterations", max_num_iterations);
    read_essential_param(nh, "keyframe_parallax", keyframe_parallax);
    min_parallax = keyframe_parallax / focal_length;
}
