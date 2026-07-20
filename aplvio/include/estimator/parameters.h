#ifndef __PARAMETERS_H
#define __PARAMETERS_H

#include <ros/ros.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "camodocal/camera_models/EquidistantCamera.h"


const int WINDOW_SIZE = 10;
extern Eigen::Vector3d G;

class Extrinsics;

class Parameters
{
public:
    Parameters() {}

    static void param_from_ros_handle(const ros::NodeHandle &);

    static std::string config_path;
    static std::string output_path;

    // imu
    static int imu_num;
    static std::string imu_topic;
    static double acc_n, acc_w, gyr_n, gyr_w;

    // cam
    static int cam_num;
    static std::string cam0_topic, cam1_topic;
    static std::vector<camodocal::CameraPtr> m_camera;
    static cv::Mat fisheye_mask;
    static double pixel_n;

    static int image_width, image_height;
    static double focal_length, init_depth;
    static double keyframe_parallax, min_parallax;
    static int max_feature_count, min_feature_distance;
    static int min_line_feature_count;
    static size_t reliable_track_count;
    static size_t reliable_line_track_count;
    static int down_sampling_raw_image, down_sampling_track_image;


    // noise and random walk of both accelerometer and gyroscope
    
    static bool is_use_imu, is_use_stereo;
    static bool is_fisheye;
    static bool is_estimate_extrinsic, is_estimate_timedelay;
    static bool is_flow_back;
    static bool is_lineflow_back;
    static bool is_show_track;
    static bool is_equalize;


    static Extrinsics extrinsics;
    // static Intrinsics intrinsics;

    static double time_delay;

    static double max_solver_time, max_num_iterations;

private:
    template <typename TName, typename TVal>
    static void read_essential_param(const ros::NodeHandle &nh, const TName &name, TVal &val)
    {
        if (!nh.getParam(name, val))
        {
            ROS_ERROR_STREAM("Read param: " << name << " failed.");
            ROS_BREAK();
        }
    }
};

#endif
