#ifndef __ROS_MASTER_H
#define __ROS_MASTER_H

#include <ros/ros.h>


extern ros::Subscriber sub_imu;
extern ros::Subscriber sub_cam0;
extern ros::Subscriber sub_cam1;

extern ros::Publisher pub_odometry, pub_latest_odometry;
extern ros::Publisher pub_path;
extern ros::Publisher pub_point_cloud, pub_margin_cloud;
extern ros::Publisher pub_left_camera_pose, pub_right_camera_pose;
extern ros::Publisher pub_camera_marker;
extern ros::Publisher pub_key_poses;
extern ros::Publisher pub_keyframe_pose;
extern ros::Publisher pub_keyframe_point;

extern ros::Publisher pub_extrinsic;

extern ros::Publisher pub_image_track;
extern ros::Publisher pub_image_line_track;

extern ros::Publisher pub_left_image;
extern ros::Publisher pub_right_image;

class Estimator;

class RosMaster
{

public:
    RosMaster();
    
    void setParameters();
    void reg(ros::NodeHandle &, Estimator &);
    void regSub(ros::NodeHandle &, Estimator &);
    void regPub(ros::NodeHandle &, Estimator &);

    static void pubOdometryFast(const double, const Eigen::Vector3d &, const Eigen::Quaterniond &, const Eigen::Vector3d &);
    static void pubTrackImage(const double, const cv::Mat &);
    static void pubLineTrackImage(const double, const cv::Mat &);
    static void pubOdometrySlow(const Estimator &);
    static void pubCameraPose(const Estimator &);
    static void pubKeyPoses(const Estimator &);
    static void pubKeyframe(const Estimator &);
    static void pubPointCloud(const Estimator &);
    static void pubLineCloud(const Estimator &);
    static void pubTF(const Estimator &);
    static void printStatistics(const Estimator &);
};

#endif
