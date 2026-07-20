#include <fstream>

#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/MarkerArray.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>

#include "odom_utils/utils.h"
#include "estimator/paramPool.h"
#include "inertial/imuProcessor.h"
#include "visual/imageProcessor.h"
#include "visual/featureManager.h"
#include "visual/lineFeatureManager.h"
#include "estimator/estimator.h"

#include "rosMaster/cameraMarker.h"
#include "rosMaster/rosMaster.h"


ros::Subscriber sub_imu;
ros::Subscriber sub_cam0;
ros::Subscriber sub_cam1;

ros::Publisher pub_odometry, pub_latest_odometry;
ros::Publisher pub_path;
ros::Publisher pub_extrinsic;

ros::Publisher pub_point_cloud, pub_margin_cloud;
ros::Publisher pub_line_cloud, pub_margin_line_cloud;

ros::Publisher pub_left_camera_pose, pub_right_camera_pose;
ros::Publisher pub_camera_marker;
ros::Publisher pub_key_poses;
ros::Publisher pub_keyframe_pose;
ros::Publisher pub_keyframe_point;

ros::Publisher pub_image_track;
ros::Publisher pub_image_line_track;

ros::Publisher pub_left_image;
ros::Publisher pub_right_image;

extern std::unique_ptr<ParamPool> shared_pool;

RosMaster::RosMaster()
{
    setParameters();
}


void RosMaster::reg(ros::NodeHandle &nh, Estimator &estimator)
{
    regSub(nh, estimator);
    regPub(nh, estimator);
}


void RosMaster::setParameters()
{
    // nothing
}


void RosMaster::regSub(ros::NodeHandle &nh, Estimator &estimator)
{
    if (shared_pool->is_use_imu)
    {
        sub_imu = nh.subscribe<sensor_msgs::Imu>(shared_pool->imu_topic, 
                            2000, 
                            boost::bind(&ImuProcessor::imu_callback, estimator.imu_processor, _1),
                            ros::VoidConstPtr(),
                            ros::TransportHints().tcpNoDelay());
    }

    sub_cam0 = nh.subscribe<sensor_msgs::Image>(shared_pool->cam0_topic, 
                            100, 
                            boost::bind(&ImageProcessor::img0_callback, estimator.image_processor, _1),
                            ros::VoidConstPtr(),
                            ros::TransportHints().tcpNoDelay());

    if (shared_pool->is_use_stereo)
    {
        sub_cam1 = nh.subscribe<sensor_msgs::Image>(shared_pool->cam1_topic, 
                            100, 
                            boost::bind(&ImageProcessor::img1_callback, estimator.image_processor, _1),
                            ros::VoidConstPtr(),
                            ros::TransportHints().tcpNoDelay());
    }
}


void RosMaster::regPub(ros::NodeHandle &nh, Estimator &estimator)
{
    pub_latest_odometry = nh.advertise<nav_msgs::Odometry>("imu_propagate", 1000);
    pub_odometry = nh.advertise<nav_msgs::Odometry>("odometry", 1000);
    pub_path = nh.advertise<nav_msgs::Path>("path", 1000);
    pub_extrinsic = nh.advertise<nav_msgs::Odometry>("extrinsic", 1000);
    pub_point_cloud = nh.advertise<sensor_msgs::PointCloud>("point_cloud", 1000);
    pub_margin_cloud = nh.advertise<sensor_msgs::PointCloud>("margin_cloud", 1000);
    pub_line_cloud = nh.advertise<visualization_msgs::Marker>("line_cloud", 1000);
    pub_margin_line_cloud = nh.advertise<visualization_msgs::Marker>("margin_line_cloud", 1000);
    pub_left_camera_pose = nh.advertise<nav_msgs::Odometry>("left_camera_pose", 1000);
    pub_right_camera_pose = nh.advertise<nav_msgs::Odometry>("right_camera_pose", 1000);
    pub_key_poses = nh.advertise<visualization_msgs::Marker>("key_poses", 1000);
    pub_keyframe_pose = nh.advertise<nav_msgs::Odometry>("keyframe_pose", 1000);
    pub_keyframe_point = nh.advertise<sensor_msgs::PointCloud>("keyframe_point", 1000);
    pub_image_track = nh.advertise<sensor_msgs::Image>("image_track", 1000);
    pub_image_line_track = nh.advertise<sensor_msgs::Image>("image_line_track", 1000);

    // kitti
    pub_left_image = nh.advertise<sensor_msgs::Image>("/kitti/cam0/image_raw", 1000);
    pub_right_image = nh.advertise<sensor_msgs::Image>("/kitti/cam1/image_raw", 1000);

    pub_camera_marker = nh.advertise<visualization_msgs::MarkerArray>("camera_marker", 1000);
}


void RosMaster::pubTrackImage(const double t, const cv::Mat &imageTrack)
{
    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(t);
    /* opencv image to ros message */
    sensor_msgs::ImagePtr imageTrackMsg = utils::image2msg<sensor_msgs::ImagePtr>(header, imageTrack);
    pub_image_track.publish(imageTrackMsg);
}


void RosMaster::pubLineTrackImage(const double t, const cv::Mat &imageTrack)
{
    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(t);
    /* opencv image to ros message */
    sensor_msgs::ImagePtr imageTrackMsg = utils::image2msg<sensor_msgs::ImagePtr>(header, imageTrack);
    pub_image_line_track.publish(imageTrackMsg);
}


void RosMaster::pubOdometryFast(const double t, const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V)
{
    nav_msgs::Odometry odometry;
    odometry.header.frame_id = "world";
    odometry.header.stamp = ros::Time(t);
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = Q.x();
    odometry.pose.pose.orientation.y = Q.y();
    odometry.pose.pose.orientation.z = Q.z();
    odometry.pose.pose.orientation.w = Q.w();
    odometry.twist.twist.linear.x = V.x();
    odometry.twist.twist.linear.y = V.y();
    odometry.twist.twist.linear.z = V.z();
    pub_latest_odometry.publish(odometry);
}


void RosMaster::pubOdometrySlow(const Estimator &estimator)
{
    if ( shared_pool->solver_flag != NONLINEAR)
        return;

    const BodyState &state = estimator.state_window[WINDOW_SIZE].body_state;

    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(state.t + shared_pool->time_delay);

    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.child_frame_id = "world";
    odometry.pose.pose.position.x = state.p.x();
    odometry.pose.pose.position.y = state.p.y();
    odometry.pose.pose.position.z = state.p.z();
    odometry.pose.pose.orientation.x = state.q.x();
    odometry.pose.pose.orientation.y = state.q.y();
    odometry.pose.pose.orientation.z = state.q.z();
    odometry.pose.pose.orientation.w = state.q.w();
    odometry.twist.twist.linear.x = state.v.x();
    odometry.twist.twist.linear.y = state.v.y();
    odometry.twist.twist.linear.z = state.v.z();
    pub_odometry.publish(odometry);

    static nav_msgs::Path path;
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = header;
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose = odometry.pose.pose;
    path.header = header;
    path.header.frame_id = "world";
    path.poses.push_back(pose_stamped);
    pub_path.publish(path);
}


void RosMaster::pubCameraPose(const Estimator &estimator)
{
    if ( shared_pool->solver_flag != NONLINEAR)
        return;

    static CameraMarker camera_marker(1, 0, 0.5); // rgb

    const BodyState &state = estimator.state_window[WINDOW_SIZE - 1].body_state;
    const Eigen::Matrix3d &R_imu_cam0 = shared_pool->extrinsics.R_imu_cam[0];
    const Eigen::Vector3d &t_imu_cam0 = shared_pool->extrinsics.t_imu_cam[0];
    const Eigen::Matrix3d &R_imu_cam1 = shared_pool->extrinsics.R_imu_cam[1];
    const Eigen::Vector3d &t_imu_cam1 = shared_pool->extrinsics.t_imu_cam[1];

    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(estimator.state_window[WINDOW_SIZE].body_state.t + shared_pool->time_delay);

    Eigen::Vector3d P = state.p + state.R * t_imu_cam0;
    Eigen::Quaterniond Q = Eigen::Quaterniond(state.R * R_imu_cam0).normalized();

    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.child_frame_id = "world";
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = Q.x();
    odometry.pose.pose.orientation.y = Q.y();
    odometry.pose.pose.orientation.z = Q.z();
    odometry.pose.pose.orientation.w = Q.w();

    pub_left_camera_pose.publish(odometry);

    camera_marker.reset();
    camera_marker.createMarker(P, Q);

    if (shared_pool->is_use_stereo)
    {
        Eigen::Vector3d P = state.p + state.R * t_imu_cam1;
        Eigen::Quaterniond Q = Eigen::Quaterniond(state.R * R_imu_cam1).normalized();

        nav_msgs::Odometry odometry;
        odometry.header = header;
        odometry.child_frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = Q.x();
        odometry.pose.pose.orientation.y = Q.y();
        odometry.pose.pose.orientation.z = Q.z();
        odometry.pose.pose.orientation.w = Q.w();
        pub_right_camera_pose.publish(odometry);
        
        camera_marker.createMarker(P, Q);
    }


	visualization_msgs::MarkerArray markerArray_msg;
    for (auto &marker : camera_marker.m_markers)
    {
        marker.header = header;
        markerArray_msg.markers.push_back(marker);
    }

    pub_camera_marker.publish(markerArray_msg);
}


void RosMaster::pubKeyPoses(const Estimator &estimator)
{
    if (shared_pool->key_poses.size() == 0)
        return;

    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(estimator.state_window[WINDOW_SIZE].body_state.t + shared_pool->time_delay);

    visualization_msgs::Marker key_poses;
    key_poses.header = header;
    key_poses.ns = "key_poses";
    key_poses.type = visualization_msgs::Marker::SPHERE_LIST;
    key_poses.action = visualization_msgs::Marker::ADD;
    key_poses.pose.orientation.w = 1.0;
    key_poses.lifetime = ros::Duration();

    // static int key_poses_id = 0;
    key_poses.id = 0; // key_poses_id++;
    key_poses.scale.x = 0.05;
    key_poses.scale.y = 0.05;
    key_poses.scale.z = 0.05;
    key_poses.color.r = 1.0;
    key_poses.color.a = 1.0;

    for (int i = 0; i != WINDOW_SIZE + 1; i++)
    {
        geometry_msgs::Point pose_marker;
        Eigen::Vector3d correct_pose;
        correct_pose = shared_pool->key_poses[i];
        pose_marker.x = correct_pose.x();
        pose_marker.y = correct_pose.y();
        pose_marker.z = correct_pose.z();
        key_poses.points.push_back(pose_marker);
    }
    pub_key_poses.publish(key_poses);
}


void RosMaster::pubKeyframe(const Estimator &estimator)
{
    // pub camera pose, 2D-3D points of keyframe
    if (shared_pool->solver_flag != NONLINEAR || shared_pool->marginalization_flag != MARGIN_OLD)
        return;

    const Eigen::Matrix3d &R_imu_cam0 = shared_pool->extrinsics.R_imu_cam[0];
    const Eigen::Vector3d &t_imu_cam0 = shared_pool->extrinsics.t_imu_cam[0];

    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(estimator.state_window[WINDOW_SIZE - 2].body_state.t + shared_pool->time_delay);

    Eigen::Vector3d P = estimator.state_window[WINDOW_SIZE - 2].body_state.p;
    Eigen::Quaterniond Q = estimator.state_window[WINDOW_SIZE - 2].body_state.q;

    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = Q.x();
    odometry.pose.pose.orientation.y = Q.y();
    odometry.pose.pose.orientation.z = Q.z();
    odometry.pose.pose.orientation.w = Q.w();

    pub_keyframe_pose.publish(odometry);


    sensor_msgs::PointCloud point_cloud;
    point_cloud.header = header;
    for (auto &feature_per_id : estimator.feature_manager->feature_list)
    {
        int frame_size = feature_per_id.feature_per_frame.size();
        if(feature_per_id.start_frame < WINDOW_SIZE - 2 && feature_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - 2 && feature_per_id.solve_flag == 1)
        {
            int frame_i = feature_per_id.start_frame;

            const Eigen::Vector3d &P = estimator.state_window[frame_i].body_state.p;
            const Eigen::Matrix3d &R = estimator.state_window[frame_i].body_state.R;

            Eigen::Vector3d pts_i = feature_per_id.feature_per_frame[0].point_0 * feature_per_id.depth;
            Eigen::Vector3d w_pts_i = R * (R_imu_cam0 * pts_i + t_imu_cam0) + P;

            geometry_msgs::Point32 p;
            p.x = w_pts_i(0);
            p.y = w_pts_i(1);
            p.z = w_pts_i(2);
            point_cloud.points.push_back(p);

            int frame_j = WINDOW_SIZE - 2 - feature_per_id.start_frame;
            sensor_msgs::ChannelFloat32 p_2d;
            p_2d.values.push_back(feature_per_id.feature_per_frame[frame_j].point_0.x());
            p_2d.values.push_back(feature_per_id.feature_per_frame[frame_j].point_0.y());
            p_2d.values.push_back(feature_per_id.feature_per_frame[frame_j].uv_0.x());
            p_2d.values.push_back(feature_per_id.feature_per_frame[frame_j].uv_0.y());
            p_2d.values.push_back(feature_per_id.feature_id);
            point_cloud.channels.push_back(p_2d);
        }
    }
    pub_keyframe_point.publish(point_cloud);
}


void RosMaster::pubPointCloud(const Estimator &estimator)
{
    /* pub latest point */
    const Eigen::Matrix3d &R_imu_cam0 = shared_pool->extrinsics.R_imu_cam[0];
    const Eigen::Vector3d &t_imu_cam0 = shared_pool->extrinsics.t_imu_cam[0];

    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(estimator.state_window[WINDOW_SIZE].body_state.t + shared_pool->time_delay);

    sensor_msgs::PointCloud point_cloud;
    point_cloud.header = header;

    for (auto &feature_per_id : estimator.feature_manager->feature_list)
    {
        if (!(feature_per_id.feature_per_frame.size() >= 2 && feature_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        if (feature_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || feature_per_id.solve_flag != 1)
            continue;
        int frame_id = feature_per_id.start_frame;

        const Eigen::Vector3d &P = estimator.state_window[frame_id].body_state.p;
        const Eigen::Matrix3d &R = estimator.state_window[frame_id].body_state.R;

        Eigen::Vector3d pts_i = feature_per_id.feature_per_frame[0].point_0 * feature_per_id.depth;
        Eigen::Vector3d w_pts_i = R * (R_imu_cam0 * pts_i + t_imu_cam0) + P;

        geometry_msgs::Point32 point;
        point.x = w_pts_i(0);
        point.y = w_pts_i(1);
        point.z = w_pts_i(2);
        point_cloud.points.push_back(point);
    }

    pub_point_cloud.publish(point_cloud);


    /* pub margined point */
    sensor_msgs::PointCloud margin_cloud;
    margin_cloud.header = header;

    for (auto &feature_per_id : estimator.feature_manager->feature_list)
    { 
        if (!(feature_per_id.feature_per_frame.size() >= 2 && feature_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        if (feature_per_id.start_frame == 0 && feature_per_id.feature_per_frame.size() <= 2 
            && feature_per_id.solve_flag == 1)
        {
            int frame_id = feature_per_id.start_frame;

            const Eigen::Vector3d &P = estimator.state_window[frame_id].body_state.p;
            const Eigen::Matrix3d &R = estimator.state_window[frame_id].body_state.R;

            Eigen::Vector3d pts_i = feature_per_id.feature_per_frame[0].point_0 * feature_per_id.depth;
            Eigen::Vector3d w_pts_i = R * (R_imu_cam0 * pts_i + t_imu_cam0) + P;

            geometry_msgs::Point32 point;
            point.x = w_pts_i(0);
            point.y = w_pts_i(1);
            point.z = w_pts_i(2);
            margin_cloud.points.push_back(point);
        }
    }

    pub_margin_cloud.publish(margin_cloud);
}


void RosMaster::pubLineCloud(const Estimator &estimator)
{
    visualization_msgs::Marker lines;

    const Eigen::Matrix3d &R_imu_cam0 = shared_pool->extrinsics.R_imu_cam[0];
    const Eigen::Vector3d &t_imu_cam0 = shared_pool->extrinsics.t_imu_cam[0];

    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(estimator.state_window[WINDOW_SIZE].body_state.t + shared_pool->time_delay);

    lines.header = header;
    lines.ns = "lines";
    lines.type = visualization_msgs::Marker::LINE_LIST;
    lines.action = visualization_msgs::Marker::ADD;
    lines.pose.orientation.w = 1.0;
    lines.lifetime = ros::Duration();

    lines.id = 0; //key_poses_id++;
    lines.scale.x = 0.03;
    lines.scale.y = 0.03;
    lines.scale.z = 0.03;
    lines.color.b = 1.0;
    lines.color.a = 1.0;

    for (auto &linefeature_per_id : estimator.linefeature_manager->linefeature_list)
    {
        if (linefeature_per_id.is_triangulate == false)
            continue;

        if (linefeature_per_id.solve_flag != 1)
            continue;

        if (linefeature_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0)
            continue;


        // int visualized_index = 0;
        int visualized_index = linefeature_per_id.linefeature_per_frame.size() / 2;

        if (linefeature_per_id.linefeature_per_frame[visualized_index].is_tracked == false)
            continue;

        int frame_id = linefeature_per_id.start_frame + visualized_index;

        const Eigen::Vector3d &t_w_imu = estimator.state_window[frame_id].body_state.p;
        const Eigen::Matrix3d &R_w_imu = estimator.state_window[frame_id].body_state.R;
        const Eigen::Vector3d t_w_cam0 = t_w_imu + R_w_imu * t_imu_cam0;
        const Eigen::Matrix3d R_w_cam0 = R_w_imu * R_imu_cam0;

        /* plucker in camera frame */
        Eigen::Matrix<double, 6, 1> plucker_c = geometry_utils::invTransformPlucker(
            linefeature_per_id.plucker_w, R_w_cam0, t_w_cam0);
        
        Eigen::Vector3d sPoint3d, ePoint3d;
        LineFeatureManager::planeTrimLine(plucker_c, linefeature_per_id.linefeature_per_frame[visualized_index].line_0, sPoint3d, ePoint3d);

        // if ((sPoint3d - ePoint3d).norm() > 15)
        //     continue;

        Eigen::Vector3d w_pts_1 = R_w_imu * (R_imu_cam0 * sPoint3d + t_imu_cam0) + t_w_imu;
        Eigen::Vector3d w_pts_2 = R_w_imu * (R_imu_cam0 * ePoint3d + t_imu_cam0) + t_w_imu;

        geometry_msgs::Point p;
        p.x = w_pts_1(0);
        p.y = w_pts_1(1);
        p.z = w_pts_1(2);
        lines.points.push_back(p);
        p.x = w_pts_2(0);
        p.y = w_pts_2(1);
        p.z = w_pts_2(2);
        lines.points.push_back(p);
    }

    pub_line_cloud.publish(lines);



    static visualization_msgs::Marker margin_lines;

    margin_lines.header = header;
    margin_lines.ns = "lines";
    margin_lines.type = visualization_msgs::Marker::LINE_LIST;
    margin_lines.action = visualization_msgs::Marker::ADD;
    margin_lines.pose.orientation.w = 1.0;
    margin_lines.lifetime = ros::Duration();

    margin_lines.scale.x = 0.025;
    margin_lines.scale.y = 0.025;
    margin_lines.scale.z = 0.025;
    margin_lines.color.r = 0.9;
    margin_lines.color.g = 0.1;
    margin_lines.color.b = 0.1;
    margin_lines.color.a = 1.0;

    for (const auto &linefeature_per_id : estimator.linefeature_manager->linefeature_list)
    {
        if (linefeature_per_id.is_published)
            continue;

        if (linefeature_per_id.is_triangulate == false)
            continue;

        if (linefeature_per_id.solve_flag != 1)
            continue;

        if (linefeature_per_id.opt_num < 4)
            continue;

        // if (linefeature_per_id.linefeature_per_frame.size() > 2)
        //     continue;


#if 1
        double max_length = 0;
        int best_frame_id = -1;
        Eigen::Vector3d best_sPoint3d, best_ePoint3d;

        for (size_t i = 0; i != linefeature_per_id.linefeature_per_frame.size(); i++)
        {
            if (linefeature_per_id.linefeature_per_frame[i].is_tracked == false)
                continue;

            // int frame_id = linefeature_per_id.start_frame;
            int frame_id = linefeature_per_id.start_frame + i;

            const Eigen::Vector3d &t_w_imu = estimator.state_window[frame_id].body_state.p;
            const Eigen::Matrix3d &R_w_imu = estimator.state_window[frame_id].body_state.R;
            const Eigen::Vector3d t_w_cam0 = t_w_imu + R_w_imu * t_imu_cam0;
            const Eigen::Matrix3d R_w_cam0 = R_w_imu * R_imu_cam0;

            // plucker in camera frame
            Eigen::Matrix<double, 6, 1> plucker_c = geometry_utils::invTransformPlucker(
                linefeature_per_id.plucker_w, R_w_cam0, t_w_cam0);

            Eigen::Vector3d sPoint3d, ePoint3d;
            LineFeatureManager::planeTrimLine(plucker_c, linefeature_per_id.linefeature_per_frame[i].line_0, sPoint3d, ePoint3d);

            double length = (sPoint3d - ePoint3d).norm();

            if (length > 15)
                continue;

            if (length > max_length)
            {
                max_length = length;
                best_frame_id = frame_id;
                best_sPoint3d = sPoint3d;
                best_ePoint3d = ePoint3d;
            }

        }

        if (best_frame_id == -1)
            continue;

        const Eigen::Vector3d &t_w_imu = estimator.state_window[best_frame_id].body_state.p;
        const Eigen::Matrix3d &R_w_imu = estimator.state_window[best_frame_id].body_state.R;

        Eigen::Vector3d w_pts_1 = R_w_imu * (R_imu_cam0 * best_sPoint3d + t_imu_cam0) + t_w_imu;
        Eigen::Vector3d w_pts_2 = R_w_imu * (R_imu_cam0 * best_ePoint3d + t_imu_cam0) + t_w_imu;
#else
        // int visualized_index = 0;
        int visualized_index = linefeature_per_id.linefeature_per_frame.size() / 3;

        if (linefeature_per_id.linefeature_per_frame[visualized_index].is_tracked == false)
            continue;
        
        
        // int frame_id = linefeature_per_id.start_frame;
        int frame_id = linefeature_per_id.start_frame + visualized_index;

        const Eigen::Vector3d &t_w_imu = estimator.state_window[frame_id].body_state.p;
        const Eigen::Matrix3d &R_w_imu = estimator.state_window[frame_id].body_state.R;
        const Eigen::Vector3d t_w_cam0 = t_w_imu + R_w_imu * t_imu_cam0;
        const Eigen::Matrix3d R_w_cam0 = R_w_imu * R_imu_cam0;

        /* plucker in camera frame */
        Eigen::Matrix<double, 6, 1> plucker_c = geometry_utils::invTransformPlucker(
            linefeature_per_id.plucker_w, R_w_cam0, t_w_cam0);

        Eigen::Vector3d sPoint3d, ePoint3d;
        LineFeatureManager::planeTrimLine(plucker_c, linefeature_per_id.linefeature_per_frame[visualized_index].line_0, sPoint3d, ePoint3d);

        if ((sPoint3d - ePoint3d).norm() > 10)
            continue;

        Eigen::Vector3d w_pts_1 = R_w_imu * (R_imu_cam0 * sPoint3d + t_imu_cam0) + t_w_imu;
        Eigen::Vector3d w_pts_2 = R_w_imu * (R_imu_cam0 * ePoint3d + t_imu_cam0) + t_w_imu;
#endif

        geometry_msgs::Point p;
        p.x = w_pts_1(0);
        p.y = w_pts_1(1);
        p.z = w_pts_1(2);
        margin_lines.points.push_back(p);
        p.x = w_pts_2(0);
        p.y = w_pts_2(1);
        p.z = w_pts_2(2);
        margin_lines.points.push_back(p);
        bool &is_published = const_cast<bool &>(linefeature_per_id.is_published);
        is_published = true;
    }

    pub_margin_line_cloud.publish(margin_lines);
}


void RosMaster::pubTF(const Estimator &estimator)
{
    if (shared_pool->solver_flag != NONLINEAR)
        return;

    const BodyState &state = estimator.state_window[WINDOW_SIZE].body_state;
    const Eigen::Matrix3d &R_imu_cam0 = shared_pool->extrinsics.R_imu_cam[0];
    const Eigen::Vector3d &t_imu_cam0 = shared_pool->extrinsics.t_imu_cam[0];
    const Eigen::Quaterniond q_imu_cam0 = Eigen::Quaterniond(R_imu_cam0).normalized();

    std_msgs::Header header;
    header.seq = 100;
    header.stamp = ros::Time(state.t + shared_pool->time_delay);
    header.frame_id = "world";

    static tf2_ros::TransformBroadcaster broadcaster;
    geometry_msgs::TransformStamped ts;

    ts.header = header;
    ts.child_frame_id = "body";

    ts.transform.translation.x = state.p.x();
    ts.transform.translation.y = state.p.y();
    ts.transform.translation.z = state.p.z();

    ts.transform.rotation.x = state.q.x();
    ts.transform.rotation.y = state.q.y();
    ts.transform.rotation.z = state.q.z();
    ts.transform.rotation.w = state.q.w();

    broadcaster.sendTransform(ts);

    ts.header.frame_id = "body";
    ts.child_frame_id = "camera";
    ts.transform.translation.x = t_imu_cam0.x();
    ts.transform.translation.y = t_imu_cam0.y();
    ts.transform.translation.z = t_imu_cam0.z();
    ts.transform.rotation.x = q_imu_cam0.x();
    ts.transform.rotation.y = q_imu_cam0.y();
    ts.transform.rotation.z = q_imu_cam0.z();
    ts.transform.rotation.w = q_imu_cam0.w();

    broadcaster.sendTransform(ts);


    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = t_imu_cam0.x();
    odometry.pose.pose.position.y = t_imu_cam0.y();
    odometry.pose.pose.position.z = t_imu_cam0.z();
    odometry.pose.pose.orientation.x = q_imu_cam0.x();
    odometry.pose.pose.orientation.y = q_imu_cam0.y();
    odometry.pose.pose.orientation.z = q_imu_cam0.z();
    odometry.pose.pose.orientation.w = q_imu_cam0.w();
    pub_extrinsic.publish(odometry);
}


void RosMaster::printStatistics(const Estimator &estimator)
{
    static std::string traj_output_path = shared_pool->output_path + "/vio.csv";
    static std::string ex_output_path = shared_pool->output_path + "/extrinsics.csv";

    static bool is_first_write = true;
    static std::ofstream traj_fout(traj_output_path, std::ios::out);

    if (is_first_write)
    {
        traj_fout << "#timestamp(s) tx ty tz qx qy qz qw" << std::endl;
        is_first_write = false;
        return;
    }

    
    if (shared_pool->solver_flag == INITIAL)
        return;

    /* print on screen */
    // const Eigen::Vector3d &p = shared_pool->latest_state.p;
    // printf("position: x = %.6f, y = %.6f, z = %.6f\n", p.x(), p.y(), p.z());
    // const Eigen::Quaterniond &q = shared_pool->latest_state.q;
    // printf("rotation: x = %6f, y = %8f, z = %6f, w = %6f\n", q.x(), q.y(), q.z(), q.w());
    // const Eigen::Vector3d ypr = math_utils::R2ypr(shared_pool->latest_state.R) * 180 / M_PI;
    // printf("Euler angles: yaw = %.6f, pitch = %.6f, roll = %.6f\n", ypr.x(), ypr.y(), ypr.z());
    // std::cout << "time delay: " << shared_pool->time_delay << std::endl;

    /* output to files */
    const BodyState &tmp_state = estimator.state_window[WINDOW_SIZE].body_state;
    traj_fout.setf(std::ios::fixed, std::ios::floatfield);
    traj_fout.precision(6);
    traj_fout << tmp_state.t + shared_pool->time_delay << " ";
    traj_fout.precision(8);
    traj_fout
        << tmp_state.p.x() << " " 
        << tmp_state.p.y() << " " 
        << tmp_state.p.z() << " "
        << tmp_state.q.x() << " " 
        << tmp_state.q.y() << " " 
        << tmp_state.q.z() << " " 
        << tmp_state.q.w() << std::endl;


    /* write extrinsics */
    static double pre_time(ros::Time::now().toSec());
    double cur_time = ros::Time::now().toSec();
    if (shared_pool->is_estimate_extrinsic && cur_time - pre_time > 5)
    {
        std::ofstream ex_fout(ex_output_path, std::ios::out);
        for (int i = 0; i != shared_pool->cam_num; i++)
        {
            Eigen::Matrix4d &Tic = shared_pool->extrinsics.T_imu_cam[i];
            Tic.block<3, 3>(0, 0) = shared_pool->extrinsics.R_imu_cam[i];
            Tic.block<3, 1>(0, 3) = shared_pool->extrinsics.t_imu_cam[i];

            ex_fout << "T_imu_cam" << i << ": [" << std::fixed << std::setprecision(12)
                << Tic(0, 0) << ", " << Tic(0, 1) << ", " << Tic(0, 2) << ", " << Tic(0, 3) << ", "
                << "\n             "
                << Tic(1, 0) << ", " << Tic(1, 1) << ", " << Tic(1, 2) << ", " << Tic(1, 3) << ", "
                << "\n             "
                << Tic(2, 0) << ", " << Tic(2, 1) << ", " << Tic(2, 2) << ", " << Tic(2, 3) << ", "
                << "\n             "
                << Tic(3, 0) << ", " << Tic(3, 1) << ", " << Tic(3, 2) << ", " << Tic(3, 3) << "]\n" 
                << std::endl;
        }
        ex_fout.close();
        pre_time = cur_time;
    }
}

