#include <std_msgs/Header.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>

#include "benchmark/parameters.h"
#include "benchmark/rosMaster.h"


ros::Subscriber sub_est_odometry;

ros::Publisher pub_odometry;
ros::Publisher pub_path;

nav_msgs::Path path;


void odom_callback(const nav_msgs::OdometryConstPtr &);

void RosMaster::regester(ros::NodeHandle &nh)
{
    regesterSub(nh);
    regesterPub(nh);
}


void RosMaster::regesterSub(ros::NodeHandle &nh)
{
    sub_est_odometry = nh.subscribe<nav_msgs::Odometry>("estimated_odometry", 100, odom_callback, 
                            ros::VoidConstPtr(), ros::TransportHints().tcpNoDelay());
}


void RosMaster::regesterPub(ros::NodeHandle &nh)
{
    pub_odometry = nh.advertise<nav_msgs::Odometry>("gt_odometry", 1000);
    pub_path = nh.advertise<nav_msgs::Path>("path", 1000);
}


void odom_callback(const nav_msgs::OdometryConstPtr &odom_msg)
{
    std::vector<Data> &benchmark = Parameters::benchmark_buffer;

    static int idx = 1;
    static int init = 0;
    static const int SKIP = 50;
    static Eigen::Quaterniond baseRgt;
    static Eigen::Vector3d baseTgt;

    if (odom_msg->header.stamp.toSec() > benchmark.back().t)
      return;

    while (idx < static_cast<int>(benchmark.size()) && benchmark[idx].t <= odom_msg->header.stamp.toSec())
        idx++;


    if (init++ < SKIP)
    {
        baseRgt = Eigen::Quaterniond(odom_msg->pose.pose.orientation.w,
                                     odom_msg->pose.pose.orientation.x,
                                     odom_msg->pose.pose.orientation.y,
                                     odom_msg->pose.pose.orientation.z) *
                  Eigen::Quaterniond(benchmark[idx - 1].qw,
                                     benchmark[idx - 1].qx,
                                     benchmark[idx - 1].qy,
                                     benchmark[idx - 1].qz).inverse();

        baseTgt = Eigen::Vector3d(odom_msg->pose.pose.position.x, 
                                  odom_msg->pose.pose.position.y, 
                                  odom_msg->pose.pose.position.z)
            - baseRgt * Eigen::Vector3d(benchmark[idx - 1].px, 
                                        benchmark[idx - 1].py, 
                                        benchmark[idx - 1].pz);
        return;
    }

    nav_msgs::Odometry odometry;
    odometry.header.stamp = ros::Time(benchmark[idx - 1].t);
    odometry.header.frame_id = "world";
    odometry.child_frame_id = "world";

    Eigen::Vector3d tmp_T = baseTgt + baseRgt * Eigen::Vector3d(benchmark[idx - 1].px, 
                                                                benchmark[idx - 1].py, 
                                                                benchmark[idx - 1].pz);
    odometry.pose.pose.position.x = tmp_T.x();
    odometry.pose.pose.position.y = tmp_T.y();
    odometry.pose.pose.position.z = tmp_T.z();

    Eigen::Quaterniond tmp_R = baseRgt * Eigen::Quaterniond(benchmark[idx - 1].qw,
                                                            benchmark[idx - 1].qx,
                                                            benchmark[idx - 1].qy,
                                                            benchmark[idx - 1].qz);
    odometry.pose.pose.orientation.w = tmp_R.w();
    odometry.pose.pose.orientation.x = tmp_R.x();
    odometry.pose.pose.orientation.y = tmp_R.y();
    odometry.pose.pose.orientation.z = tmp_R.z();
    pub_odometry.publish(odometry);

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = odometry.header;
    pose_stamped.pose = odometry.pose.pose;
    path.header = odometry.header;
    path.poses.push_back(pose_stamped);
    pub_path.publish(path);
}
