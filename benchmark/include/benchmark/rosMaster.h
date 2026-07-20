#ifndef __BENCHMARK_ROS_MASTER_H
#define __BENCHMARK_ROS_MASTER_H

#include <ros/ros.h>
#include <ros/ros.h>
#include <Eigen/Dense>


class RosMaster
{
public:
    void regester(ros::NodeHandle &);
    void regesterSub(ros::NodeHandle &);
    void regesterPub(ros::NodeHandle &);
};

#endif
