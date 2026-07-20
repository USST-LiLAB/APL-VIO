#ifndef __BENCHMARK_PARAMETERS_H
#define __BENCHMARK_PARAMETERS_H

#include <ros/ros.h>
#include <Eigen/Dense>

struct Data
{
    Data(std::string &f)
    {
        std::stringstream ss(f);
        ss >> t >> px >> py >> pz >> qx >> qy >> qz >> qw;
    }
    double t;
    float px, py, pz;
    float qx, qy, qz, qw;
};

class Parameters
{
public:
    Parameters() {}

    static void param_from_ros_handle(const ros::NodeHandle &);

    static std::string sequence_path;
    static std::vector<Data> benchmark_buffer;

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
