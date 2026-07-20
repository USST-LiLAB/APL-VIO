#include <fstream>
#include "benchmark/parameters.h"

std::string Parameters::sequence_path;
std::vector<Data> Parameters::benchmark_buffer;

void Parameters::param_from_ros_handle(const ros::NodeHandle &nh)
{
    std::cout << std::endl;
    read_essential_param(nh, "sequence_path", sequence_path);
    ROS_INFO_STREAM("sequence_path: \n" << sequence_path);

    std::fstream fs;
    fs.open(sequence_path, std::ios::in);
    if (!fs.is_open())
    {
        ROS_WARN("Can NOT load ground truth. Wrong path!!!");
        ROS_BREAK();
    }
    std::string str;
    while (getline(fs, str))
    {
        benchmark_buffer.emplace_back(str);
    }
    fs.close();

    ROS_INFO("Data loaded: %d", static_cast<int>(benchmark_buffer.size()));
}
