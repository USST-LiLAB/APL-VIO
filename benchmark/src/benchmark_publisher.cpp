#include <ros/ros.h>
#include "benchmark/parameters.h"
#include "benchmark/rosMaster.h"


int main(int argc, char *argv[])
{
    ros::init(argc, argv, "benchmark");
    ros::NodeHandle nh("~");

    Parameters::param_from_ros_handle(nh);

    RosMaster reg;
    reg.regester(nh);

    ros::spin();
    
    return 0;
}
