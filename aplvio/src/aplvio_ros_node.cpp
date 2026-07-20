#include "estimator/paramPool.h"
#include "estimator/estimator.h"
#include "rosMaster/rosMaster.h"


std::unique_ptr<ParamPool> shared_pool(new ParamPool);

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "aplvio");
    ros::NodeHandle nh("~");

    Parameters::param_from_ros_handle(nh);

    shared_pool->setParameters();

    Estimator estimator;
    estimator.setParameters();
    
    RosMaster rosMaster;
    rosMaster.reg(nh, estimator);

    ROS_WARN("waiting for image and imu...");

    estimator.setThreads();

    ros::spin();

    return 0;
}
