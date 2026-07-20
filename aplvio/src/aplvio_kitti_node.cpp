#include <cv_bridge/cv_bridge.h>

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

	std::string dataPath = std::string(argv[1]) + "/";
    
    RosMaster rosMaster;
    rosMaster.reg(nh, estimator);

    ROS_WARN("waiting for image and imu...");

    estimator.setThreads();


	/* load image list */
	FILE* file = std::fopen((dataPath + "times.txt").c_str() , "r");
	if(file == NULL)
    {
	    printf("cannot find file: %stimes.txt\n", dataPath.c_str());
	    ROS_BREAK();
	    return 0;          
	}
	double imageTime;
	std::vector<double> imageTimeList;
	while (fscanf(file, "%lf", &imageTime) != EOF)
	{
	    imageTimeList.push_back(imageTime);
	}
	std::fclose(file);

	std::string leftImagePath, rightImagePath;
	cv::Mat imLeft, imRight;
    for (size_t i = 0; i < imageTimeList.size() && ros::ok(); ++i)
    {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << i;
        leftImagePath = dataPath + "image_0/" + ss.str() + ".png";
        rightImagePath = dataPath + "image_1/" + ss.str() + ".png";
        // std::cout << "load " << i << "-th image" << std::endl;

        imLeft = cv::imread(leftImagePath, cv::IMREAD_GRAYSCALE);
        sensor_msgs::ImagePtr imLeftMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", imLeft).toImageMsg();
        imLeftMsg->header.stamp = ros::Time(imageTimeList[i]);
        pub_left_image.publish(imLeftMsg);

        imRight = cv::imread(rightImagePath, cv::IMREAD_GRAYSCALE);
        sensor_msgs::ImagePtr imRightMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", imRight).toImageMsg();
        imRightMsg->header.stamp = ros::Time(imageTimeList[i]);
        pub_right_image.publish(imRightMsg);

        ros::spinOnce();

        std::chrono::milliseconds dura(30);
        std::this_thread::sleep_for(dura);
    }
    std::cout << "All images have been loaded!" << std::endl;
    ros::spin();

    return 0;
}
