#ifndef __UTILS_H
#define __UTILS_H

#include <ctime>
#include <chrono>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include "odom_utils/math_utils.h"
#include "odom_utils/geometry_utils.h"


namespace utils
{

struct TicToc
{
    TicToc() { begin = std::chrono::system_clock::now(); }

    double toc() { return std::chrono::duration<double>(std::chrono::system_clock::now() - begin).count() * 1000; }

    std::chrono::time_point<std::chrono::system_clock> begin;
};


struct TimeCost
{
    TimeCost() { num = 0; total_cost = 0.0; max_cost = 0.0; }
    inline double avgCost(double cur_cost) { total_cost += cur_cost; return (total_cost / ++num); }
    inline double maxCost(double cur_cost) { max_cost = cur_cost > max_cost ? cur_cost : max_cost; return max_cost; }

    int num;
    double total_cost, max_cost;
};


/** \brief Convert image type from ROS to OpenCV
 * @param pMsg Pointer to ros-type image msg 
 */
template <typename TMat>
TMat msg2image(const sensor_msgs::ImageConstPtr &pMsg)
{
    // pointer to cv-type image
    cv_bridge::CvImageConstPtr pImg;

    if (pMsg->encoding == "8UC1")
    {
        sensor_msgs::Image msg;
        msg.header = pMsg->header;
        msg.height = pMsg->height;
        msg.width = pMsg->width;
        msg.is_bigendian = pMsg->is_bigendian;
        msg.step = pMsg->step;
        msg.data = pMsg->data;
        msg.encoding = "mono8";
        pImg = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
    }
    else
        pImg = cv_bridge::toCvCopy(pMsg, sensor_msgs::image_encodings::MONO8);

    return pImg->image.clone();
}


/** \brief Convert image type from OpenCV to ROS
 * @param header message header
 * @param img cv-type image
 */
template <typename TMsg>
TMsg image2msg(const std_msgs::Header &header, const cv::Mat &img)
{
    return cv_bridge::CvImage(header, "bgr8", img).toImageMsg();
}


template <typename Tp_>
Tp_ dist(const cv::Point_<Tp_> &point1, const cv::Point_<Tp_> &point2)
{
    cv::Point_<Tp_> dp = point1 - point2;
    return sqrt(dp.x * dp.x + dp.y * dp.y);
}


template <typename Tp_>
bool inside(const cv::Mat &image, const cv::Point_<Tp_> &pt)
{
    return (1 < pt.x && pt.x < image.cols - 1) && (1 < pt.y && pt.y < image.rows - 1);
    // return 1 <= cvRound(pt.x) && cvRound(pt.x) < image.cols - 1 && 1 <= cvRound(pt.y) && cvRound(pt.y) < image.rows - 1;
}

template <typename TName>
void resizeVector(TName &vec, std::vector<uchar> &status)
{
    int j = 0;
    for (size_t i = 0; i != vec.size(); i++)
    {
        if (status.at(i))
        {
            vec.at(j++) = vec.at(i);
        }
    }
    vec.resize(j);
}

};

#endif
