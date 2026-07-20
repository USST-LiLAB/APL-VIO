#ifndef __IMAGE_PROCESSOR_H
#define __IMAGE_PROCESSOR_H

#include <mutex>
#include <atomic>
#include <thread>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include "estimator/customType.h"


class FeatureTrackerBase;
class LineFeatureTrackerBase;

class ImageProcessor
{
public:
    ImageProcessor();
    ~ImageProcessor();

    void clearState();
    void setParameters();
    void setThreads();
    
    void img0_callback(const sensor_msgs::ImageConstPtr &);
    void img1_callback(const sensor_msgs::ImageConstPtr &);
    
    void processImage();
    bool syncImage(double &, cv::Mat &, cv::Mat &);
    bool getImageFrame(ImageFrame &);
    void trackFeature(const double, const cv::Mat &, const cv::Mat &);
    void trackLineFeature(const double, const cv::Mat &, const cv::Mat &);

    static bool fundmantalMatrixRANSAC(const int, const std::vector<cv::Point2f> &, const std::vector<cv::Point2f> &, std::vector<uchar> &);
    static bool undistortImagePoints(const int, const std::vector<cv::Point2f> &, std::vector<cv::Point2f> &);
    static void undistortImage(const int, const cv::Mat &, cv::Mat &);
    static void showUndistortImage(const int, const cv::Mat &, const std::vector<cv::Point2f> &, std::vector<cv::Point2f> &);


    FrameBuffer frame_buffer;
    FrameLineBuffer frame_line_buffer;
    std::mutex frame_buffer_mutex; // image mutex
    std::mutex frame_line_buffer_mutex; // image mutex

    static std::vector<cv::Mat> maps1, maps2;
    static std::vector<cv::Mat> K_vec;

private:
    int new_frame_id; // image id is unique
    double new_frame_time;
    cv::Mat image0, image1;
    sensor_msgs::ImageConstPtr img0, img1; // left and right camera image

    std::unique_ptr<FeatureTrackerBase> feature_tracker;
    std::unique_ptr<LineFeatureTrackerBase> linefeature_tracker;
    
    std::thread process_thread, track_thread, line_track_thread;
    std::atomic<bool> point_track_atomic_lock, line_track_atomic_lock;

    std::mutex img0_buffer_mutex, img1_buffer_mutex; // image mutex
    std::queue<sensor_msgs::ImageConstPtr> img0_buffer, img1_buffer; // image buffer
    FrameFeatures frame_features;
};

#endif