#include "catkin_settings.h"
#include "odom_utils/utils.h"
#include "estimator/paramPool.h"

#include "visual/imageProcessor.h"
#include "visual/featureTracker.h"
#include "visual/lineFeatureTracker_DDM.h"
#include "visual/lineFeatureTracker_LOFT.h"

#if ENABLE_CUDA
    #include <opencv2/cudawarping.hpp>
#endif


#define ENABLE_LOFT 1

extern std::unique_ptr<ParamPool> shared_pool;
std::vector<cv::Mat> ImageProcessor::maps1, ImageProcessor::maps2;
std::vector<cv::Mat> ImageProcessor::K_vec;

ImageProcessor::ImageProcessor() : 
    new_frame_id(0), new_frame_time(0.0), 
    feature_tracker(new FeatureTracker), linefeature_tracker(nullptr),
    point_track_atomic_lock(false), line_track_atomic_lock(false)
{
#if ENABLE_LOFT
    linefeature_tracker.reset(new LineFeatureTrackerLOFT);
#else
    linefeature_tracker.reset(new LineFeatureTrackerDDM);
#endif
    // nothing
}


ImageProcessor::~ImageProcessor()
{
    // process_thread.join();
    // std::cout << "join image process thread!" << std::endl;
    // track_thread.join();
    // std::cout << "join point track thread!" << std::endl;
    // line_track_thread.join();
    // std::cout << "join line track thread!" << std::endl;
}


/** \brief left camera image msg callback
 * @param pMsg Pointer to ros-type image msg 
 */
void ImageProcessor::img0_callback(const sensor_msgs::ImageConstPtr &pMsg)
{
    std::lock_guard<std::mutex> img0_buffer_guard(img0_buffer_mutex);
    img0_buffer.push(pMsg);
}


/** \brief right camera image msg callback
 * @param pMsg Pointer to ros-type image msg 
 */
void ImageProcessor::img1_callback(const sensor_msgs::ImageConstPtr &pMsg)
{
    std::lock_guard<std::mutex> img1_buffer_guard(img1_buffer_mutex);
    img1_buffer.push(pMsg);
}


void ImageProcessor::clearState()
{
    new_frame_id = 0;

    feature_tracker->clearState();
#if ENABLE_LINE_FEATURE
    linefeature_tracker->clearState();
#endif

    point_track_atomic_lock.store(false);
    line_track_atomic_lock.store(false);

    img0_buffer_mutex.lock();
    img0_buffer = std::queue<sensor_msgs::ImageConstPtr>();
    img0_buffer_mutex.unlock();

    img1_buffer_mutex.lock();
    img1_buffer = std::queue<sensor_msgs::ImageConstPtr>();
    img1_buffer_mutex.unlock();
    
    frame_buffer_mutex.lock();
    frame_buffer = FrameBuffer();
    frame_buffer_mutex.unlock();
}

void ImageProcessor::setParameters()
{
    feature_tracker->setParameters();
    linefeature_tracker->setParameters();

    for (int i = 0; i != shared_pool->cam_num; i++)
    {
        cv::Mat map1, map2;
        cv::Mat K = shared_pool->m_camera[i]->initUndistortRectifyMap(map1, map2);

        maps1.push_back(map1);
        maps2.push_back(map2);
        K_vec.push_back(K);
    }
}


void ImageProcessor::setThreads()
{
    /* start a new thread to synchronize and track image */
    process_thread = std::thread([this]() -> void {
        while (true) {
            this->processImage();
        }
    });
    // process_thread.detach();


    track_thread = std::thread([&]() -> void {
        while (true) {
            this->trackFeature(new_frame_time, image0, image1);
        }
    });
    // track_thread.detach();

#if ENABLE_LINE_FEATURE
    line_track_thread = std::thread([&]() -> void {
        while (true) {
            this->trackLineFeature(new_frame_time, image0, image1);
        }
    });
    // line_track_thread.detach();
#endif
}


/** \brief Synchronize two images and track images include previous image0, current image0 and current image1 */
void ImageProcessor::processImage()
{
    /* wait until the last point and line tracking have completed */
    if (point_track_atomic_lock.load() || line_track_atomic_lock.load())
    {
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
        return;
    }

    if (!syncImage(new_frame_time, image0, image1))
        return;

    new_frame_id++;
    
    point_track_atomic_lock.store(true);
#if ENABLE_LINE_FEATURE
    line_track_atomic_lock.store(true);
#endif

#if !ENABLE_MULTITHREAD
    this->trackFeature(new_frame_time, image0, image1);
#if ENABLE_LINE_FEATURE
    this->trackLineFeature(new_frame_time, image0, image1);
#endif
#endif
}


/** \brief Synchronize two images
 * @param frame_time Output frame time which is equal to synchronized image0 time
 * @param image0 Output synchronized image0 within 0.003 second
 * @param image1 Output synchronized image1 within 0.003 second
 * @return Synchronize images success or not
 */
bool ImageProcessor::syncImage(double &frame_time, cv::Mat &image0, cv::Mat &image1)
{
    std::lock(img0_buffer_mutex, img1_buffer_mutex);
    std::lock_guard<std::mutex> img0_buffer_guard(img0_buffer_mutex, std::adopt_lock);
    std::lock_guard<std::mutex> img1_buffer_guard(img1_buffer_mutex, std::adopt_lock);

    image0 = cv::Mat();
    image1 = cv::Mat();

    if (shared_pool->is_use_stereo)
    {
        /* Synchronize images */
        if (!img0_buffer.empty() && !img1_buffer.empty())
        {
            double t0 = img0_buffer.front()->header.stamp.toSec();
            double t1 = img1_buffer.front()->header.stamp.toSec();

            if (t0 + 0.003 < t1)
            {
                image0 = utils::msg2image<cv::Mat>(img0_buffer.front());
                image1 = cv::Mat();
                img0_buffer.pop();
            }
            else if (t1 + 0.003 < t0)
            {
                img1_buffer.pop();
                std::cout << "throw img1\n";
            }
            else
            {
                /* image0_time is defined as frame_time */
                frame_time = img0_buffer.front()->header.stamp.toSec();
                image0 = utils::msg2image<cv::Mat>(img0_buffer.front());
                image1 = utils::msg2image<cv::Mat>(img1_buffer.front());
                img0_buffer.pop();
                img1_buffer.pop();
            }
        }
    }
    else
    {
        if (!img0_buffer.empty())
        {
            frame_time = img0_buffer.front()->header.stamp.toSec();
            image0 = utils::msg2image<cv::Mat>(img0_buffer.front());
            img0_buffer.pop();
        }
    }

    if (image0.empty())
        return false;
    
    ROS_ASSERT_MSG(image0.rows == shared_pool->image_height && image0.cols == shared_pool->image_width, 
        "Wrong image size. Please check the image size!!!");

    return true;
}

/** \brief Track point features in images
 * @param frame_time Frame time which is equal to Synchronized image0 time
 * @param image0 Synchronized image0 used for point feature tracking
 * @param image1 Synchronized image1 used for point feature tracking
 */
void ImageProcessor::trackFeature(const double frame_time, const cv::Mat &image0, const cv::Mat &image1)
{
    /* wait until the images have been synchronized */
    if (!point_track_atomic_lock.load())
    {
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
        return;
    }

    if (new_frame_id % shared_pool->down_sampling_raw_image == 0) // down sampling
    {
        // track image feature and add it to feature_buffer
        if (feature_tracker->trackImage(frame_time, image0, image1))
        {
            if (new_frame_id % shared_pool->down_sampling_track_image == 0)
            {
                std::lock_guard<std::mutex> frame_buffer_guard(frame_buffer_mutex);
                frame_buffer.push(feature_tracker->frame_features);
            }
        }
    }

    point_track_atomic_lock.store(false);
}

/** \brief Track line features in images
 * @param frame_time Frame time which is equal to Synchronized image0 time
 * @param image0 Synchronized image0 used for line feature tracking
 * @param image1 Synchronized image1 used for line feature tracking
 */
void ImageProcessor::trackLineFeature(const double frame_time, const cv::Mat &image0, const cv::Mat &image1)
{
    /* wait until the images have been synchronized */
    if (!line_track_atomic_lock.load())
    {
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
        return;
    }

    if (new_frame_id % shared_pool->down_sampling_raw_image == 0) // down sampling
    {
        if (linefeature_tracker->trackImage(frame_time, image0, image1))
        {
            if (new_frame_id % shared_pool->down_sampling_track_image == 0)
            {
                std::lock_guard<std::mutex> frame_line_buffer_guard(frame_line_buffer_mutex);
                frame_line_buffer.push(linefeature_tracker->frame_linefeatures);
            }
        }
    }

    line_track_atomic_lock.store(false);
}

/** \brief Synchronize and obtain point and line features in the images at the same time
 * @param image_frame Output image frame containing the information of point and line features in the Synchronized images
 * \return Obtain the point and line features in the Synchronized images successfully or not
 */
bool ImageProcessor::getImageFrame(ImageFrame &image_frame)
{
#if ENABLE_LINE_FEATURE
    std::lock(frame_buffer_mutex, frame_line_buffer_mutex);
    std::lock_guard<std::mutex> frame_buffer_guard(frame_buffer_mutex, std::adopt_lock);
    std::lock_guard<std::mutex> frame_line_buffer_guard(frame_line_buffer_mutex, std::adopt_lock);

    if (frame_buffer.empty() || frame_line_buffer.empty())
        return false;

    if (frame_buffer.front().first == frame_line_buffer.front().first)
    {
        image_frame.t = frame_buffer.front().first;
        image_frame.feature_points = frame_buffer.front().second;
        frame_buffer.pop();

        image_frame.feature_lines = frame_line_buffer.front().second;
        frame_line_buffer.pop();
    }
    else if (frame_buffer.front().first < frame_line_buffer.front().first)
    {
        std::cout << "throw tracking points" << std::endl;
        frame_buffer.pop();
        return false;
    }
    else if (frame_buffer.front().first > frame_line_buffer.front().first)
    {
        std::cout << "throw tracking lines" << std::endl;
        frame_line_buffer.pop();
        return false;
    }
#else
    std::lock_guard<std::mutex> frame_buffer_guard(frame_buffer_mutex);
    
    if (frame_buffer.empty())
        return false;

    image_frame.t = frame_buffer.front().first;
    image_frame.feature_points = frame_buffer.front().second;
    frame_buffer.pop();
#endif

    return true;
}


bool ImageProcessor::fundmantalMatrixRANSAC(const int camera_id, 
                                            const std::vector<cv::Point2f> &matched_2d_pts_0,
                                            const std::vector<cv::Point2f> &matched_2d_pts_1,
                                            std::vector<uchar> &status)
{
    size_t n = matched_2d_pts_0.size();
    status.clear();

    if (n >= 8)
    {
        double tmp_x, tmp_y;
        std::vector<cv::Point2f> matched_2d_unpts_0(n), matched_2d_unpts_1(n);
        ImageProcessor::undistortImagePoints(camera_id, matched_2d_pts_0, matched_2d_unpts_0);
        ImageProcessor::undistortImagePoints(camera_id, matched_2d_pts_1, matched_2d_unpts_1);
        for (size_t i = 0; i != n; i++)
        {
            tmp_x = shared_pool->focal_length * matched_2d_unpts_0.at(i).x + shared_pool->image_width / 2.0;
            tmp_y = shared_pool->focal_length * matched_2d_unpts_0.at(i).y + shared_pool->image_height / 2.0;
            matched_2d_unpts_0.at(i) = cv::Point2f(tmp_x, tmp_y);

            tmp_x = shared_pool->focal_length * matched_2d_unpts_1.at(i).x + shared_pool->image_width / 2.0;
            tmp_y = shared_pool->focal_length * matched_2d_unpts_1.at(i).y + shared_pool->image_height / 2.0;
            matched_2d_unpts_1.at(i) = cv::Point2f(tmp_x, tmp_y);
        }
        cv::findFundamentalMat(matched_2d_unpts_0, matched_2d_unpts_1, cv::FM_RANSAC, 1.0, 0.99, status);
        
        // int cnt = std::accumulate(status.begin(), status.end(), 0);
        // std::cout << "fundmantalMatrixRANSAC delete cnt:" << status.size() - cnt << std::endl;
        
        return true;
    }

    return false;
}


/** \brief Undistort image
 * @param camera_id camera id.
 * @param image_pts Input vector of 2D points of the distorted image.
 * @param image_unpts Output vector of 2D points of the undistorted image.
 */
bool ImageProcessor::undistortImagePoints(const int camera_id, 
    const std::vector<cv::Point2f> &image_pts, std::vector<cv::Point2f> &image_unpts)
{
    if (image_pts.size() == 0)
        return false;

    image_unpts.clear();

    for (auto &pt : image_pts)
    {
        Eigen::Vector3d un_pt;
        shared_pool->m_camera[camera_id]->liftProjective(Eigen::Vector2d(pt.x, pt.y), un_pt);
        image_unpts.push_back(cv::Point2f(un_pt.x() / un_pt.z(), un_pt.y() / un_pt.z()));
    }

    return true;
}


void ImageProcessor::undistortImage(const int camera_id, const cv::Mat &distorted_image, cv::Mat &undistorted_image)
{
    utils::TicToc remap_tictoc;
#if ENABLE_CUDA
    cv::cuda::GpuMat distorted_gpu_image(distorted_image);
    cv::cuda::GpuMat undistorted_gpu_image(undistorted_image);
    cv::cuda::GpuMat gpu_map1(maps1[camera_id]);
    cv::cuda::GpuMat gpu_map2(maps2[camera_id]);
    cv::cuda::remap(distorted_gpu_image, undistorted_gpu_image, gpu_map1, gpu_map2, cv::INTER_LINEAR);
    undistorted_gpu_image.download(undistorted_image);
#else
    cv::remap(distorted_image, undistorted_image, maps1[camera_id], maps2[camera_id], cv::INTER_LINEAR);
#endif
}


void ImageProcessor::showUndistortImage(const int camera_id, const cv::Mat &distorted_image, 
    const std::vector<cv::Point2f> &image_pts, std::vector<cv::Point2f> &image_unpts)
{
    const cv::Matx33d &K = K_vec[camera_id];

    cv::Mat undistorted_image(distorted_image.rows, distorted_image.cols, CV_8UC1, cv::Scalar(0));

    undistortImage(camera_id, distorted_image, undistorted_image);

    cv::Mat concat_image;
    cv::vconcat(distorted_image, undistorted_image, concat_image);
    cv::cvtColor(concat_image, concat_image, cv::COLOR_GRAY2RGB);

    /* track point in distort image */
    for (size_t j = 0; j != image_pts.size(); j++)
    {
        cv::circle(concat_image, image_pts[j], 2, cv::Scalar(0, 0, 255), 2); // color format: BGR
    }
    /* track point in undistort image */
    for (size_t j = 0; j != image_unpts.size(); j++)
    {
        cv::Point2f tmp_point(K(0, 0) * image_unpts[j].x + K(0, 2),
                              K(1, 1) * image_unpts[j].y + K(1, 2));

        if (utils::inside(undistorted_image, tmp_point))
            cv::circle(concat_image, tmp_point + cv::Point2f(0, distorted_image.rows), 2, cv::Scalar(0, 255, 0), 2); // color format: BGR
    }

    cv::imshow("distorted and undistorted images", concat_image);
    cv::waitKey(2);
}
