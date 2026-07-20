#include "catkin_settings.h"
#include "odom_utils/utils.h"
#include "estimator/paramPool.h"
#include "rosMaster/rosMaster.h"

#include "visual/featureTracker.h" 
#include "visual/imageProcessor.h"


#if ENABLE_CUDA
    #include <opencv2/cudaoptflow.hpp>
    #include <opencv2/cudaimgproc.hpp>
    #include <opencv2/cudaarithm.hpp>
#endif


#define SAVE_POINT_TRACKING_COST 0

extern std::unique_ptr<ParamPool> shared_pool;

void FeatureTracker::clearState()
{
    new_feature_id = 0;

    pre_frame_time = 0.0;
    pre_image0 = cv::Mat();
    pre_image0_pts.clear();
    pre_image0_ids_unpts_map.clear();
    pre_image0_ids_pts_map.clear();
    pre_image1_ids_unpts_map.clear();

    cur_frame_time = 0.0;
    cur_image0 = cv::Mat();
    cur_image1 = cv::Mat();
    cur_image0_ids.clear();
    cur_image1_ids.clear();
    cur_image0_pts.clear();
    cur_image1_pts.clear();
    cur_image0_unpts.clear();
    cur_image1_unpts.clear();
    cur_image0_ids_unpts_map.clear();
    cur_image1_ids_unpts_map.clear();

    new_image0_pts.clear();
    track_counters.clear();

    image0_unpts_vel.clear();
    image1_unpts_vel.clear();

    mask = cv::Mat();
    imageTrack = cv::Mat();
}


void FeatureTracker::setParameters()
{
    // nothing
}


/** \brief track feature between the previous left camera image and current left camera image,
 *                       between the current left camera image and current right camera image.
 * @param image0_time left camera image time.
 * @param image0 current left camera image.
 * @param image1 current right camera image
 */
bool FeatureTracker::trackImage(const double image0_time, const cv::Mat &image0, const cv::Mat &image1)
{
    utils::TicToc point_track_tictoc;

    cur_image0 = image0.clone();

    if (shared_pool->is_equalize)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(cur_image0, cur_image0);
    }

    std::vector<uchar> status;
    std::vector<float> err;

    cur_frame_time = image0_time;
    cur_image0_pts.clear();
    image0_unpts_vel.clear();


    /* 1.track feature between previous image and current image */
    if (pre_image0_pts.size() > 0)
    {
#if ENABLE_CUDA
        cv::cuda::GpuMat pre_gpu_image0(pre_image0);
        cv::cuda::GpuMat cur_gpu_image0(cur_image0);
        cv::cuda::GpuMat pre_gpu_image0_pts(pre_image0_pts);
        cv::cuda::GpuMat cur_gpu_image0_pts(cur_image0_pts);
        cv::cuda::GpuMat gpu_status(status);
        cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> ptr_sparse_pyrLK_flow 
            = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(21, 21), 3, 30, false);
        ptr_sparse_pyrLK_flow->calc(pre_gpu_image0, cur_gpu_image0, pre_gpu_image0_pts, cur_gpu_image0_pts, gpu_status);

        cur_image0_pts = std::vector<cv::Point2f>(cur_gpu_image0_pts.cols);
        cur_gpu_image0_pts.download(cur_image0_pts);

        status = std::vector<uchar>(gpu_status.cols);
        gpu_status.download(status);

        if (shared_pool->is_flow_back)
        {
            cv::cuda::GpuMat reverse_gpu_status;
            cv::cuda::GpuMat reverse_gpu_image0_pts = pre_gpu_image0_pts;

            cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> ptr_sparse_pyrLK_flow 
                = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(21, 21), 1, 30, true);
            ptr_sparse_pyrLK_flow->calc(cur_gpu_image0, pre_gpu_image0, cur_gpu_image0_pts, reverse_gpu_image0_pts, reverse_gpu_status);

            std::vector<cv::Point2f> reverse_image0_pts(reverse_gpu_image0_pts.cols);
            reverse_gpu_image0_pts.download(reverse_image0_pts);
            std::vector<uchar> reverse_status(reverse_gpu_status.cols);
            reverse_gpu_status.download(reverse_status);

            for (size_t i = 0; i != status.size(); i++)
                status[i] = ((status[i] && reverse_status[i] && utils::dist(pre_image0_pts[i], reverse_image0_pts[i]) <= 0.5) ? 1 : 0);
        }
#else
        cv::calcOpticalFlowPyrLK(pre_image0, cur_image0, pre_image0_pts, cur_image0_pts, status, err, cv::Size(21, 21), 3);

        if (shared_pool->is_flow_back)
        {
            std::vector<uchar> reverse_status;
            std::vector<cv::Point2f> reverse_image0_pts = pre_image0_pts;
            cv::calcOpticalFlowPyrLK(cur_image0, pre_image0, cur_image0_pts, reverse_image0_pts, reverse_status, err, cv::Size(21, 21), 1,
                cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

            for(size_t i = 0; i != status.size(); i++)
                status[i] = (status[i] && reverse_status[i] && utils::dist(pre_image0_pts[i], reverse_image0_pts[i]) <= 0.5) ? 1 : 0;
        }
#endif
        for (size_t i = 0; i != status.size(); i++)
            if (status[i] && !utils::inside(cur_image0, cur_image0_pts[i]))
                status[i] = 0;

        utils::resizeVector(pre_image0_pts, status);
        utils::resizeVector(cur_image0_pts, status);
        utils::resizeVector(cur_image0_ids, status);
        utils::resizeVector(track_counters, status);

        for (auto &counter : track_counters)
            counter++;
    }

    
    /* 2. ransac rejection */
    std::vector<uchar> ransac_status;
    if (ImageProcessor::fundmantalMatrixRANSAC(0, pre_image0_pts, cur_image0_pts, ransac_status))
    {
        utils::resizeVector(pre_image0_pts, ransac_status);
        utils::resizeVector(cur_image0_pts, ransac_status);
        utils::resizeVector(cur_image0_ids, ransac_status);
        utils::resizeVector(track_counters, ransac_status);
    }


    /* 3.feature extraction and supplementation */
    int new_max_feature_cnt = shared_pool->max_feature_count - cur_image0_pts.size();
    if (new_max_feature_cnt > 0)
    {
        setMask(mask);
        new_image0_pts.clear();
#if ENABLE_CUDA
        cv::cuda::GpuMat gpu_mask(mask);
        cv::cuda::GpuMat cur_gpu_image0(cur_image0);
        cv::cuda::GpuMat new_gpu_image0_pts(new_image0_pts);

        cv::Ptr<cv::cuda::CornersDetector> p_detector = 
            cv::cuda::createGoodFeaturesToTrackDetector(cur_gpu_image0.type(), new_max_feature_cnt, 0.01, shared_pool->min_feature_distance);
        p_detector->detect(cur_gpu_image0, new_gpu_image0_pts, gpu_mask);
        // p_detector->detect(cur_gpu_image0, new_gpu_image0_pts, cv::Mat());

        if (!new_gpu_image0_pts.empty())
        {
            new_image0_pts = std::vector<cv::Point2f>(new_gpu_image0_pts.cols);
            new_gpu_image0_pts.download(new_image0_pts);
        }
#else
        /* Searching for feature points in areas outside the mask with Shi-Tomasi feature detection algorithm */
        cv::goodFeaturesToTrack(cur_image0, new_image0_pts, new_max_feature_cnt, 0.01, shared_pool->min_feature_distance, mask);
        // cv::goodFeaturesToTrack(cur_image0, new_image0_pts, new_max_feature_cnt, 0.01, shared_pool->min_feature_distance, cv::Mat());
#endif
        for (auto &pt : new_image0_pts)
        {
            cur_image0_ids.push_back(new_feature_id++);
            cur_image0_pts.push_back(pt);
            track_counters.push_back(1);
        }
    }

    if (!ImageProcessor::undistortImagePoints(0, cur_image0_pts, cur_image0_unpts))
    {
        ROS_WARN_STREAM("No tracking feature in left image!");
        return false;
    }
#if 0
    ImageProcessor::showUndistortImage(0, cur_image0, cur_image0_pts, cur_image0_unpts);
#endif
    calcPointsVelecity(cur_image0_ids, cur_image0_unpts, pre_image0_ids_unpts_map, cur_image0_ids_unpts_map, image0_unpts_vel);


    /* 4.track feature between left camera image and right camera image */
    if (!image1.empty() && shared_pool->is_use_stereo)
    {
        cur_image1 = image1.clone();
        
        if (shared_pool->is_equalize)
        {
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
            clahe->apply(cur_image1, cur_image1);
        }

        cur_image1_ids.clear();
        cur_image1_pts.clear();
        image1_unpts_vel.clear();

        if (!cur_image0_pts.empty())
        {
            status.clear();
            err.clear();
#if ENABLE_CUDA
            cv::cuda::GpuMat cur_gpu_image0(cur_image0);
            cv::cuda::GpuMat cur_gpu_image1(cur_image1);
            cv::cuda::GpuMat cur_gpu_image0_pts(cur_image0_pts);
            cv::cuda::GpuMat cur_gpu_image1_pts(cur_image1_pts);
            cv::cuda::GpuMat gpu_status(status);
            cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> ptr_sparse_pyrLK_flow = 
                cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(21, 21), 3, 30, false);
            ptr_sparse_pyrLK_flow->calc(cur_gpu_image0, cur_gpu_image1, cur_gpu_image0_pts, cur_gpu_image1_pts, gpu_status);

            cur_image1_pts = std::vector<cv::Point2f>(cur_gpu_image1_pts.cols);
            cur_gpu_image1_pts.download(cur_image1_pts);

            status = std::vector<uchar>(gpu_status.cols);
            gpu_status.download(status);

            if (shared_pool->is_flow_back)
            {
                cv::cuda::GpuMat reverse_gpu_status;
                cv::cuda::GpuMat reverse_gpu_image0_pts = cur_gpu_image0_pts;
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> ptr_sparse_pyrLK_flow = 
                    cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(21, 21), 3, 30, true);
                ptr_sparse_pyrLK_flow->calc(cur_gpu_image1, cur_gpu_image0, cur_gpu_image1_pts, reverse_gpu_image0_pts, reverse_gpu_status);

                std::vector<cv::Point2f> reverse_image0_pts(reverse_gpu_image0_pts.cols);
                reverse_gpu_image0_pts.download(reverse_image0_pts);

                std::vector<uchar> reverse_status(reverse_gpu_status.cols);
                reverse_gpu_status.download(reverse_status);

                for (size_t i = 0; i != status.size(); i++) 
                    status[i] = (status[i] && reverse_status[i] && utils::dist(cur_image0_pts[i], reverse_image0_pts[i]) <= 0.5) ? 1 : 0;
            }
#else
            /* cur left ---- cur right */
            cv::calcOpticalFlowPyrLK(cur_image0, cur_image1, cur_image0_pts, cur_image1_pts, status, err, cv::Size(21, 21), 3);

            if (shared_pool->is_flow_back)
            {
                std::vector<uchar> reverse_status;
                std::vector<cv::Point2f> reverse_image0_pts = cur_image1_pts;
                cv::calcOpticalFlowPyrLK(cur_image1, cur_image0, cur_image1_pts, reverse_image0_pts, reverse_status, err, cv::Size(21, 21), 3,
                    cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
                
                for (size_t i = 0; i != status.size(); i++) 
                    status[i] = (status[i] && reverse_status[i] && utils::dist(cur_image0_pts[i], reverse_image0_pts[i]) <= 0.5) ? 1 : 0;
            }
#endif
            for(size_t i = 0; i != status.size(); i++) 
                if (status[i] && !utils::inside(cur_image1, cur_image1_pts[i]))
                    status[i] = 0;

            cur_image1_ids = cur_image0_ids;
            utils::resizeVector(cur_image1_pts, status);
            utils::resizeVector(cur_image1_ids, status);


            if (!ImageProcessor::undistortImagePoints(1, cur_image1_pts, cur_image1_unpts))
            {
                ROS_WARN_STREAM("No tracking feature in right image!");
            }
#if 0
            ImageProcessor::showUndistortImage(1, cur_image1, cur_image1_pts, cur_image1_unpts);
#endif
            calcPointsVelecity(cur_image1_ids, cur_image1_unpts, pre_image1_ids_unpts_map, cur_image1_ids_unpts_map, image1_unpts_vel);
        }
    }

    if (shared_pool->is_show_track)
    {
        drawTrack(cur_image0, cur_image1);
        RosMaster::pubTrackImage(cur_frame_time, imageTrack);
    }
    
    pre_image0 = cur_image0;
    pre_image0_pts = cur_image0_pts;
    pre_frame_time = cur_frame_time;


    {
#if SAVE_POINT_TRACKING_COST
        static std::ofstream fout(shared_pool->output_path + "/point_tracking_costs.txt", std::ofstream::out);
        fout << point_track_tictoc.toc() << std::endl;
#endif
        // static utils::TimeCost avg_point_track_cost;
        // printf("average point tracking cost: %.6f ms\n", avg_point_track_cost.avgCost(point_track_tictoc.toc()));
        // printf("max tracking cost: %.6f ms\n", avg_point_track_cost.maxCost(point_track_tictoc.toc()));
    }


    /* 5.store all tracking features */
    FeaturePoints feature_points;
    for (size_t i = 0; i != cur_image0_ids.size(); i++)
    {
        int feature_id = cur_image0_ids[i];
        int camera_id = 0;
        double x, y, z;
        x = cur_image0_unpts[i].x;
        y = cur_image0_unpts[i].y;
        z = 1;
        double p_u, p_v;
        p_u = cur_image0_pts[i].x;
        p_v = cur_image0_pts[i].y;
        double v_x, v_y;
        v_x = image0_unpts_vel[i].x;
        v_y = image0_unpts_vel[i].y;

        Eigen::Matrix<double, 7, 1> xyz_uv_vel;
        xyz_uv_vel << x, y, z, p_u, p_v, v_x, v_y;
        feature_points[feature_id].emplace_back(camera_id, xyz_uv_vel);
    }

    if (!image1.empty() && shared_pool->is_use_stereo)
    {
        for (size_t i = 0; i != cur_image1_ids.size(); i++)
        {
            int feature_id = cur_image1_ids[i];
            int camera_id = 1;
            double x, y, z;
            x = cur_image1_unpts[i].x;
            y = cur_image1_unpts[i].y;
            z = 1;
            double p_u, p_v;
            p_u = cur_image1_pts[i].x;
            p_v = cur_image1_pts[i].y;
            double v_x, v_y;
            v_x = image1_unpts_vel[i].x;
            v_y = image1_unpts_vel[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_vel;
            xyz_uv_vel << x, y, z, p_u, p_v, v_x, v_y;
            feature_points[feature_id].emplace_back(camera_id, xyz_uv_vel);
        }
    }

    frame_features = std::make_pair(cur_frame_time, feature_points);

    return true;
}


/** \brief Set a mask map for successfully tracked feature points */
void FeatureTracker::setMask(cv::Mat &mask)
{
    if (shared_pool->is_fisheye)
        mask = shared_pool->fisheye_mask.clone();
    else
        mask = cv::Mat(cur_image0.rows, cur_image0.cols, CV_8UC1, cv::Scalar(255));

    /* prefer to keep features that are tracked for long time */
    std::vector<CntIdPts> cnt_id_pts;

    for (size_t i = 0; i != cur_image0_pts.size(); i++)
        cnt_id_pts.push_back(std::make_pair(track_counters[i], std::make_pair(cur_image0_ids[i], cur_image0_pts[i])));

    sort(cnt_id_pts.begin(), cnt_id_pts.end(), [](const CntIdPts &a, const CntIdPts &b) {
        return a.first > b.first;
    });

    /* clear */
    cur_image0_pts.clear();
    cur_image0_ids.clear();
    track_counters.clear();

    /* regenerate */
    for (auto &pt : cnt_id_pts)
    {
        if (mask.at<uchar>(pt.second.second) == 255)
        {
            cur_image0_ids.push_back(pt.second.first);
            cur_image0_pts.push_back(pt.second.second);
            track_counters.push_back(pt.first);
            cv::circle(mask, pt.second.second, shared_pool->min_feature_distance, 0, -1);
        }
    }
}


/** \brief Reset hash map: feature_id --> feature_point.
 * @param ids Input vector of feature ids.
 * @param unpts Input vector of 2d fearure points.
 * @param ids_unpts_map Output vector of hash map : feature_id --> feature_point.
 */
void FeatureTracker::mapIdsPoints(const std::vector<int> &ids, const std::vector<cv::Point2f> &unpts, std::map<int, cv::Point2f> &ids_unpts_map)
{
    ids_unpts_map.clear();
    for (size_t i = 0; i != ids.size(); i++)
    {
        ids_unpts_map.insert(std::make_pair(ids.at(i),  unpts.at(i)));
    }
}


/** \brief Calculate the average velocity of feature points in two successive image.
 * @param ids Input vector of feature ids.
 * @param unpts Input vector of 2d fearure points.
 * @param pre_image_ids_unpts_map Input vector of hash map in previous image : feature_id --> feature_point.
 * @param cur_image_ids_unpts_map Input vector of hash map in current image : feature_id --> feature_point.
 * @param image_unpts_vel Output vector of the average velocity of feature points in two successive image.
 */
void FeatureTracker::calcPointsVelecity(const std::vector<int> &ids, const std::vector<cv::Point2f> &unpts, 
    std::map<int, cv::Point2f> &pre_image_ids_unpts_map, std::map<int, cv::Point2f> &cur_image_ids_unpts_map,
    std::vector<cv::Point2f> &image_unpts_vel)
{
    mapIdsPoints(ids, unpts, cur_image_ids_unpts_map);

    if (!cur_image_ids_unpts_map.empty())
    {
        double vel_x, vel_y;
        double dt = cur_frame_time - pre_frame_time;
        for (size_t i = 0; i != ids.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator iter = pre_image_ids_unpts_map.find(ids[i]);
            if (iter != pre_image_ids_unpts_map.end())
            {
                vel_x = (unpts[i].x - iter->second.x) / dt;
                vel_y = (unpts[i].y - iter->second.y) / dt;
                image_unpts_vel.emplace_back(vel_x, vel_y);
            }
            else
                image_unpts_vel.emplace_back(0.0, 0.0);
        }
    }
    else
    {
        for (size_t i = 0; i != ids.size(); i++)
            image_unpts_vel.emplace_back(0.0, 0.0);
    }
    
    pre_image_ids_unpts_map = cur_image_ids_unpts_map;
}


/** \brief Draw the tracking results
 * @param image0 Input left image.
 * @param image1 Input right image.
 */
void FeatureTracker::drawTrack(const cv::Mat &image0, const cv::Mat &image1)
{
    if (!image1.empty() && shared_pool->is_use_stereo)
        cv::hconcat(image0, image1, imageTrack); // concat two images
    else
        imageTrack = image0.clone();
    cv::cvtColor(imageTrack, imageTrack, cv::COLOR_GRAY2RGB);

    for (size_t j = 0; j != cur_image0_pts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * (track_counters[j] - 1) / 20);
        /* As the track counts increase, feature color: blue --> red */
        cv::circle(imageTrack, cur_image0_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2); // color format: BGR
    }
    if (!image1.empty() && shared_pool->is_use_stereo)
    {
        for (size_t i = 0; i != cur_image1_pts.size(); i++)
        {
            /* set feature color as green */
            cv::circle(imageTrack, cur_image1_pts[i] + cv::Point2f(image0.cols, 0), 2, cv::Scalar(0, 255, 0), 2);
        }
    }
    
    std::map<int, cv::Point2f>::iterator iter;
    for (size_t i = 0; i != cur_image0_ids.size(); i++)
    {
        int id = cur_image0_ids[i];
        iter = pre_image0_ids_pts_map.find(id);
        if (iter != pre_image0_ids_pts_map.end())
        {
            /* draw arrows in left camera image based on the tracking points */
            cv::arrowedLine(imageTrack, cur_image0_pts[i], iter->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }

    mapIdsPoints(cur_image0_ids, cur_image0_pts, pre_image0_ids_pts_map);

    // namedWindow("KLT tracking", cv::WINDOW_NORMAL);
    // cv::imshow("tracking", imageTrack);
    // cv::waitKey(2);

    // static std::string image_output_path = shared_pool->output_path + "/tracking_point.png";
    // cv::imwrite(image_output_path, imageTrack);



    // static int draw_frame_count = 0;

    // // std::vector<int> seq_set = {530, 1045, 1145,1890};
    // std::vector<int> seq_set = {50}; // real

    // static int count = 0;
    // if (++draw_frame_count == seq_set[count])
    // {
    //     std::string image_output_path = shared_pool->output_path + "/tracking_point" + std::to_string(++count) + ".png";
    //     cv::imwrite(image_output_path, imageTrack);
    // }
}
