#include <fstream>

#include "catkin_settings.h"
#include "odom_utils/utils.h"
#include "estimator/paramPool.h"
#include "rosMaster/rosMaster.h"
#include "line_descriptor.hpp"

#if ENABLE_CUDA
    #include <opencv2/cudaoptflow.hpp>
    #include <opencv2/cudaimgproc.hpp>
    #include <opencv2/cudaarithm.hpp>
#endif

#include "visual/imageProcessor.h"
#include "visual/lineFeatureTracker_DDM.h"


#define USE_EDLINES 0
#define PARALLEL_LINE_DMM 1
#define DETECTING_TEST 0
#define MATCHING_TEST 0
#define SAVE_LINE_TRACKING_COST 0

extern std::unique_ptr<ParamPool> shared_pool;

void LineFeatureTrackerDDM::clearState()
{
    new_line_id = 0;

    pre_frame_time = 0.0;
    pre_image0 = cv::Mat();
    pre_image0_line_ids.clear();
    pre_image0_lines.clear();
    pre_image0_describers = cv::Mat();
    pre_image0_track_counters.clear();

    cur_frame_time = 0.0;
    cur_image0 = cv::Mat();
    cur_image1 = cv::Mat();
    cur_image0_line_ids.clear();
    cur_image1_line_ids.clear();
    cur_image0_lines.clear();
    cur_image1_lines.clear();
    cur_image0_describers = cv::Mat();
    cur_image1_lbd_describers = cv::Mat();


    matched_cur_image0_line_ids.clear();
    matched_cur_image0_lines.clear();
    matched_cur_image0_describers = cv::Mat();
    matched_cur_image0_track_counters.clear();

    matched_cur_image1_line_ids.clear();
    matched_cur_image1_lines.clear();
    matched_cur_image1_lbd_describers = cv::Mat();
    matched_cur_image1_track_counters.clear();


    imageTrack = cv::Mat();
}


void LineFeatureTrackerDDM::setParameters()
{
    // nothing
}


/** \brief track feature between the previous left camera image and current left camera image,
 *                       between the current left camera image and current right camera image.
 * @param image0_time left camera image time.
 * @param image0 current left camera image.
 * @param image1 current right camera image
 */
bool LineFeatureTrackerDDM::trackImage(const double image0_time, const cv::Mat &image0, const cv::Mat &image1)
{
    utils::TicToc line_track_tictoc;

    cur_image0 = image0.clone();
    ImageProcessor::undistortImage(0, image0, cur_image0);

    if (shared_pool->is_equalize)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(cur_image0, cur_image0);
    }

    cur_frame_time = image0_time;

    cur_image0_line_ids.clear();
    cur_image0_lines.clear();
    cur_image0_describers = cv::Mat();

    matched_cur_image0_line_ids.clear();
    matched_cur_image0_lines.clear();
    matched_cur_image0_describers = cv::Mat();
    matched_cur_image0_track_counters.clear();

    /* 1. Detect line features using LSD/EDlines detector */    
#if PARALLEL_LINE_DMM
    if (!image1.empty() && shared_pool->is_use_stereo)
    {
        parallel_thread = std::thread([&]() -> void {
            cur_image1 = image1.clone();
            ImageProcessor::undistortImage(1, image1, cur_image1);

            if (shared_pool->is_equalize)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
                clahe->apply(cur_image1, cur_image1);
            }

            cur_image1_line_ids.clear();
            cur_image1_lines.clear();
            cur_image1_lbd_describers = cv::Mat();
            matched_cur_image1_line_ids.clear();
            matched_cur_image1_lines.clear();
            matched_cur_image1_lbd_describers = cv::Mat();
            matched_cur_image1_track_counters.clear();

            detectLine(cur_image1, cur_image1_lines);

            describeLine(cur_image1, cur_image1_lines, cur_image1_lbd_describers);
        });
    }
#endif

    utils::TicToc detect_tictoc;
    detectLine(cur_image0, cur_image0_lines);
    // std::cout << "lsd costs: " << detect_tictoc.toc() << std::endl;
    // static utils::TimeCost avg_detect_cost;
    // printf("average left detect cost: %.6f ms\n", avg_detect_cost.avgCost(detect_tictoc.toc()));

    /* 2. Describe line features using LBD descriptor */
    utils::TicToc lbd_tictoc;
    describeLine(cur_image0, cur_image0_lines, cur_image0_describers); 
    // std::cout << "lbd costs: " << lbd_tictoc.toc() << std::endl;
    // static utils::TimeCost avg_lbd_cost;
    // printf("average left lbd cost: %.6f ms\n", avg_lbd_cost.avgCost(lbd_tictoc.toc()));

    if (cur_image0_lines.empty())
    {
        pre_image0_line_ids.clear();
        pre_image0_lines.clear();
        pre_image0_describers = cv::Mat();
    }

    /* 3. Match line features using the binary descriptor matcher (KNN) */
    if (!pre_image0_lines.empty())
    {
        utils::TicToc match_tictoc;
        std::vector<cv::DMatch> line_matches;
        matchLine(cur_image0_describers, pre_image0_describers, line_matches);
        // std::cout << "match costs: " << match_tictoc.toc() << std::endl;
        // static utils::TimeCost avg_match_cost;
        // printf("average left match cost: %.6f ms\n", avg_match_cost.avgCost(match_tictoc.toc()));
        // static utils::TimeCost avg_lbd_match_cost;
        // printf("average left lbd + match cost: %.6f ms\n", avg_lbd_match_cost.avgCost(lbd_tictoc.toc()));

        std::vector<cv::DMatch> good_matches;
        checkGoodMatches(pre_image0_lines, cur_image0_lines, line_matches, good_matches);

        std::vector<uchar> track_status(cur_image0_lines.size(), 0);
        for (auto &good_match : good_matches) 
        {
            matched_cur_image0_line_ids.push_back(pre_image0_line_ids[good_match.trainIdx]);
            matched_cur_image0_lines.push_back(cur_image0_lines[good_match.queryIdx]);
            matched_cur_image0_describers.push_back(cur_image0_describers.row(good_match.queryIdx));
            matched_cur_image0_track_counters.push_back(pre_image0_track_counters[good_match.trainIdx] + 1);
            track_status[good_match.queryIdx] = 1;
        }

#if MATCHING_TEST
        static size_t total_match_line_num = 0;
        static size_t total_good_match_line_num = 0;
        static size_t total_length = 0;
        total_match_line_num += line_matches.size();
        total_good_match_line_num += good_matches.size();
        for (auto l : matched_cur_image0_lines)
            total_length += l.lineLength;
        std::cout << "total_match_line_num = " << total_match_line_num << std::endl;
        std::cout << "total_good_match_line_num = " << total_good_match_line_num << std::endl;
        std::cout << "average len = " << 1.0 * total_length / total_good_match_line_num << std::endl;
        std::cout << "prop = " << 1.0 * total_good_match_line_num / total_match_line_num << std::endl;
#endif

        randomQuickSort(matched_cur_image0_track_counters, 0, matched_cur_image0_track_counters.size() - 1);

        int new_line_cnt = shared_pool->min_line_feature_count - static_cast<int>(matched_cur_image0_lines.size());
        for (size_t i = 0; i < cur_image0_lines.size() && new_line_cnt > 25; i++)
        {
            if (!track_status[i])
            {
                matched_cur_image0_line_ids.push_back(new_line_id++);
                matched_cur_image0_lines.push_back(cur_image0_lines[i]);
                matched_cur_image0_describers.push_back(cur_image0_describers.row(i));
                matched_cur_image0_track_counters.push_back(1);
                new_line_cnt--;
            }
        }
    }
    else
    {
        for (size_t i = 0; i < cur_image0_lines.size(); i++)
        {
            matched_cur_image0_line_ids.push_back(new_line_id++);
            matched_cur_image0_track_counters.push_back(0);
        }
        matched_cur_image0_lines = cur_image0_lines;
        matched_cur_image0_describers = cur_image0_describers;
    }

    pre_image0 = cur_image0;

    
    if (!image1.empty() && shared_pool->is_use_stereo)
    {
#if 1
#if PARALLEL_LINE_DMM
        parallel_thread.join();
#else
        utils::TicToc undistorted_tictoc;
        cur_image1 = image1.clone();
        ImageProcessor::undistortImage(1, image1, cur_image1);

        if (shared_pool->is_equalize)
        {
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
            clahe->apply(cur_image1, cur_image1);
        }

        cur_image1_line_ids.clear();
        cur_image1_lines.clear();
        cur_image1_lbd_describers = cv::Mat();
        matched_cur_image1_line_ids.clear();
        matched_cur_image1_lines.clear();
        matched_cur_image1_lbd_describers = cv::Mat();
        matched_cur_image1_track_counters.clear();

        utils::TicToc detect_tictoc;
        detectLine(cur_image1, cur_image1_lines);
        // std::cout << "lsd costs: " << detect_tictoc.toc() << std::endl;
        // static utils::TimeCost avg_right_lsd_cost;
        // printf("average right lsd cost: %.6f ms\n", avg_right_lsd_cost.avgCost(detect_tictoc.toc()));

        utils::TicToc lbd_tictoc;
        describeLine(cur_image1, cur_image1_lines, cur_image1_lbd_describers);
        // std::cout << "lbd costs: " << lbd_tictoc.toc() << std::endl;
        // static utils::TimeCost avg_right_lbd_cost;
        // printf("average right lbd cost: %.6f ms\n", avg_right_lbd_cost.avgCost(lbd_tictoc.toc()));
#endif

        utils::TicToc match_tictoc;
        std::vector<cv::DMatch> line_matches;
        matchLine(cur_image1_lbd_describers, matched_cur_image0_describers, line_matches);

        // static utils::TimeCost avg_right_match_cost;
        // printf("average right match cost: %.6f ms\n", avg_right_match_cost.avgCost(match_tictoc.toc()));

        std::vector<cv::DMatch> good_matches;
        checkGoodMatches(matched_cur_image0_lines, cur_image1_lines, line_matches, good_matches);
        
        for (auto &good_match : good_matches) 
        {
            matched_cur_image1_lines.push_back(cur_image1_lines[good_match.queryIdx]);
            matched_cur_image1_line_ids.push_back(matched_cur_image0_line_ids[good_match.trainIdx]);
            matched_cur_image1_lbd_describers.push_back(cur_image1_lbd_describers.row(good_match.queryIdx));
            matched_cur_image1_track_counters.push_back(matched_cur_image0_track_counters[good_match.trainIdx]);
        }
#else

        utils::TicToc undistorted_tictoc;
        cur_image1 = image1.clone();
        ImageProcessor::undistortImage(1, image1, cur_image1);

        if (shared_pool->is_equalize)
        {
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
            clahe->apply(cur_image1, cur_image1);
        }

        cur_image1_line_ids.clear();
        cur_image1_lines.clear();
        cur_image1_lbd_describers = cv::Mat();
        matched_cur_image1_line_ids.clear();
        matched_cur_image1_lines.clear();
        matched_cur_image1_lbd_describers = cv::Mat();
        matched_cur_image1_track_counters.clear();

        std::vector<cv::Point2f> cur_image0_line_pts;
        std::vector<cv::Point2f> cur_image1_line_pts;

        auto iter = std::find_if(matched_cur_image0_track_counters.begin(), matched_cur_image0_track_counters.end(),
            [](int &track_counter) { return track_counter < 1; });
        int n = static_cast<int>(iter - matched_cur_image0_track_counters.begin());

        if (n > 0)
        {
            for (int i = 0; i != n; i++)
            {
                cur_image0_line_pts.push_back(matched_cur_image0_lines[i].getStartPoint());
                cur_image0_line_pts.push_back(matched_cur_image0_lines[i].getEndPoint());
            }
            std::vector<uchar> status;
            std::vector<float> err;
            utils::TicToc klt_tictoc;
#if ENABLE_CUDA
            cv::cuda::GpuMat cur_gpu_image0(cur_image0);
            cv::cuda::GpuMat cur_gpu_image1(cur_image1);
            cv::cuda::GpuMat cur_gpu_image0_line_pts(cur_image0_line_pts);
            cv::cuda::GpuMat cur_gpu_image1_line_pts(cur_image1_line_pts);
            cv::cuda::GpuMat gpu_status(status);
            cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> ptr_sparse_pyrLK_flow = 
                cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(21, 21), 3, 30, false);
            ptr_sparse_pyrLK_flow->calc(cur_gpu_image0, cur_gpu_image1, cur_gpu_image0_line_pts, cur_gpu_image1_line_pts, gpu_status);

            cur_image1_line_pts = std::vector<cv::Point2f>(cur_gpu_image1_line_pts.cols);
            cur_gpu_image1_line_pts.download(cur_image1_line_pts);
            status = std::vector<uchar>(gpu_status.cols);
            gpu_status.download(status);

            // flow back
            cv::cuda::GpuMat reverse_gpu_status;
            cv::cuda::GpuMat reverse_gpu_image0_line_pts = cur_gpu_image0_line_pts;
            cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> ptr_reverse_sparse_pyrLK_flow = 
                cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(21, 21), 1, 30, true);
            ptr_reverse_sparse_pyrLK_flow->calc(cur_gpu_image1, cur_gpu_image0, cur_gpu_image1_line_pts, reverse_gpu_image0_line_pts, reverse_gpu_status);

            std::vector<cv::Point2f> reverse_image0_line_pts(reverse_gpu_image0_line_pts.cols);
            reverse_gpu_image0_line_pts.download(reverse_image0_line_pts);

            std::vector<uchar> reverse_status(reverse_gpu_status.cols);
            reverse_gpu_status.download(reverse_status);
#else
            cv::calcOpticalFlowPyrLK(cur_image0, cur_image1, cur_image0_line_pts, cur_image1_line_pts, status, err, cv::Size(21, 21), 3);

            // flow back
            std::vector<uchar> reverse_status;
            std::vector<cv::Point2f> reverse_image0_line_pts = cur_image1_line_pts;
            cv::calcOpticalFlowPyrLK(cur_image1, cur_image0, cur_image1_line_pts, reverse_image0_line_pts, reverse_status, err, cv::Size(21, 21), 1,
                cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
#endif
            // std::cout << "optical flow costs: " << klt_tictoc.toc() << std::endl;
            static utils::TimeCost avg_right_klt_cost;
            // printf("average right klt cost: %.6f ms\n", avg_right_klt_cost.avgCost(klt_tictoc.toc()));


            utils::TicToc refine_tictoc;
            for (int i = 0, j = 0; i != n; j = (++i << 1))
            {
                if (!(status[j] && status[j + 1] && reverse_status[j] && reverse_status[j + 1]))
                {
                    status[j] = status[j + 1] = 0;
                    continue;
                }

                float diff_len1 = fabs(matched_cur_image0_lines[i].lineLength 
                    - utils::dist(reverse_image0_line_pts[j], reverse_image0_line_pts[j + 1]));
                float diff_len2 = fabs(matched_cur_image0_lines[i].lineLength 
                    - utils::dist(cur_image1_line_pts[j], cur_image1_line_pts[j + 1]));

                auto diff_point1 = reverse_image0_line_pts[j + 1] - reverse_image0_line_pts[j];
                auto tmp_angle1 = std::atan2(diff_point1.y, diff_point1.x);
                while (tmp_angle1 < -M_PI)
                    tmp_angle1 += M_PI * 2;
                float diff_angle1 = fabs(tmp_angle1 - matched_cur_image0_lines[i].angle);

                auto diff_point2 = cur_image1_line_pts[j + 1] - cur_image1_line_pts[j];
                auto tmp_angle2 = std::atan2(diff_point2.y, diff_point2.x);
                while (tmp_angle2 < -M_PI)
                    tmp_angle2 += M_PI * 2;
                float diff_angle2 = fabs(tmp_angle2 - matched_cur_image0_lines[i].angle);
                
                float dist_start = utils::dist(reverse_image0_line_pts[j], cur_image0_line_pts[j]);
                float dist_end = utils::dist(reverse_image0_line_pts[j + 1], cur_image0_line_pts[j + 1]);
                if ((diff_angle1 > 0.05 && M_PI - diff_angle1 > 0.05) ||
                    (diff_angle2 > 0.05 && M_PI - diff_angle2 > 0.05) ||
                    diff_len1 > 10 || diff_len2 > 20 || dist_start > 10 || dist_end > 10)
                {
                    status[j] = status[j + 1] = 0;
                }
            }
            static utils::TimeCost avg_right_refine_cost;
            // printf("average right refine cost: %.6f ms\n", avg_right_refine_cost.avgCost(refine_tictoc.toc()));

            for (int i = 0, j = 0; i != n; j = (++i << 1))
            {
                if (status[j])
                {
                    KeyLine keyline;
                    keyline.startPointX = cur_image1_line_pts[j].x;
                    keyline.startPointY = cur_image1_line_pts[j].y;
                    keyline.endPointX = cur_image1_line_pts[j + 1].x;
                    keyline.endPointY = cur_image1_line_pts[j + 1].y;

                    matched_cur_image1_lines.push_back(keyline);
                    matched_cur_image1_line_ids.push_back(matched_cur_image0_line_ids[i]);
                    matched_cur_image1_track_counters.push_back(matched_cur_image0_track_counters[i]);
                }
            }
        }
#endif
    }

    // std::cout << "line feature tracking costs: " << line_track_tictoc.toc() << std::endl;

    pre_image0_line_ids = matched_cur_image0_line_ids;
    pre_image0_lines = matched_cur_image0_lines;
    pre_image0_describers = matched_cur_image0_describers;
    pre_image0_track_counters = matched_cur_image0_track_counters;


    {
#if SAVE_LINE_TRACKING_COST
        static std::ofstream fout(shared_pool->output_path + "/line_tracking_costs.txt", std::ofstream::out);
        fout << line_track_tictoc.toc() << std::endl;
#endif
        // static utils::TimeCost avg_line_track_cost;
        // printf("average line tracking cost: %.6f ms\n", avg_line_track_cost.avgCost(line_track_tictoc.toc()));
        // printf("max tracking cost: %.6f ms\n", avg_line_track_cost.maxCost(line_track_tictoc.toc()));
    }


    if (shared_pool->is_show_track)
    {
        drawTrack(cur_image0, cur_image1);
        RosMaster::pubLineTrackImage(cur_frame_time, imageTrack);
    }

    pre_frame_time = cur_frame_time;


    FeatureLines feature_lines;
    for (size_t i = 0; i != matched_cur_image0_line_ids.size(); i++)
    {
        int line_id = matched_cur_image0_line_ids[i];
        int camera_id = 0;
        const cv::Matx33d &K = ImageProcessor::K_vec[camera_id];
        double s_x, s_y;
        s_x = (matched_cur_image0_lines[i].startPointX - K(0, 2)) / K(0, 0);
        s_y = (matched_cur_image0_lines[i].startPointY - K(1, 2)) / K(1, 1);
        double e_x, e_y;
        e_x = (matched_cur_image0_lines[i].endPointX - K(0, 2)) / K(0, 0);
        e_y = (matched_cur_image0_lines[i].endPointY - K(1, 2)) / K(1, 1);
        Eigen::Vector4d xy0_xy1;
        xy0_xy1 << s_x, s_y, e_x, e_y;
        feature_lines[line_id].emplace_back(camera_id, xy0_xy1);
    }

    if (!image1.empty() && shared_pool->is_use_stereo)
    {
        for (size_t i = 0; i != matched_cur_image1_line_ids.size(); i++)
        {
            // continue;
            int line_id = matched_cur_image1_line_ids[i];
            int camera_id = 1;
            const cv::Matx33d &K = ImageProcessor::K_vec[camera_id];
            double s_x, s_y;
            s_x = (matched_cur_image1_lines[i].startPointX - K(0, 2)) / K(0, 0);
            s_y = (matched_cur_image1_lines[i].startPointY - K(1, 2)) / K(1, 1);
            double e_x, e_y;
            e_x = (matched_cur_image1_lines[i].endPointX - K(0, 2)) / K(0, 0);
            e_y = (matched_cur_image1_lines[i].endPointY - K(1, 2)) / K(1, 1);
            Eigen::Vector4d xy0_xy1;
            xy0_xy1 << s_x, s_y, e_x, e_y;
            feature_lines[line_id].emplace_back(camera_id, xy0_xy1);
        }
    }

    frame_linefeatures = std::make_pair(cur_frame_time, feature_lines);

    return true;
}


#if DETECTING_TEST
/** \brief Detect line features in the image.
 * @param image the image to be detect.
 * @param image_lines Output the detected lsd lines.
 */
void LineFeatureTrackerDDM::detectLine(const cv::Mat &image, std::vector<KeyLine> &image_lines)
{
#if !USE_EDLINES
    utils::TicToc detect_tictoc;
    cv::line_descriptor_custom::LSDParam lsd_param;
    lsd_param.scale        = 0.5;   //0.8   The scale of the image that will be used to find the lines. Range (0..1].
    lsd_param.sigma_scale  = 0.6;	//0.6  	Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
    lsd_param.quant        = 2.0;	//2.0   Bound to the quantization error on the gradient norm
    lsd_param.ang_th       = 22.5;	//22.5	Gradient angle tolerance in degrees
    lsd_param.log_eps      = 1.0;	//0		Detection threshold: -log10(NFA) > log_eps. Used only when advance refinement is chosen
    lsd_param.density_th   = 0.6;	//0.7	Minimal density of aligned region points in the enclosing rectangle.
    lsd_param.n_bins       = 1024;	//1024 	Number of bins in pseudo-ordering of gradient modulus.
    float min_length = 0.125 * std::min(image.rows, image.cols);
    cv::Ptr<cv::line_descriptor_custom::LSDDetector> lsd_detecter
        = cv::line_descriptor_custom::LSDDetector::createLSDDetector(lsd_param);
    lsd_detecter->detect(image, image_lines, 2, 1, min_length, cv::Mat());

    // lsd_param.scale        = 0.5;   //0.8   The scale of the image that will be used to find the lines. Range (0..1].
    // lsd_param.sigma_scale  = 0.6;	//0.6  	Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
    // lsd_param.quant        = 2.0;	//2.0   Bound to the quantization error on the gradient norm
    // lsd_param.ang_th       = 22.5;	//22.5	Gradient angle tolerance in degrees
    // lsd_param.log_eps      = 1.0;	//0		Detection threshold: -log10(NFA) > log_eps. Used only when advance refinement is chosen
    // lsd_param.density_th   = 0.6;	//0.7	Minimal density of aligned region points in the enclosing rectangle.
    // lsd_param.n_bins       = 1024;	//1024 	Number of bins in pseudo-ordering of gradient modulus.
    // float min_length = 0.125 * std::min(image.rows, image.cols);

    // cv::Ptr<cv::line_descriptor_custom::LSDDetector> lsd_detecter
    //     = cv::line_descriptor_custom::LSDDetector::createLSDDetector(lsd_param);
    // lsd_detecter->detect(image, image_lines, 2, 1, min_length, cv::Mat());

    removeEdgeLine(image, image_lines);

// #if 1

//     static int lsd_draw_frame_count = 0;
//     std::vector<int> edline_seq_set = {50, 550, 1045, 1610, 1890}; // MH_04 corridor1 kitti00
//     static int lsd_count = 0;
//     if (++lsd_draw_frame_count == edline_seq_set[lsd_count])
//     {
//         std::cout << "lsd costs: " << detect_tictoc.toc() << std::endl;
//         double total_len = 0;
//         int long_line_num = 0;
//         for (size_t i = 0; i < image_lines.size(); ++i)
//         {
//             total_len += image_lines[i].lineLength;
//         }
//         ++lsd_count;
//         std::cout << "nums: " << image_lines.size() << std::endl;
//         std::cout << "avg len: " << total_len / image_lines.size() << std::endl;
//         std::cout << "long line prop: " << (double) long_line_num / image_lines.size() << std::endl;
//         std::cout << std::endl;
//     }
// #endif

#elif USE_EDLINES
    utils::TicToc edlines_detect_tictoc;
    cv::line_descriptor_custom::BinaryDescriptor::EDLineParam edline_param;
    edline_param.ksize = 5;
    edline_param.sigma = 1.0;
    edline_param.gradientThreshold = 32;
    edline_param.anchorThreshold = 16;
    edline_param.scanIntervals = 1;
    edline_param.minLineLen = 0.125 * (std::min(image.cols, image.rows));
    edline_param.lineFitErrThreshold = 3.0;

    cv::Ptr<cv::line_descriptor_custom::BinaryDescriptor> edlines_detector
        = cv::line_descriptor_custom::BinaryDescriptor::createBinaryDescriptor(edline_param);
    edlines_detector->detect(image, image_lines, cv::Mat());
    removeEdgeLine(image, image_lines);

#if 0
    cv::Mat edlines_detect_imageTrack = image.clone();
    cv::Mat edlines_detect_background = cv::Mat(image.rows, image.cols, CV_8UC1, 255); // lines mask;
    cv::cvtColor(edlines_detect_imageTrack, edlines_detect_imageTrack, cv::COLOR_GRAY2RGB);
    cv::cvtColor(edlines_detect_background, edlines_detect_background, cv::COLOR_GRAY2RGB);

    static int edline_draw_frame_count = 0;
    std::vector<int> edline_seq_set = {50, 550, 1045, 1610, 1890}; // MH_04 corridor1 kitti00
    static int edlines_count = 0;
    if (++edline_draw_frame_count == edline_seq_set[edlines_count])
    {
        std::cout << "edlines costs: " << edlines_detect_tictoc.toc() << std::endl;
        double total_len = 0;
        int long_line_num = 0;
        for (size_t i = 0; i < image_lines.size(); ++i)
        {
            total_len += image_lines[i].lineLength;
            if (image_lines[i].lineLength >= 25)
            {
                ++long_line_num;
                cv::line(edlines_detect_imageTrack, image_lines[i].getStartPoint(), image_lines[i].getEndPoint(), cv::Scalar(0, 220, 0), 2); // color format: BGR
                cv::line(edlines_detect_background, image_lines[i].getStartPoint(), image_lines[i].getEndPoint(), cv::Scalar(0, 220, 0), 2); // color format: BGR
            }
            else
            {
                cv::line(edlines_detect_imageTrack, image_lines[i].getStartPoint(), image_lines[i].getEndPoint(), cv::Scalar(0, 0, 255), 2); // color format: BGR
                cv::line(edlines_detect_background, image_lines[i].getStartPoint(), image_lines[i].getEndPoint(), cv::Scalar(0, 0, 255), 2); // color format: BGR
            }

        }
        ++edlines_count;
        std::cout << "nums: " << image_lines.size() << std::endl;
        std::cout << "avg len: " << total_len / image_lines.size() << std::endl;
        std::cout << "long line prop: " << (double) long_line_num / image_lines.size() << std::endl;
        std::cout << std::endl;

        std::string image_output_path = shared_pool->output_path + "/edlines_detect_line" + std::to_string(edlines_count) + ".png";
        cv::imwrite(image_output_path, edlines_detect_imageTrack);
        std::string bg_image_output_path = shared_pool->output_path + "/edlines_detect_bg" + std::to_string(edlines_count) + ".png";
        cv::imwrite(bg_image_output_path, edlines_detect_background);
        std::string raw_image_output_path = shared_pool->output_path + "/raw_image" + std::to_string(edlines_count) + ".png";
        cv::imwrite(raw_image_output_path, image);
    }
#endif


#elif 0
    cv::Ptr<cv::ximgproc::FastLineDetector> fld_detecter = cv::ximgproc::createFastLineDetector();
    image_lines.clear();
    fld_detecter->detect(image, image_lines);
    
#endif


    return;
}

#else

/** \brief Detect line features in the image.
 * @param image the image to be detect.
 * @param image_lines Output the detected lsd lines.
 */
void LineFeatureTrackerDDM::detectLine(const cv::Mat &image, std::vector<KeyLine> &image_lines)
{
#if !USE_EDLINES
    utils::TicToc detect_tictoc;
    cv::line_descriptor_custom::LSDParam lsd_param;
    lsd_param.scale        = 0.5;   //0.8   The scale of the image that will be used to find the lines. Range (0..1].
    lsd_param.sigma_scale  = 0.6;	//0.6  	Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
    lsd_param.quant        = 2.0;	//2.0   Bound to the quantization error on the gradient norm
    lsd_param.ang_th       = 22.5;	//22.5	Gradient angle tolerance in degrees
    lsd_param.log_eps      = 1.0;	//0		Detection threshold: -log10(NFA) > log_eps. Used only when advance refinement is chosen
    lsd_param.density_th   = 0.6;	//0.7	Minimal density of aligned region points in the enclosing rectangle.
    lsd_param.n_bins       = 1024;	//1024 	Number of bins in pseudo-ordering of gradient modulus.
    float min_length = 0.125 * std::min(image.rows, image.cols);
    cv::Ptr<cv::line_descriptor_custom::LSDDetector> lsd_detecter
        = cv::line_descriptor_custom::LSDDetector::createLSDDetector(lsd_param);
    lsd_detecter->detect(image, image_lines, 2, 1, min_length, cv::Mat());

    removeEdgeLine(image, image_lines);

#elif USE_EDLINES
    utils::TicToc edlines_detect_tictoc;
    cv::line_descriptor_custom::BinaryDescriptor::EDLineParam edline_param;
    edline_param.ksize = 5;
    edline_param.sigma = 1.0;
    edline_param.gradientThreshold = 32;
    edline_param.anchorThreshold = 16;
    edline_param.scanIntervals = 1;
    edline_param.minLineLen = 0.125 * (std::min(image.cols, image.rows));
    edline_param.lineFitErrThreshold = 3.0;

    cv::Ptr<cv::line_descriptor_custom::BinaryDescriptor> edlines_detector
        = cv::line_descriptor_custom::BinaryDescriptor::createBinaryDescriptor(edline_param);
    edlines_detector->detect(image, image_lines, cv::Mat());
    removeEdgeLine(image, image_lines);

#elif 0
    cv::Ptr<cv::ximgproc::FastLineDetector> fld_detecter = cv::ximgproc::createFastLineDetector();
    image_lines.clear();
    fld_detecter->detect(image, image_lines);
    
#endif


    return;
}
#endif

/** \brief Detect line features in the image.
 * @param image the image to be detect.
 * @param image_lines Input the detected lsd lines.
 * @param image_lbd_describers Output the image line lbd describers.
 */
void LineFeatureTrackerDDM::describeLine(const cv::Mat &image, std::vector<KeyLine> &image_lines, cv::Mat &image_lbd_describers)
{
    cv::line_descriptor_custom::BinaryDescriptor::Params lbd_param;
    lbd_param.widthOfBand_ = 7;

    cv::Ptr<cv::line_descriptor_custom::BinaryDescriptor> lbd_descriptor
        = cv::line_descriptor_custom::BinaryDescriptor::createBinaryDescriptor(lbd_param);
    lbd_descriptor->compute(image, image_lines, image_lbd_describers);

    return;
}


/** \brief Detect line features in the image.
 * @param query_image_lbd_describers Query descriptors.
 * @param train_image_lbd_describers Dataset of descriptors furnished by user.
 * @param lsd_matches Output match results.
 */
void LineFeatureTrackerDDM::matchLine(cv::Mat &query_image_lbd_describers, cv::Mat &train_image_lbd_describers, std::vector<cv::DMatch> &lsd_matches)
{
    cv::Ptr<cv::line_descriptor_custom::BinaryDescriptorMatcher> lbd_matcher = 
        cv::line_descriptor_custom::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    // lbd_matcher->match(cur_image1_lbd_describers, matched_cur_image0_describers, lsd_matches);
    lbd_matcher->match(query_image_lbd_describers, train_image_lbd_describers, lsd_matches);

    return;
}


void LineFeatureTrackerDDM::removeEdgeLine(const cv::Mat &image, std::vector<KeyLine> &image_lines)
{
    std::vector<uchar> bound_status(image_lines.size(), 1);
    int col_thickness = 0.05 * image.cols;
    int row_thickness = 0.05 * image.rows;
    for (size_t i = 0; i != image_lines.size(); i++)
    {
        const KeyLine &cur_line0 = image_lines[i];
        if (cur_line0.startPointX <= col_thickness && cur_line0.endPointX <= col_thickness)
            bound_status[i] = 0;
        if (cur_line0.startPointY <= row_thickness && cur_line0.endPointY <= row_thickness)
            bound_status[i] = 0;
        if (cur_line0.startPointX >= image.cols - col_thickness && cur_line0.endPointX >= image.cols - col_thickness)
            bound_status[i] = 0;
        if (cur_line0.startPointY >= image.rows - row_thickness && cur_line0.endPointY >= image.rows - row_thickness)
            bound_status[i] = 0;
    }
    utils::resizeVector(image_lines, bound_status);
}


void LineFeatureTrackerDDM::checkGoodMatches(
    const std::vector<KeyLine> &train_image_line, const std::vector<KeyLine> &query_image_line, 
    const std::vector<cv::DMatch> &line_matches, std::vector<cv::DMatch> &good_matches)
{
    good_matches.clear();
    for (const auto &line_match : line_matches)
    {
        // std::cout << "distance = " << line_match.distance << std::endl;
        if (line_match.distance > 30)
            continue;
        
        const KeyLine &cur_line0 = train_image_line[line_match.trainIdx];
        const KeyLine &cur_line1 = query_image_line[line_match.queryIdx];

        if (((utils::dist(cur_line1.getStartPoint(), cur_line0.getEndPoint()) < 200 &&
            utils::dist(cur_line1.getEndPoint(), cur_line0.getStartPoint()) < 200) ||
            (utils::dist(cur_line1.getStartPoint(), cur_line0.getStartPoint()) < 200 &&
            utils::dist(cur_line1.getEndPoint(), cur_line0.getEndPoint()) < 200)) &&
            (fabs(cur_line0.angle - cur_line1.angle) < 0.1 || M_PI - fabs(cur_line0.angle - cur_line1.angle) < 0.1))
            good_matches.push_back(line_match);
    }
}


/** \brief fast sort from large to small
 * @param group the vector need to be sorted
 * @param first the first element index
 * @param last the last element index
 */
void LineFeatureTrackerDDM::randomQuickSort(std::vector<int> &group, int first, int last)
{
    if (first >= last)
        return;
    int left = first, right = last;
    while (left < right)
    {
        while (left < right && group[left] > group[right])
            ++left;
        if (left < right)
        {
            std::swap(matched_cur_image0_line_ids[left], matched_cur_image0_line_ids[right]);
            std::swap(matched_cur_image0_lines[left], matched_cur_image0_lines[right]);
            auto tmp = matched_cur_image0_describers.row(left).clone();
            matched_cur_image0_describers.row(left) = matched_cur_image0_describers.row(right) + 0;
            matched_cur_image0_describers.row(right) = tmp + 0;
            std::swap(group[left], group[right--]);
        }
        while (left < right && group[left] > group[right])
            --right;
        if (left < right)
        {
            std::swap(matched_cur_image0_line_ids[left], matched_cur_image0_line_ids[right]);
            std::swap(matched_cur_image0_lines[left], matched_cur_image0_lines[right]);
            auto tmp = matched_cur_image0_describers.row(left).clone();
            matched_cur_image0_describers.row(left) = matched_cur_image0_describers.row(right) + 0;
            matched_cur_image0_describers.row(right) = tmp + 0;
            std::swap(group[left++], group[right]);

        }
    }
    randomQuickSort(group, first, left - 1);
    randomQuickSort(group, right + 1, last);
}


void LineFeatureTrackerDDM::drawTrack(const cv::Mat &image0, const cv::Mat &image1)
{
    if (!image1.empty() && shared_pool->is_use_stereo)
        cv::hconcat(image0, image1, imageTrack); // concat two images
    else
        imageTrack = image0.clone();
    cv::cvtColor(imageTrack, imageTrack, cv::COLOR_GRAY2RGB);

    for (size_t i = 0; i != matched_cur_image0_lines.size(); i++)
    {
        double len = std::min(1.0, 1.0 * (matched_cur_image0_track_counters[i] - 1) / 5);
        /* As the track counts increase, feature color: blue --> red */
        cv::line(imageTrack, matched_cur_image0_lines[i].getStartPoint(), matched_cur_image0_lines[i].getEndPoint(), 
            cv::Scalar(255 * (1 - len), 0, 255 * len), 3); // color format: BGR
    }
    if (!image1.empty() && shared_pool->is_use_stereo)
    {
        for (size_t i = 0; i != matched_cur_image1_lines.size(); i++)
        {
            double len = std::min(1.0, 1.0 * matched_cur_image1_track_counters[i] / 5);
            cv::line(imageTrack, 
                matched_cur_image1_lines[i].getStartPoint() + cv::Point2f(image0.cols, 0), 
                matched_cur_image1_lines[i].getEndPoint() + cv::Point2f(image0.cols, 0), 
                cv::Scalar(255 * (1 - len), 0, 255 * len), 3); // color format: BGR
        }
    }

    // namedWindow("LSD tracking", cv::WINDOW_NORMAL);
    // cv::imshow("LSD tracking", imageTrack);
    // cv::waitKey(2);
}
