#include <fstream>

#include <opencv2/features2d.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "catkin_settings.h"
#if ENABLE_CUDA
    #include <opencv2/cudaoptflow.hpp>
    #include <opencv2/cudaimgproc.hpp>
    #include <opencv2/cudaarithm.hpp>
#endif
#include "line_descriptor.hpp"

#include "odom_utils/utils.h"
#include "estimator/paramPool.h"
#include "rosMaster/rosMaster.h"

#include "visual/lineOpticalFlowTracker.h"
#include "visual/lineFeatureTracker_LOFT.h"
#include "visual/imageProcessor.h"


#define USE_EDLINES 1
#define PARALLEL_LINE_LOFT 0
#define SHOW_LINEFLOW 1
#define DETECTING_TEST 0
#define TRACKING_TEST 1
#define SAVE_LINE_TRACKING_COST 0

extern std::unique_ptr<ParamPool> shared_pool;

void LineFeatureTrackerLOFT::clearState()
{
    new_line_id = 0;

    pre_frame_time = 0.0;
    pre_image0 = cv::Mat();

    cur_frame_time = 0.0;
    cur_image0 = cv::Mat();
    cur_image1 = cv::Mat();
    cur_image0_lines.clear();
    cur_image1_lines.clear();
    cur_image0_line_ids.clear();
    cur_image1_line_ids.clear();
    cur_image0_track_counters.clear();
    cur_image1_track_counters.clear();

    /* greydoor in LOFT, more smaller more strict!!! */
    image0_greydoor = 5.0;
    image1_greydoor = 5.0;
    /* used for uniform distribution line feature!!! */
    lines_mask_thickness = 30; // 15
    /* used to merge line feature!!! */
    lines_index_thickness = 8; // 8

    cur_image0_lineflows.clear();
    cur_image1_lineflows.clear();

    imageTrack = cv::Mat();
}


void LineFeatureTrackerLOFT::setParameters()
{
    // nothing
}


/** \brief track feature between the previous left camera image and current left camera image,
 *                       between the current left camera image and current right camera image.
 * @param image0_time left camera image time.
 * @param image0 current left camera image.
 * @param image1 current right camera image
 */
bool LineFeatureTrackerLOFT::trackImage(const double image0_time, const cv::Mat &image0, const cv::Mat &image1)
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
    cur_image0_lines.clear();

    std::vector<cv::Vec4f> new_lines;


#if PARALLEL_LINE_LOFT
    parallel_thread = std::thread([&]() -> void {
        detectLine(cur_image0, new_lines);

        if (!image1.empty() && shared_pool->is_use_stereo)
        {
            cur_image1 = image1.clone();
            ImageProcessor::undistortImage(1, image1, cur_image1);

            if (shared_pool->is_equalize)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
                clahe->apply(cur_image1, cur_image1);
            }

            cur_image1_lines.clear();
        }
    });
#endif

    utils::TicToc reconstruct_tictoc;

    /* 1.track line feature between previous image and current image */
    if (!pre_image0_lines.empty())
    {
        utils::TicToc loft_tictoc;
        std::vector<uchar> cur_image0_line_status;
        LineOpticalFlowTracker::OpticalFlowMultiLevel(pre_image0, cur_image0, pre_image0_lines, cur_image0_lines, cur_image0_line_status, false, image0_greydoor);
        // std::cout << "left loft costs: " << loft_tictoc.toc() << std::endl;
        // static utils::TimeCost avg_loft_cost;
        // printf("average left loft cost: %.6f ms\n", avg_loft_cost.avgCost(loft_tictoc.toc()));


        utils::TicToc reconstruct_tictoc;
        reconstructLine(cur_image0, pre_image0_lines, cur_image0_lines, cur_image0_line_status);
        // std::cout << "reconstruct costs: " << reconstruct_tictoc.toc() << std::endl;
        // static utils::TimeCost avg_reconstruct_cost;
        // printf("average reconstruct cost: %.6f ms\n", avg_reconstruct_cost.avgCost(reconstruct_tictoc.toc()));


        // ROS_ASSERT(cur_image0_lines.size() < 255);
        utils::resizeVector(cur_image0_lines, cur_image0_line_status);
        utils::resizeVector(cur_image0_line_ids, cur_image0_line_status);
        utils::resizeVector(cur_image0_track_counters, cur_image0_line_status);
#if SHOW_LINEFLOW
        utils::resizeVector(cur_image0_lineflows, cur_image0_line_status);
#endif

#if TRACKING_TEST
        static size_t total_track_line_num = 0;
        static size_t total_good_track_line_num = 0;
        static size_t total_length = 0;
        total_track_line_num += pre_image0_lines.size();
        total_good_track_line_num += cur_image0_lines.size();
        for (auto l : cur_image0_lines)
            total_length += l.getLength();
        std::cout << "total_track_line_num = " << total_track_line_num << std::endl;
        std::cout << "total_good_track_line_num = " << total_good_track_line_num << std::endl;
        std::cout << "average len = " << 1.0 * total_length / total_good_track_line_num << std::endl;
        std::cout << "prop = " << 1.0 * total_good_track_line_num / total_track_line_num << std::endl;
#endif

        if (shared_pool->is_lineflow_back)
        {
            std::vector<Line> reverse_pre_image0_lines;
            std::vector<uchar> reverse_cur_image0_line_status;
            LineOpticalFlowTracker::OpticalFlowMultiLevel(cur_image0, pre_image0, cur_image0_lines, reverse_pre_image0_lines, reverse_cur_image0_line_status, false, image0_greydoor);

            cv::Mat imageMagnitude, imageAngle;
            cv::Mat x_arr, y_arr;
            cv::Sobel(pre_image0, x_arr, CV_32F, 1, 0);
            cv::Sobel(pre_image0, y_arr, CV_32F, 0, 1);
            cv::cartToPolar(x_arr, y_arr, imageMagnitude, imageAngle, true);
            for (size_t i = 0; i < reverse_pre_image0_lines.size(); i++)
            {
                if (!checkGoodLine(reverse_pre_image0_lines[i].key_points, imageMagnitude, imageAngle))
                {
                    reverse_cur_image0_line_status[i] = 0;
                    continue;
                }
            }

            utils::resizeVector(cur_image0_lines, reverse_cur_image0_line_status);
            utils::resizeVector(cur_image0_line_ids, reverse_cur_image0_line_status);
            utils::resizeVector(cur_image0_track_counters, reverse_cur_image0_line_status);
#if SHOW_LINEFLOW
            utils::resizeVector(cur_image0_lineflows, reverse_cur_image0_line_status);
#endif
        }

        for (auto &counter : cur_image0_track_counters)
            counter++;

#if SHOW_LINEFLOW
        for (size_t i = 0; i < cur_image0_lines.size(); ++i)
        {
            cv::Vec4f lineflow{cur_image0_lines[i].sPoint.x, cur_image0_lines[i].sPoint.y, cur_image0_lines[i].ePoint.x, cur_image0_lines[i].ePoint.y};
            cur_image0_lineflows[i].first.push_back(lineflow);
        }
#endif

        // image0_greydoor = 15 * (1.0 - 1.0 * cur_image0_lines.size() / shared_pool->min_line_feature_count);
        // image0_greydoor = std::max(image0_greydoor, 5.0);
        // image0_greydoor = std::min(image0_greydoor, 15.0);
        // std::cout << "image0_greydoor = " << image0_greydoor << std::endl;
    }

#if PARALLEL_LINE_LOFT
    parallel_thread.join();
#endif


    /* 2.extract and supplement line feature */
    int new_line_num = shared_pool->min_line_feature_count - cur_image0_lines.size();
    if (new_line_num > 10)
    {
#if !PARALLEL_LINE_LOFT
        utils::TicToc detect_tictoc;
        std::vector<cv::Vec4f> new_lines;
        detectLine(cur_image0, new_lines);
        // static utils::TimeCost avg_detect_cost;
        // printf("average detect cost: %.6f ms\n", avg_detect_cost.avgCost(detect_tictoc.toc()));
#endif

        cv::Mat lines_mask_image = cv::Mat(cur_image0.rows, cur_image0.cols, CV_8UC1, 255); // used for uniform distribution line feature
        cv::Mat lines_index_image = cv::Mat(cur_image0.rows, cur_image0.cols, CV_8UC1, 255); // used to merge line feature and record the line feature index
        setMask(lines_mask_image, cur_image0_lines); // set mask for the existing lines
        setIndex(lines_index_image, cur_image0_lines); // set the index for the existing lines

        int line_index = cur_image0_lines.size(); // line index in each image. NOT line id!!!
        for (auto new_line : new_lines)
        {
            Line line(new_line);

            /* merge each detected line */
            if (mergeLine(line, cur_image0_lines, lines_index_image, lines_mask_image))
                continue;

            if (new_line_num <= 0)
                continue;

            /* uniform distribution line feature */
            if ((lines_mask_image.at<uchar>(line.mPoint) != 255 && 
                lines_mask_image.at<uchar>(line.sPoint) != 255) ||
                (lines_mask_image.at<uchar>(line.mPoint) != 255 && 
                lines_mask_image.at<uchar>(line.ePoint) != 255))
                continue;

            /* add new lines */
            cur_image0_lines.push_back(line);
            cur_image0_line_ids.push_back(new_line_id++);
            cur_image0_track_counters.push_back(1);
#if SHOW_LINEFLOW
            cv::Vec4f lineflow{line.sPoint.x, line.sPoint.y, line.ePoint.x, line.ePoint.y};
            cur_image0_lineflows.push_back(std::make_pair<std::vector<cv::Vec4f>, cv::Scalar>(std::vector<cv::Vec4f>{lineflow}, cv::Scalar(rand() % 256, rand() % 256, rand() % 256)));
#endif

            new_line_num--;

            /* draw new lines */
            cv::line(lines_mask_image, line.sPoint, line.ePoint, 0, lines_mask_thickness);
            // cv::imshow("lines_mask_image", lines_mask_image);
            // cv::waitKey(2);
            drawLineIndex(line, lines_index_image, line_index++);
        }
    }


    if (!image1.empty() && shared_pool->is_use_stereo)
    {
        cur_image1 = image1.clone();
        ImageProcessor::undistortImage(1, image1, cur_image1);

        if (shared_pool->is_equalize)
        {
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
            clahe->apply(cur_image1, cur_image1);
        }

        cur_image1_lines.clear();

        if (!cur_image0_lines.empty())
        {
            utils::TicToc loft_tictoc;
            std::vector<uchar> cur_image1_line_status;
            LineOpticalFlowTracker::OpticalFlowMultiLevel(cur_image0, cur_image1, cur_image0_lines, cur_image1_lines, cur_image1_line_status, false, image1_greydoor);
            // std::cout << "right loft costs: " << loft_tictoc.toc() << std::endl;
            // static utils::TimeCost avg_loft_cost;
            // printf("average right loft cost: %.6f ms\n", avg_loft_cost.avgCost(loft_tictoc.toc()));

            reconstructLine(cur_image1, cur_image0_lines, cur_image1_lines, cur_image1_line_status);

            for (size_t i = 0; i < cur_image0_track_counters.size(); ++i)
                if (cur_image0_track_counters[i] <= 1)
                    cur_image1_line_status[i] = 0;

            // ROS_ASSERT(cur_image1_lines.size() < 255);
            cur_image1_line_ids = cur_image0_line_ids;
            cur_image1_track_counters = cur_image0_track_counters;
            utils::resizeVector(cur_image1_lines, cur_image1_line_status);
            utils::resizeVector(cur_image1_line_ids, cur_image1_line_status);
            utils::resizeVector(cur_image1_track_counters, cur_image1_line_status);

#if SHOW_LINEFLOW
            cur_image1_lineflows = cur_image0_lineflows;
            utils::resizeVector(cur_image1_lineflows, cur_image1_line_status);
#endif

            if (shared_pool->is_lineflow_back)
            {
                std::vector<Line> reverse_cur_image0_lines;
                std::vector<uchar> reverse_cur_image1_line_status;
                LineOpticalFlowTracker::OpticalFlowMultiLevel(cur_image1, cur_image0, cur_image1_lines, reverse_cur_image0_lines, reverse_cur_image1_line_status, false, image1_greydoor);

                cv::Mat imageMagnitude, imageAngle;
                cv::Mat x_arr, y_arr;
                cv::Sobel(cur_image0, x_arr, CV_32F, 1, 0);
                cv::Sobel(cur_image0, y_arr, CV_32F, 0, 1);
                cv::cartToPolar(x_arr, y_arr, imageMagnitude, imageAngle, true);
                for (size_t i = 0; i < reverse_cur_image0_lines.size(); i++)
                {
                    if (!checkGoodLine(reverse_cur_image0_lines[i].key_points, imageMagnitude, imageAngle))
                    {
                        reverse_cur_image1_line_status[i] = 0;
                        continue;
                    }
                }

                utils::resizeVector(cur_image1_lines, reverse_cur_image1_line_status);
                utils::resizeVector(cur_image1_line_ids, reverse_cur_image1_line_status);
                utils::resizeVector(cur_image1_track_counters, reverse_cur_image1_line_status);
#if SHOW_LINEFLOW
                utils::resizeVector(cur_image1_lineflows, reverse_cur_image1_line_status);
#endif
            }

            // image1_greydoor = 10 * (1.0 - 1.0 * cur_image1_lines.size() / shared_pool->min_line_feature_count);
            // image1_greydoor = std::max(image1_greydoor, 5.0);
            // image1_greydoor = std::min(image1_greydoor, 10.0);
            // std::cout << "image1_greydoor = " << image1_greydoor << std::endl;
    }
    }
    
    pre_image0 = cur_image0;
    pre_frame_time = cur_frame_time;
    pre_image0_lines = cur_image0_lines;


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

    FeatureLines feature_lines;
    for (size_t i = 0; i != cur_image0_line_ids.size(); i++)
    {
        int line_id = cur_image0_line_ids[i];
        int camera_id = 0;
        const cv::Matx33d &K = ImageProcessor::K_vec[camera_id];
        double s_x, s_y;
        s_x = (cur_image0_lines[i].sPoint.x - K(0, 2)) / K(0, 0);
        s_y = (cur_image0_lines[i].sPoint.y - K(1, 2)) / K(1, 1);
        double e_x, e_y;
        e_x = (cur_image0_lines[i].ePoint.x - K(0, 2)) / K(0, 0);
        e_y = (cur_image0_lines[i].ePoint.y - K(1, 2)) / K(1, 1);
        Eigen::Vector4d xy0_xy1;
        xy0_xy1 << s_x, s_y, e_x, e_y;
        feature_lines[line_id].emplace_back(camera_id, xy0_xy1);
    }

    if (!image1.empty() && shared_pool->is_use_stereo)
    {
        for (size_t i = 0; i != cur_image1_line_ids.size(); i++)
        {
            // continue;
            int line_id = cur_image1_line_ids[i];
            int camera_id = 1;
            const cv::Matx33d &K = ImageProcessor::K_vec[camera_id];
            double s_x, s_y;
            s_x = (cur_image1_lines[i].sPoint.x - K(0, 2)) / K(0, 0);
            s_y = (cur_image1_lines[i].sPoint.y - K(1, 2)) / K(1, 1);
            double e_x, e_y;
            e_x = (cur_image1_lines[i].ePoint.x - K(0, 2)) / K(0, 0);
            e_y = (cur_image1_lines[i].ePoint.y - K(1, 2)) / K(1, 1);
            Eigen::Vector4d xy0_xy1;
            xy0_xy1 << s_x, s_y, e_x, e_y;
            feature_lines[line_id].emplace_back(camera_id, xy0_xy1);
        }
    }

    frame_linefeatures = std::make_pair(cur_frame_time, feature_lines);

    return true;
}


void LineFeatureTrackerLOFT::setMask(cv::Mat &mask, std::vector<Line> &image_lines)
{
    for (size_t i = 0; i < image_lines.size(); i++)
    {
        cv::Point2f sPoint = image_lines[i].key_points[1];
        cv::Point2f ePoint = image_lines[i].key_points.back();

        cv::line(mask, sPoint, ePoint, 0, lines_mask_thickness);
    }

    // cv::imshow("mask", mask);
    // cv::waitKey(2);
}


/** \brief Set line feature index in the index image
 * @param index_image Recode the line feature index
 */
void LineFeatureTrackerLOFT::setIndex(cv::Mat &index_image, std::vector<Line> &image_lines)
{
    int line_index = 0;
    for (unsigned int i = 0; i < image_lines.size(); i++)
        drawLineIndex(image_lines[i], index_image, line_index++);

    // cv::imshow("index_image", index_image);
    // cv::waitKey(2);
}



#if DETECTING_TEST

/** \brief Detect line features in the image.
 * @param image the image to be detect.
 * @param image_lines Output the detected image lines.
 */
void LineFeatureTrackerLOFT::detectLine(const cv::Mat &image, std::vector<cv::Vec4f> &image_lines)
{
// #if !USE_EDLINES
{
    utils::TicToc detect_tictoc;
    cv::line_descriptor_custom::LSDParam lsd_param;
    lsd_param.scale        = 0.5;   //0.8   The scale of the image that will be used to find the lines. Range (0..1].
    lsd_param.sigma_scale  = 0.6;	//0.6  	Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
    lsd_param.quant        = 2.0;	//2.0   Bound to the quantization error on the gradient norm
    lsd_param.ang_th       = 22.5;	//22.5	Gradient angle tolerance in degrees
    lsd_param.log_eps      = 1.0;	//0		Detection threshold: -log10(NFA) > log_eps. Used only when advance refinement is chosen
    lsd_param.density_th   = 0.6;	//0.7	Minimal density of aligned region points in the enclosing rectangle.
    lsd_param.n_bins       = 1024;	//1024 	Number of bins in pseudo-ordering of gradient modulus.
    // float min_length = 0.125 * std::min(image.rows, image.cols);

    std::vector<KeyLine> image_lsd_lines;
    // cv::Ptr<cv::line_descriptor_custom::LSDDetector> lsd_detecter
    //     = cv::line_descriptor_custom::LSDDetector::createLSDDetector(lsd_param);
    // lsd_detecter->detect(image, image_lsd_lines, 2, 1, min_length, cv::Mat());
    cv::Ptr<cv::line_descriptor_custom::LSDDetector> lsd_detecter
        = cv::line_descriptor_custom::LSDDetector::createLSDDetector();
    lsd_detecter->detect(image, image_lsd_lines, 2, 1, cv::Mat());

    image_lines.clear();
    for (auto &lsd_line : image_lsd_lines)
        image_lines.emplace_back(lsd_line.startPointX, lsd_line.startPointY, lsd_line.endPointX, lsd_line.endPointY);

    removeEdgeLine(image, image_lines);

#if 1
    cv::Mat detect_imageTrack = image.clone();
    cv::Mat detect_background = cv::Mat(image.rows, image.cols, CV_8UC1, 255); // lines mask;
    cv::cvtColor(detect_imageTrack, detect_imageTrack, cv::COLOR_GRAY2RGB);
    cv::cvtColor(detect_background, detect_background, cv::COLOR_GRAY2RGB);

    static int draw_frame_count = 0;
    // std::vector<int> seq_set = {50, 550, 1045, 1610, 1890}; // MH_04 corridor1 kitti00
    std::vector<int> seq_set = {50, 550, 880, 1145, 1890}; // MH_04 corridor1 kitti00
    static int count = 0;
    if (++draw_frame_count == seq_set[count])
    {
        std::cout << "lsd costs: " << detect_tictoc.toc() << std::endl;
        double total_len = 0;
        int long_line_num = 0;
        for (size_t i = 0; i < image_lines.size(); ++i)
        {
            Line line(image_lines[i]);
            total_len += line.getLength();
            if (line.getLength() >= 25)
            {
                ++long_line_num;
                cv::line(detect_imageTrack, line.sPoint, line.ePoint, cv::Scalar(220, 0, 0), 2); // color format: BGR
                cv::line(detect_background, line.sPoint, line.ePoint, cv::Scalar(220, 0, 0), 2); // color format: BGR
            }
            else
            {
                cv::line(detect_imageTrack, line.sPoint, line.ePoint, cv::Scalar(0, 0, 255), 2); // color format: BGR
                cv::line(detect_background, line.sPoint, line.ePoint, cv::Scalar(0, 0, 255), 2); // color format: BGR
            }

        }
        ++count;
        std::cout << "nums: " << image_lines.size() << std::endl;
        std::cout << "avg len: " << total_len / image_lines.size() << std::endl;
        std::cout << "long line prop: " << (double) long_line_num / image_lines.size() << std::endl;
        std::cout << std::endl;

        std::string image_output_path = shared_pool->output_path + "/lsd_detect_line" + std::to_string(count) + ".png";
        cv::imwrite(image_output_path, detect_imageTrack);
        std::string bg_image_output_path = shared_pool->output_path + "/lsd_detect_bg" + std::to_string(count) + ".png";
        cv::imwrite(bg_image_output_path, detect_background);
        std::string raw_image_output_path = shared_pool->output_path + "/raw_image" + std::to_string(count) + ".png";
        cv::imwrite(raw_image_output_path, image);
    }
#endif
}


// #elif USE_EDLINES
{
    utils::TicToc edlines_detect_tictoc;
    cv::Ptr<cv::ximgproc::EdgeDrawing> edline_detecter = cv::ximgproc::createEdgeDrawing();
    edline_detecter->params.EdgeDetectionOperator = cv::ximgproc::EdgeDrawing::SOBEL;
    // edline_detecter->params.GradientThresholdValue = 32;
    // edline_detecter->params.AnchorThresholdValue = 16;
    // edline_detecter->params.ScanInterval = 2;
    // edline_detecter->params.MinLineLength = 0.125 * (std::min(image.cols, image.rows));

    image_lines.clear();
    edline_detecter->detectEdges(image);
    edline_detecter->detectLines(image_lines);
    // if (image_lines.size() < 50)
    // {
    //     edline_detecter->params.AnchorThresholdValue *= 0.5;
    //     edline_detecter->params.ScanInterval *= 0.5;
        
    //     edline_detecter->detectEdges(image);
    //     edline_detecter->detectLines(image_lines);
    // }
    removeEdgeLine(image, image_lines);

#if 1
    cv::Mat edlines_detect_imageTrack = image.clone();
    cv::Mat edlines_detect_background = cv::Mat(image.rows, image.cols, CV_8UC1, 255); // lines mask;
    cv::cvtColor(edlines_detect_imageTrack, edlines_detect_imageTrack, cv::COLOR_GRAY2RGB);
    cv::cvtColor(edlines_detect_background, edlines_detect_background, cv::COLOR_GRAY2RGB);

    static int edline_draw_frame_count = 0;
    // std::vector<int> edline_seq_set = {50, 550, 1045, 1610, 1890}; // MH_04 corridor1 kitti00
    std::vector<int> edline_seq_set = {50, 550, 880, 1145, 1890}; // MH_04 corridor1 kitti00
    static int edlines_count = 0;
    if (++edline_draw_frame_count == edline_seq_set[edlines_count])
    {
        std::cout << "edlines costs: " << edlines_detect_tictoc.toc() << std::endl;
        double total_len = 0;
        int long_line_num = 0;
        for (size_t i = 0; i < image_lines.size(); ++i)
        {
            Line line(image_lines[i]);
            total_len += line.getLength();
            if (line.getLength() >= 25)
            {
                ++long_line_num;
                cv::line(edlines_detect_imageTrack, line.sPoint, line.ePoint, cv::Scalar(0, 220, 0), 2); // color format: BGR
                cv::line(edlines_detect_background, line.sPoint, line.ePoint, cv::Scalar(0, 220, 0), 2); // color format: BGR
            }
            else
            {
                cv::line(edlines_detect_imageTrack, line.sPoint, line.ePoint, cv::Scalar(0, 0, 255), 2); // color format: BGR
                cv::line(edlines_detect_background, line.sPoint, line.ePoint, cv::Scalar(0, 0, 255), 2); // color format: BGR
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
    }
#endif
}

{
    utils::TicToc edlines_detect_tictoc;
    cv::Ptr<cv::ximgproc::EdgeDrawing> edline_detecter = cv::ximgproc::createEdgeDrawing();
    edline_detecter->params.EdgeDetectionOperator = cv::ximgproc::EdgeDrawing::SOBEL;
    edline_detecter->params.MinLineLength = 0.125 * (std::min(image.cols, image.rows));

    image_lines.clear();
    edline_detecter->detectEdges(image);
    edline_detecter->detectLines(image_lines);
    // if (image_lines.size() < 50)
    // {
    //     edline_detecter->params.AnchorThresholdValue *= 0.5;
    //     edline_detecter->params.ScanInterval *= 0.5;
        
    //     edline_detecter->detectEdges(image);
    //     edline_detecter->detectLines(image_lines);
    // }
    removeEdgeLine(image, image_lines);

#if 0
    static int frame_count = 0;
    static utils::TimeCost avg_detect_cost;
    printf("average detect cost: %.6f ms\n", avg_detect_cost.avgCost(edlines_detect_tictoc.toc()));

    static size_t total_detect_line_num = 0;
    static size_t total_length = 0;
    total_detect_line_num += image_lines.size();
    for (auto l : image_lines)
        total_length += Line(l).getLength();
    ++frame_count;
    std::cout << "total_detect_line_num = " << total_detect_line_num << std::endl;
    std::cout << "frame_count = " << frame_count << std::endl;
    std::cout << "average_detect_line_num = " << 1.0 * total_detect_line_num / frame_count << std::endl;
    std::cout << "average len = " << 1.0 * total_length / total_detect_line_num << std::endl;
#endif

#if 1
    cv::Mat edlines_detect_imageTrack = image.clone();
    cv::cvtColor(edlines_detect_imageTrack, edlines_detect_imageTrack, cv::COLOR_GRAY2RGB);

    static int edline_draw_frame_count = 0;
    // std::vector<int> edline_seq_set = {50, 550, 1045, 1610, 1890}; // MH_04 corridor1 kitti00
    std::vector<int> edline_seq_set = {50, 550, 880, 1145, 1890}; // MH_04 corridor1 kitti00
    static int edlines_count = 0;
    if (++edline_draw_frame_count == edline_seq_set[edlines_count])
    {
        double total_len = 0;
        int long_line_num = 0;
        for (size_t i = 0; i < image_lines.size(); ++i)
        {
            Line line(image_lines[i]);
            total_len += line.getLength();
            if (line.getLength() >= 25)
            {
                ++long_line_num;
                cv::line(edlines_detect_imageTrack, line.sPoint, line.ePoint, cv::Scalar(0, 220, 0), 2); // color format: BGR
            }
            else
            {
                cv::line(edlines_detect_imageTrack, line.sPoint, line.ePoint, cv::Scalar(0, 0, 255), 2); // color format: BGR
            }

        }
        ++edlines_count;

        std::string image_output_path = shared_pool->output_path + "/long_edlines_detect_line" + std::to_string(edlines_count) + ".png";
        cv::imwrite(image_output_path, edlines_detect_imageTrack);
    }
#endif
}



{
    utils::TicToc edlines_detect_tictoc;
    cv::Ptr<cv::ximgproc::EdgeDrawing> edline_detecter = cv::ximgproc::createEdgeDrawing();
    edline_detecter->params.EdgeDetectionOperator = cv::ximgproc::EdgeDrawing::SOBEL;
    edline_detecter->params.GradientThresholdValue = 32;
    edline_detecter->params.AnchorThresholdValue = 16;
    edline_detecter->params.ScanInterval = 1;
    edline_detecter->params.MinLineLength = 0.125 * (std::min(image.cols, image.rows));

    image_lines.clear();
    edline_detecter->detectEdges(image);
    edline_detecter->detectLines(image_lines);
    // if (image_lines.size() < 50)
    // {
    //     edline_detecter->params.AnchorThresholdValue *= 0.5;
    //     edline_detecter->params.ScanInterval *= 0.5;
        
    //     edline_detecter->detectEdges(image);
    //     edline_detecter->detectLines(image_lines);
    // }
    removeEdgeLine(image, image_lines);

#if 1
    cv::Mat edlines_detect_imageTrack = image.clone();
    cv::cvtColor(edlines_detect_imageTrack, edlines_detect_imageTrack, cv::COLOR_GRAY2RGB);

    static int edline_draw_frame_count = 0;
    // std::vector<int> edline_seq_set = {50, 550, 1045, 1610, 1890}; // MH_04 corridor1 kitti00
    std::vector<int> edline_seq_set = {50, 550, 880, 1145, 1890}; // MH_04 corridor1 kitti00
    static int edlines_count = 0;
    if (++edline_draw_frame_count == edline_seq_set[edlines_count])
    {
        double total_len = 0;
        int long_line_num = 0;
        for (size_t i = 0; i < image_lines.size(); ++i)
        {
            Line line(image_lines[i]);
            total_len += line.getLength();
            if (line.getLength() >= 25)
            {
                ++long_line_num;
                cv::line(edlines_detect_imageTrack, line.sPoint, line.ePoint, cv::Scalar(0, 220, 0), 2); // color format: BGR
            }
            else
            {
                cv::line(edlines_detect_imageTrack, line.sPoint, line.ePoint, cv::Scalar(0, 0, 255), 2); // color format: BGR
            }

        }
        ++edlines_count;

        std::string image_output_path = shared_pool->output_path + "/adjust_edlines_detect_line" + std::to_string(edlines_count) + ".png";
        cv::imwrite(image_output_path, edlines_detect_imageTrack);
    }
#endif
}


// #elif 0
//     cv::Ptr<cv::ximgproc::FastLineDetector> fld_detecter = cv::ximgproc::createFastLineDetector();
//     image_lines.clear();
//     fld_detecter->detect(image, image_lines);

//     removeEdgeLine(image, image_lines);
// #endif


    std::vector<int> len_vec(image_lines.size());
    for (size_t i = 0; i < image_lines.size(); ++i)
    {
        int dx = image_lines[i][0] - image_lines[i][2];
        int dy = image_lines[i][1] - image_lines[i][3];
        len_vec[i] = dx * dx + dy * dy;
    }

    randomQuickSort(image_lines, len_vec, 0, image_lines.size() - 1);

    return;
}

#else

/** \brief Detect line features in the image.
 * @param image the image to be detect.
 * @param image_lines Output the detected image lines.
 */
void LineFeatureTrackerLOFT::detectLine(const cv::Mat &image, std::vector<cv::Vec4f> &image_lines)
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

    std::vector<KeyLine> image_lsd_lines;
    cv::Ptr<cv::line_descriptor_custom::LSDDetector> lsd_detecter
        = cv::line_descriptor_custom::LSDDetector::createLSDDetector(lsd_param);
    lsd_detecter->detect(image, image_lsd_lines, 2, 1, min_length, cv::Mat());

    image_lines.clear();
    for (auto &lsd_line : image_lsd_lines)
        image_lines.emplace_back(lsd_line.startPointX, lsd_line.startPointY, lsd_line.endPointX, lsd_line.endPointY);

    removeEdgeLine(image, image_lines);

#elif USE_EDLINES
{
#if 1
    utils::TicToc edlines_detect_tictoc;
    cv::Ptr<cv::ximgproc::EdgeDrawing> edline_detecter = cv::ximgproc::createEdgeDrawing();
    edline_detecter->params.EdgeDetectionOperator = cv::ximgproc::EdgeDrawing::SOBEL;
    edline_detecter->params.GradientThresholdValue = 32; // 32
    edline_detecter->params.AnchorThresholdValue = 16; // 16
    edline_detecter->params.ScanInterval = 1; // 1
    edline_detecter->params.MinLineLength = 0.125 * (std::min(image.cols, image.rows));

    image_lines.clear();
    edline_detecter->detectEdges(image);
    edline_detecter->detectLines(image_lines);
    // if (image_lines.size() < 50)
    // {
    //     edline_detecter->params.AnchorThresholdValue *= 0.5;
    //     edline_detecter->params.ScanInterval *= 0.5;
        
    //     edline_detecter->detectEdges(image);
    //     edline_detecter->detectLines(image_lines);
    // }
    removeEdgeLine(image, image_lines);
#else
    utils::TicToc edlines_detect_tictoc;
    cv::line_descriptor_custom::BinaryDescriptor::EDLineParam edline_param;
    edline_param.ksize = 5;
    edline_param.sigma = 1.0;
    edline_param.gradientThreshold = 32;
    edline_param.anchorThreshold = 1;
    edline_param.scanIntervals = 1;
    edline_param.minLineLen = 0.125 * (std::min(image.cols, image.rows));
    edline_param.lineFitErrThreshold = 4.0;

    std::vector<KeyLine> image_ed_lines;
    cv::Ptr<cv::line_descriptor_custom::BinaryDescriptor> edlines_detector
        = cv::line_descriptor_custom::BinaryDescriptor::createBinaryDescriptor(edline_param);
    edlines_detector->detect(image, image_ed_lines, cv::Mat());

    image_lines.clear();
    for (auto &ed_line : image_ed_lines)
        image_lines.emplace_back(ed_line.startPointX, ed_line.startPointY, ed_line.endPointX, ed_line.endPointY);

    removeEdgeLine(image, image_lines);
#endif


}


#elif 0
    cv::Ptr<cv::ximgproc::FastLineDetector> fld_detecter = cv::ximgproc::createFastLineDetector();
    image_lines.clear();
    fld_detecter->detect(image, image_lines);
    
    removeEdgeLine(image, image_lines);
#endif


    std::vector<int> len_vec(image_lines.size());
    for (size_t i = 0; i < image_lines.size(); ++i)
    {
        int dx = image_lines[i][0] - image_lines[i][2];
        int dy = image_lines[i][1] - image_lines[i][3];
        len_vec[i] = dx * dx + dy * dy;
    }

    randomQuickSort(image_lines, len_vec, 0, image_lines.size() - 1);

    return;
}
#endif

void LineFeatureTrackerLOFT::randomQuickSort(std::vector<cv::Vec4f> &image_lines, std::vector<int> &len_vec, int first, int last)
{
    if (first >= last)
        return;
    int index = rand() % (last - first + 1) + first;
    std::swap(len_vec[index], len_vec[last]);
    std::swap(image_lines[index], image_lines[last]);
    int left = first, right = last, pivot = len_vec[right];
    while (left < right)
    {
        while (left < right && len_vec[left] > pivot)
            ++left;
        if (left < right)
        {
            std::swap(image_lines[left], image_lines[right]);
            std::swap(len_vec[left], len_vec[right--]);
        }
        while (left < right && len_vec[left] > pivot)
            --right;
        if (left < right)
        {
            std::swap(image_lines[left], image_lines[right]);
            std::swap(len_vec[left++], len_vec[right]);
        }
    }
    randomQuickSort(image_lines, len_vec, first, left - 1);
    randomQuickSort(image_lines, len_vec, right + 1, last);
}


/** \brief remove image boundary lines
 * @param image Input image
 * @param image_lines Input & Output image lines
 */
void LineFeatureTrackerLOFT::removeEdgeLine(const cv::Mat &image, std::vector<cv::Vec4f> &image_lines)
{
    std::vector<uchar> bound_status(image_lines.size(), 1);
    int col_thickness = 0.05 * image.cols;
    int row_thickness = 0.05 * image.rows;
    for (size_t i = 0; i != image_lines.size(); i++)
    {
        const cv::Vec4f &cur_line0 = image_lines[i];
        if (cur_line0[0] <= col_thickness && cur_line0[2] <= col_thickness)
            bound_status[i] = 0;
        if (cur_line0[1] <= row_thickness && cur_line0[3] <= row_thickness)
            bound_status[i] = 0;
        if (cur_line0[0] >= image.cols - col_thickness && cur_line0[2] >= image.cols - col_thickness)
            bound_status[i] = 0;
        if (cur_line0[1] >= image.rows - row_thickness && cur_line0[3] >= image.rows - row_thickness)
            bound_status[i] = 0;
    }
    utils::resizeVector(image_lines, bound_status);
}


void LineFeatureTrackerLOFT::reconstructLine(const cv::Mat &image, std::vector<Line> &old_image_lines, std::vector<Line> &new_image_lines, std::vector<uchar> &status)
{
    cv::Mat imageMagnitude, imageAngle;
    cv::Mat x_arr, y_arr;
    cv::Sobel(image, x_arr, CV_32F, 1, 0);
    cv::Sobel(image, y_arr, CV_32F, 0, 1);
    cv::cartToPolar(x_arr, y_arr, imageMagnitude, imageAngle, true);
    // cv::imshow("magnitude", magnitude);
    // cv::imshow("angle", angle);
    // cv::waitKey(2);

    std::vector<Line> tmp_lines;
    cv::Mat lines_mask_image = cv::Mat(image.rows, image.cols, CV_8UC1, 255); // lines mask
    cv::Mat merged_lines_index = cv::Mat(image.rows, image.cols, CV_8UC1, 255); // record the line feature index
    int lineIndex = 0;

    for (size_t i = 0; i < new_image_lines.size(); i++)
    {
        if (status[i] != 1)
            continue;

        if (!checkGoodLine(new_image_lines[i].key_points, imageMagnitude, imageAngle))
        {
            status[i] = 0;
            continue;
        }

        // if (mergeLine(new_image_lines[i], tmp_lines, merged_lines_index, lines_mask_image))
        // {
        //     status[i] = 0;
        //     continue;
        // }
        
        /* angle rejection */
        if (abs(new_image_lines[i].getAngle() - old_image_lines[i].getAngle()) > M_PI / 30 && 
            M_PI - abs(new_image_lines[i].getAngle() - old_image_lines[i].getAngle()) > M_PI / 30)
        {
            status[i] = 0;
            continue;
        }

        /* length rejection */
        if (new_image_lines[i].length < 0.1 * std::min(image.cols, image.rows))
        {
            status[i] = 0;
            continue;
        }

        if (!pruneLine(new_image_lines[i], image, imageMagnitude, imageAngle))
        {
            status[i] = 0;
            continue;
        }

        /* try to extend line and recover line */
        if (extendLine(new_image_lines[i], imageMagnitude, imageAngle))
        {
            // new_image_lines[i].reset();
        }

        tmp_lines.push_back(new_image_lines[i]);
        drawLineIndex(new_image_lines[i], merged_lines_index, lineIndex++);
    }
}


bool LineFeatureTrackerLOFT::checkGoodLine(const std::vector<cv::Point2f> &line, const cv::Mat &imageMagnitude, const cv::Mat imageAngle)
{
    int pointNum = 16;
    int pointNumber = 0;

    cv::Point2f start = line[1];
    cv::Point2f end = line[line.size() - 1];

    float dx0 = end.x - start.x;
    float dy0 = end.y - start.y;

    cv::Point2f DistPoint0[pointNum];

    for (int i = 0; i < pointNum; i++)
    {
        cv::Point2f InsertPoint = cv::Point2f(start.x + i * dx0 / pointNum, start.y + i * dy0 / pointNum);
        if (InsertPoint.x < 1 || InsertPoint.x > imageMagnitude.cols - 1 ||
            InsertPoint.y < 1 || InsertPoint.y > imageMagnitude.rows - 1)
        {
            continue;
        }
        DistPoint0[pointNumber++] = InsertPoint;
    }

    if (pointNumber == 0)
        return false;

    float mag = 0, angle = 0;
    for (int i = 0; i < pointNumber; i++)
    {
        mag += abs(imageMagnitude.at<float>(DistPoint0[i]) / pointNumber);
        angle += imageAngle.at<float>(DistPoint0[i]) / pointNumber;
    }


    cv::Mat downsampled_imageMagnitude;
    cv::resize(imageMagnitude, downsampled_imageMagnitude, cv::Size(), 0.1, 0.1, cv::INTER_NEAREST);
    // int threshold = cv::mean(downsampled_imageMagnitude)[0] * 2; // cv::mean(downsampled_imageMagnitude)[0]
    int threshold = cv::mean(downsampled_imageMagnitude)[0]; // cv::mean(downsampled_imageMagnitude)[0]
    
    if (mag < threshold) // 128
        return false;

    float da = math_utils::wrap2pi(angle / 180.0 * M_PI - atan2(dy0, dx0));

    if (!(abs(da - M_PI_2) < M_PI / 18 || abs(da + M_PI_2) < M_PI / 18))
        return false;

    return true;
}


bool LineFeatureTrackerLOFT::pruneLine(Line &line, const cv::Mat &image, const cv::Mat &imageMagnitude, const cv::Mat &imageAngle)
{
    std::vector<cv::Point2f> &key_points = line.key_points;

    /* Adjust the lines to be located in the image */
    line.sPoint = key_points[1];
    while (line.sPoint.x < 3 || line.sPoint.x > image.cols - 4 ||
        line.sPoint.y < 3 || line.sPoint.y > image.rows - 4)
    {
        for (size_t i = 1; i < key_points.size(); i++)
        {
            key_points[i] = 0.9 * key_points[i] + 0.1 * key_points[key_points.size() - 1];
        }
        key_points[0] = 0.5 * (key_points[1] + key_points[key_points.size() - 1]);
        line.sPoint = key_points[1];
        if (utils::dist(key_points[1], key_points[key_points.size() - 1]) < 0.1 * std::min(image.cols, image.rows))
            return false;
    }

    line.ePoint = key_points[key_points.size() - 1];
    while (line.ePoint.x < 3 || line.ePoint.x > image.cols - 4 ||
        line.ePoint.y < 3 || line.ePoint.y > image.rows - 4)
    {
        for (size_t i = key_points.size() - 1; i > 0; i--)
        {
            key_points[i] = 0.9 * key_points[i] + 0.1 * key_points[1];
        }
        key_points[0] = 0.5 * (key_points[1] + key_points[key_points.size() - 1]);
        line.ePoint = key_points[key_points.size() - 1];
        if (utils::dist(key_points[1], key_points[key_points.size() - 1]) < 0.1 * std::min(image.cols, image.rows))
            return false;
    }

#if 0
    float angThreshold = 10;
    int halflength = 1;
    int pixel_num = (2 * halflength + 1) * (2 * halflength + 1);

    float nowAngle = 0;
    cv::Point2f nowStart = key_points[1];
    cv::Point2f nowEnd = key_points.back();
    float lineAngle = atan2(nowEnd.y - key_points[1].y, nowEnd.x - key_points[1].x) * 180 / M_PI;
    lineAngle += lineAngle < 0 ? 180 : 0;
    for (size_t i = 1; i < key_points.size() / 2; i++)
    {
        nowStart = key_points[i];
        nowAngle = 0;
        for (int j = -halflength; j <= halflength; j++)
        {
            for (int k = -halflength; k <= halflength; k++)
            {
                cv::Point2f now = cv::Point2f(nowStart.x + j, nowStart.y  + k);
                if (!utils::inside(imageAngle, now))
                    continue;
                nowAngle += imageAngle.at<float>(now);
            }
        }
        nowAngle /= pixel_num;
        nowAngle -= nowAngle > 180 ? 180 : 0;

        if (!(abs(nowAngle - lineAngle - 90) > angThreshold &&
              abs(nowAngle - lineAngle + 90) > angThreshold))
            break;
    }
    line.sPoint = nowStart;

    for (size_t i = key_points.size() - 1; i > key_points.size() / 2; i--)
    {
        nowEnd = key_points[i];
        nowAngle = 0;
        for (int j = -halflength; j <= halflength; j++)
        {
            for (int k = -halflength; k <= halflength; k++)
            {
                cv::Point2f now = cv::Point2f(nowEnd.x + j, nowEnd.y  + k);
                if (!utils::inside(imageAngle, now))
                    continue;
                nowAngle += imageAngle.at<float>(now);
            }
        }
        nowAngle /= pixel_num;
        nowAngle -= nowAngle > 180 ? 180 : 0;

        if (!(abs(nowAngle - lineAngle - 90) > angThreshold &&
              abs(nowAngle - lineAngle + 90) > angThreshold))
            break;
    }
    line.ePoint = nowEnd;
#endif

    return true;
}


/** \brief Extend the line.
 * @param line Input & Output: input the original line, output the extended line.
 * @param imageMagnitude Magnitude information.
 * @param angle Angle information.
 * \return Extend line successfully or not.
 */
bool LineFeatureTrackerLOFT::extendLine(Line &line, const cv::Mat &imageMagnitude, const cv::Mat &imageAngle)
{
    int halflength = 1;
    int step = 5;
    bool ret = false;

    cv::Point2f start = line.key_points[1];
    cv::Point2f end = line.key_points.back();
    cv::Point2f mid = 0.5 * (start + end);
    // ROS_WARN("begin extend\n");

    /* The direction of line expansion for the start point and end point */
    float lengthDir = cv::norm(start - mid); // length
    float startDirX = step * (start.x - mid.x) / lengthDir; // cos(\theta)
    float startDirY = step * (start.y - mid.y) / lengthDir; // sin(\theta)
    lengthDir = cv::norm(end - mid); // length
    float endDirX = step * (end.x - mid.x) / lengthDir; // cos(\theta)
    float endDirY = step * (end.y - mid.y) / lengthDir; // cos(\theta)

    /* compute sample gradient */
    // ROS_WARN("compute sample gradient\n");
    int sampleSegLength = 4; // sampling interval
    int samplePointNum = line.length / sampleSegLength;
    float sampleDirX = sampleSegLength * (end.x - start.x) / line.length;
    float sampleDirY = sampleSegLength * (end.y - start.y) / line.length;

    float lineAngle = atan2(sampleDirY, sampleDirX) * 180 / M_PI;
    lineAngle += lineAngle < 0 ? 180 : 0;

    /* record the imageMagnitude and angle data for the whole line */
    std::vector<float> magSample, angSample;
    for (int i = 0; i < samplePointNum; i++)
    {
        for (int j = -halflength; j <= halflength; j++)
        {
            for (int k = -halflength; k <= halflength; k++)
            {
                cv::Point2f now = cv::Point2f(start.x + sampleDirX * i + j, start.y + sampleDirY * i + k);
            
                if (!utils::inside(imageMagnitude, now))
                    continue;
                magSample.push_back(imageMagnitude.at<float>(now));

                float nowAngle = imageAngle.at<float>(now);
                nowAngle -= nowAngle > 180 ? 180 : 0;
                angSample.push_back(nowAngle);
                // ROS_WARN("(%f %f) sampleMagnitude:%f  sampleAngle:%f\n", now.x, now.y, imageMagnitude.at<float>(now), nowAngle);
            }
        }
    }
    // ROS_WARN("sampleMagnitude:%d  sampleAngle:%d\n", sampleMagnitude.size(), sampleAngle.size());

    if (magSample.empty() || angSample.empty())
        return false;


    /* set the magnitude and angle threshold */
    float magMean = 0, magStdev = 0;
    for (float mag : magSample)
        magMean += mag;
    magMean /= magSample.size();
    for (float mag : magSample)
        magStdev += (mag - magMean) * (mag - magMean);
    magStdev = sqrt(magStdev / magSample.size());
    
    float magThreshold = magMean - 2 * magStdev;
    float angThreshold = 3;

    /* start to extend line */
    int pixel_num = (2 * halflength + 1) * (2 * halflength + 1);
    int extendPointNum = 5; 
    float nowMagnitude, nowAngle;

    // ROS_WARN("extend start point\n");
    for (int i = 0; i < extendPointNum; i++) // extend line using points
    {
        nowMagnitude = nowAngle = 0;
        
        cv::Point2f now = cv::Point2f(start.x + startDirX * i, start.y + startDirY * i);
        if (!utils::inside(imageMagnitude, now))
            break;

        for (int j = -halflength; j <= halflength; j++)
        {
            for (int k = -halflength; k <= halflength; k++)
            {
                cv::Point2f now = cv::Point2f(start.x + startDirX * i + j, start.y + startDirY * i + k);
                if (!utils::inside(imageMagnitude, now))
                    continue;
                nowMagnitude += imageMagnitude.at<float>(now);
                nowAngle += imageAngle.at<float>(now);
            }
        }
        nowMagnitude /= pixel_num;
        nowAngle /= pixel_num;
        nowAngle -= nowAngle > 180 ? 180 : 0;

        if (nowMagnitude < magThreshold)
            break;

        if (abs(nowAngle - lineAngle - 90) > angThreshold &&
            abs(nowAngle - lineAngle + 90) > angThreshold)
            break;

        line.sPoint = now;
        ret = true;
    }

    // ROS_WARN("extend start point\n");
    for (int i = 0; i < extendPointNum; i++) // extend line using points
    {
        nowMagnitude = nowAngle = 0;

        cv::Point2f now = cv::Point2f(end.x + endDirX * i, end.y + endDirY * i);
        if (!utils::inside(imageMagnitude, now))
            break;
        for (int j = -halflength; j <= halflength; j++)
        {
            for (int k = -halflength; k <= halflength; k++)
            {
                cv::Point2f now = cv::Point2f(end.x + endDirX * i + j, end.y + endDirY * i + k);
                if (!utils::inside(imageMagnitude, now))
                    continue;
                nowMagnitude += imageMagnitude.at<float>(now);
                nowAngle += imageAngle.at<float>(now);
            }
        }
        nowMagnitude /= pixel_num;
        nowAngle /= pixel_num;
        nowAngle -= nowAngle > 180 ? 180 : 0;

        if (nowMagnitude < magThreshold)
            break;

        if (abs(nowAngle - lineAngle - 90) > angThreshold &&
            abs(nowAngle - lineAngle + 90) > angThreshold)
            break;

        line.ePoint = now;
        ret = true;
    }
    return ret;
}

/** \brief Merge the line.
 * @param new_line Input: input the new detected line.
 * @param cur_image_lines Current lines.
 * @param lines_index_image Line index.
 * @param lines_mask_image Line mask.
 * \return false if no intersect, true if merged or intersect but not merged.
 */
bool LineFeatureTrackerLOFT::mergeLine(const Line &new_line, std::vector<Line> &cur_image_lines, 
    const cv::Mat &lines_index_image, cv::Mat &lines_mask_image)
{
    if (!checkIntersect(new_line, lines_index_image))
        return false;

    if (!utils::inside(lines_index_image, new_line.sPoint) || 
        !utils::inside(lines_index_image, new_line.mPoint) || 
        !utils::inside(lines_index_image, new_line.ePoint))
        return true;
    // std::cout << "new_line.sPoint = " << new_line.sPoint << std::endl;

    std::vector<int> pixelValues(3);
    pixelValues[0] = lines_index_image.at<uchar>(new_line.sPoint);
    pixelValues[1] = lines_index_image.at<uchar>(new_line.mPoint);
    pixelValues[2] = lines_index_image.at<uchar>(new_line.ePoint);

    int line_index = 0;
    for (int pixelValue : pixelValues)
    {
        line_index = pixelValue;
        
        if (line_index > static_cast<int>(cur_image_lines.size())) // line index == 255
        {
            line_index = -1;
            continue;
        }
        cv::Point2f &merged_sPoint = cur_image_lines[line_index].sPoint;
        cv::Point2f &merged_ePoint = cur_image_lines[line_index].ePoint;

        float mergeAngle = atan2(merged_ePoint.y - merged_sPoint.y, merged_ePoint.x - merged_sPoint.x);
        mergeAngle += mergeAngle < 0 ? M_PI : 0;

        float angle_diff = fabs(mergeAngle - new_line.angle);
        if (angle_diff < M_PI / 180.0 || M_PI - angle_diff < M_PI / 180.0)
            break;
        line_index = -1;
    }

    if (line_index == -1 || line_index > static_cast<int>(cur_image_lines.size()))
        return true;


    cv::Point2f &merged_sPoint = cur_image_lines[line_index].sPoint;
    cv::Point2f &merged_ePoint = cur_image_lines[line_index].ePoint;

    if (new_line.angle < 45 * M_PI / 180.0 || new_line.angle > 135 * M_PI / 180.0)
    {
        if (merged_sPoint.x > merged_ePoint.x)
        {
            if (new_line.sPoint.x > merged_sPoint.x)
                merged_sPoint = new_line.sPoint;
            else if (new_line.sPoint.x < merged_ePoint.x)
                merged_ePoint = new_line.sPoint;
            if (new_line.ePoint.x > merged_sPoint.x)
                merged_sPoint = new_line.ePoint;
            else if (new_line.ePoint.x < merged_ePoint.x)
                merged_ePoint = new_line.ePoint;
        }
        else if (merged_sPoint.x < merged_ePoint.x)
        {
            if (new_line.sPoint.x < merged_sPoint.x)
                merged_sPoint = new_line.sPoint;
            else if (new_line.sPoint.x > merged_ePoint.x)
                merged_ePoint = new_line.sPoint;
            if (new_line.ePoint.x < merged_sPoint.x)
                merged_sPoint = new_line.ePoint;
            else if (new_line.ePoint.x > merged_ePoint.x)
                merged_ePoint = new_line.ePoint;
        }
        else
        {
            std::vector<float> yArr = {merged_sPoint.y, merged_ePoint.y, new_line.sPoint.y, new_line.ePoint.y};
            sort(yArr.begin(), yArr.end());
            merged_sPoint = cv::Point2f(merged_sPoint.x, yArr[0]);
            merged_ePoint = cv::Point2f(merged_ePoint.x, yArr[yArr.size() - 1]);
        }
    }
    else
    {
        if (merged_sPoint.y > merged_ePoint.y)
        {
            if (new_line.sPoint.y > merged_sPoint.y)
                merged_sPoint = new_line.sPoint;
            else if (new_line.sPoint.y < merged_ePoint.y)
                merged_ePoint = new_line.sPoint;
            if (new_line.ePoint.y > merged_sPoint.y)
                merged_sPoint = new_line.ePoint;
            else if (new_line.ePoint.y < merged_ePoint.y)
                merged_ePoint = new_line.ePoint;
        }
        else if (merged_sPoint.y < merged_ePoint.y)
        {
            if (new_line.sPoint.y < merged_sPoint.y)
                merged_sPoint = new_line.sPoint;
            else if (new_line.sPoint.y > merged_ePoint.y)
                merged_ePoint = new_line.sPoint;
            if (new_line.ePoint.y < merged_sPoint.y)
                merged_sPoint = new_line.ePoint;
            else if (new_line.ePoint.y > merged_ePoint.y)
                merged_ePoint = new_line.ePoint;
        }
        else
        {
            std::vector<float> yArr = {merged_sPoint.x, merged_ePoint.x, new_line.sPoint.x, new_line.ePoint.x};
            sort(yArr.begin(), yArr.end());
            merged_sPoint = cv::Point2f(yArr[0], merged_sPoint.y);
            merged_ePoint = cv::Point2f(yArr[yArr.size() - 1], merged_ePoint.y);
        }
    }
    
    cur_image_lines[line_index].reset();

    cv::line(lines_mask_image, merged_sPoint, merged_ePoint, 0, lines_mask_thickness);
    // cv::imshow("lines_mask_image", lines_mask_image);
    // cv::waitKey(2);

    return true;
}


/** \brief Check whether the input line intersects with the existing line segments in the image.
 * @param lines_index_image Input the existing line segments in the image.
 * @param line Input new line.
 * \return true if the line intersects with the existing line segments, otherwise, false.
 */
bool LineFeatureTrackerLOFT::checkIntersect(const Line &line, const cv::Mat &lines_index_image)
{
    return (lines_index_image.at<uchar>(line.sPoint) != 255 ||
            lines_index_image.at<uchar>(line.ePoint) != 255 ||
            lines_index_image.at<uchar>(line.mPoint) != 255);
}


/** \brief Draw the merged lines in the mask image with the corresponding index, this operation will extend the line.
 * @param line current line.
 * @param lines_index_image Merged lines in the mask image with the corresponding index.
 * @param index Index of line.
 */
void LineFeatureTrackerLOFT::drawLineIndex(const Line &line, cv::Mat &merged_lines_image, int index)
{
    float lenthThreshod = 20; // for line fusion

    if (index > 255)
    {
        ROS_WARN("index > 255 : %d\n", index);
        return;
    }

    int row = merged_lines_image.rows - 1;
    int col = merged_lines_image.cols - 1;
    cv::Point2f sPoint = line.sPoint;
    cv::Point2f ePoint = line.ePoint;
    cv::Point2f draw_sPoint, draw_ePoint;

    // cos(theta) * lenthThreshod
    float dealtx = abs(sPoint.x - ePoint.x) / line.length * lenthThreshod;

    // extend start point and end point
    if (sPoint.x > ePoint.x)
    {
        float extend_sPoint = abs(col - sPoint.x) > dealtx ? (col - sPoint.x) / abs(col - sPoint.x) * dealtx : (col - sPoint.x);
        float extend_ePoint = abs(0 - ePoint.x)   > dealtx ? (0 - ePoint.x)   / abs(0 - ePoint.x)   * dealtx : (0 - ePoint.x);

        draw_sPoint = sPoint + extend_sPoint / (sPoint.x - ePoint.x) * (sPoint - ePoint);
        draw_ePoint = ePoint + extend_ePoint / (ePoint.x - sPoint.x) * (ePoint - sPoint);
    }
    else if (sPoint.x < ePoint.x)
    {
        float extend_sPoint = abs(0 - sPoint.x)   > dealtx ? (0 - sPoint.x)   / abs(0 - sPoint.x)   * dealtx : (0 - sPoint.x);
        float extend_ePoint = abs(col - ePoint.x) > dealtx ? (col - ePoint.x) / abs(col - ePoint.x) * dealtx : (col - ePoint.x);

        draw_sPoint = sPoint + extend_sPoint / (sPoint.x - ePoint.x) * (sPoint - ePoint);
        draw_ePoint = ePoint + extend_ePoint / (ePoint.x - sPoint.x) * (ePoint - sPoint);
    }
    else
    {
        draw_sPoint = cv::Point2f(sPoint.x, 0);
        draw_ePoint = cv::Point2f(sPoint.x, row);
    }

    /* check start point whether outside the boundary in y direction */
    if (draw_sPoint.y > row)
        draw_sPoint = sPoint + (float)(row - sPoint.y) / (sPoint.y - ePoint.y) * (sPoint - ePoint);
    else if (draw_sPoint.y < 0)
        draw_sPoint = sPoint + (float)(0 - sPoint.y)   / (sPoint.y - ePoint.y) * (sPoint - ePoint);

    /* check end point whether outside the boundary in y direction */
    if (draw_ePoint.y > row)
        draw_ePoint = ePoint + (float)(row - ePoint.y) / (ePoint.y - sPoint.y) * (ePoint - sPoint);
    else if (draw_ePoint.y < 0)
        draw_ePoint = ePoint + (float)(0 - ePoint.y)   / (ePoint.y - sPoint.y) * (ePoint - sPoint);

    /* draw line with index */
    cv::line(merged_lines_image, draw_sPoint, draw_ePoint, index, lines_index_thickness);
    // cv::imshow("merged_lines", merged_lines_image);
    // cv::waitKey(2);
}


void LineFeatureTrackerLOFT::drawTrack(const cv::Mat &image0, const cv::Mat &image1)
{
    if (!image1.empty() && shared_pool->is_use_stereo)
        cv::hconcat(image0, image1, imageTrack); // concat two images
    else
        imageTrack = image0.clone();
    cv::cvtColor(imageTrack, imageTrack, cv::COLOR_GRAY2RGB);

    for (size_t i = 0; i != cur_image0_lines.size(); i++)
    {
        double len = std::min(1, (cur_image0_track_counters[i] - 1) / 2);
        // double len = std::min(1.0, 1.0 * (cur_image0_track_counters[i] - 1) / 20);

        cv::putText(imageTrack, std::to_string(
            cur_image0_line_ids[i]), cur_image0_lines[i].key_points[0], 
            cv::FONT_HERSHEY_SIMPLEX, 0.45, CV_RGB(255, 230, 0), 1.8);
        cv::line(imageTrack, cur_image0_lines[i].sPoint, cur_image0_lines[i].ePoint, 
            cv::Scalar(0, 255 * (1 - len), 255 * len), 3); // color format: BGR

#if SHOW_LINEFLOW
        for (size_t j = 0; j < cur_image0_lineflows[i].first.size() - 1; ++j)
        {
            cv::Point2f sPoint(cur_image0_lineflows[i].first[j][0], cur_image0_lineflows[i].first[j][1]);
            cv::Point2f ePoint(cur_image0_lineflows[i].first[j][2], cur_image0_lineflows[i].first[j][3]);
            cv::line(imageTrack, sPoint, ePoint, cur_image0_lineflows[i].second, 1); // color format: BGR
        }
#endif
    }
    if (!image1.empty() && shared_pool->is_use_stereo)
    {
        for (size_t i = 0; i != cur_image1_lines.size(); i++)
        {
            double len = std::min(1, (cur_image1_track_counters[i] - 1) / 2);
            // double len = std::min(1.0, 1.0 * (cur_image1_track_counters[i] - 1) / 20);

            cv::putText(imageTrack, std::to_string(cur_image1_line_ids[i]), 
                cur_image1_lines[i].key_points[0] + cv::Point2f(image0.cols, 0), 
                cv::FONT_HERSHEY_SIMPLEX, 0.45, CV_RGB(255, 230, 0), 1.8);
            cv::line(imageTrack, 
                cur_image1_lines[i].sPoint + cv::Point2f(image0.cols, 0), 
                cur_image1_lines[i].ePoint + cv::Point2f(image0.cols, 0), 
                cv::Scalar(0, 255 * (1 - len), 255 * len), 3); // color format: BGR

#if SHOW_LINEFLOW
            for (size_t j = 0; j < cur_image1_lineflows[i].first.size() - 1; ++j)
            {
                cv::Point2f sPoint(cur_image1_lineflows[i].first[j][0], cur_image1_lineflows[i].first[j][1]);
                cv::Point2f ePoint(cur_image1_lineflows[i].first[j][2], cur_image1_lineflows[i].first[j][3]);
                cv::line(imageTrack, sPoint + cv::Point2f(image0.cols, 0), ePoint + cv::Point2f(image0.cols, 0), cur_image1_lineflows[i].second, 1); // color format: BGR
            }
#endif
        }
    }

    // cv::imshow("line tracking", imageTrack);
    // cv::waitKey(2);


#if TRACKING_TEST
    cv::Mat save_imageTrack = image0.clone();
    cv::cvtColor(save_imageTrack, save_imageTrack, cv::COLOR_GRAY2RGB);

    static int draw_frame_count = 0;
    std::vector<int> seq_set = {50, 550, 1045, 1610, 1890}; // MH_04 corridor1 kitti00
    // std::vector<int> seq_set = {50}; // MH_04 corridor1 kitti00
    static int count = 0;
    if (++draw_frame_count == seq_set[count])
    {
#if SHOW_LINEFLOW
        std::vector<cv::Scalar> bgr_vec(cur_image0_lines.size());

        for (size_t i = 0; i < cur_image0_lines.size(); ++i)
        {
            bgr_vec[i] = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
            for (size_t j = 0; j < cur_image0_lineflows[i].first.size() - 1; ++j)
            {
                cv::Point sPoint(cur_image0_lineflows[i].first[j][0], cur_image0_lineflows[i].first[j][1]);
                cv::Point ePoint(cur_image0_lineflows[i].first[j][2], cur_image0_lineflows[i].first[j][3]);
                cv::line(save_imageTrack, sPoint, ePoint, cur_image0_lineflows[i].second, 1); // color format: BGR
            }
        }
#endif
        for (size_t i = 0; i < cur_image0_lines.size(); ++i)
        {
#if SHOW_LINEFLOW
            // if (cur_image0_lineflows[i].size() == 1)
            //     cv::line(save_imageTrack, cur_image0_lines[i].sPoint, cur_image0_lines[i].ePoint, 
            //         cv::Scalar(0, 220, 0), 2); // color format: BGR 
            // else
            //     cv::line(save_imageTrack, cur_image0_lines[i].sPoint, cur_image0_lines[i].ePoint, 
            //         cv::Scalar(0, 0, 255), 2); // color format: BGR
            cv::line(save_imageTrack, cur_image0_lines[i].sPoint, cur_image0_lines[i].ePoint, 
                cur_image0_lineflows[i].second, 3); // color format: BGR 
#else
            double len = std::min(1.0, 1.0 * (cur_image0_track_counters[i] - 1) / 2);
            cv::line(save_imageTrack, cur_image0_lines[i].sPoint, cur_image0_lines[i].ePoint, 
                cv::Scalar(0, 255 * (1 - len), 255 * len), 2); // color format: BGR
            // cv::line(save_imageTrack, cur_image0_lines[i].sPoint, cur_image0_lines[i].ePoint, 
            //     cv::Scalar(0, 220, 0), 2); // color format: BGR 
#endif
            // for (size_t j = 0; j < cur_image0_lines[i].key_points.size(); ++j)
            // {
            //     cv::circle(save_imageTrack, cur_image0_lines[i].key_points[j], 2, 
            //         cv::Scalar(0, 200, 230), 2); // color format: BGR
            //     // cv::circle(save_imageTrack, cur_image0_lines[i].key_points[j], 2, 
            //     //     cv::Scalar(255, 150, 0), 2); // color format: BGR
            // }
        }
        ++count;

        std::string image_output_path = shared_pool->output_path + "/detect_line" + std::to_string(count) + ".png";
        cv::imwrite(image_output_path, save_imageTrack);

        std::string imageTrack_output_path = shared_pool->output_path + "/imTrack_line" + std::to_string(count) + ".png";
        cv::imwrite(imageTrack_output_path, imageTrack);

        cv::Mat lines_mask_image = cv::Mat(cur_image0.rows, cur_image0.cols, CV_8UC1, 255); // used for uniform distribution line feature
        cv::Mat lines_index_image = cv::Mat(cur_image0.rows, cur_image0.cols, CV_8UC1, 255); // used to merge line feature and record the line feature index
        setMask(lines_mask_image, cur_image0_lines); // set mask for the existing lines
        setIndex(lines_index_image, cur_image0_lines); // set the index for the existing lines

        std::string imageIndexMask_output_path = shared_pool->output_path + "/imIndexMask_line" + std::to_string(count) + ".png";
        cv::imwrite(imageIndexMask_output_path, lines_index_image);

        std::string imageMask_output_path = shared_pool->output_path + "/imMask_line" + std::to_string(count) + ".png";
        cv::imwrite(imageMask_output_path, lines_mask_image);
    }
#endif
}
