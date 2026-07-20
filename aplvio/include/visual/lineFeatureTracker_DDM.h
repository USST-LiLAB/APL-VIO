/*******************************************************
 * The LineFeatureTrackerDDM class utilizing the LSD/EDLines to Detect,
 * the LBD to Describe, and the KNN to Match (DMM) the line features
 * 
 * Author: Wu Lifeng (hcxygxgn@163.com)
 *******************************************************/

#ifndef __LINE_FEATURE_TRACKER_DDM_H
#define __LINE_FEATURE_TRACKER_DDM_H

#include <numeric>
#include <thread>

#include "line_descriptor.hpp"
#include "lineFeatureTrackerBase.h"


typedef cv::line_descriptor_custom::KeyLine KeyLine;

class LineFeatureTrackerDDM : public LineFeatureTrackerBase
{
public:
    LineFeatureTrackerDDM() {}

    virtual void clearState() override;
    virtual void setParameters() override;

    virtual bool trackImage(const double, const cv::Mat &, const cv::Mat &image1 = cv::Mat()) override;
    void randomQuickSort(std::vector<int> &, int, int);
    void detectLine(const cv::Mat &, std::vector<KeyLine> &);
    void describeLine(const cv::Mat &, std::vector<KeyLine> &, cv::Mat &);
    void matchLine(cv::Mat &, cv::Mat &, std::vector<cv::DMatch> &);
    void removeEdgeLine(const cv::Mat &, std::vector<KeyLine> &);
    void checkGoodMatches(const std::vector<KeyLine> &, const std::vector<KeyLine> &, const std::vector<cv::DMatch> &, std::vector<cv::DMatch> &);
    void drawTrack(const cv::Mat &, const cv::Mat &);

private:
    // previous frame
    std::vector<int> pre_image0_line_ids, pre_image0_track_counters;
    std::vector<KeyLine> pre_image0_lines;
    cv::Mat pre_image0_describers;

    // current frame
    std::vector<int> cur_image0_line_ids, cur_image1_line_ids;
    std::vector<KeyLine> cur_image0_lines, cur_image1_lines;
    cv::Mat cur_image0_describers, cur_image1_lbd_describers;

    // matched lines
    std::vector<int> matched_cur_image0_line_ids, matched_cur_image0_track_counters;
    std::vector<KeyLine> matched_cur_image0_lines;
    cv::Mat matched_cur_image0_describers;
    std::vector<int> matched_cur_image1_line_ids, matched_cur_image1_track_counters;
    std::vector<KeyLine> matched_cur_image1_lines;
    cv::Mat matched_cur_image1_lbd_describers;

    std::thread parallel_thread;

    cv::Mat imageTrack;
};

#endif
