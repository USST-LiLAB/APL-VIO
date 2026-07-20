/*******************************************************
 * The LineFeatureTrackerLOFT class utilizing the line 
 * line optical flow tracker (LOFT) to track line features.
 * 
 * Author: Wu Lifeng (hcxygxgn@163.com)
 *******************************************************/

#ifndef __LINE_FEATURE_FLOW_TRACKER_LOFT_H
#define __LINE_FEATURE_FLOW_TRACKER_LOFT_H

#include <numeric>
#include <thread>

#include "line_descriptor.hpp"
#include "estimator/customType.h"

#include "lineFeatureTrackerBase.h"


#define ADAPTIVE_LENGTH_ADJUSTMENT 1

struct Line
{
    cv::Point2f sPoint;
    cv::Point2f ePoint;
    cv::Point2f mPoint;
    float angle; // rad
    float length;
    std::vector<cv::Point2f> key_points; // key points of line

    Line() {}

    Line(const cv::Vec4f &line) : 
        sPoint(cv::Point2f(line[0], line[1])), ePoint(cv::Point2f(line[2], line[3]))
    {
        reset();
    }

    void reset()
    {
        mPoint = 0.5 * (sPoint + ePoint);
        length = getLength();
        angle = getAngle();

#if ADAPTIVE_LENGTH_ADJUSTMENT
        key_points = getKeypoints();
#else
        key_points = std::vector<cv::Point2f>(5);
        key_points[0] = mPoint;
        // key_points[1] = 0.95 * sPoint + 0.05 * ePoint;
        key_points[2] = 0.75 * sPoint + 0.25 * ePoint;
        key_points[3] = 0.25 * sPoint + 0.75 * ePoint;
        // key_points[4] = 0.05 * sPoint + 0.95 * ePoint;
        key_points[1] = sPoint - 5 * (sPoint - mPoint) / length / 2;
        key_points[4] = ePoint - 5 * (ePoint - mPoint) / length / 2;
#endif
    }

    float getLength() const
    {
        cv::Point2f delta_pt = ePoint - sPoint;
        return sqrt(delta_pt.x * delta_pt.x + delta_pt.y * delta_pt.y);
    }

    float getAngle() const
    {
        float angle = atan2(ePoint.y - sPoint.y, ePoint.x - sPoint.x);
        return angle < 0 ? angle + M_PI : angle;
    }


    std::vector<cv::Point2f> getKeypoints()
    {
        // set some parameters to distribute the keypoints
        
        static const int tip_size = 3;
        static const int min_keypoints_num = 3;
        static const int max_keypoints_num = 9;
        // static const int min_keypoints_num = 5;
        // static const int max_keypoints_num = 5;
        // static const int interval = 18;
        static const int interval = std::max(Parameters::image_width, Parameters::image_height) / (max_keypoints_num * 2);

        float length = getLength();
        
        int keypoints_num = length / interval;
        if (keypoints_num > max_keypoints_num)
        {
            keypoints_num = max_keypoints_num;
        }
        else if (keypoints_num < min_keypoints_num)
        {
            keypoints_num = min_keypoints_num;
        }
        else if (keypoints_num % 2 == 0)
        {
            keypoints_num++;
        }
        // std::cout << "keypoints_num = " << keypoints_num << std::endl;

        std::vector<cv::Point2f> kps(keypoints_num);

        cv::Point2f inner_sPoint, inner_ePoint;
        // float tip = 5 / length / 2 / 2;
        double tip = (tip_size / length) < (1.0 / (keypoints_num + 1)) ? (tip_size / length) : 1.0 / (keypoints_num + 1);
        inner_sPoint = (1.0 - tip) * sPoint + tip * ePoint;
        inner_ePoint = (1.0 - tip) * ePoint + tip * sPoint;
        // std::cout << "tip = " << tip << std::endl;

        kps[0] = 0.5 * (inner_sPoint + inner_ePoint);

        for (int i = 0; i < keypoints_num / 2; i++)
        {
            float prop = 1.0 * (i) / (keypoints_num / 2);
            kps[i + 1] = (1.0 - prop) * inner_sPoint + prop * kps[0];
        }

        for (int i = 0; i < keypoints_num / 2; i++)
        {
            float prop = 1.0 * (i + 1) / (keypoints_num / 2);
            kps[i + 1 + keypoints_num / 2] = (1.0 - prop) * kps[0] + prop * inner_ePoint;
        }

        return kps;
    }
};


typedef cv::line_descriptor_custom::KeyLine KeyLine;

class LineFeatureTrackerLOFT : public LineFeatureTrackerBase
{
public:
    LineFeatureTrackerLOFT() {}

    virtual void clearState() override;
    virtual void setParameters() override;

    virtual bool trackImage(const double, const cv::Mat &, const cv::Mat &image1 = cv::Mat()) override;
    void detectLine(const cv::Mat &, std::vector<cv::Vec4f> &);
    void removeEdgeLine(const cv::Mat &, std::vector<cv::Vec4f> &);
    void setMask(cv::Mat &, std::vector<Line> &);
    void setIndex(cv::Mat &, std::vector<Line> &);
    void drawTrack(const cv::Mat &, const cv::Mat &);

    void reconstructLine(const cv::Mat &, std::vector<Line> &, std::vector<Line> &, std::vector<uchar> &);
    bool checkGoodLine(const std::vector<cv::Point2f> &, const cv::Mat &, const cv::Mat);
    bool pruneLine(Line &, const cv::Mat &,  const cv::Mat &, const cv::Mat &);
    bool extendLine(Line &, const cv::Mat &, const cv::Mat &);
    bool mergeLine(const Line &, std::vector<Line> &, const cv::Mat &, cv::Mat &);
    void randomQuickSort(std::vector<cv::Vec4f> &, std::vector<int> &, int, int);
    bool checkIntersect(const Line &, const cv::Mat &);
    void drawLineIndex(const Line &, cv::Mat &, int);

private:
    // previous frame
    std::vector<Line> pre_image0_lines;

    // current frame
    std::vector<Line> cur_image0_lines, cur_image1_lines;
    std::vector<int> cur_image0_line_ids, cur_image1_line_ids;
    std::vector<int> cur_image0_track_counters, cur_image1_track_counters;

    double image0_greydoor, image1_greydoor;
    int lines_mask_thickness, lines_index_thickness;

    std::thread parallel_thread;

    // line flow
    std::vector<std::pair<std::vector<cv::Vec4f>, cv::Scalar>> cur_image0_lineflows, cur_image1_lineflows;

    cv::Mat imageTrack;
};

#endif
