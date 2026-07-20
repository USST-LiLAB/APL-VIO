#ifndef __LINE_FEATURE_TRACKER_BASE_H
#define __LINE_FEATURE_TRACKER_BASE_H

#include <numeric>

#include "estimator/customType.h"


class LineFeatureTrackerBase
{
public:
    LineFeatureTrackerBase() : new_line_id(0), pre_frame_time(0.0), cur_frame_time(0.0) {}
    virtual ~LineFeatureTrackerBase() {}

    virtual void clearState() = 0;
    virtual void setParameters() = 0;

    virtual bool trackImage(const double, const cv::Mat &, const cv::Mat &image1 = cv::Mat()) = 0;

    FrameLineFeatures frame_linefeatures; // store results
    
protected:
    int new_line_id; // line id is unique
    
    // previous frame
    double pre_frame_time;
    cv::Mat pre_image0;

    // current frame
    double cur_frame_time;
    cv::Mat cur_image0, cur_image1;
};

#endif
