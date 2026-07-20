#ifndef __FEATURE_TRACKER_BASE_H
#define __FEATURE_TRACKER_BASE_H

#include "estimator/customType.h"


class FeatureTrackerBase
{
public:
    FeatureTrackerBase() : new_feature_id(0), pre_frame_time(0.0), cur_frame_time(0.0) {}
    virtual ~FeatureTrackerBase() {}

    virtual void clearState() = 0;
    virtual void setParameters() = 0;

    virtual bool trackImage(const double, const cv::Mat &, const cv::Mat &image1 = cv::Mat()) = 0;

    FrameFeatures frame_features; // store results
    
protected:
    int new_feature_id; // feature id is unique

    // previous frame
    double pre_frame_time;
    cv::Mat pre_image0;

    // current frame
    double cur_frame_time;
    cv::Mat cur_image0, cur_image1;
};

#endif
