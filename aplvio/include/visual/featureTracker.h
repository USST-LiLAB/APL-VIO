#ifndef __FEATURE_TRACKER_H
#define __FEATURE_TRACKER_H

#include "estimator/customType.h"
#include "featureTrackerBase.h"


class FeatureTracker : public FeatureTrackerBase
{
public:
    FeatureTracker() {}

    virtual void clearState() override;
    virtual void setParameters() override;

    virtual bool trackImage(const double, const cv::Mat &, const cv::Mat &image1 = cv::Mat()) override;

    void setMask(cv::Mat &);
    void mapIdsPoints(const std::vector<int> &, const std::vector<cv::Point2f> &, std::map<int, cv::Point2f> &);
    void calcPointsVelecity(const std::vector<int> &, const std::vector<cv::Point2f> &, std::map<int, cv::Point2f> &, std::map<int, cv::Point2f> &, std::vector<cv::Point2f> &);
    void drawTrack(const cv::Mat &, const cv::Mat &);

private:
    // previous frame
    std::vector<cv::Point2f> pre_image0_pts;
    std::map<int, cv::Point2f> pre_image0_ids_unpts_map, pre_image1_ids_unpts_map; // map: id --> 2d point
    std::map<int, cv::Point2f> pre_image0_ids_pts_map; // map: id --> 2d point

    // current frame
    std::vector<int> cur_image0_ids, cur_image1_ids, track_counters;
    std::vector<cv::Point2f> cur_image0_pts, cur_image1_pts, new_image0_pts;
    std::vector<cv::Point2f> cur_image0_unpts, cur_image1_unpts;
    std::map<int, cv::Point2f> cur_image0_ids_unpts_map, cur_image1_ids_unpts_map; // map: id --> 2d point
    std::vector<cv::Point2f> image0_unpts_vel, image1_unpts_vel;

    cv::Mat mask;
    cv::Mat imageTrack;
};

#endif
