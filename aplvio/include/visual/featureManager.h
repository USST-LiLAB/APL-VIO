#ifndef __FEATURE_MANAGER_H
#define __FEATURE_MANAGER_H

#include "estimator/customType.h"


class FeatureManager
{
public:
    FeatureManager();

    void clearState();
    void setParameters();

    void insertFeature(const int, const FeaturePoints &);
    bool isKeyframe(const int) const;
    double calcParallax(const int) const;

    FeaturePairs getFeaturePairs(const int, const int);
    void initFramePoseByPnP(const int, FrameWindow &);
    void triangulate(const FrameWindow &);
    double reprojectError(const Eigen::Vector3d &, const Eigen::Matrix3d &, const Eigen::Vector3d &, const Eigen::Matrix3d &, const Eigen::Vector3d &,
                          const Eigen::Vector3d &, const Eigen::Matrix3d &, const Eigen::Vector3d &, const Eigen::Matrix3d &, const Eigen::Vector3d &, 
                          const double);
    void detectOutliers(const FrameWindow &);
    void removeOutliers();
    
    void clearDepth();
    void getInvDepth(std::vector<double> &);
    void setDepth(std::vector<double> &);

    void removeFailures();
    void removeOldShiftDepth(const Eigen::Matrix3d &, const Eigen::Vector3d &, const Eigen::Matrix3d &, const Eigen::Vector3d &);
    void removeOld();
    void removeSecondNew(int);

    static bool solvePoseByPnP(std::vector<cv::Point3f> &, std::vector<cv::Point2f> &, Eigen::Matrix3d &, Eigen::Vector3d &, const bool);
    static void triangulatePoint(Eigen::Matrix<double, 3, 4> &, Eigen::Matrix<double, 3, 4> &, Eigen::Vector2d &, Eigen::Vector2d &, Eigen::Vector3d &);

    FeatureList feature_list;

private:
    std::set<int> outlierIndex;
    int track_num, new_track_num, long_track_num;
};

#endif
