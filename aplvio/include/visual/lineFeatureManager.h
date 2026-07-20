#ifndef __LINE_FEATURE_MANAGER_H
#define __LINE_FEATURE_MANAGER_H

#include "estimator/customType.h"


class LineFeatureManager
{
public:
    LineFeatureManager();

    void clearState();
    void setParameters();

    void insertLineFeature(const int, const FeatureLines &);
    LineFeaturePerId *matchLine(const std::pair<int, std::vector<FeatureLine>> &);
    bool isKeyframe(const int) const;
    void triangulate(const FrameWindow &);
    double reprojectError(const Eigen::Vector4d &, const Eigen::Matrix<double, 6, 1> &, const Eigen::Matrix3d &, const Eigen::Vector3d &);
    void detectOutliers(const FrameWindow &);
    void removeOutliers();
    void removeFailures();

    void getOrthoLine(std::vector<Eigen::Vector4d> &, const FrameWindow &);
    void setOrthoLine(std::vector<Eigen::Vector4d> &, const FrameWindow &);

    void removeOld();
    void removeSecondNew(int);

    static void planeTrimLine(const Eigen::Matrix<double, 6, 1> &, const Eigen::Vector4d &, Eigen::Vector3d &, Eigen::Vector3d &);

    LineFeatureList linefeature_list;

private:
    std::set<int> outlierIndex;

    int track_num, new_track_num, long_track_num;
};

#endif
