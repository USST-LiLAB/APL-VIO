#ifndef __CUSTOM_TYPE_H
#define __CUSTOM_TYPE_H

#include "estimator/parameters.h"


class IntegrationBase;

enum StateOrder
{
    P_order = 0,  // Poistion covariance
    R_order = 3,  // Rotation covariance
    V_order = 6,  // Velocity covariance
    BA_order = 9, // acc bais covariance
    BG_order = 12 // gyr bais covariance
};

enum NoiseOrder
{
    AN_order = 0, // acc noise
    GN_order = 3, // gyr noise
    AW_order = 6, // acc random walk
    GW_order = 9  // gyr random walk
};

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1,
    SIZE_LINEFEATURE = 4
};

enum SolverFlag
{
    INITIAL = 0, // initial period
    NONLINEAR = 1 // nonlinear optimization period
};

enum MarginalizationFlag
{
    MARGIN_OLD = 0, // margin the oldest keyframe
    MARGIN_SECOND_NEW = 1 // margin the second newest non-keyframe
};


typedef std::pair<int, std::pair<int, cv::Point2f>> CntIdPts;

// FeaturePoint is a camera_id and xyz_uv_velocity pair
typedef std::pair<int, Eigen::Matrix<double, 7, 1>> FeaturePoint;
// FeaturePoints is a map from feature_id to FeaturePoint
typedef std::map<int, std::vector<FeaturePoint>> FeaturePoints;
// FrameFeatures is a frame_time and FeaturePoints pair
typedef std::pair<double, FeaturePoints> FrameFeatures;
// FrameBuffer is a queue of FrameFeatures
typedef std::queue<FrameFeatures> FrameBuffer;

// FeatureLine is a camera_id and xy0_xy1 pair
typedef std::pair<int, Eigen::Vector4d> FeatureLine;
// FeatureLines is a map from line_id to FeatureLine
typedef std::map<int, std::vector<FeatureLine>> FeatureLines;
// FrameLineFeatures is a frame_time and FeatureLines pair
typedef std::pair<double, FeatureLines> FrameLineFeatures;
// FrameLineBuffer is a queue of FrameLineFeatures
typedef std::queue<FrameLineFeatures> FrameLineBuffer;


struct Extrinsics
{
    Eigen::Matrix4d T_imu_cam[2]; // transform from cam to imu
    Eigen::Matrix3d R_imu_cam[2]; // rotation from cam to imu
    Eigen::Vector3d t_imu_cam[2]; // translation from cam to imu
};


struct ImuData
{
    double t;
    Eigen::Vector3d acc;
    Eigen::Vector3d gyr;
};
typedef std::queue<ImuData> ImuBuffer;


struct BodyState
{
    BodyState();
    BodyState &operator=(const BodyState &);

    void clearState();
    void swap(BodyState &);

    double t;
    Eigen::Vector3d p;
    Eigen::Quaterniond q;
    Eigen::Matrix3d R;
    Eigen::Vector3d v;
    Eigen::Vector3d ba;
    Eigen::Vector3d bg;
    Eigen::Vector3d acc;
    Eigen::Vector3d gyr;
};


struct FrameState
{
    BodyState body_state;
    std::shared_ptr<IntegrationBase> pre_integration; // select shared_ptr for safety

    FrameState();

    void clearState();
    void swap(FrameState &);
};
typedef FrameState FrameWindow[WINDOW_SIZE + 1];


struct FeaturePerFrame
{
    FeaturePerFrame() = delete;
    FeaturePerFrame(const std::vector<FeaturePoint> &, double);

    bool is_stereo;
    double td;
    Eigen::Vector3d point_0, point_1;
    Eigen::Vector2d uv_0, uv_1;
    Eigen::Vector2d vel_0, vel_1;
};


struct FeaturePerId
{
    FeaturePerId(const int, const int);

    inline int endFrame() const { return start_frame + feature_per_frame.size() - 1; };

    const int feature_id;
    int start_frame;
    std::vector<FeaturePerFrame> feature_per_frame;
    double depth; // estimated feature depth relative to start_frame
    int solve_flag; // 0 haven't solve yet; 1 solve successfully; -1 solve failed;
};
typedef std::list<FeaturePerId> FeatureList;
typedef std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> FeaturePairs;


struct LineFeaturePerFrame
{
    LineFeaturePerFrame();
    LineFeaturePerFrame(const std::vector<FeatureLine> &);

    bool is_stereo;
    bool is_tracked;
    Eigen::Vector4d line_0, line_1; // line in the left and right image
};


struct LineFeaturePerId
{
    LineFeaturePerId() = delete;
    LineFeaturePerId(const int, const int);

    inline int endFrame() const { return start_frame + linefeature_per_frame.size() - 1; };

    int line_id;
    int start_frame;
    int track_count;
    std::vector<LineFeaturePerFrame> linefeature_per_frame;
    Eigen::Matrix<double, 6, 1> plucker_w; // plucher in world frame
    bool is_triangulate;
    bool is_published;
    int opt_num; // number of times for optimization
    int solve_flag; // 0 haven't solve yet; 1 solve successfully; -1 solve failed;
};
typedef std::list<LineFeaturePerId> LineFeatureList;


struct SFMFeature
{
	SFMFeature(const FeaturePerId &);

    bool is_triangulate;
    int feature_id;
    std::vector<std::pair<int, Eigen::Vector2d>> covisible_frames;
    double para_point3d[3];
    double depth;
};


struct ImageFrame
{
    ImageFrame();

    double t;
    FeaturePoints feature_points;
    FeatureLines feature_lines;

    Eigen::Matrix3d R_w_imu;
    Eigen::Vector3d t_w_imu;
    
    std::shared_ptr<IntegrationBase> pre_integration;
    
    bool is_keyframe;
};
typedef std::list<ImageFrame> ImageFrameList;

#endif
