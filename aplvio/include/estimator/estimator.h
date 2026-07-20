#ifndef __ESTIMATOR_H
#define __ESTIMATOR_H

#include <thread>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "estimator/customType.h"


class MarginalizationInfo;
class Initializer;
class ImuProcessor;
class ImageProcessor;
class FeatureManager;
class LineFeatureManager;

class Estimator
{
public:
    Estimator();
    ~Estimator();

    void clearState();
    void setParameters();

    void setThreads();
    void process();
    void lineBundleAdjust();
    void optimization();
    void state2ceres();
    void ceres2state();
    void addParameterBlock(ceres::Problem &);
    void addResidualBlock(ceres::Problem &, ceres::LossFunction *);
    void marginalization(ceres::Problem &, ceres::LossFunction *);
    void shiftMargAddrOld(MarginalizationInfo *);
    void shiftMargAddrNew(MarginalizationInfo *);
    void slideWindow();
    void slideWindowOld();
    void slideWindowNew();

    void updateLatestStates();
    void pubLatestStates();

    std::shared_ptr<Initializer> initializer;
    std::shared_ptr<ImuProcessor> imu_processor;
    std::shared_ptr<ImageProcessor> image_processor;
    std::shared_ptr<FeatureManager> feature_manager;
    std::shared_ptr<LineFeatureManager> linefeature_manager;

    // slide windows
    FrameWindow state_window;

private:
    std::thread process_thread;
    std::mutex process_mutex;

    bool have_init_imu;

    int cur_frame_count;
    double pre_frame_time, cur_frame_time; // frame time is equal to instant0 time

    ImageFrameList image_frame_list;

    std::vector<double> invdepth_vector;
    std::vector<Eigen::Vector4d> ortholine_vector;

    // ceres solver
    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[1000][SIZE_FEATURE];
    double para_LineFeature[1000][SIZE_LINEFEATURE];
    double para_Ex_Pose[2][SIZE_POSE];
    double para_Td[1][1];

    MarginalizationInfo *last_marginalization_info = nullptr;

    BodyState marg_state;

    std::vector<Eigen::Vector3d> key_poses;
};

#endif
