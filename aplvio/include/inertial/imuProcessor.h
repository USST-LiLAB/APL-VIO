#ifndef __IMU_PROCESSOR_H
#define __IMU_PROCESSOR_H

#include <thread>
#include <sensor_msgs/Imu.h>

#include "rosMaster/rosMaster.h"


class ImuProcessor
{
public:
    void clearState();
    void setParameters();
    void imu_callback(const sensor_msgs::ImuConstPtr &);

    bool checkImu(double) const;
    bool getInterframeImuData(const double, const double, std::vector<ImuData> &);
    void interframeIntegration(FrameState &, const double, const double, const std::vector<ImuData> &);
    void fastPredict(BodyState &, const double, const ImuData &);
    bool initImuState(const std::vector<ImuData> &, BodyState &);
    static Eigen::Matrix3d alignGravity(const Eigen::Vector3d &);

    ImuBuffer imu_buffer;
    mutable std::mutex imu_buffer_mutex;
    
private:
    ImuData imu_data;
};

#endif
