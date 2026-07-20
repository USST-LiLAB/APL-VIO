#ifndef __INTERGRATION_BASE_H
#define __INTERGRATION_BASE_H

#include <vector>
#include <Eigen/Dense>


class IntegrationBase
{
public:    
    IntegrationBase() = delete;
    IntegrationBase(const Eigen::Vector3d &, const Eigen::Vector3d &, const Eigen::Vector3d &, const Eigen::Vector3d &);
    IntegrationBase(const IntegrationBase &) = default;

    void clearState();

    void push_back(const double, const Eigen::Vector3d &, const Eigen::Vector3d &);
    void propagate(const double, const Eigen::Vector3d &, const Eigen::Vector3d &);
    void updateJacobian(const double, const Eigen::Matrix3d &, const Eigen::Matrix3d &, const Eigen::Matrix3d &, const Eigen::Matrix3d &, const Eigen::Matrix3d &);

    void repropagate(const Eigen::Vector3d &, const Eigen::Vector3d &);

    Eigen::Matrix<double, 15, 1> Evaluate(
        const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
        const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj);

    double dt; // dt from previous imu timestamp to current imu timestamp
    Eigen::Vector3d pre_acc, pre_gyr; // acc and gyr at previous imu timestamp
    // Eigen::Vector3d cur_acc, cur_gyr; // acc and gyr at current imu timestamp

    std::vector<double> dt_vector; // vector of dt during previous frame time to current frame time
    std::vector<Eigen::Vector3d> acc_vector; // vector of acceleration during previous frame time to current frame time
    std::vector<Eigen::Vector3d> gyr_vector; // vector of gyroscope during previous frame time to current frame time

    Eigen::Vector3d pre_frame_acc, pre_frame_gyr; // acceleration and gyroscope at previous frame time
    Eigen::Vector3d ba, bg; // bias of acceleration and gyroscope at previous imu timestamp

    double sum_dt; // sum t from previous frame time to current imu timestamp
    Eigen::Vector3d sum_alpha; // sum p from previous frame time to current imu timestamp
    Eigen::Vector3d sum_beta; // sum v from previous frame time to current imu timestamp
    Eigen::Quaterniond sum_gamma; // sum q from previous frame time to current imu timestamp

    Eigen::Matrix<double, 15, 15> state_jacobian;
    Eigen::Matrix<double, 15, 15> state_cov;
    Eigen::Matrix<double, 18, 18> noise_cov;
};

#endif
