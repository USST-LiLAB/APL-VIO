#include "odom_utils/utils.h"
#include "estimator/paramPool.h"

#include "factor/integrationBase.h"


extern std::unique_ptr<ParamPool> shared_pool;

/** \brief Integration base
 * @param _pre_frame_acc acceleration at previous frame time
 * @param _pre_frame_gyr gyroscope at previous frame time
 * @param _ba bias of acceleration at previous frame time
 * @param _bg bias of gyroscope at previous frame time
 */
IntegrationBase::IntegrationBase(const Eigen::Vector3d &_pre_frame_acc, const Eigen::Vector3d &_pre_frame_gyr, const Eigen::Vector3d &_ba, const Eigen::Vector3d &_bg)
    : pre_acc(_pre_frame_acc), pre_gyr(_pre_frame_gyr), pre_frame_acc(_pre_frame_acc), pre_frame_gyr(_pre_frame_gyr), ba(_ba), bg(_bg),
      sum_dt(0.0), sum_alpha(Eigen::Vector3d::Zero()), sum_beta(Eigen::Vector3d::Zero()), sum_gamma(Eigen::Quaterniond::Identity()),
      state_jacobian(Eigen::Matrix<double, 15, 15>::Identity()), state_cov(Eigen::Matrix<double, 15, 15>::Zero()), 
      noise_cov(Eigen::Matrix<double, 18, 18>::Zero())
{
    noise_cov.block<3, 3>(0, 0) = (shared_pool->acc_n * shared_pool->acc_n) * Eigen::Matrix3d::Identity();
    noise_cov.block<3, 3>(3, 3) = (shared_pool->gyr_n * shared_pool->gyr_n) * Eigen::Matrix3d::Identity();
    noise_cov.block<3, 3>(6, 6) = (shared_pool->acc_n * shared_pool->acc_n) * Eigen::Matrix3d::Identity();
    noise_cov.block<3, 3>(9, 9) = (shared_pool->gyr_n * shared_pool->gyr_n) * Eigen::Matrix3d::Identity();
    noise_cov.block<3, 3>(12, 12) = (shared_pool->acc_w * shared_pool->acc_w) * Eigen::Matrix3d::Identity();
    noise_cov.block<3, 3>(15, 15) = (shared_pool->gyr_w * shared_pool->gyr_w) * Eigen::Matrix3d::Identity();
}


void IntegrationBase::clearState()
{
    pre_acc = pre_frame_acc;
    pre_gyr = pre_frame_gyr;

    sum_dt = 0.0;
    sum_alpha.setZero();
    sum_beta.setZero();
    sum_gamma.setIdentity();

    state_jacobian.setIdentity();
    state_cov.setZero();
}


void IntegrationBase::push_back(const double dt, const Eigen::Vector3d &cur_acc, const Eigen::Vector3d &cur_gyr)
{
    dt_vector.push_back(dt);
    acc_vector.push_back(cur_acc);
    gyr_vector.push_back(cur_gyr);
    propagate(dt, cur_acc, cur_gyr);
}


/** \brief propagate alpha, beta and gamma
 * @param dt dt from previous imu timestamp to current imu timestamp
 * @param cur_acc acceleration from current imu timestamp
 * @param cur_gyr gyroscope at current imu timestamp
 */
void IntegrationBase::propagate(const double dt, const Eigen::Vector3d &cur_acc, const Eigen::Vector3d &cur_gyr)
{
    // gamma
    Eigen::Vector3d unbias_gyr = 0.5 * (pre_gyr + cur_gyr) - bg;
    Eigen::Quaterniond gamma = math_utils::theta2dq(unbias_gyr * dt);
    Eigen::Quaterniond result_sum_gamma = sum_gamma * gamma;
    
    Eigen::Vector3d pre_unbias_acc = sum_gamma * (pre_acc - ba);
    Eigen::Vector3d cur_unbias_acc = result_sum_gamma * (cur_acc - ba);
    Eigen::Vector3d avg_unbias_acc = 0.5 * (pre_unbias_acc + cur_unbias_acc);
    // beta
    Eigen::Vector3d beta = avg_unbias_acc * dt;
    Eigen::Vector3d result_sum_beta = sum_beta + beta;

    // alpha
    Eigen::Vector3d alpha = 0.5 * beta * dt;
    Eigen::Vector3d result_sum_alpha = sum_alpha + sum_beta * dt + alpha;

    // update_jacobian
    {
        Eigen::Matrix3d R0 = sum_gamma.toRotationMatrix();
        Eigen::Matrix3d R1 = result_sum_gamma.toRotationMatrix();
        Eigen::Matrix3d w_x = math_utils::skewSymmetric(0.5 * (pre_gyr + cur_gyr) - bg);
        Eigen::Matrix3d a0_x = math_utils::skewSymmetric(pre_acc - ba);
        Eigen::Matrix3d a1_x = math_utils::skewSymmetric(cur_acc - ba);
        Eigen::Matrix3d R0_a0_x = R0 * a0_x;
        Eigen::Matrix3d R1_a1_x = R1 * a1_x;
        
        updateJacobian(dt, w_x, R0, R1, R0_a0_x, R1_a1_x);
    }

    sum_dt += dt;
    sum_alpha = result_sum_alpha;
    sum_beta = result_sum_beta;
    sum_gamma = result_sum_gamma.normalized();
    pre_acc = cur_acc;
    pre_gyr = cur_gyr;
}


/** \brief propagate jacobian for pre-integration
 * @param dt delta t
 * @param w_x [0.5*(w_k + w_{k+1}) - bg_k]_x
 * @param R0 R_k
 * @param R1 R_{k+1}
 * @param R0_a0_x R0 * [a_k - ba_k]_x
 * @param R1_a1_x R1 * [a_{k+1} - ba_{k+1}]_x
 */
void IntegrationBase::updateJacobian(const double dt, const Eigen::Matrix3d &w_x, 
                                     const Eigen::Matrix3d &R0, const Eigen::Matrix3d &R1, 
                                     const Eigen::Matrix3d &R0_a0_x, const Eigen::Matrix3d &R1_a1_x)
{
    /**********************************************************************************************************************************************
        w_x = [0.5*(w_k + w_{k+1}) - bg_k]_x
        R0 = R_k; 
        R1 = R_{k+1}
        R0_a0_x = R0 * [a_k - ba_k]_x
        R1_a1_x = R1 * [a_{k+1} - ba_{k+1}]_x

        --     --   ---                                                                                                           ---   --     --
        |d_alpha|   |   I   -0.25*(R0_a0_x + R1_a1_x*(I-w_x*dt))*dt*dt  dt     -0.25*(R0 + R1)*dt*dt  0.25*R1_a1_x*dt*dt*dt         |   |d_alpha|
        |       |   |                                                                                                               |   |       |
        |d_theta|   |   0   I-w_x*dt                                    0       0                           -dt                     |   |d_theta|
        |       |   |                                                                                                               |   |       |
        |d_beta | = |   0   -0.5*(R0_a0_x + R1_a1_x*(I-w_x*dt))*dt      I       -0.5*(R0 + R1)*dt     0.5*R1_a1_x*dt*dt             | * |d_beta |
        |       |   |                                                                                                               |   |       |
        |d_ba   |   |   0   0                                           0       I                           0                       |   |d_ba   |
        |       |   |                                                                                                               |   |       |
        |d_bg   |   |   0   0                                           0       0                           I                       |   |d_bg   |
        --     --   ---                                                                                                           ---   --     --
                                                                                                                                        --  --
                    ---                                                                                                           ---   |n_a |
                    |   -0.25*R0*dt*dt      0.25*R1_a1_x*dt*dt*0.5*dt   -0.25*R1_a1_x*dt*dt     0.25*R1_a1_x*dt*dt*0.5*dt   0   0   |   |    |
                    |                                                                                                               |   |n_w |
                    |   0                   -0.5*dt                     0                       -0.5*dt                     0   0   |   |    |
                    |                                                                                                               | * |n_a |
                  - |   -0.5*R0*dt          0.25*R1_a1_x*dt*dt          -0.5*R1*dt              0.25*R1_a1_x*dt*dt          0   0   |   |    |
                    |                                                                                                               |   |n_w |
                    |   0                   0                           0                       0                           dt  0   |   |    |
                    |                                                                                                               |   |n_ba|
                    |   0                   0                           0                       0                           0   dt  |   |    |
                    ---                                                                                                           ---   |n_bg|
                                                                                                                                        --  --
    **********************************************************************************************************************************************/

    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(15, 15);
    F.block<3, 3>(0, 0)     =   Eigen::Matrix3d::Identity();
    F.block<3, 3>(0, 3)     =   -0.25 * (R0_a0_x + R1_a1_x * (Eigen::Matrix3d::Identity() - w_x * dt)) * dt * dt;
    F.block<3, 3>(0, 6)     =   Eigen::MatrixXd::Identity(3, 3) * dt;
    F.block<3, 3>(0, 9)     =   -0.25 * (R0 + R1) * dt * dt;
    F.block<3, 3>(0, 12)    =   -0.25 * R1_a1_x * dt * dt * -dt;
    F.block<3, 3>(3, 3)     =   Eigen::Matrix3d::Identity() - w_x * dt;
    F.block<3, 3>(3, 12)    =   Eigen::MatrixXd::Identity(3,3) * -dt;
    F.block<3, 3>(6, 3)     =   -0.5 * (R0_a0_x + R1_a1_x * (Eigen::Matrix3d::Identity() - w_x * dt)) * dt;
    F.block<3, 3>(6, 6)     =   Eigen::Matrix3d::Identity();
    F.block<3, 3>(6, 9)     =   -0.5 * (R0 + R1) * dt;
    F.block<3, 3>(6, 12)    =   0.5 * R1_a1_x * dt * dt;
    F.block<3, 3>(9, 9)     =   Eigen::Matrix3d::Identity();
    F.block<3, 3>(12, 12)   =   Eigen::Matrix3d::Identity();

    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(15,18);
    V.block<3, 3>(0, 0)     =   0.25 * R0 * dt * dt;
    V.block<3, 3>(0, 3)     =   0.25 * -R1_a1_x  * dt * dt * 0.5 * dt;
    V.block<3, 3>(0, 6)     =   0.25 * R1 * dt * dt;
    V.block<3, 3>(0, 9)     =   V.block<3, 3>(0, 3);
    V.block<3, 3>(3, 3)     =   0.5 * Eigen::MatrixXd::Identity(3,3) * dt;
    V.block<3, 3>(3, 9)     =   V.block<3, 3>(3, 3);
    V.block<3, 3>(6, 0)     =   0.5 * R0 * dt;
    V.block<3, 3>(6, 3)     =   0.25 * -R1_a1_x  * dt * dt;
    V.block<3, 3>(6, 6)     =   0.5 * R1 * dt;
    V.block<3, 3>(6, 9)     =   V.block<3, 3>(6, 3);
    V.block<3, 3>(9, 12)    =   Eigen::MatrixXd::Identity(3, 3) * -dt;
    V.block<3, 3>(12, 15)   =   V.block<3, 3>(9, 12);

    state_jacobian = F * state_jacobian;
    state_cov = F * state_cov * F.transpose() + V * noise_cov * V.transpose();

    // ROS_ASSERT(fabs(state_jacobian.maxCoeff()) < 1e8);
    // ROS_ASSERT(fabs(state_jacobian.minCoeff()) < 1e8);
}


/** \brief repropagate alpha, beta and gamma
 * @param _ba bias of acceleration with significant change
 * @param _bg bias of gyroscope with significant change
 */
void IntegrationBase::repropagate(const Eigen::Vector3d &_ba, const Eigen::Vector3d &_bg)
{
    clearState();
    
    ba = _ba;
    bg = _bg;

    for (size_t i = 0; i != dt_vector.size(); i++)
        propagate(dt_vector[i], acc_vector[i], gyr_vector[i]);
}


Eigen::Matrix<double, 15, 1> IntegrationBase::Evaluate(
    const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
    const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj)
{
    Eigen::Matrix<double, 15, 1> residuals;

    const Eigen::Matrix3d &dp_dba = state_jacobian.block<3, 3>(P_order, BA_order);
    const Eigen::Matrix3d &dp_dbg = state_jacobian.block<3, 3>(P_order, BG_order);

    const Eigen::Matrix3d &dq_dbg = state_jacobian.block<3, 3>(R_order, BG_order);

    const Eigen::Matrix3d &dv_dba = state_jacobian.block<3, 3>(V_order, BA_order);
    const Eigen::Matrix3d &dv_dbg = state_jacobian.block<3, 3>(V_order, BG_order);

    const Eigen::Vector3d dba = Bai - ba;
    const Eigen::Vector3d dbg = Bgi - bg;


    Eigen::Quaterniond corrected_delta_q = sum_gamma * math_utils::theta2dq(dq_dbg * dbg);
    Eigen::Vector3d corrected_delta_v = sum_beta + dv_dba * dba + dv_dbg * dbg;
    Eigen::Vector3d corrected_delta_p = sum_alpha + dp_dba * dba + dp_dbg * dbg;

    residuals.block<3, 1>(P_order, 0) = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
    residuals.block<3, 1>(R_order, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
    residuals.block<3, 1>(V_order, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
    residuals.block<3, 1>(BA_order, 0) = Baj - Bai;
    residuals.block<3, 1>(BG_order, 0) = Bgj - Bgi;

    return residuals;
}