#include "odom_utils/utils.h"
#include "factor/integrationBase.h"
#include "estimator/paramPool.h"
#include "estimator/customType.h"

#include "factor/imuFactor.h"


Eigen::Matrix<double, 15, 15> ImuFactor::sqrt_info;

ImuFactor::ImuFactor(std::shared_ptr<IntegrationBase> _pre_integration) : pre_integration(_pre_integration)
{
    // nothing
}


bool ImuFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    const Eigen::Map<const Eigen::Vector3d> Pi(&parameters[0][0]);
    const Eigen::Map<const Eigen::Quaterniond> Qi(&parameters[0][3]); // array order: (x, y, z, w)

    const Eigen::Map<const Eigen::Vector3d> Vi(&parameters[1][0]);
    const Eigen::Map<const Eigen::Vector3d> Bai(&parameters[1][3]);
    const Eigen::Map<const Eigen::Vector3d> Bgi(&parameters[1][6]);

    const Eigen::Map<const Eigen::Vector3d> Pj(&parameters[2][0]);
    const Eigen::Map<const Eigen::Quaterniond> Qj(&parameters[2][3]); // array order: (x, y, z, w)

    const Eigen::Map<const Eigen::Vector3d> Vj(&parameters[3][0]);
    const Eigen::Map<const Eigen::Vector3d> Baj(&parameters[3][3]);
    const Eigen::Map<const Eigen::Vector3d> Bgj(&parameters[3][6]);


#if 0
    if ((Bai - pre_integration->ba).norm() > 0.10 ||
        (Bgi - pre_integration->bg).norm() > 0.01)
    {
        pre_integration->repropagate(Bai, Bgi);
        std::cout << "repropagate pre_integration" << std::endl;
    }
#endif

    /* calculate residual */
    Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
    residual = pre_integration->Evaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);
    sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration->state_cov.inverse()).matrixL().transpose();
    // sqrt_info.setIdentity(); // unweighted
    residual = sqrt_info * residual; // residual wighted with Mahalanobis distance


    /* calculate jacobians */
    if (jacobians)
    {
        double sum_dt = pre_integration->sum_dt;
        const Eigen::Matrix3d &dp_dba = pre_integration->state_jacobian.template block<3, 3>(P_order, BA_order);
        const Eigen::Matrix3d &dp_dbg = pre_integration->state_jacobian.template block<3, 3>(P_order, BG_order);

        const Eigen::Matrix3d &dq_dbg = pre_integration->state_jacobian.template block<3, 3>(R_order, BG_order);

        const Eigen::Matrix3d &dv_dba = pre_integration->state_jacobian.template block<3, 3>(V_order, BA_order);
        const Eigen::Matrix3d &dv_dbg = pre_integration->state_jacobian.template block<3, 3>(V_order, BG_order);

        if (pre_integration->state_jacobian.maxCoeff() > 1e8 || pre_integration->state_jacobian.minCoeff() < -1e8)
            ROS_WARN("numerical unstable in preintegration");

        // ROS_ASSERT(fabs(pre_integration->state_jacobian.maxCoeff()) < 1e8);
        // ROS_ASSERT(fabs(pre_integration->state_jacobian.minCoeff()) < 1e8);


        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
            jacobian_pose_i.setZero();

            jacobian_pose_i.block<3, 3>(P_order, P_order) = -Qi.inverse().toRotationMatrix();
            jacobian_pose_i.block<3, 3>(P_order, R_order) = math_utils::skewSymmetric(Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));
#if 0
            jacobian_pose_i.block<3, 3>(R_order, R_order) = -(Qj.inverse() * Qi).toRotationMatrix();
#else
            Eigen::Quaterniond corrected_delta_q = pre_integration->sum_gamma * math_utils::theta2dq(dq_dbg * (Bgi - pre_integration->bg));
            jacobian_pose_i.block<3, 3>(R_order, R_order) = -(math_utils::Qleft(Qj.inverse() * Qi) * math_utils::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();
#endif
            jacobian_pose_i.block<3, 3>(V_order, R_order) = math_utils::skewSymmetric(Qi.inverse() * (G * sum_dt + Vj - Vi));

            jacobian_pose_i = sqrt_info * jacobian_pose_i;

            if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                ROS_WARN("numerical unstable in preintegration");

            // ROS_ASSERT(fabs(jacobian_pose_i.maxCoeff()) < 1e8);
            // ROS_ASSERT(fabs(jacobian_pose_i.minCoeff()) < 1e8);
        }
        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
            jacobian_speedbias_i.setZero();
            jacobian_speedbias_i.block<3, 3>(P_order, V_order - V_order) = -Qi.inverse().toRotationMatrix() * sum_dt;
            jacobian_speedbias_i.block<3, 3>(P_order, BA_order - V_order) = -dp_dba;
            jacobian_speedbias_i.block<3, 3>(P_order, BG_order - V_order) = -dp_dbg;
#if 0
            jacobian_speedbias_i.block<3, 3>(R_order, BG_order - V_order) = -dq_dbg;
#else
            //Eigen::Quaterniond corrected_delta_q = pre_integration->sum_gamma * math_utils::theta2dq(dq_dbg * (Bgi - pre_integration->bg));
            //jacobian_speedbias_i.block<3, 3>(R_order, BG_order - V_order) = -math_utils::Qleft(Qj.inverse() * Qi * corrected_delta_q).bottomRightCorner<3, 3>() * dq_dbg;
            jacobian_speedbias_i.block<3, 3>(R_order, BG_order - V_order) = -math_utils::Qleft(Qj.inverse() * Qi * pre_integration->sum_gamma).bottomRightCorner<3, 3>() * dq_dbg;
#endif
            jacobian_speedbias_i.block<3, 3>(V_order, V_order - V_order) = -Qi.inverse().toRotationMatrix();
            jacobian_speedbias_i.block<3, 3>(V_order, BA_order - V_order) = -dv_dba;
            jacobian_speedbias_i.block<3, 3>(V_order, BG_order - V_order) = -dv_dbg;

            jacobian_speedbias_i.block<3, 3>(BA_order, BA_order - V_order) = -Eigen::Matrix3d::Identity();
            jacobian_speedbias_i.block<3, 3>(BG_order, BG_order - V_order) = -Eigen::Matrix3d::Identity();

            jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;

            // ROS_ASSERT(fabs(jacobian_speedbias_i.maxCoeff()) < 1e8);
            // ROS_ASSERT(fabs(jacobian_speedbias_i.minCoeff()) < 1e8);
        }
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);
            jacobian_pose_j.setZero();

            jacobian_pose_j.block<3, 3>(P_order, P_order) = Qi.inverse().toRotationMatrix();
#if 0
            jacobian_pose_j.block<3, 3>(R_order, R_order) = Eigen::Matrix3d::Identity();
#else
            Eigen::Quaterniond corrected_delta_q = pre_integration->sum_gamma * math_utils::theta2dq(dq_dbg * (Bgi - pre_integration->bg));
            jacobian_pose_j.block<3, 3>(R_order, R_order) = math_utils::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
#endif
            jacobian_pose_j = sqrt_info * jacobian_pose_j;

            // ROS_ASSERT(fabs(jacobian_pose_j.maxCoeff()) < 1e8);
            // ROS_ASSERT(fabs(jacobian_pose_j.minCoeff()) < 1e8);
        }
        if (jacobians[3])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[3]);
            jacobian_speedbias_j.setZero();

            jacobian_speedbias_j.block<3, 3>(V_order, V_order - V_order) = Qi.inverse().toRotationMatrix();
            jacobian_speedbias_j.block<3, 3>(BA_order, BA_order - V_order) = Eigen::Matrix3d::Identity();
            jacobian_speedbias_j.block<3, 3>(BG_order, BG_order - V_order) = Eigen::Matrix3d::Identity();
            jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;

            // ROS_ASSERT(fabs(jacobian_speedbias_j.maxCoeff()) < 1e8);
            // ROS_ASSERT(fabs(jacobian_speedbias_j.minCoeff()) < 1e8);
        }
    }

    return true;
}