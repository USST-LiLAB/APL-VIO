#include "odom_utils/utils.h"

#include "factor/projectionTwoFrameTwoCamFactor.h"


Eigen::Matrix2d ProjectionTwoFrameTwoCamFactor::sqrt_info;

ProjectionTwoFrameTwoCamFactor::ProjectionTwoFrameTwoCamFactor(
    const Eigen::Vector3d &_pts_i, const Eigen::Vector2d &_velocity_i, const double _td_i,
    const Eigen::Vector3d &_pts_j, const Eigen::Vector2d &_velocity_j, const double _td_j) : 
    pts_i(_pts_i), pts_j(_pts_j), td_i(_td_i), td_j(_td_j)
{
    velocity_i.x() = _velocity_i.x();
    velocity_i.y() = _velocity_i.y();
    velocity_i.z() = 0;
    velocity_j.x() = _velocity_j.x();
    velocity_j.y() = _velocity_j.y();
    velocity_j.z() = 0;

#ifdef UNIT_SPHERE_ERROR
    Eigen::Vector3d b1, b2;
    Eigen::Vector3d a = pts_j.normalized();
    Eigen::Vector3d tmp(0, 0, 1);
    if (a == tmp)
        tmp << 1, 0, 0;
    b1 = (tmp - a * (a.transpose() * tmp)).normalized();
    b2 = a.cross(b1);
    tangent_base.block<1, 3>(0, 0) = b1.transpose();
    tangent_base.block<1, 3>(1, 0) = b2.transpose();
#endif
};

bool ProjectionTwoFrameTwoCamFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    const Eigen::Map<const Eigen::Vector3d> Pi(&parameters[0][0]);
    const Eigen::Map<const Eigen::Quaterniond> Qi(&parameters[0][3]); // array order: (x, y, z, w)

    const Eigen::Map<const Eigen::Vector3d> Pj(&parameters[1][0]);
    const Eigen::Map<const Eigen::Quaterniond> Qj(&parameters[1][3]); // array order: (x, y, z, w)

    const Eigen::Map<const Eigen::Vector3d> tic(&parameters[2][0]);
    const Eigen::Map<const Eigen::Quaterniond> qic(&parameters[2][3]); // array order: (x, y, z, w)

    const Eigen::Map<const Eigen::Vector3d> tic2(&parameters[3][0]);
    const Eigen::Map<const Eigen::Quaterniond> qic2(&parameters[3][3]); // array order: (x, y, z, w)

    const double inv_dep_i = parameters[4][0];

    const double td = parameters[5][0];


    /* project feature from left image of frame_i to right image of frame_j */
    Eigen::Vector3d pts_i_td, pts_j_td;
    pts_i_td = pts_i - (td - td_i) * velocity_i;
    pts_j_td = pts_j - (td - td_j) * velocity_j;
    Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic2.inverse() * (pts_imu_j - tic2);


    /* calculate residual */
    Eigen::Map<Eigen::Vector2d> residual(residuals);
#ifdef UNIT_SPHERE_ERROR 
    residual =  tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
    const double inv_dep_j = 1.0 / pts_camera_j.z();
    residual = (pts_camera_j * inv_dep_j).head<2>() - pts_j_td.head<2>();
#endif
    residual = sqrt_info * residual;

    
    /* calculate jacobians */
    if (jacobians)
    {
        Eigen::Matrix3d Ri = Qi.toRotationMatrix();
        Eigen::Matrix3d Rj = Qj.toRotationMatrix();
        Eigen::Matrix3d ric = qic.toRotationMatrix();
        Eigen::Matrix3d ric2 = qic2.toRotationMatrix();

        Eigen::Matrix<double, 2, 3> reduce(2, 3);
#ifdef UNIT_SPHERE_ERROR
        double norm = pts_camera_j.norm();
        Eigen::Matrix3d norm_jaco;
        double x1, x2, x3;
        x1 = pts_camera_j(0);
        x2 = pts_camera_j(1);
        x3 = pts_camera_j(2);
        norm_jaco << 1.0 / norm - x1 * x1 / pow(norm, 3), - x1 * x2 / pow(norm, 3),            - x1 * x3 / pow(norm, 3),
                     - x1 * x2 / pow(norm, 3),            1.0 / norm - x2 * x2 / pow(norm, 3), - x2 * x3 / pow(norm, 3),
                     - x1 * x3 / pow(norm, 3),            - x2 * x3 / pow(norm, 3),            1.0 / norm - x3 * x3 / pow(norm, 3);
        reduce = tangent_base * norm_jaco;
#else
        reduce << 
            inv_dep_j, 0, -pts_camera_j(0) * (inv_dep_j * inv_dep_j),
            0, inv_dep_j, -pts_camera_j(1) * (inv_dep_j * inv_dep_j);
#endif
        reduce = sqrt_info * reduce;

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

            Eigen::Matrix<double, 3, 6> jaco_i;
            jaco_i.leftCols<3>() = ric2.transpose() * Rj.transpose();
            jaco_i.rightCols<3>() = ric2.transpose() * Rj.transpose() * Ri * -math_utils::skewSymmetric(pts_imu_i);

            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }

        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

            Eigen::Matrix<double, 3, 6> jaco_j;
            jaco_j.leftCols<3>() = ric2.transpose() * -Rj.transpose();
            jaco_j.rightCols<3>() = ric2.transpose() * math_utils::skewSymmetric(pts_imu_j);

            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
            Eigen::Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = ric2.transpose() * Rj.transpose() * Ri; 
            jaco_ex.rightCols<3>() = ric2.transpose() * Rj.transpose() * Ri * ric * -math_utils::skewSymmetric(pts_camera_i);
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        if (jacobians[3])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose1(jacobians[3]);
            Eigen::Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = - ric2.transpose();
            jaco_ex.rightCols<3>() = math_utils::skewSymmetric(pts_camera_j);
            jacobian_ex_pose1.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose1.rightCols<1>().setZero();
        }
        if (jacobians[4])
        {
            Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[4]);
#if 1
            jacobian_feature = reduce * ric2.transpose() * Rj.transpose() * Ri * ric * pts_i_td * -1.0 / (inv_dep_i * inv_dep_i);
#else
            jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i;
#endif
        }
        if (jacobians[5])
        {
            Eigen::Map<Eigen::Vector2d> jacobian_td(jacobians[5]);
            jacobian_td = reduce * ric2.transpose() * Rj.transpose() * Ri * ric * velocity_i / inv_dep_i * -1.0  +
                          sqrt_info * velocity_j.head(2);
        }
    }

    return true;
}
