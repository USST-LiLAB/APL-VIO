#include "odom_utils/utils.h"

#include "factor/projectionLineFactor.h"


Eigen::Matrix2d ProjectionLineFactor::sqrt_info;

ProjectionLineFactor::ProjectionLineFactor(const Eigen::Vector4d &_line_i) : line_i(_line_i)
{
    // pass
}

#if !USE_NORMAL_VECTOR_RESIDUAL
bool ProjectionLineFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{

    const Eigen::Map<const Eigen::Vector3d> Pi(&parameters[0][0]);
    const Eigen::Map<const Eigen::Quaterniond> Qi(&parameters[0][3]); // array order: (x, y, z, w)

    const Eigen::Map<const Eigen::Vector3d> tic(&parameters[1][0]);
    const Eigen::Map<const Eigen::Quaterniond> qic(&parameters[1][3]); // array order: (x, y, z, w)

    const Eigen::Map<const Eigen::Vector4d> line_orth(&parameters[2][0]);


    Eigen::Matrix<double, 6, 1> line_w = geometry_utils::ortho2plucker(line_orth);

    const Eigen::Vector3d &t_w_imu = Pi;
    const Eigen::Matrix3d &R_w_imu = Qi.toRotationMatrix();
    
    Eigen::Matrix<double, 6, 1> line_b = geometry_utils::invTransformPlucker(line_w, R_w_imu, t_w_imu);

    const Eigen::Vector3d &t_imu_cam = tic;
    const Eigen::Matrix3d &R_imu_cam = qic.toRotationMatrix();

    Eigen::Matrix<double, 6, 1> line_c = geometry_utils::invTransformPlucker(line_b, R_imu_cam, t_imu_cam);

    // K is identity matrix
    Eigen::Vector3d nc = line_c.head(3);
    double l_squarednorm = nc(0) * nc(0) + nc(1) * nc(1);
    double l_norm = sqrt(l_squarednorm);
    double l_trinorm = l_squarednorm * l_norm;

    double e1 = line_i(0) * nc(0) + line_i(1) * nc(1) + nc(2);
    double e2 = line_i(2) * nc(0) + line_i(3) * nc(1) + nc(2);

    Eigen::Map<Eigen::Vector2d> residual(residuals);
    residual(0) = e1 / l_norm;
    residual(1) = e2 / l_norm;

    residual = sqrt_info * residual;


    /* calculate jacobians */
    if (jacobians)
    {
        Eigen::Matrix<double, 2, 3> jaco_e_l(2, 3);

        jaco_e_l << 
            (line_i(0) / l_norm - nc(0) * e1 / l_trinorm ), (line_i(1) / l_norm - nc(1) * e1 / l_trinorm ), 1.0 / l_norm,
            (line_i(2) / l_norm - nc(0) * e2 / l_trinorm ), (line_i(3) / l_norm - nc(1) * e2 / l_trinorm ), 1.0 / l_norm;

        jaco_e_l = sqrt_info * jaco_e_l;

        Eigen::Matrix<double, 3, 6> jaco_l_Lc(3, 6);
        jaco_l_Lc.setZero();
        jaco_l_Lc.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();

        Eigen::Matrix<double, 2, 6> jaco_e_Lc;
        jaco_e_Lc = jaco_e_l * jaco_l_Lc;

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

            Eigen::Matrix<double, 6, 6> invTbc;
            invTbc << 
                R_imu_cam.transpose(), -R_imu_cam.transpose() * math_utils::skewSymmetric(t_imu_cam),
                Eigen::Matrix3d::Zero(),  R_imu_cam.transpose();

            Eigen::Vector3d nw = line_w.head(3);
            Eigen::Vector3d dw = line_w.tail(3);

            Eigen::Matrix<double, 6, 6> jaco_Lc_pose;

            jaco_Lc_pose.setZero();
            jaco_Lc_pose.block(0, 0, 3, 3) = R_w_imu.transpose() * math_utils::skewSymmetric(dw);
            jaco_Lc_pose.block(0, 3, 3, 3) = math_utils::skewSymmetric(R_w_imu.transpose() * (nw + math_utils::skewSymmetric(dw) * t_w_imu));
            jaco_Lc_pose.block(3, 3, 3, 3) = math_utils::skewSymmetric(R_w_imu.transpose() * dw);

            jaco_Lc_pose = invTbc * jaco_Lc_pose;

            jacobian_pose_i.leftCols<6>() = jaco_e_Lc * jaco_Lc_pose;
            jacobian_pose_i.rightCols<1>().setZero();
        }
        if (jacobians[1])
        {

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[1]);

            Eigen::Vector3d nb = line_b.head(3);
            Eigen::Vector3d db = line_b.tail(3);
            Eigen::Matrix<double, 6, 6> jaco_Lc_ex;

            jaco_Lc_ex.setZero();
            jaco_Lc_ex.block(0, 0, 3, 3) = R_imu_cam.transpose() * math_utils::skewSymmetric(db);   // Lc_t
            jaco_Lc_ex.block(0, 3, 3, 3) = math_utils::skewSymmetric(R_imu_cam.transpose() * (nb + math_utils::skewSymmetric(db) * t_imu_cam));  // Lc_theta
            jaco_Lc_ex.block(3, 3, 3, 3) = math_utils::skewSymmetric(R_imu_cam.transpose() * db);

            jacobian_ex_pose.leftCols<6>() = jaco_e_Lc * jaco_Lc_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jacobian_lineOrth(jacobians[2]);

            Eigen::Matrix3d R_w_cam = R_w_imu * R_imu_cam;
            Eigen::Vector3d t_w_cam = R_w_imu * t_imu_cam + t_w_imu;
            Eigen::Matrix<double, 6, 6> invTwc;

            invTwc << 
                R_w_cam.transpose(), -R_w_cam.transpose() * math_utils::skewSymmetric(t_w_cam),
                Eigen::Matrix3d::Zero(),  R_w_cam.transpose();

            Eigen::Vector3d nw = line_w.head(3);
            Eigen::Vector3d vw = line_w.tail(3);
            Eigen::Vector3d u1 = nw / nw.norm();
            Eigen::Vector3d u2 = vw / vw.norm();
            Eigen::Vector3d u3 = u1.cross(u2);
            Eigen::Vector2d w(nw.norm(), vw.norm());
            w = w / w.norm();

            Eigen::Matrix<double, 6, 4> jaco_Lw_orth;
            jaco_Lw_orth.setZero();
            jaco_Lw_orth.block(3, 0, 3, 1) = w(1) * u3;
            jaco_Lw_orth.block(0, 1, 3, 1) = -w(0) * u3;
            jaco_Lw_orth.block(0, 2, 3, 1) = w(0) * u2;
            jaco_Lw_orth.block(3, 2, 3, 1) = -w(1) * u1;
            jaco_Lw_orth.block(0, 3, 3, 1) = -w(1) * u1;
            jaco_Lw_orth.block(3, 3, 3, 1) = w(0) * u2;

            jacobian_lineOrth = jaco_e_Lc * invTwc * jaco_Lw_orth;
        }
    }
    
    return true;
}

#else

bool ProjectionLineFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{

    const Eigen::Map<const Eigen::Vector3d> Pi(&parameters[0][0]);
    const Eigen::Map<const Eigen::Quaterniond> Qi(&parameters[0][3]); // array order: (x, y, z, w)

    const Eigen::Map<const Eigen::Vector3d> tic(&parameters[1][0]);
    const Eigen::Map<const Eigen::Quaterniond> qic(&parameters[1][3]); // array order: (x, y, z, w)

    const Eigen::Map<const Eigen::Vector4d> line_orth(&parameters[2][0]);


    Eigen::Matrix<double, 6, 1> line_w = geometry_utils::ortho2plucker(line_orth);

    const Eigen::Vector3d &t_w_imu = Pi;
    const Eigen::Matrix3d &R_w_imu = Qi.toRotationMatrix();
    
    Eigen::Matrix<double, 6, 1> line_b = geometry_utils::invTransformPlucker(line_w, R_w_imu, t_w_imu);

    const Eigen::Vector3d &t_imu_cam = tic;
    const Eigen::Matrix3d &R_imu_cam = qic.toRotationMatrix();

    Eigen::Matrix<double, 6, 1> line_c = geometry_utils::invTransformPlucker(line_b, R_imu_cam, t_imu_cam);

    // K is identity matrix
    Eigen::Vector3d nc = line_c.head(3);
    Eigen::Vector3d obs_l1(line_i(0), line_i(1), 1);
    Eigen::Vector3d obs_l2(line_i(2), line_i(3), 1);
    Eigen::Vector3d obs_nc = obs_l1.cross(obs_l2);
    obs_nc *= obs_nc.dot(nc) < 0 ? -1 : 1;

    double obs_norm = obs_nc.norm();
    double squarednorm = nc.squaredNorm();
    double norm = sqrt(squarednorm);
    double trinorm = squarednorm * norm;
    double dot_obs_norm_trinorm = obs_nc.dot(nc) / (obs_norm * trinorm);
    double obs_norm_norm = obs_norm * norm;
    
    double *residual(residuals);
    residual[0] = 1 - abs(obs_nc.dot(nc) / (obs_norm * norm));
    residual[0] = sqrt_info(0) * residual[0];


    /* calculate jacobians */
    if (jacobians)
    {
        // Eigen::Matrix<double, 1, 3> jaco_e_l(1, 3);
        // jaco_e_l << 
        //     nc(0) * dot_obs_norm_trinorm - obs_nc(0) / obs_norm_norm,
        //     nc(1) * dot_obs_norm_trinorm - obs_nc(1) / obs_norm_norm,
        //     nc(2) * dot_obs_norm_trinorm - obs_nc(2) / obs_norm_norm;

        Eigen::Matrix<double, 1, 3> jaco_e_l = nc.transpose() * dot_obs_norm_trinorm - obs_nc.transpose() / obs_norm_norm;
        jaco_e_l = sqrt_info(0) * jaco_e_l;

        Eigen::Matrix<double, 3, 6> jaco_l_Lc(3, 6);
        jaco_l_Lc.setZero();
        jaco_l_Lc.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();

        Eigen::Matrix<double, 1, 6> jaco_e_Lc;
        jaco_e_Lc = jaco_e_l * jaco_l_Lc;

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

            Eigen::Matrix<double, 6, 6> invTbc;
            invTbc << 
                R_imu_cam.transpose(), -R_imu_cam.transpose() * math_utils::skewSymmetric(t_imu_cam),
                Eigen::Matrix3d::Zero(),  R_imu_cam.transpose();

            Eigen::Vector3d nw = line_w.head(3);
            Eigen::Vector3d dw = line_w.tail(3);

            Eigen::Matrix<double, 6, 6> jaco_Lc_pose;

            jaco_Lc_pose.setZero();
            jaco_Lc_pose.block(0, 0, 3, 3) = R_w_imu.transpose() * math_utils::skewSymmetric(dw);
            jaco_Lc_pose.block(0, 3, 3, 3) = math_utils::skewSymmetric(R_w_imu.transpose() * (nw + math_utils::skewSymmetric(dw) * t_w_imu));
            jaco_Lc_pose.block(3, 3, 3, 3) = math_utils::skewSymmetric(R_w_imu.transpose() * dw);

            jaco_Lc_pose = invTbc * jaco_Lc_pose;

            jacobian_pose_i.leftCols<6>() = jaco_e_Lc * jaco_Lc_pose;
            jacobian_pose_i.rightCols<1>().setZero();
        }
        if (jacobians[1])
        {

            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[1]);

            Eigen::Vector3d nb = line_b.head(3);
            Eigen::Vector3d db = line_b.tail(3);
            Eigen::Matrix<double, 6, 6> jaco_Lc_ex;

            jaco_Lc_ex.setZero();
            jaco_Lc_ex.block(0, 0, 3, 3) = R_imu_cam.transpose() * math_utils::skewSymmetric(db);   // Lc_t
            jaco_Lc_ex.block(0, 3, 3, 3) = math_utils::skewSymmetric(R_imu_cam.transpose() * (nb + math_utils::skewSymmetric(db) * t_imu_cam));  // Lc_theta
            jaco_Lc_ex.block(3, 3, 3, 3) = math_utils::skewSymmetric(R_imu_cam.transpose() * db);

            jacobian_ex_pose.leftCols<6>() = jaco_e_Lc * jaco_Lc_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>> jacobian_lineOrth(jacobians[2]);

            Eigen::Matrix3d R_w_cam = R_w_imu * R_imu_cam;
            Eigen::Vector3d t_w_cam = R_w_imu * t_imu_cam + t_w_imu;
            Eigen::Matrix<double, 6, 6> invTwc;

            invTwc << 
                R_w_cam.transpose(), -R_w_cam.transpose() * math_utils::skewSymmetric(t_w_cam),
                Eigen::Matrix3d::Zero(),  R_w_cam.transpose();

            Eigen::Vector3d nw = line_w.head(3);
            Eigen::Vector3d vw = line_w.tail(3);
            Eigen::Vector3d u1 = nw / nw.norm();
            Eigen::Vector3d u2 = vw / vw.norm();
            Eigen::Vector3d u3 = u1.cross(u2);
            Eigen::Vector2d w(nw.norm(), vw.norm());
            w = w / w.norm();

            Eigen::Matrix<double, 6, 4> jaco_Lw_orth;
            jaco_Lw_orth.setZero();
            jaco_Lw_orth.block(3, 0, 3, 1) = w(1) * u3;
            jaco_Lw_orth.block(0, 1, 3, 1) = -w(0) * u3;
            jaco_Lw_orth.block(0, 2, 3, 1) = w(0) * u2;
            jaco_Lw_orth.block(3, 2, 3, 1) = -w(1) * u1;
            jaco_Lw_orth.block(0, 3, 3, 1) = -w(1) * u1;
            jaco_Lw_orth.block(3, 3, 3, 1) = w(0) * u2;

            jacobian_lineOrth = jaco_e_Lc * invTwc * jaco_Lw_orth;
        }
    }
    
    return true;
}
#endif
