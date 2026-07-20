#ifndef __MATH_UTILS_H
#define __MATH_UTILS_H

#include <Eigen/Dense>


namespace math_utils
{

/* Quaternion perturbation */
template <typename Derived>
Eigen::Quaternion<typename Derived::Scalar> theta2dq(const Eigen::MatrixBase<Derived> &theta)
{
    Eigen::Quaternion<typename Derived::Scalar> dq;
    Eigen::Matrix<typename Derived::Scalar, 3, 1> half_theta = theta;
    half_theta *= static_cast<typename Derived::Scalar>(0.5);
    dq.w() = static_cast<typename Derived::Scalar>(1.0);
    dq.x() = half_theta.x();
    dq.y() = half_theta.y();
    dq.z() = half_theta.z();
    return dq;
}


template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> &q)
{
    Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
    ans << typename Derived::Scalar(0), -q(2), q(1),
           q(2), typename Derived::Scalar(0), -q(0),
           -q(1), q(0), typename Derived::Scalar(0);
    return ans;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived> &q)
{
    Eigen::Quaternion<typename Derived::Scalar> qq = q;
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = qq.w(), ans.template block<1, 3>(0, 1) = -qq.vec().transpose();
    ans.template block<3, 1>(1, 0) = qq.vec(), ans.template block<3, 3>(1, 1) = qq.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() + skewSymmetric(qq.vec());
    return ans;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Eigen::QuaternionBase<Derived> &p)
{
    Eigen::Quaternion<typename Derived::Scalar> pp = p;
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = pp.w(), ans.template block<1, 3>(0, 1) = -pp.vec().transpose();
    ans.template block<3, 1>(1, 0) = pp.vec(), ans.template block<3, 3>(1, 1) = pp.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() - skewSymmetric(pp.vec());
    return ans;
}


template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::DenseBase<Derived>& ypr)
{
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
    EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime == 3, THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
    EIGEN_STATIC_ASSERT(Derived::ColsAtCompileTime == 1, THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);

    typename Derived::Scalar c, s;

    Eigen::Matrix<typename Derived::Scalar, 3, 3> Rz = Eigen::Matrix<typename Derived::Scalar, 3, 3>::Zero();
    typename Derived::Scalar y = ypr(0);
    c = cos(y);
    s = sin(y);
    Rz(0, 0) = c;
    Rz(1, 0) = s;
    Rz(0, 1) = -s;
    Rz(1, 1) = c;
    Rz(2, 2) = 1;

    Eigen::Matrix<typename Derived::Scalar, 3, 3> Ry = Eigen::Matrix<typename Derived::Scalar, 3, 3>::Zero();
    typename Derived::Scalar p = ypr(1);
    c = cos(p);
    s = sin(p);
    Ry(0, 0) = c;
    Ry(2, 0) = -s;
    Ry(0, 2) = s;
    Ry(2, 2) = c;
    Ry(1, 1) = 1;

    Eigen::Matrix<typename Derived::Scalar, 3, 3> Rx = Eigen::Matrix<typename Derived::Scalar, 3, 3>::Zero();
    typename Derived::Scalar r = ypr(2);
    c = cos(r);
    s = sin(r);
    Rx(1, 1) = c;
    Rx(2, 1) = s;
    Rx(1, 2) = -s;
    Rx(2, 2) = c;
    Rx(0, 0) = 1;

    Eigen::Matrix<typename Derived::Scalar, 3, 3> R = Rz * Ry * Rx;
    return R;
}


template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1> R2ypr(const Eigen::DenseBase<Derived>& R)
{
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
    EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime == 3, THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
    EIGEN_STATIC_ASSERT(Derived::ColsAtCompileTime == 3, THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);

    Eigen::Matrix<typename Derived::Scalar, 3, 1> n = R.col(0);
    Eigen::Matrix<typename Derived::Scalar, 3, 1> o = R.col(1);
    Eigen::Matrix<typename Derived::Scalar, 3, 1> a = R.col(2);

    Eigen::Matrix<typename Derived::Scalar, 3, 1> ypr(3);
    typename Derived::Scalar y = atan2(n(1), n(0));
    typename Derived::Scalar p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    typename Derived::Scalar r =
        atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr;
}


template <typename Scalar>
Scalar wrap2pi(Scalar angle)
{
    while (angle < -M_PI)
        angle += M_PI * 2;
    while (angle > M_PI)
        angle -= M_PI * 2;

    return angle;
}

}

#endif
