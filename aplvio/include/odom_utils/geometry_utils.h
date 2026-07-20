#ifndef __GEOMETRY_UTILS_H
#define __GEOMETRY_UTILS_H

#include "odom_utils/math_utils.h"


namespace geometry_utils
{

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 6, 1> ortho2line(const Eigen::MatrixBase<Derived> &ortho)
{
    Eigen::Matrix<typename Derived::Scalar, 6, 1> line;

    Eigen::Matrix<typename Derived::Scalar, 3, 1> theta = ortho.head(3);
    typename Derived::Scalar phi = ortho[3];

    typename Derived::Scalar s1 = sin(theta[0]);
    typename Derived::Scalar c1 = cos(theta[0]);
    typename Derived::Scalar s2 = sin(theta[1]);
    typename Derived::Scalar c2 = cos(theta[1]);
    typename Derived::Scalar s3 = sin(theta[2]);
    typename Derived::Scalar c3 = cos(theta[2]);
    Eigen::Matrix<typename Derived::Scalar, 3, 3> R;
    R << 
        c2 * c3, s1 * s2 * c3 - c1 * s3, c1 * s2 * c3 + s1 * s3,
        c2 * s3, s1 * s2 * s3 + c1 * c3, c1 * s2 * s3 - s1 * c3,
        -s2, s1 * c2, c1 * c2;

    typename Derived::Scalar w1 = cos(phi);
    typename Derived::Scalar w2 = sin(phi);
    typename Derived::Scalar d = w1 / w2;

    line.head(3) = -R.col(2) * d;
    line.tail(3) = R.col(1);

    return line;
}


template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 4, 1> line2ortho(const Eigen::MatrixBase<Derived> &line)
{
    Eigen::Matrix<typename Derived::Scalar, 4, 1> ortho;

    Eigen::Matrix<typename Derived::Scalar, 3, 1> p = line.head(3);
    Eigen::Matrix<typename Derived::Scalar, 3, 1> v = line.tail(3);
    Eigen::Matrix<typename Derived::Scalar, 3, 1> n = p.cross(v);

    Eigen::Matrix<typename Derived::Scalar, 3, 1> u1 = n / n.norm();
    Eigen::Matrix<typename Derived::Scalar, 3, 1> u2 = v / v.norm();
    Eigen::Matrix<typename Derived::Scalar, 3, 1> u3 = u1.cross(u2);

    ortho[0] = atan2(u2(2), u3(2));
    ortho[1] = asin(-u1(2));
    ortho[2] = atan2(u1(1), u1(0));

    Eigen::Matrix<typename Derived::Scalar, 2, 1> w(n.norm(), v.norm());
    w /= w.norm();

    ortho[3] = asin(w(1));

    return ortho;
}


template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 6, 1> ortho2plucker(const Eigen::MatrixBase<Derived> &ortho)
{
    Eigen::Matrix<typename Derived::Scalar, 6, 1> plucker;

    Eigen::Matrix<typename Derived::Scalar, 3, 1> theta = ortho.head(3);
    typename Derived::Scalar phi = ortho[3];

    typename Derived::Scalar s1 = sin(theta[0]);
    typename Derived::Scalar c1 = cos(theta[0]);
    typename Derived::Scalar s2 = sin(theta[1]);
    typename Derived::Scalar c2 = cos(theta[1]);
    typename Derived::Scalar s3 = sin(theta[2]);
    typename Derived::Scalar c3 = cos(theta[2]);
    Eigen::Matrix<typename Derived::Scalar, 3, 3> R;
    R << 
        c2 * c3, s1 * s2 * c3 - c1 * s3, c1 * s2 * c3 + s1 * s3,
        c2 * s3, s1 * s2 * s3 + c1 * c3, c1 * s2 * s3 - s1 * c3,
        -s2, s1 * c2, c1 * c2;

    typename Derived::Scalar w1 = cos(phi);
    typename Derived::Scalar w2 = sin(phi);
    // typename Derived::Scalar d = w1 / w2;

    Eigen::Matrix<typename Derived::Scalar, 3, 1> u1 = R.col(0);
    Eigen::Matrix<typename Derived::Scalar, 3, 1> u2 = R.col(1);

    Eigen::Matrix<typename Derived::Scalar, 3, 1> n = w1 * u1;
    Eigen::Matrix<typename Derived::Scalar, 3, 1> v = w2 * u2;

    plucker.head(3) = n;
    plucker.tail(3) = v;

    return plucker;
}


template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 4, 1> plucker2ortho(const Eigen::MatrixBase<Derived> &line)
{
    Eigen::Matrix<typename Derived::Scalar, 4, 1> ortho;

    Eigen::Matrix<typename Derived::Scalar, 3, 1> n = line.head(3);
    Eigen::Matrix<typename Derived::Scalar, 3, 1> v = line.tail(3);

    Eigen::Matrix<typename Derived::Scalar, 3, 1> u1 = n / n.norm();
    Eigen::Matrix<typename Derived::Scalar, 3, 1> u2 = v / v.norm();
    Eigen::Matrix<typename Derived::Scalar, 3, 1> u3 = u1.cross(u2);

    ortho[0] = atan2(u2(2), u3(2));
    ortho[1] = asin(-u1(2));
    ortho[2] = atan2(u1(1), u1(0));

    Eigen::Matrix<typename Derived::Scalar, 2, 1> w(n.norm(), v.norm());
    w /= w.norm();

    ortho[3] = asin(w(1));

    return ortho;
}


template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 4, 1> ptptpt2plane(
    const Eigen::MatrixBase<Derived> &pt1, 
    const Eigen::MatrixBase<Derived> &pt2, 
    const Eigen::MatrixBase<Derived> &pt3
)
{
    Eigen::Matrix<typename Derived::Scalar, 4, 1> plane;
    
    plane << (pt1 - pt3).cross(pt2 - pt3), -pt3.dot(pt1.cross(pt2));

    return plane;
}


template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 6, 1> planeplane2plucker(
    const Eigen::MatrixBase<Derived> &plane1, 
    const Eigen::MatrixBase<Derived> &plane2
)
{
    Eigen::Matrix<typename Derived::Scalar, 6, 1> plucker;
    
    Eigen::Matrix<typename Derived::Scalar, 4, 4> plucker_matrix
        = plane1 * plane2.transpose() - plane2 * plane1.transpose();

    plucker << 
        plucker_matrix(0, 3), plucker_matrix(1, 3), plucker_matrix(2, 3), 
        plucker_matrix(2, 1), plucker_matrix(0, 2), plucker_matrix(1, 0);

    return plucker;
}


template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 6, 1> transformPlucker(
    const Eigen::MatrixBase<Derived> &plucker, 
    const Eigen::Matrix<typename Derived::Scalar, 3, 3> &R, 
    const Eigen::Matrix<typename Derived::Scalar, 3, 1> &t
)
{
    Eigen::Matrix<typename Derived::Scalar, 6, 1> ans;
    
    ans.head(3) = R * plucker.head(3) + math_utils::skewSymmetric(t) * R * plucker.tail(3);
    ans.tail(3) = R * plucker.tail(3);

    return ans;
}


template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 6, 1> invTransformPlucker(
    const Eigen::MatrixBase<Derived> &plucker, 
    const Eigen::Matrix<typename Derived::Scalar, 3, 3> &R, 
    const Eigen::Matrix<typename Derived::Scalar, 3, 1> &t
)
{
    return transformPlucker(plucker, R.transpose(), R.transpose() * -t);
}


template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1> transformPoint(
    const Eigen::MatrixBase<Derived> &point, 
    const Eigen::Matrix<typename Derived::Scalar, 3, 3> &R, 
    const Eigen::Matrix<typename Derived::Scalar, 3, 1> &t
)
{
    Eigen::Matrix<typename Derived::Scalar, 3, 1> ans;
    
    ans = R * point + t;

    return ans;
}


template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1> invTransformPoint(
    const Eigen::MatrixBase<Derived> &point, 
    const Eigen::Matrix<typename Derived::Scalar, 3, 3> &R, 
    const Eigen::Matrix<typename Derived::Scalar, 3, 1> &t
)
{
    return transformPoint(point, R.transpose(), R.transpose() * -t);
}


template <typename Derived>
typename Derived::Scalar Point2Line(
    const Eigen::MatrixBase<Derived> &line, 
    const Eigen::Matrix<typename Derived::Scalar, 2, 1> &point
)
{
    typename Derived::Scalar ans;

    const Eigen::Matrix<typename Derived::Scalar, 2, 1> &sPoint = line.head(2);
    const Eigen::Matrix<typename Derived::Scalar, 2, 1> &ePoint = line.tail(2);

    if (sPoint.x() == ePoint.x())
    {
        ans = fabs(point.x() - sPoint.x());
    }
    else if (sPoint.y() == ePoint.y())
    {
        ans = fabs(point.y() - sPoint.y());
    }
    else
    {
        typename Derived::Scalar k = (ePoint.y() - sPoint.y()) / (ePoint.x() - sPoint.x());
        ans = fabs((k * (point.x() - sPoint.x()) - (point.y() - sPoint.y())) / sqrt(1 + k * k));
    }

    return ans;
}


template <typename Derived>
typename Derived::Scalar dist(
    const Eigen::MatrixBase<Derived> &point1, 
    const Eigen::MatrixBase<Derived> &point2
)
{
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, 1> dp = point1 - point2;
    return dp.norm();
}


}

#endif