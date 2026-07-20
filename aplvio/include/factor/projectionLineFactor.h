#ifndef __PROJECTION_LINE_FACTOR_H
#define __PROJECTION_LINE_FACTOR_H

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>


#define USE_NORMAL_VECTOR_RESIDUAL 0

constexpr int residual_dimension = USE_NORMAL_VECTOR_RESIDUAL == 1 ? 1 : 2;

class ProjectionLineFactor : public ceres::SizedCostFunction<residual_dimension, 7, 7, 4>
{
public:
    ProjectionLineFactor(const Eigen::Vector4d &_line_i);
      
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Eigen::Vector4d line_i;
    Eigen::Matrix<double, 2, 3> tangent_base;
    
    static Eigen::Matrix2d sqrt_info;
};

#endif
