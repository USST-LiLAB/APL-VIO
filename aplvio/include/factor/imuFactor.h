#ifndef __IMU_FACTOR_H
#define __IMU_FACTOR_H

#include <memory>
#include <Eigen/Dense>
#include <ros/assert.h>
#include <ceres/ceres.h>


class IntegrationBase;

class ImuFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9>
{
public:
    ImuFactor() = delete;
    ImuFactor(std::shared_ptr<IntegrationBase> _pre_integration);

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    static Eigen::Matrix<double, 15, 15> sqrt_info; // sqrt of information matrix
    std::shared_ptr<IntegrationBase> pre_integration;
};

#endif
