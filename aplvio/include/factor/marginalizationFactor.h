#ifndef __MARGINALIZATION_FACTOR_H
#define __MARGINALIZATION_FACTOR_H

#include <thread>
#include <ros/ros.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include <Eigen/Dense>


const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *, ceres::LossFunction *, std::vector<double *>, std::vector<int>);

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks; // all parameter blocks
    std::vector<int> drop_set;

    /* jacobians for ceres-solver */
    double **raw_jacobians; 
    /* store all jacobian blocks */
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians; 
    Eigen::VectorXd residuals;
};

struct ThreadsStruct
{
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::vector<ResidualBlockInfo *> sub_factors;
    std::unordered_map<long, int> *ptr_parameter_block_size;
    std::unordered_map<long, int> *ptr_parameter_block_idx;
};

class MarginalizationInfo
{
public:
    MarginalizationInfo();
    ~MarginalizationInfo();

    void addResidualBlockInfo(ResidualBlockInfo *);
    void keepParameterBlocks(std::unordered_map<long, double *> &);

    void preMarginalize();
    void marginalize();

    void setParameterBlockIndex();
    void schurComplement();

    static void buildAb(std::vector<ResidualBlockInfo *>, Eigen::MatrixXd &, Eigen::VectorXd &, std::unordered_map<long, int> &, std::unordered_map<long, int> &);
    static inline int localSize(const int size) { return size == 7 ? 6 : size; }
    static inline int globalSize(const int size) { return size == 6 ? 7 : size; }

    // all factors containing margianization, imu and visual factors
    std::vector<ResidualBlockInfo *> factors;
    
    std::thread thread_ids[NUM_THREADS];
    ThreadsStruct threadsstruct[NUM_THREADS];

    int m, r, n; // margin num, reserve and total parameter num
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    /* (global size) hash map for all parameter_blocks: parameter_block_address --> parameter_block_size */
    std::unordered_map<long, int> parameter_block_size;
    std::unordered_map<long, int> parameter_block_idx; //local size
    std::unordered_map<long, double *> parameter_block_data;

    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;
    std::vector<double *> keep_block_addr;

    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;
    bool valid;
};


class MarginalizationFactor : public ceres::CostFunction
{
public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};

#endif
