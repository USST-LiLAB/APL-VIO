#include "odom_utils/utils.h"

#include "factor/marginalizationFactor.h"


/** \brief Constructor of ResidualBlockInfo
 * @param _cost_function cost function
 * @param _loss_function lost function
 * @param _parameter_blocks parameter blocks
 * @param _drop_set parameters ready to drop
 */
ResidualBlockInfo::ResidualBlockInfo(
    ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
    : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set)
{
    // nothing
}


void ResidualBlockInfo::Evaluate()
{
    // set the dim of residuals
    residuals.resize(cost_function->num_residuals());

    // (glabol size) size of each parameter block
    const std::vector<int> &param_block_sizes = cost_function->parameter_block_sizes();
    // set num of jacobians
    jacobians.resize(param_block_sizes.size());
    // set the pointers array of jacobian_blocks for ceres-solver
    raw_jacobians = new double *[param_block_sizes.size()];

    for (size_t i = 0; i != param_block_sizes.size(); i++)
    {
        // set the row and col num for each jacobian
        jacobians[i].resize(cost_function->num_residuals(), param_block_sizes[i]);
        // set the pointer of each jacobian_block
        raw_jacobians[i] = jacobians[i].data();
    }
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);


    if (loss_function)
    {
        double residual_scaling_, alpha_sq_norm_;

        double sq_norm, rho[3]; // rho, rho', rho"

        sq_norm = residuals.squaredNorm();
        loss_function->Evaluate(sq_norm, rho);
        //printf("sq_norm: %f, rho[0]: %f, rho[1]: %f, rho[2]: %f\n", sq_norm, rho[0], rho[1], rho[2]);

        double sqrt_rho1_ = sqrt(rho[1]);

        if ((sq_norm == 0.0) || (rho[2] <= 0.0))
        {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        }
        else
        {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }

        for (size_t i = 0; i != parameter_blocks.size(); i++)
        {
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
        }

        residuals *= residual_scaling_;
    }
}


MarginalizationInfo::MarginalizationInfo() : valid(true)
{
    // nothing
}


MarginalizationInfo::~MarginalizationInfo()
{
    for (auto iter = parameter_block_data.begin(); iter != parameter_block_data.end(); iter++)
        delete iter->second;

    for (size_t i = 0; i != factors.size(); i++)
    {
        delete[] factors[i]->raw_jacobians;
        
        delete factors[i]->cost_function;

        delete factors[i];
    }
}


/** \brief add marginalization_info from last marginalization, imu and visual residual_block_info
 * @param residual_block_info pointer to residual_block_info
 */
void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo *residual_block_info)
{
    factors.push_back(residual_block_info);

    const std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks;
    const std::vector<int> &param_block_sizes = residual_block_info->cost_function->parameter_block_sizes();

    for (size_t i = 0; i != residual_block_info->parameter_blocks.size(); i++)
        parameter_block_size[reinterpret_cast<long>(parameter_blocks[i])] = param_block_sizes[i];

    // add all parameters to be marginalized
    for (size_t i = 0; i != residual_block_info->drop_set.size(); i++)
        parameter_block_idx[reinterpret_cast<long>(parameter_blocks[residual_block_info->drop_set[i]])] = 0;
}


void MarginalizationInfo::preMarginalize()
{
    for (auto iter : factors)
    {
        iter->Evaluate();

        // prepare memory for parameter_blocks
        std::vector<int> param_block_sizes = iter->cost_function->parameter_block_sizes();
        for (size_t i = 0; i != param_block_sizes.size(); i++)
        {
            const long param_address_i = reinterpret_cast<long>(iter->parameter_blocks[i]);
            if (parameter_block_data.find(param_address_i) == parameter_block_data.end())
            {
                double *data = new double[param_block_sizes[i]];
                memcpy(data, iter->parameter_blocks[i], param_block_sizes[i] * sizeof(double));
                parameter_block_data[param_address_i] = data;
            }
        }
    }
}


void MarginalizationInfo::marginalize()
{
    setParameterBlockIndex();

    if (m == 0)
    {
        valid = false;
        ROS_WARN_STREAM("unstable tracking...\n");
        return;
    }

    /* construct A and b for marginalization */
    // A = J'J, b = J'e
    utils::TicToc buildAb_tictoc;
    A = Eigen::MatrixXd::Zero(n, n);
    b = Eigen::VectorXd::Zero(n);
#if 1 /* ENABLE_MULTITHREAD  */
    int thread_id = 0;
    for (auto iter : factors)
    {
        threadsstruct[thread_id].sub_factors.push_back(iter);
        thread_id = (thread_id + 1) % NUM_THREADS;
    }
    for (int i = 0; i != NUM_THREADS; i++)
    {
        threadsstruct[i].A = A;
        threadsstruct[i].b = b;
        threadsstruct[i].ptr_parameter_block_size = &parameter_block_size;
        threadsstruct[i].ptr_parameter_block_idx = &parameter_block_idx;
        thread_ids[i] = std::thread([this, i]() {
            this->buildAb(threadsstruct[i].sub_factors, threadsstruct[i].A, threadsstruct[i].b, 
            *threadsstruct[i].ptr_parameter_block_idx, *threadsstruct[i].ptr_parameter_block_size); 
        });
    }
    for (int i = NUM_THREADS - 1; i >= 0; i--)  
    {
        thread_ids[i].join();
        A += threadsstruct[i].A;
        b += threadsstruct[i].b;
    }
#else /* UNABLE_MULTITHREAD */
    buildAb(factors, A, b, parameter_block_idx, parameter_block_size);
#endif
    // std::cout << "marginalization build A and b time costs: " << buildAb_tictoc.toc() << " ms" << std::endl;


    /* schur complement */
    schurComplement();


    /* Preparation for calculating residual in optimization */
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A); // A = U * S * U' = J‘ * J

    Eigen::VectorXd S((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0)); // saes2 > eps ? saes2 : 0
    Eigen::VectorXd S_inv((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

    // J = S^{1/2} * U' (A = U * S * U' = J' * J)
    linearized_jacobians = S.cwiseSqrt().asDiagonal() * saes2.eigenvectors().transpose();
    // e = (J'){-1} * b (b = J' * e)
    linearized_residuals = S_inv.cwiseSqrt().asDiagonal() * saes2.eigenvectors().transpose() * b;
}


void MarginalizationInfo::setParameterBlockIndex()
{
    // margin parameter num
    n = 0;
    for (auto &iter : parameter_block_idx)
    {
        iter.second = n;
        n += localSize(parameter_block_size[iter.first]);
    }
    m = n;

    // reserved parameter num
    for (const auto &iter : parameter_block_size)
    {
        if (parameter_block_idx.find(iter.first) == parameter_block_idx.end())
        {
            parameter_block_idx[iter.first] = n;
            n += localSize(iter.second);
        }
    }
    r = n - m;
}


void MarginalizationInfo::buildAb(std::vector<ResidualBlockInfo *> factors, Eigen::MatrixXd &A, Eigen::VectorXd &b,
    std::unordered_map<long, int> &parameter_block_idx, std::unordered_map<long, int> &parameter_block_size)
{
    for (auto iter : factors)
    {
        for (size_t i = 0; i != iter->parameter_blocks.size(); i++)
        {
            const long param_address_i = reinterpret_cast<long>(iter->parameter_blocks[i]);
            const int idx_i = parameter_block_idx[param_address_i];
            const int size_i = MarginalizationInfo::localSize(parameter_block_size[param_address_i]);
            const Eigen::MatrixXd &jacobian_i = iter->jacobians[i].leftCols(size_i);

            for (size_t j = i; j != iter->parameter_blocks.size(); j++)
            {
                const long param_address_j = reinterpret_cast<long>(iter->parameter_blocks[j]);
                const int idx_j = parameter_block_idx[param_address_j];
                const int size_j = MarginalizationInfo::localSize(parameter_block_size[param_address_j]);
                const Eigen::MatrixXd &jacobian_j = iter->jacobians[j].leftCols(size_j);
                
                if (i == j)
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    A.block(idx_j, idx_i, size_j, size_i) = A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            b.segment(idx_i, size_i) += jacobian_i.transpose() * iter->residuals;
        }
    }
}


void MarginalizationInfo::schurComplement()
{
    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

    //ROS_ASSERT_MSG(saes.eigenvalues().minCoeff() >= -1e-4, "min eigenvalue %f", saes.eigenvalues().minCoeff());

    const Eigen::MatrixXd Amm_inv = saes.eigenvectors() 
        * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal()
        * saes.eigenvectors().transpose();
    //printf("error1: %f\n", (Amm * Amm_inv - Eigen::MatrixXd::Identity(m, m)).sum());

    const Eigen::MatrixXd &Amr = A.block(0, m, m, r);
    const Eigen::MatrixXd &Arm = A.block(m, 0, r, m);
    const Eigen::MatrixXd &Arr = A.block(m, m, r, r);
    const Eigen::VectorXd &bmm = b.segment(0, m);
    const Eigen::VectorXd &brr = b.segment(m, r);
    A = Arr - Arm * Amm_inv * Amr;
    b = brr - Arm * Amm_inv * bmm;
}


void MarginalizationInfo::keepParameterBlocks(std::unordered_map<long, double *> &addr_shift)
{
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();
    keep_block_addr.clear();

    for (const auto &iter : parameter_block_idx)
    {
        if (iter.second >= m)
        {
            keep_block_size.push_back(parameter_block_size[iter.first]);
            keep_block_idx.push_back(parameter_block_idx[iter.first]);
            keep_block_data.push_back(parameter_block_data[iter.first]);
            keep_block_addr.push_back(addr_shift[iter.first]);
        }
    }
}



MarginalizationFactor::MarginalizationFactor(MarginalizationInfo* _marginalization_info)
    : marginalization_info(_marginalization_info)
{
    int cnt = 0;
    for (auto iter : marginalization_info->keep_block_size)
    {
        mutable_parameter_block_sizes()->push_back(iter);
        cnt += iter;
    }
    //printf("residual size: %d, %d\n", cnt, r);
    set_num_residuals(marginalization_info->r);
};


bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    const int r = marginalization_info->r;
    const int m = marginalization_info->m;
    Eigen::VectorXd dx(r);
    
    for (size_t i = 0; i != marginalization_info->keep_block_size.size(); i++)
    {
        const int size_i = marginalization_info->keep_block_size[i];
        const int idx_i = marginalization_info->keep_block_idx[i] - m;
        const Eigen::VectorXd &x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size_i);
        const Eigen::VectorXd &x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size_i);
        
        if (size_i != 7)
            dx.segment(idx_i, size_i) = x - x0;
        else
        {
            Eigen::Quaterniond tmp_dq(Eigen::Quaterniond(x0.tail<4>()).inverse() * Eigen::Quaterniond(x.tail<4>()));
            dx.segment<3>(idx_i + 0) = x.head<3>() - x0.head<3>();
            dx.segment<3>(idx_i + 3) = 2.0 * tmp_dq.vec() * (tmp_dq.w() >= 0 ? 1 : -1);
        }
    }
    Eigen::Map<Eigen::VectorXd>(residuals, r) = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;

    if (jacobians)
    {
        for (size_t i = 0; i < marginalization_info->keep_block_size.size(); i++)
        {
            if (jacobians[i])
            {
                int size_i = marginalization_info->keep_block_size[i];
                int idx_i = marginalization_info->keep_block_idx[i] - m;
                int local_size_i = MarginalizationInfo::localSize(size_i);
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], r, size_i);
                jacobian.setZero();
                jacobian.leftCols(local_size_i) = marginalization_info->linearized_jacobians.middleCols(idx_i, local_size_i);
            }
        }
    }
    return true;
}
