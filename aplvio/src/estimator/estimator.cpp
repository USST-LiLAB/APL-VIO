#include <fstream>

#include "catkin_settings.h"
#include "odom_utils/utils.h"
#include "factor/imuFactor.h"
#include "factor/marginalizationFactor.h"
#include "factor/poseLocalParameterization.h"
#include "factor/lineLocalParameterization.h"
#include "factor/projectionOneFrameTwoCamFactor.h"
#include "factor/projectionTwoFrameOneCamFactor.h"
#include "factor/projectionTwoFrameTwoCamFactor.h"
#include "factor/projectionLineFactor.h"
#include "factor/integrationBase.h"
#include "estimator/paramPool.h"

#include "initializer/initializer.h"
#include "inertial/imuProcessor.h"
#include "visual/imageProcessor.h"
#include "visual/featureManager.h"
#include "visual/lineFeatureManager.h"

#include "estimator/estimator.h"


#define SAVE_ESTIMATE_COST 0

extern std::unique_ptr<ParamPool> shared_pool;

Estimator::Estimator() : 
    initializer(new Initializer), imu_processor(new ImuProcessor), image_processor(new ImageProcessor), 
    feature_manager(new FeatureManager), linefeature_manager(new LineFeatureManager),
    have_init_imu(false), cur_frame_count(0), pre_frame_time(0.0), cur_frame_time(0.0)
{
    clearState();
    // nothing
}


Estimator::~Estimator()
{
    // process_thread.join();
    // std::cout << "join estimator process thread" << std::endl;
}


void Estimator::clearState()
{
    shared_pool->setParameters();

    have_init_imu = false;

    initializer->clearState();
    imu_processor->clearState();
    image_processor->clearState();
    feature_manager->clearState();
    linefeature_manager->clearState();

    cur_frame_count = 0;
    pre_frame_time = 0.0;
    cur_frame_time = 0.0;

    invdepth_vector.clear();

    for (int i = 0; i != WINDOW_SIZE; i++)
    {
        state_window[i].clearState();
    }

    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    last_marginalization_info = nullptr;
}


void Estimator::setParameters()
{
    initializer->setParameters();
    imu_processor->setParameters();
    image_processor->setParameters();
    feature_manager->setParameters();
    linefeature_manager->setParameters();
}


void Estimator::setThreads()
{
    process_thread = std::thread([this]() -> void {
        while (true) {
            this->process();
        }
    });
    process_thread.detach();
    
#if ENABLE_MULTITHREAD
    image_processor->setThreads();
#endif
}


void Estimator::process()
{

#if !ENABLE_MULTITHREAD
    image_processor->processImage();
#endif

    utils::TicToc process_tictoc;

    ImageFrame image_frame;
    // get image frame point and line features
    if (!image_processor->getImageFrame(image_frame))
    {
#if ENABLE_MULTITHREAD
        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
#endif
        return;
    }

    /* process inertial data */
    std::vector<ImuData> imu_vector; // interframe imu data
    if (shared_pool->is_use_imu)
    {
        cur_frame_time = image_frame.t + shared_pool->time_delay;

        if (!imu_processor->getInterframeImuData(pre_frame_time, cur_frame_time, imu_vector)) 
            return;

        if (!have_init_imu)
        {
            if (!imu_processor->initImuState(imu_vector, state_window[0].body_state))
                return;
            have_init_imu = true;
        }
        FrameState &cur_frame_state = state_window[cur_frame_count];
        if (cur_frame_state.pre_integration == nullptr)
        {
            cur_frame_state.pre_integration.reset(new IntegrationBase(
                cur_frame_state.body_state.acc, cur_frame_state.body_state.gyr, 
                cur_frame_state.body_state.ba,  cur_frame_state.body_state.bg));
        }

        if (cur_frame_count != 0)
            imu_processor->interframeIntegration(cur_frame_state, pre_frame_time, cur_frame_time, imu_vector);

        image_frame.pre_integration.reset(new IntegrationBase(*cur_frame_state.pre_integration));

        pre_frame_time = cur_frame_time;
    }

    state_window[cur_frame_count].body_state.t = image_frame.t; // image time without timedelay

    image_frame_list.push_back(image_frame);


    /* process visual data */
    process_mutex.lock();
    feature_manager->insertFeature(cur_frame_count, image_frame.feature_points);
#if ENABLE_LINE_FEATURE
    linefeature_manager->insertLineFeature(cur_frame_count, image_frame.feature_lines);
    shared_pool->marginalization_flag = 
        (feature_manager->isKeyframe(cur_frame_count) || linefeature_manager->isKeyframe(cur_frame_count)) 
        ? MARGIN_OLD : MARGIN_SECOND_NEW;
#else
    shared_pool->marginalization_flag = feature_manager->isKeyframe(cur_frame_count) ? MARGIN_OLD : MARGIN_SECOND_NEW;
#endif

    switch (shared_pool->solver_flag)
    {
    case INITIAL:
    {
        // monocular + imu initilization
        if (!shared_pool->is_use_stereo && shared_pool->is_use_imu)
        {
            // struct from motion
            if (cur_frame_count == WINDOW_SIZE)
            {
                if (image_frame.t - initializer->initial_timestamp > 0.1)
                {
                    bool result = false;

                    result = initializer->initialStructure(image_frame_list, state_window, feature_manager);

                    if (result)
                    {
                        optimization();
                        updateLatestStates();
                        feature_manager->removeFailures();
                        shared_pool->solver_flag = NONLINEAR;
                        ROS_INFO("Initialization finish!");
                    }
                    initializer->initial_timestamp = image_frame.t;
                }
            }
        }

        // stereo + imu initilization
        if (shared_pool->is_use_stereo && shared_pool->is_use_imu)
        {
            feature_manager->initFramePoseByPnP(cur_frame_count, state_window);
            feature_manager->triangulate(state_window);

            if (cur_frame_count == WINDOW_SIZE)
            {
                int i = 0;
                for (auto iter = image_frame_list.begin(); iter != image_frame_list.end(); iter++, i++)
                {
                    iter->R_w_imu = state_window[i].body_state.R;
                    iter->t_w_imu = state_window[i].body_state.p;
                }
                initializer->solveGyrBias(image_frame_list, state_window);

                optimization();
                updateLatestStates();
                feature_manager->removeFailures();

                shared_pool->solver_flag = NONLINEAR;
                ROS_INFO("Initialization finish!");
            }
        }

        // stereo only initilization
        if (shared_pool->is_use_stereo && !shared_pool->is_use_imu)
        {
            feature_manager->initFramePoseByPnP(cur_frame_count, state_window);
            feature_manager->triangulate(state_window);

            optimization();

            if (cur_frame_count == WINDOW_SIZE)
            {
                updateLatestStates();
                feature_manager->removeFailures();
                shared_pool->solver_flag = NONLINEAR;
                ROS_INFO("Initialization finish!");
            }
        }

        slideWindow();

        break;
    }
    case NONLINEAR:
    {
        if (!shared_pool->is_use_imu)
            feature_manager->initFramePoseByPnP(cur_frame_count, state_window);
        feature_manager->triangulate(state_window);
#if ENABLE_LINE_FEATURE
        linefeature_manager->triangulate(state_window);
        linefeature_manager->detectOutliers(state_window);
        linefeature_manager->removeOutliers();
#endif

#if ENABLE_LINE_FEATURE
        lineBundleAdjust();
        linefeature_manager->detectOutliers(state_window);
        linefeature_manager->removeOutliers();
        linefeature_manager->removeFailures();
#endif
        optimization();
        
        feature_manager->detectOutliers(state_window);
        feature_manager->removeOutliers();
        feature_manager->removeFailures();
#if ENABLE_LINE_FEATURE
        lineBundleAdjust();
        linefeature_manager->detectOutliers(state_window);
        linefeature_manager->removeOutliers();
        linefeature_manager->removeFailures();
#endif

        slideWindow();
        updateLatestStates();

        {
#if SAVE_ESTIMATE_COST
            static std::ofstream fout(shared_pool->output_path + "/process_costs.txt", std::ofstream::out);
            fout << process_tictoc.toc() << std::endl;
#endif
            // static utils::TimeCost avg_process_cost;
            // printf("average process measurement costs: %.6f ms\n", avg_process_cost.avgCost(process_tictoc.toc()));
            // printf("max process measurement costs: %.6f ms\n", avg_process_cost.maxCost(process_tictoc.toc()));
            // if (process_tictoc.toc() > 100)
            //     printf("process measurement costs: %.6f ms\n", process_tictoc.toc());
            // printf("process measurement costs: %.6f ms\n", process_tictoc.toc());
        }

        break;
    }
    default:
        break;
    }
    process_mutex.unlock();

    utils::TicToc print_tictoc;
    pubLatestStates();
    // printf("print costs: %.6f ms\n", print_tictoc.toc());
}


void Estimator::state2ceres()
{
    for (int i = 0; i != WINDOW_SIZE + 1; i++)
    {
        para_Pose[i][0] = state_window[i].body_state.p.x();
        para_Pose[i][1] = state_window[i].body_state.p.y();
        para_Pose[i][2] = state_window[i].body_state.p.z();
        para_Pose[i][3] = state_window[i].body_state.q.x();
        para_Pose[i][4] = state_window[i].body_state.q.y();
        para_Pose[i][5] = state_window[i].body_state.q.z();
        para_Pose[i][6] = state_window[i].body_state.q.w();


        if (shared_pool->is_use_imu)
        {
            para_SpeedBias[i][0] = state_window[i].body_state.v.x();
            para_SpeedBias[i][1] = state_window[i].body_state.v.y();
            para_SpeedBias[i][2] = state_window[i].body_state.v.z();
            para_SpeedBias[i][3] = state_window[i].body_state.ba.x();
            para_SpeedBias[i][4] = state_window[i].body_state.ba.y();
            para_SpeedBias[i][5] = state_window[i].body_state.ba.z();
            para_SpeedBias[i][6] = state_window[i].body_state.bg.x();
            para_SpeedBias[i][7] = state_window[i].body_state.bg.y();
            para_SpeedBias[i][8] = state_window[i].body_state.bg.z();
        }
    }


    for (int i = 0; i != shared_pool->cam_num; i++)
    {
        para_Ex_Pose[i][0] = shared_pool->extrinsics.t_imu_cam[i].x();
        para_Ex_Pose[i][1] = shared_pool->extrinsics.t_imu_cam[i].y();
        para_Ex_Pose[i][2] = shared_pool->extrinsics.t_imu_cam[i].z();
        Eigen::Quaterniond q(shared_pool->extrinsics.R_imu_cam[i]);
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    feature_manager->getInvDepth(invdepth_vector);
    for (size_t i = 0; i != invdepth_vector.size(); i++)
        para_Feature[i][0] = invdepth_vector[i];

#if ENABLE_LINE_FEATURE
    linefeature_manager->getOrthoLine(ortholine_vector, state_window);
    for (size_t i = 0; i != ortholine_vector.size(); i++)
    {
        para_LineFeature[i][0] = ortholine_vector[i][0];
        para_LineFeature[i][1] = ortholine_vector[i][1];
        para_LineFeature[i][2] = ortholine_vector[i][2];
        para_LineFeature[i][3] = ortholine_vector[i][3];
    }
#endif

    para_Td[0][0] = shared_pool->time_delay;
}


void Estimator::ceres2state()
{
    Eigen::Vector3d ypr_0 = math_utils::R2ypr(state_window[0].body_state.R);
    Eigen::Vector3d para_ypr0 = math_utils::R2ypr(Eigen::Map<const Eigen::Quaterniond>(&para_Pose[0][3]).toRotationMatrix());

    // since yaw is not observable, the corresponding optimized result is NOT reliable
    double yaw_diff = ypr_0.x() - para_ypr0.x();
    Eigen::Matrix3d rot_diff = math_utils::ypr2R(Eigen::Vector3d(yaw_diff, 0, 0));

    if (shared_pool->is_use_imu)
    {
        if (abs(abs(ypr_0.y() * 180 / M_PI) - 90) < 1.0 || abs(abs(para_ypr0.y() * 180 / M_PI) - 90) < 1.0)
        {
            ROS_WARN_STREAM("euler singular point!");
            rot_diff = state_window[0].body_state.R * Eigen::Map<const Eigen::Quaterniond>(&para_Pose[0][3]).toRotationMatrix().transpose();
        }

        for (int i = 0; i != WINDOW_SIZE + 1; i++)
        {
            state_window[i].body_state.p = state_window[0].body_state.p + rot_diff * 
                (Eigen::Map<const Eigen::Vector3d>(&para_Pose[i][0]) - Eigen::Map<const Eigen::Vector3d>(&para_Pose[0][0]));

            state_window[i].body_state.R = rot_diff * Eigen::Map<const Eigen::Quaterniond>(&para_Pose[i][3]).normalized().toRotationMatrix();
            state_window[i].body_state.q = Eigen::Quaterniond(state_window[i].body_state.R).normalized();

            state_window[i].body_state.v = rot_diff * Eigen::Map<const Eigen::Vector3d>(&para_SpeedBias[i][0]);
            state_window[i].body_state.ba = Eigen::Map<const Eigen::Vector3d>(&para_SpeedBias[i][3]);
            state_window[i].body_state.bg = Eigen::Map<const Eigen::Vector3d>(&para_SpeedBias[i][6]);
        }
    }
    else
    {
        for (int i = 0; i != WINDOW_SIZE + 1; i++)
        {
            state_window[i].body_state.p = Eigen::Map<const Eigen::Vector3d>(&para_Pose[i][0]);

            state_window[i].body_state.q = Eigen::Map<const Eigen::Quaterniond>(&para_Pose[i][3]).normalized();
            state_window[i].body_state.R = state_window[i].body_state.q.toRotationMatrix();
        }
    }

    if (shared_pool->is_use_imu)
    {
        for (int i = 0; i != shared_pool->cam_num; i++)
        {
            shared_pool->extrinsics.t_imu_cam[i] = Eigen::Map<const Eigen::Vector3d>(&para_Ex_Pose[i][0]);
            shared_pool->extrinsics.R_imu_cam[i] = Eigen::Map<const Eigen::Quaterniond>(&para_Ex_Pose[i][3]).normalized().toRotationMatrix();
        }
    }

    for (size_t i = 0; i != invdepth_vector.size(); i++)
        invdepth_vector[i] = para_Feature[i][0];
    feature_manager->setDepth(invdepth_vector);

#if ENABLE_LINE_FEATURE
    Eigen::Vector3d t_diff = state_window[0].body_state.p
        + rot_diff * -Eigen::Map<const Eigen::Vector3d>(&para_Pose[0][0]);
    for (size_t i = 0; i != ortholine_vector.size(); i++)
    {
        const Eigen::Vector4d &ortholine = Eigen::Map<const Eigen::Vector4d>(&para_LineFeature[i][0]);
        ortholine_vector[i] = geometry_utils::plucker2ortho(
            geometry_utils::transformPlucker(geometry_utils::ortho2plucker(ortholine), rot_diff, t_diff));
    }
    linefeature_manager->setOrthoLine(ortholine_vector, state_window);
#endif

    shared_pool->time_delay = para_Td[0][0];
}


void Estimator::lineBundleAdjust()
{
    state2ceres();

    utils::TicToc refine_line_tictoc;
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1.0);

    for (int i = 0; i != cur_frame_count + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.SetParameterBlockConstant(para_Pose[i]);
    }

    /* extrinsic parameter block */
    for (int i = 0; i != shared_pool->cam_num; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        problem.SetParameterBlockConstant(para_Ex_Pose[i]);
    }

    int linefeature_index = -1;
    for (auto &linefeature_per_id : linefeature_manager->linefeature_list)
    {
        if (linefeature_per_id.linefeature_per_frame.size() < shared_pool->reliable_line_track_count)
            continue;

        if (!linefeature_per_id.is_triangulate)
            continue;

        ++linefeature_index;

        int frame_i = linefeature_per_id.start_frame, frame_j = frame_i - 1;


        for (auto &linefeature_per_frame : linefeature_per_id.linefeature_per_frame)
        {
            frame_j++;

            if (linefeature_per_frame.is_tracked == false)
                continue;
            
            // if (frame_i != frame_j)
            {
                Eigen::Vector4d line_j = linefeature_per_frame.line_0;
                    
                ProjectionLineFactor *f = new ProjectionLineFactor(line_j);

                problem.AddResidualBlock(f, loss_function, para_Pose[frame_j], para_Ex_Pose[0], para_LineFeature[linefeature_index]);
            }
            if (shared_pool->is_use_stereo && linefeature_per_frame.is_stereo)
            {
                Eigen::Vector4d line_j_right = linefeature_per_frame.line_1;
                
                ProjectionLineFactor *f = new ProjectionLineFactor(line_j_right);

                problem.AddResidualBlock(f, loss_function, para_Pose[frame_j], para_Ex_Pose[1], para_LineFeature[linefeature_index]);
            }
        }
    }
    ProjectionLineFactor::sqrt_info = shared_pool->focal_length / shared_pool->pixel_n * Eigen::Matrix2d::Identity();

    if(linefeature_index < 3)
        return;

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = shared_pool->max_num_iterations;
    options.max_solver_time_in_seconds = shared_pool->max_solver_time;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // std::cout << "refine line time: " << std::setw(6) << refine_line_tictoc.toc() << " ms" << std::endl;

    ceres2state();
}


void Estimator::optimization()
{
    state2ceres();

    utils::TicToc solver_tictoc;
    ceres::Problem problem;
    // error = error_norm < 1 ? linear error : sqrt error
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0); // huber kernel function

    addParameterBlock(problem);

    addResidualBlock(problem, loss_function);

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = shared_pool->max_num_iterations;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (shared_pool->marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = shared_pool->max_solver_time * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = shared_pool->max_solver_time;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // std::cout << "solver time: " << std::setw(6) << solver_tictoc.toc() << " ms" << std::endl;

    ceres2state();

    if (cur_frame_count < WINDOW_SIZE)
        return;

    utils::TicToc margin_tictoc;
    marginalization(problem, loss_function);
    // std::cout << "marginalization time: " << std::setw(6) << margin_tictoc.toc() << " ms" << std::endl;
}


void Estimator::addParameterBlock(ceres::Problem &problem)
{
    /* pose parameter block */
    for (int i = 0; i != cur_frame_count + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);

        if (shared_pool->is_use_imu)
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }

    /* imu parameter block */
    if (!shared_pool->is_use_imu)
        problem.SetParameterBlockConstant(para_Pose[0]); // fix init pose

    /* extrinsic parameter block */
    static bool openExEstimation = false;
    for (int i = 0; i != shared_pool->cam_num; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);

        if ((shared_pool->is_estimate_extrinsic && cur_frame_count == WINDOW_SIZE && state_window[0].body_state.v.norm() > 0.2) || openExEstimation)
            openExEstimation = true;
        else
            problem.SetParameterBlockConstant(para_Ex_Pose[i]); // fix extrinsic param
    }

    /* timedelay parameter block */
    problem.AddParameterBlock(para_Td[0], 1);
    if (!shared_pool->is_estimate_timedelay || state_window[0].body_state.v.norm() < 0.2)
        problem.SetParameterBlockConstant(para_Td[0]);

#if ENABLE_LINE_FEATURE
    int linefeature_index = -1;
    for (auto &linefeature_per_id : linefeature_manager->linefeature_list)
    {
        if (linefeature_per_id.linefeature_per_frame.size() < shared_pool->reliable_line_track_count)
            continue;

        if (!linefeature_per_id.is_triangulate)
            continue;

        ++linefeature_index;

        ceres::LocalParameterization *local_parameterization = new LineLocalParameterization();
        problem.AddParameterBlock(para_LineFeature[linefeature_index], SIZE_LINEFEATURE, local_parameterization);
    }
#endif
}


void Estimator::addResidualBlock(ceres::Problem &problem, ceres::LossFunction *loss_function)
{
    /* marginalization residual block */
    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);

        problem.AddResidualBlock(marginalization_factor, NULL, last_marginalization_info->keep_block_addr);
    }
    
    /* imu residual block */
    if (shared_pool->is_use_imu)
    {
        for (int i = 0; i < cur_frame_count; i++)
        {
            int j = i + 1;
            if (state_window[j].pre_integration->sum_dt > 10.0)
                continue;
            ImuFactor* imu_factor = new ImuFactor(state_window[j].pre_integration);

            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
    }

    /* visual residual block */
    int feature_opt_count = 0;
    int feature_index = -1;
    for (auto &feature_per_id : feature_manager->feature_list)
    {
        if (feature_per_id.feature_per_frame.size() < shared_pool->reliable_track_count)
            continue;

        ++feature_index;

        int frame_i = feature_per_id.start_frame, frame_j = frame_i - 1;
        Eigen::Vector3d pts_i = feature_per_id.feature_per_frame[0].point_0;

        for (auto &feature_per_frame : feature_per_id.feature_per_frame)
        {
            frame_j++;
            if (frame_i != frame_j)
            {
                Eigen::Vector3d pts_j = feature_per_frame.point_0;

                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(
                    pts_i,  feature_per_id.feature_per_frame[0].vel_0,  feature_per_id.feature_per_frame[0].td,
                    pts_j,  feature_per_frame.vel_0,                    feature_per_frame.td);

                problem.AddResidualBlock(f_td, loss_function, para_Pose[frame_i], para_Pose[frame_j], 
                    para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                
                feature_opt_count++;
            }
            if (shared_pool->is_use_stereo && feature_per_frame.is_stereo)
            {
                Eigen::Vector3d pts_j_right = feature_per_frame.point_1;
                
                if (frame_i != frame_j)
                {
                    ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(
                        pts_i,          feature_per_id.feature_per_frame[0].vel_0,  feature_per_id.feature_per_frame[0].td,
                        pts_j_right,    feature_per_frame.vel_1,                    feature_per_frame.td);

                    problem.AddResidualBlock(f, loss_function, para_Pose[frame_i], para_Pose[frame_j], 
                        para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                    
                    // feature_opt_count++;
                }
                else
                {
                    ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(
                        pts_i,          feature_per_id.feature_per_frame[0].vel_0,  feature_per_id.feature_per_frame[0].td,
                        pts_j_right,    feature_per_frame.vel_1,                    feature_per_frame.td);

                    problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], 
                        para_Feature[feature_index], para_Td[0]);
                    
                    // feature_opt_count++;
                }
            }
        }
    }
    // std::cout << "opt point feature num: " << feature_index << std::endl;

#if ENABLE_LINE_FEATURE
    int linefeature_opt_count = 0;
    int linefeature_index = -1;
    for (auto &linefeature_per_id : linefeature_manager->linefeature_list)
    {
        if (linefeature_per_id.linefeature_per_frame.size() < shared_pool->reliable_line_track_count)
            continue;

        if (!linefeature_per_id.is_triangulate)
            continue;

        ++linefeature_index;

        int frame_i = linefeature_per_id.start_frame, frame_j = frame_i - 1;

        for (auto &linefeature_per_frame : linefeature_per_id.linefeature_per_frame)
        {
            frame_j++;

            if (linefeature_per_frame.is_tracked == false)
                continue;

            // if (frame_i != frame_j)
            {
                Eigen::Vector4d line_j = linefeature_per_frame.line_0;

                ProjectionLineFactor *f = new ProjectionLineFactor(line_j);

                problem.AddResidualBlock(f, loss_function, para_Pose[frame_j], para_Ex_Pose[0], para_LineFeature[linefeature_index]);
                
                linefeature_opt_count++;
            }
            if (shared_pool->is_use_stereo && linefeature_per_frame.is_stereo)
            {
                Eigen::Vector4d line_j_right = linefeature_per_frame.line_1;

                ProjectionLineFactor *f = new ProjectionLineFactor(line_j_right);

                problem.AddResidualBlock(f, loss_function, para_Pose[frame_j], para_Ex_Pose[1], para_LineFeature[linefeature_index]);
                
                // linefeature_opt_count++;
            }
        }
    }

    double ratio = 1;

    if (shared_pool->is_use_stereo)
    {
        ratio = 50.0 * (10 * linefeature_opt_count + 1) / (feature_opt_count + 10 * linefeature_opt_count + 1);
        ratio = std::min(std::max(ratio, 1.0), 50.0);
    }
    else
    {
        ratio = 50.0 * (10 * linefeature_opt_count + 1) / (feature_opt_count + 10 * linefeature_opt_count + 1);
        ratio = std::min(std::max(ratio, 1.0), 50.0);
    }

    if (feature_opt_count > 0 && linefeature_opt_count > 0)
    {
        ProjectionLineFactor::sqrt_info = ratio * shared_pool->focal_length / shared_pool->pixel_n * Eigen::Matrix2d::Identity();
        ProjectionTwoFrameOneCamFactor::sqrt_info = shared_pool->focal_length / shared_pool->pixel_n * Eigen::Matrix2d::Identity();
        ProjectionTwoFrameTwoCamFactor::sqrt_info = shared_pool->focal_length / shared_pool->pixel_n * Eigen::Matrix2d::Identity();
        ProjectionOneFrameTwoCamFactor::sqrt_info = shared_pool->focal_length / shared_pool->pixel_n * Eigen::Matrix2d::Identity();
    }
    else
    {
        ProjectionLineFactor::sqrt_info = shared_pool->focal_length / shared_pool->pixel_n * Eigen::Matrix2d::Identity();
        ProjectionTwoFrameOneCamFactor::sqrt_info = shared_pool->focal_length / shared_pool->pixel_n * Eigen::Matrix2d::Identity();
        ProjectionTwoFrameTwoCamFactor::sqrt_info = shared_pool->focal_length / shared_pool->pixel_n * Eigen::Matrix2d::Identity();
        ProjectionOneFrameTwoCamFactor::sqrt_info = shared_pool->focal_length / shared_pool->pixel_n * Eigen::Matrix2d::Identity();
    }

    // std::cout << "feature_opt_count = " << feature_opt_count << std::endl;
    // std::cout << "linefeature_opt_count = " << linefeature_opt_count << std::endl;

#endif
}


void Estimator::marginalization(ceres::Problem &problem, ceres::LossFunction *loss_function)
{
    switch (shared_pool->marginalization_flag)
    {
    case MARGIN_OLD:
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo(); // select raw pointer for efficiency
        state2ceres();

        /* marginalizetion factors */
        /* propagate the last_marginalization_info to the marginalization_info */
        if (last_marginalization_info && last_marginalization_info->valid)
        {
            /* the oldest frame will always be marginalized */
            std::vector<int> drop_set;
            for (size_t i = 0; i != last_marginalization_info->keep_block_addr.size(); i++)
            {
                if (last_marginalization_info->keep_block_addr[i] == para_Pose[0] ||
                    last_marginalization_info->keep_block_addr[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }

            /* construct new marginlization_factor */
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);

            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                marginalization_factor, NULL, last_marginalization_info->keep_block_addr, drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        /* imu factors */
        if (shared_pool->is_use_imu)
        {
            if (state_window[1].pre_integration->sum_dt < 10.0)
            {
                ImuFactor* imu_factor = new ImuFactor(state_window[1].pre_integration);

                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                    std::vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                    std::vector<int>{0, 1});

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        /* visual factors */
        int feature_index = -1;
        for (auto &feature_per_id : feature_manager->feature_list)
        {
            if (feature_per_id.feature_per_frame.size() < shared_pool->reliable_track_count)
                continue;

            ++feature_index;

            if (feature_per_id.start_frame != 0)
                continue;

            int frame_i = feature_per_id.start_frame, frame_j = frame_i - 1;

            Eigen::Vector3d pts_i = feature_per_id.feature_per_frame[0].point_0;

            for (auto &feature_per_frame : feature_per_id.feature_per_frame)
            {
                frame_j++;
                if (frame_i != frame_j)
                {
                    Eigen::Vector3d pts_j = feature_per_frame.point_0;

                    ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(
                        pts_i,  feature_per_id.feature_per_frame[0].vel_0,  feature_per_id.feature_per_frame[0].td,
                        pts_j,  feature_per_frame.vel_0,                    feature_per_frame.td);

                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                        std::vector<double *>{para_Pose[frame_i], para_Pose[frame_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                        std::vector<int>{0, 3});
                        
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
                if (shared_pool->is_use_stereo && feature_per_frame.is_stereo)
                {
                    Eigen::Vector3d pts_j_right = feature_per_frame.point_1;
                    if (frame_i != frame_j)
                    {
                        ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(
                            pts_i,          feature_per_id.feature_per_frame[0].vel_0,  feature_per_id.feature_per_frame[0].td,
                            pts_j_right,    feature_per_frame.vel_1,                    feature_per_frame.td);

                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                            std::vector<double *>{para_Pose[frame_i], para_Pose[frame_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                            std::vector<int>{0, 4});

                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(
                            pts_i,          feature_per_id.feature_per_frame[0].vel_0,  feature_per_id.feature_per_frame[0].td,
                            pts_j_right,    feature_per_frame.vel_1,                    feature_per_frame.td);

                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                            std::vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                            std::vector<int>{2});

                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }
    
#if ENABLE_LINE_FEATURE
        int linefeature_index = -1;
        for (auto &linefeature_per_id : linefeature_manager->linefeature_list)
        {
            if (linefeature_per_id.linefeature_per_frame.size() < shared_pool->reliable_line_track_count)
                continue;

            if (!linefeature_per_id.is_triangulate)
                continue;

            ++linefeature_index;

            if (linefeature_per_id.start_frame != 0)
                continue;
            
            int frame_i = linefeature_per_id.start_frame, frame_j = frame_i - 1;

            for (auto &linefeature_per_frame : linefeature_per_id.linefeature_per_frame)
            {
                frame_j++;

                if (linefeature_per_frame.is_tracked == false)
                    continue;
                
                // if (frame_i != frame_j)
                {
                    Eigen::Vector4d line_j = linefeature_per_frame.line_0;

                    ProjectionLineFactor *f = new ProjectionLineFactor(line_j);
                    
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                        std::vector<double *>{para_Pose[frame_j], para_Ex_Pose[0], para_LineFeature[linefeature_index]},
                        std::vector<int>{2});
                    
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
                if (shared_pool->is_use_stereo && linefeature_per_frame.is_stereo)
                {
                    continue; // have marginalized int previous step
                    Eigen::Vector4d line_j_right = linefeature_per_frame.line_1;

                    ProjectionLineFactor *f = new ProjectionLineFactor(line_j_right);
                    
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                        std::vector<double *>{para_Pose[frame_j], para_Ex_Pose[1], para_LineFeature[linefeature_index]},
                        std::vector<int>{2});
                    
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
            }
        }
#endif

        marginalization_info->preMarginalize();

        marginalization_info->marginalize();


        shiftMargAddrOld(marginalization_info);


        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;

        break;
    }
    case MARGIN_SECOND_NEW:
    {
        if (!last_marginalization_info || std::count(std::begin(last_marginalization_info->keep_block_addr),
            std::end(last_marginalization_info->keep_block_addr), para_Pose[WINDOW_SIZE - 1]) == 0)
            return;

        MarginalizationInfo *marginalization_info = new MarginalizationInfo(); // select raw pointer for efficiency
        state2ceres();

        if (last_marginalization_info && last_marginalization_info->valid)
        {
            std::vector<int> drop_set;
            for (size_t i = 0; i != last_marginalization_info->keep_block_addr.size(); i++)
            {
                ROS_ASSERT(last_marginalization_info->keep_block_addr[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                if (last_marginalization_info->keep_block_addr[i] == para_Pose[WINDOW_SIZE - 1])
                    drop_set.push_back(i);
            }
            /* construct new marginlization_factor */
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);

            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                last_marginalization_info->keep_block_addr, drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        marginalization_info->preMarginalize();

        marginalization_info->marginalize();
        

        shiftMargAddrNew(marginalization_info);


        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;

        break;
    }
    default:
        break;
    }
}


void Estimator::shiftMargAddrOld(MarginalizationInfo *marginalization_info)
{
    std::unordered_map<long, double *> addr_shift;
    for (int i = 1; i != WINDOW_SIZE + 1; i++)
    {
        addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
        if (shared_pool->is_use_imu)
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
    }
    for (int i = 0; i != shared_pool->cam_num; i++)
        addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

    addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

    marginalization_info->keepParameterBlocks(addr_shift);
}


void Estimator::shiftMargAddrNew(MarginalizationInfo *marginalization_info)
{
    std::unordered_map<long, double *> addr_shift;
    for (int i = 0; i != WINDOW_SIZE + 1; i++)
    {
        if (i == WINDOW_SIZE - 1)
            continue;
        else if (i == WINDOW_SIZE)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            if (shared_pool->is_use_imu)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        else
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
            if (shared_pool->is_use_imu)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
        }
    }
    for (int i = 0; i != shared_pool->cam_num; i++)
        addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

    addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

    marginalization_info->keepParameterBlocks(addr_shift);
}


void Estimator::slideWindow()
{
    if (cur_frame_count != WINDOW_SIZE)
    {
        int pre_frame_count = cur_frame_count++;
        state_window[cur_frame_count].body_state = state_window[pre_frame_count].body_state;
        
        return;
    }

    utils::TicToc slideWindow_tictoc;
    switch (shared_pool->marginalization_flag)
    {
    case MARGIN_OLD:
    {
        marg_state = state_window[0].body_state;

        /* [0, WINDOW_SIZE - 1] */
        for (int i = 0; i != WINDOW_SIZE; i++)
            state_window[i].swap(state_window[i + 1]);

        /* [WINDOW_SIZE - 1, WINDOW_SIZE] */
        state_window[WINDOW_SIZE].body_state = state_window[WINDOW_SIZE - 1].body_state;

        if (shared_pool->is_use_imu)
        {
            state_window[WINDOW_SIZE].pre_integration = nullptr;
        }

        double t_0 = marg_state.t;
        auto iter = find_if(image_frame_list.begin(), image_frame_list.end(), 
            [&t_0](const ImageFrame &iter) { return iter.t == t_0; });
    
        image_frame_list.erase(image_frame_list.begin(), iter);

        slideWindowOld();

        break;
    }
    case MARGIN_SECOND_NEW:
    {
        FrameState &pre_frame_state = state_window[cur_frame_count - 1];
        FrameState &cur_frame_state = state_window[cur_frame_count];

        pre_frame_state.body_state = cur_frame_state.body_state;

        if (shared_pool->is_use_imu)
        {
            const std::vector<double> &dt_vector = cur_frame_state.pre_integration->dt_vector;
            const std::vector<Eigen::Vector3d> &acc_vector = cur_frame_state.pre_integration->acc_vector;
            const std::vector<Eigen::Vector3d> &gyr_vector = cur_frame_state.pre_integration->gyr_vector;

            for (size_t i = 0; i != dt_vector.size(); i++)
                pre_frame_state.pre_integration->push_back(dt_vector[i], acc_vector[i], gyr_vector[i]);
            
            cur_frame_state.pre_integration = nullptr;
        }

        slideWindowNew();

        break;
    }
    default:
        break;
    }
}


void Estimator::slideWindowOld()
{
    if (shared_pool->solver_flag == NONLINEAR)
    {
        Eigen::Matrix3d margR_w_cam0, R0_w_cam0;
        Eigen::Vector3d margP_w_cam0, P0_w_cam0;
        margR_w_cam0 = marg_state.R * shared_pool->extrinsics.R_imu_cam[0];
        margP_w_cam0 = marg_state.p + marg_state.R * shared_pool->extrinsics.t_imu_cam[0];
        R0_w_cam0 = state_window[0].body_state.R * shared_pool->extrinsics.R_imu_cam[0];
        P0_w_cam0 = state_window[0].body_state.p + state_window[0].body_state.R * shared_pool->extrinsics.t_imu_cam[0];
        feature_manager->removeOldShiftDepth(margR_w_cam0, margP_w_cam0, R0_w_cam0, P0_w_cam0);
    }
    else
    {
        feature_manager->removeOld();
    }
#if ENABLE_LINE_FEATURE
    linefeature_manager->removeOld();
#endif
}


void Estimator::slideWindowNew()
{
    feature_manager->removeSecondNew(cur_frame_count);
#if ENABLE_LINE_FEATURE
    linefeature_manager->removeSecondNew(cur_frame_count);
#endif
}


void Estimator::updateLatestStates()
{
    shared_pool->propagate_mutex.lock();

    imu_processor->imu_buffer_mutex.lock();
    ImuBuffer tmp_imu_buffer = imu_processor->imu_buffer;
    imu_processor->imu_buffer_mutex.unlock();

    shared_pool->latest_state = state_window[WINDOW_SIZE].body_state;
    while(!tmp_imu_buffer.empty())
    {
        ImuData imu_data = tmp_imu_buffer.front();
        double dt = imu_data.t - shared_pool->latest_state.t;
        imu_processor->fastPredict(shared_pool->latest_state, dt, imu_data);
        tmp_imu_buffer.pop();
    }

    shared_pool->propagate_mutex.unlock();

    shared_pool->key_poses.clear();
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
        shared_pool->key_poses.push_back(state_window[i].body_state.p);

}


void Estimator::pubLatestStates()
{
    RosMaster::pubOdometrySlow(*this);
    RosMaster::pubCameraPose(*this);
    RosMaster::pubKeyPoses(*this);
    RosMaster::pubKeyframe(*this);
    RosMaster::pubPointCloud(*this);
#if ENABLE_LINE_FEATURE
    RosMaster::pubLineCloud(*this);
#endif
    RosMaster::pubTF(*this);
    RosMaster::printStatistics(*this);
}
