#include "odom_utils/utils.h"
#include "factor/integrationBase.h"
#include "estimator/paramPool.h"

#include "initializer/initializer.h"


extern std::unique_ptr<ParamPool> shared_pool;

Initializer::Initializer()
{
    void clearState();
}


void Initializer::clearState()
{
	ref_count = 0;
    g = G;
    sfm_features.clear();

    for (int i = 0; i != WINDOW_SIZE + 1; i++)
    {
        Q[i].setIdentity();
        R[i].setIdentity();
        P[i].setZero();
        cQ[i].setIdentity();
        cR[i].setIdentity();
        cP[i].setZero();
        cT[i].setZero();
    }
}


void Initializer::setParameters()
{
    // nothing
}


bool Initializer::initialStructure(ImageFrameList &image_frame_list, FrameWindow &state_window, std::shared_ptr<FeatureManager> feature_manager)
{
    clearState();

    for (const auto &feature_per_id : feature_manager->feature_list)
    {
        sfm_features.emplace_back(feature_per_id);
    }

    if (!selectReferanceFrame(feature_manager))
    {
        ROS_INFO("Not enough features or parallax, move device around");
        return false;
    }

    if (!construct())
    {
        ROS_WARN("SFM construct failed!");
        return false;
    }

    if (!optimization())
    {
        ROS_WARN("SFM optimization failed!");
        return false;
    }

    if (!reconstruct(image_frame_list, state_window))
    {
        ROS_WARN("SFM reconstruct failed!");
        return false;
    }

    //solve scale
    if (!visualInertialAlignment(image_frame_list, state_window))
    {
        ROS_WARN("misalign visual structure with inertial");
        return false;
    }

    if (!refineGravity(image_frame_list))
    {
        ROS_WARN("refine gravity failed!");
        return false;
    }

    if (!recoveryStructure(image_frame_list, state_window, feature_manager))
    {
        ROS_WARN("recovery structure failed");
        return false;
    }

    return true;
}

/** \brief select referance frame via average parallax between referance frame and WINDOW_SIZE frame
 * @param feature_manager feature manager
 * @param Rotation output rotation from WINDOW_SIZE frame to referance frame
 * @param translation output translation from WINDOW_SIZE frame to referance frame
 * @param ref_count output the frame_id of referance frame
 */
bool Initializer::selectReferanceFrame(std::shared_ptr<FeatureManager> feature_manager)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i != WINDOW_SIZE; i++)
    {
        FeaturePairs feature_pairs = feature_manager->getFeaturePairs(i, WINDOW_SIZE);
        int n = feature_pairs.size();

        if (n > 20)
        {
            double sum_parallax = 0;
            // calculate average parallax
            for (int j = 0; j != n; j++)
            {
                const Eigen::Vector2d &pts_ref = feature_pairs[j].first.head<2>();
                const Eigen::Vector2d &pts_new = feature_pairs[j].second.head<2>();
                sum_parallax += (pts_ref - pts_new).norm();
            }

            double average_parallax = sum_parallax / n;
            // calculate rotation and translation of referance frame by 5-points method, if the average parallax is enough
            if (average_parallax * 460 > 30 && solveRelativeRt(feature_pairs, R[WINDOW_SIZE], P[WINDOW_SIZE]))
            {
                ref_count = i; // determine the referance frame count
                Q[WINDOW_SIZE] = R[WINDOW_SIZE];
                // std::cout << "average_parallax " <<  average_parallax * 460 << " choose referance frame "
                //     << ref_count << " and newest frame to triangulate the whole structure\n";
                return true;
            }
        }
    }
    return false;
}


bool Initializer::solveRelativeRt(const FeaturePairs &feature_pairs, Eigen::Matrix3d &Rotation, Eigen::Vector3d &translation)
{
    if (feature_pairs.size() < 15)
        return false;
    
    std::vector<cv::Point2f> pts_ref, pts_new;
    for (int i = 0; i < int(feature_pairs.size()); i++)
    {
        pts_ref.emplace_back(feature_pairs[i].first(0), feature_pairs[i].first(1));
        pts_new.emplace_back(feature_pairs[i].second(0), feature_pairs[i].second(1));
    }

    cv::Mat mask;
    cv::Mat E = cv::findFundamentalMat(pts_ref, pts_new, cv::FM_RANSAC, 0.3 / 460, 0.99, mask);
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    cv::Mat cv_Rotation, cv_translation;
    int inlier_cnt = cv::recoverPose(E, pts_ref, pts_new, cameraMatrix, cv_Rotation, cv_translation, mask);

    if (inlier_cnt > 12)
    {
        cv::cv2eigen(cv_Rotation, Rotation);
        cv::cv2eigen(cv_translation, translation);

        Rotation.transposeInPlace();
        translation = Rotation * -translation;

        return true;
    }

    return false;
}


bool Initializer::construct()
{
	cQ[ref_count] = Q[ref_count].inverse();
	cR[ref_count] = cQ[ref_count].toRotationMatrix();
	cP[ref_count] = cR[ref_count] * -P[ref_count];
	cT[ref_count].block<3, 3>(0, 0) = cR[ref_count];
	cT[ref_count].block<3, 1>(0, 3) = cP[ref_count];

	cQ[WINDOW_SIZE] = Q[WINDOW_SIZE].inverse();
	cR[WINDOW_SIZE] = cQ[WINDOW_SIZE].toRotationMatrix();
	cP[WINDOW_SIZE] = cR[WINDOW_SIZE] * -P[WINDOW_SIZE];
	cT[WINDOW_SIZE].block<3, 3>(0, 0) = cR[WINDOW_SIZE];
	cT[WINDOW_SIZE].block<3, 1>(0, 3) = cP[WINDOW_SIZE];


    /* triangulate the covisible points between (frame_count --- WINDOW_SIZE - 1) and (WINDOW_SIZE) */
    /* solve pnp for frame (ref_count + 1 ---- WINDOW_SIZE - 1) */
	for (int i = ref_count; i != WINDOW_SIZE; i++)
	{
		// solve pnp
		if (i > ref_count)
		{
			cR[i] = cR[i - 1];
			cP[i] = cP[i - 1];

			if (!solveFrameByPnP(cR[i], cP[i], i))
				return false;

			cQ[i] = cR[i];
			cT[i].block<3, 3>(0, 0) = cR[i];
			cT[i].block<3, 1>(0, 3) = cP[i];
		}

		triangulateTwoFrames(i, cT[i], WINDOW_SIZE, cT[WINDOW_SIZE]);
	}
	

    /* triangulate the covisible points between (ref_count) and (ref_count + 1 --- WINDOW_SIZE - 1) */
	for (int i = ref_count + 1; i != WINDOW_SIZE; i++)
		triangulateTwoFrames(ref_count, cT[ref_count], i, cT[i]);
	

    /* solve pnp for frame (ref_count - 1 ---- 0) */
    /* triangulate the covisible points between (ref_count - 1 ---- 0) and (ref_count) */
	for (int i = ref_count - 1; i >= 0; i--)
	{
		//solve pnp
		cR[i] = cR[i + 1];
		cP[i] = cP[i + 1];

		if (!solveFrameByPnP(cR[i], cP[i], i))
			return false;

		cQ[i] = cR[i];
		cT[i].block<3, 3>(0, 0) = cR[i];
		cT[i].block<3, 1>(0, 3) = cP[i];

		triangulateTwoFrames(i, cT[i], ref_count, cT[ref_count]);
	}


	/* triangulate all other points */
    for (auto &sfm_feature : sfm_features)
    {
		if (sfm_feature.is_triangulate)
			continue;

		if (sfm_feature.covisible_frames.size() < 2)
            continue;

        int frame_0 = sfm_feature.covisible_frames[0].first;
        int frame_1 = sfm_feature.covisible_frames.back().first;
        Eigen::Vector2d point0 = sfm_feature.covisible_frames[0].second;
        Eigen::Vector2d point1 = sfm_feature.covisible_frames.back().second;

        Eigen::Vector3d point3d;
        FeatureManager::triangulatePoint(cT[frame_0], cT[frame_1], point0, point1, point3d);
        sfm_feature.is_triangulate = true;
        sfm_feature.para_point3d[0] = point3d(0);
        sfm_feature.para_point3d[1] = point3d(1);
        sfm_feature.para_point3d[2] = point3d(2);
    }

    return true;
}


/** \brief 2d-3d PnP
 * @param Rotation Input & Output: input initial rotation, output rotation solved by PnP.
 * @param translation Input & Output: input initial translation, output translation solved by PnP.
 * @param frame_count the frame count
 * @param sfm_features sfm features
 * @return true if solve frame pose by PnP successfully, false if not
 */
bool Initializer::solveFrameByPnP(Eigen::Matrix3d &Rotation, Eigen::Vector3d &translation, int frame_count)
{
	// get all 2d-3d point pairs
	std::vector<cv::Point2f> pts2D;
	std::vector<cv::Point3f> pts3D;

    for (const auto &sfm_feature : sfm_features)
    {
        if (!sfm_feature.is_triangulate)
            continue;

        for (const auto &covisible_frame : sfm_feature.covisible_frames)
        {
            if (covisible_frame.first == frame_count)
            {
                const Eigen::Vector3d &point3d = Eigen::Map<const Eigen::Vector3d>(sfm_feature.para_point3d);
                const Eigen::Vector2d &point2d = covisible_frame.second;
                pts3D.emplace_back(point3d.x(), point3d.y(), point3d.z());
                pts2D.emplace_back(point2d.x(), point2d.y());
                break;
            }
        }
    }

	// enough 2d-3d point pairs for precision
	if (pts2D.size() < 15)
	{
		std::cout << "unstable features tracking, please slowly move you device!" << std::endl;
        return false;
	}

    if (!FeatureManager::solvePoseByPnP(pts3D, pts2D, Rotation, translation, false))
    {
        return false;
    }

	return true;
}


/** \brief triangulate points which is covisible for two frames
 * @param frame0 frame0 count
 * @param Pose0 transform from referance frame to frame0
 * @param frame1 frame1 count
 * @param Pose1 transform from referance frame to frame1
 * @param sfm_features sfm features
 */
void Initializer::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
									   int frame1, Eigen::Matrix<double, 3, 4> &Pose1)
{
    for (auto &sfm_feature : sfm_features)
    {
		if (sfm_feature.is_triangulate)
			continue;

		Eigen::Vector2d point0, point1;
		bool has_0 = false, has_1 = false;

		for (const auto &covisible_frame : sfm_feature.covisible_frames)
		{
			if (covisible_frame.first == frame0)
			{
				point0 = covisible_frame.second;
				has_0 = true;
			}
			if (covisible_frame.first == frame1)
			{
				point1 = covisible_frame.second;
				has_1 = true;
			}
		}

		if (has_0 && has_1)
		{
			Eigen::Vector3d point_3d;
			FeatureManager::triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			sfm_feature.is_triangulate = true;
			sfm_feature.para_point3d[0] = point_3d(0);
			sfm_feature.para_point3d[1] = point_3d(1);
			sfm_feature.para_point3d[2] = point_3d(2);
		}
    }
}


bool Initializer::optimization()
{
	// full BA
	ceres::Problem problem;
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();

	double para_cQ[WINDOW_SIZE + 1][4];
	double para_cP[WINDOW_SIZE + 1][3];
	/* set parameter block */
	for (int i = 0; i < WINDOW_SIZE + 1; i++)
	{
		para_cP[i][0] = cP[i].x();
		para_cP[i][1] = cP[i].y();
		para_cP[i][2] = cP[i].z();
		para_cQ[i][0] = cQ[i].w();
		para_cQ[i][1] = cQ[i].x();
		para_cQ[i][2] = cQ[i].y();
		para_cQ[i][3] = cQ[i].z();


		problem.AddParameterBlock(para_cQ[i], 4, local_parameterization);
		problem.AddParameterBlock(para_cP[i], 3);
		if (i == ref_count)
		{
			problem.SetParameterBlockConstant(para_cQ[i]);
		}
		if (i == ref_count || i == WINDOW_SIZE)
		{
			problem.SetParameterBlockConstant(para_cP[i]);
		}
	}

	/* set residual block */
    for (auto &sfm_feature : sfm_features)
    {
		if (!sfm_feature.is_triangulate)
			continue;

        for (auto &covisible_frame : sfm_feature.covisible_frames)
        {
			int frame_count = covisible_frame.first;

			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
                covisible_frame.second.x(),
                covisible_frame.second.y());

    		problem.AddResidualBlock(cost_function, NULL, para_cQ[frame_count], para_cP[frame_count], sfm_feature.para_point3d);
        }
    }


	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	//options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.2;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
	{
        ROS_INFO("vision only BA converge");
	}
	else
	{
        ROS_WARN("vision only BA NOT converge");
		return false;
	}
    
	for (int i = 0; i < WINDOW_SIZE + 1; i++)
	{
		// Q[i] = Eigen::Map<const Eigen::Quaterniond>(&para_cQ[i][0]).inverse();
        Q[i].w() = para_cQ[i][0];
        Q[i].x() = para_cQ[i][1];
        Q[i].y() = para_cQ[i][2];
        Q[i].z() = para_cQ[i][3];
        Q[i] = Q[i].inverse();
		P[i] = Q[i] * -Eigen::Map<const Eigen::Vector3d>(&para_cP[i][0]);
	}

    return true;
}


bool Initializer::reconstruct(ImageFrameList &image_frame_list, FrameWindow &state_window)
{
    // solve pnp for all frame
    int frame_count = 0;
    for (auto &image_frame : image_frame_list)
    {
        if (image_frame.t == state_window[frame_count].body_state.t)
        {
            image_frame.is_keyframe = true;
            image_frame.R_w_imu = Q[frame_count].toRotationMatrix() * shared_pool->extrinsics.R_imu_cam[0].transpose();
            image_frame.t_w_imu = P[frame_count];
            frame_count++;
            continue;
        }

        if (image_frame.t > state_window[frame_count].body_state.t)
        {
            frame_count++;
        }

        image_frame.is_keyframe = false;
        std::vector<cv::Point3f> pts3D;
        std::vector<cv::Point2f> pts2D;
        for (auto &id_pts : image_frame.feature_points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                auto iter = std::find_if(sfm_features.begin(), sfm_features.end(), [feature_id](SFMFeature &sfm_feature) -> bool {
                    return (sfm_feature.is_triangulate && sfm_feature.feature_id == feature_id);
                });


                if (iter != sfm_features.end())
                {
                    cv::Point3f point3d(iter->para_point3d[0], iter->para_point3d[1], iter->para_point3d[2]);
                    cv::Point2f point2d(i_p.second.x(), i_p.second.y());
                    pts3D.push_back(point3d);
                    pts2D.push_back(point2d);
                }
            }
        }

        if (pts3D.size() < 6)
        {
            ROS_WARN("Not enough points for solve pnp !");
            return false;
        }

        Eigen::Matrix3d R_pnp = Q[frame_count].toRotationMatrix();
        Eigen::Vector3d T_pnp = P[frame_count];

        if (!FeatureManager::solvePoseByPnP(pts3D, pts2D, R_pnp, T_pnp, true))
        {
            return false;
        }

        image_frame.R_w_imu = R_pnp * shared_pool->extrinsics.R_imu_cam[0].transpose();
        image_frame.t_w_imu = T_pnp;
    }

    return true;
}


/** \brief visual inertial alignment (optimize gyr bias, velocity, gravity and scale)
 * @param image_frame_list all image frames for initializition
 * @param state_window slide window state
 * \note https://zhuanlan.zhihu.com/p/111600755?utm_id=0
 */
bool Initializer::visualInertialAlignment(ImageFrameList &image_frame_list, FrameWindow &state_window)
{
    solveGyrBias(image_frame_list, state_window);

    int n = image_frame_list.size() * 3 + 3 + 1; // velocity + gravity + scale

    Eigen::MatrixXd sum_A = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd sum_b = Eigen::VectorXd::Zero(n);

    ImageFrameList::iterator frame_i;
    ImageFrameList::iterator frame_j;
    int i = 0;
    for (frame_i = image_frame_list.begin(); next(frame_i) != image_frame_list.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        Eigen::MatrixXd tmp_A = Eigen::MatrixXd::Zero(6, 10);
        Eigen::VectorXd tmp_b = Eigen::VectorXd::Zero(6);

        double dt = frame_j->pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->R_w_imu.transpose() * dt * dt / 2 * Eigen::Matrix3d::Identity();
        // dividing by 100 is for numerical stability of the scale
        tmp_A.block<3, 1>(0, 9) = frame_i->R_w_imu.transpose() * (frame_j->t_w_imu - frame_i->t_w_imu) / scale_power; 
        
        tmp_b.block<3, 1>(0, 0) = frame_j->pre_integration->sum_alpha + frame_i->R_w_imu.transpose() * frame_j->R_w_imu * shared_pool->extrinsics.t_imu_cam[0] - shared_pool->extrinsics.t_imu_cam[0];
        tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->R_w_imu.transpose() * frame_j->R_w_imu;
        tmp_A.block<3, 3>(3, 6) = frame_i->R_w_imu.transpose() * dt * Eigen::Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->pre_integration->sum_beta;

        Eigen::Matrix<double, 6, 6> cov_inv = Eigen::Matrix<double, 6, 6>::Identity();

        Eigen::MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        Eigen::VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        // velocity
        sum_A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        sum_b.segment<6>(i * 3) += r_b.head<6>();

        // gravity and scale
        sum_A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        sum_b.tail<4>() += r_b.tail<4>();

        // velocity, gravity and scale cross-covariance
        sum_A.block<6, 4>(i * 3, n - 4) += r_A.topRightCorner<6, 4>();
        sum_A.block<4, 6>(n - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }

    // multiply by 1000 for numerical stability
    sum_A = sum_A * 1000.0;
    sum_b = sum_b * 1000.0;
    x = sum_A.ldlt().solve(sum_b); // variable state with velocity, gravity and scale

    // unexpanded scale
    s = (x.tail<1>())(0) / scale_power;
    // estimated gravity
    g = x.segment<3>(n - 4);
    
    if (fabs(g.norm() - G.norm()) > 0.5 || s < 0.0)
    {
        return false;
    }

    return true;
}


void Initializer::solveGyrBias(ImageFrameList &image_frame_list, FrameWindow &state_window)
{
    Eigen::Matrix3d sum_A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d sum_b = Eigen::Vector3d::Zero();
    Eigen::Vector3d delta_bg = Eigen::Vector3d::Zero();

    ImageFrameList::iterator frame_i, frame_j;
    for (frame_i = image_frame_list.begin(); next(frame_i) != image_frame_list.end(); frame_i = frame_j)
    {

        frame_j = next(frame_i);
        
        Eigen::Quaterniond q_ij(frame_i->R_w_imu.transpose() * frame_j->R_w_imu);
        
        Eigen::Matrix3d tmp_A = frame_j->pre_integration->state_jacobian.template block<3, 3>(R_order, BG_order);
        Eigen::Vector3d tmp_b = 2 * (frame_j->pre_integration->sum_gamma.inverse() * q_ij).vec();

        if (tmp_A.array().isNaN().any() || tmp_b.array().isNaN().any())
            continue;

        sum_A += tmp_A.transpose() * tmp_A;
        sum_b += tmp_A.transpose() * tmp_b;
    }

    delta_bg = sum_A.ldlt().solve(sum_b);


    if (!delta_bg.array().isNaN().any())
    {
        // correct gyr bias
        for (int i = 0; i != WINDOW_SIZE + 1; i++)
        {
            state_window[i].body_state.bg += delta_bg;
            state_window[i].pre_integration->repropagate(Eigen::Vector3d::Zero(), state_window[i].body_state.bg);
        }

        // re-integration for all interframe imu data
        for (frame_i = next(image_frame_list.begin()); next(frame_i) != image_frame_list.end( ); frame_i++)
        {
            frame_i->pre_integration->repropagate(Eigen::Vector3d::Zero(), state_window[0].body_state.bg);
        }
    }
    else
        delta_bg = Eigen::Vector3d::Zero();

    ROS_WARN_STREAM("initial gyr bias: " << std::fixed << std::setprecision(6) << delta_bg.transpose());
}


/** \brief refine gravity
 * @param image_frame_list all image frames
 * \note https://zhuanlan.zhihu.com/p/109965927
 */
bool Initializer::refineGravity(ImageFrameList &image_frame_list)
{
    Eigen::Vector3d g0 = g.normalized() * G.norm();
    Eigen::Vector3d lx, ly;
    int n = image_frame_list.size() * 3 + 2 + 1;

    Eigen::MatrixXd sum_A = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd sum_b = Eigen::VectorXd::Zero(n);

    ImageFrameList::iterator frame_i;
    ImageFrameList::iterator frame_j;
    for(int k = 0; k < 4; k++)
    {
        Eigen::MatrixXd lxly(3, 2);
        lxly = tangentBasis(g0);
        int i = 0;
        for (frame_i = image_frame_list.begin(); next(frame_i) != image_frame_list.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            Eigen::MatrixXd tmp_A = Eigen::MatrixXd::Zero(6, 9);
            Eigen::VectorXd tmp_b = Eigen::VectorXd::Zero(6);

            double dt = frame_j->pre_integration->sum_dt;


            tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->R_w_imu.transpose() * dt * dt / 2 * Eigen::Matrix3d::Identity() * lxly;
            // dividing by 100 is for numerical stability of the scale
            tmp_A.block<3, 1>(0, 8) = frame_i->R_w_imu.transpose() * (frame_j->t_w_imu - frame_i->t_w_imu) / scale_power;
            tmp_b.block<3, 1>(0, 0) = frame_j->pre_integration->sum_alpha + frame_i->R_w_imu.transpose() * frame_j->R_w_imu * shared_pool->extrinsics.t_imu_cam[0] - shared_pool->extrinsics.t_imu_cam[0] - frame_i->R_w_imu.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->R_w_imu.transpose() * frame_j->R_w_imu;
            tmp_A.block<3, 2>(3, 6) = frame_i->R_w_imu.transpose() * dt * Eigen::Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->pre_integration->sum_beta - frame_i->R_w_imu.transpose() * dt * Eigen::Matrix3d::Identity() * g0;


            Eigen::Matrix<double, 6, 6> cov_inv = Eigen::Matrix<double, 6, 6>::Identity();
            cov_inv.setIdentity();

            Eigen::MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            Eigen::VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            sum_A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            sum_b.segment<6>(i * 3) += r_b.head<6>();

            sum_A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            sum_b.tail<3>() += r_b.tail<3>();

            sum_A.block<6, 3>(i * 3, n - 3) += r_A.topRightCorner<6, 3>();
            sum_A.block<3, 6>(n - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
        sum_A = sum_A * 1000.0;
        sum_b = sum_b * 1000.0;
        x = sum_A.ldlt().solve(sum_b); // variable state with velocity, gravity, tangentBasis perturbation and scale

        Eigen::VectorXd dg = x.segment<2>(n - 3);
        g0 = (g0 + lxly * dg).normalized() * G.norm();
    }

    g = g0;
    // unexpanded scale
    s = (x.tail<1>())(0) / scale_power;
    
    if (fabs(g.norm() - G.norm()) > 0.5 || s < 0.0)
    {
        return false;
    }

    return true;
}


/** \brief get the orthogonal basis of the tangent plane of gravity
 * @param g0 gravity vector
 * @return the orthogonal basis of the tangent plane of gravity
 */
Eigen::MatrixXd Initializer::tangentBasis(Eigen::Vector3d &g0)
{
    Eigen::Vector3d b, c;
    Eigen::Vector3d a = g0.normalized();
    Eigen::Vector3d tmp(0, 0, 1);
    if (a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    Eigen::MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

/** \brief recovery scale
 * @param image_frame_list
 * @param state_window
 * @param feature_manager
 * @return true if recovery scale successfully, false if not
 */
bool Initializer::recoveryStructure(ImageFrameList &image_frame_list, FrameWindow &state_window, std::shared_ptr<FeatureManager> feature_manager)
{
    /************************ Initialize successfully so far ************************/
    int image_frame_count = 0, frame_count = 0;
    for (auto &image_frame : image_frame_list)
    {
        if (image_frame.is_keyframe)
        {
            state_window[frame_count].body_state.R = image_frame.R_w_imu;
            state_window[frame_count].body_state.p = image_frame.t_w_imu;
            state_window[frame_count].body_state.v = image_frame.R_w_imu * x.segment<3>(image_frame_count * 3);

            if (frame_count++ == WINDOW_SIZE)
                break;
        }
        image_frame_count++;
    }
    std::cout << "frame_count = " << frame_count << std::endl;

    // repropagate pre_integration
    for (int i = 0; i != WINDOW_SIZE + 1; i++)
    {
        state_window[i].pre_integration->repropagate(Eigen::Vector3d::Zero(), state_window[i].body_state.bg);
    }
    

    // recovery scale and set the first frame as referance frame
    for (int i = WINDOW_SIZE; i >= 0; i--)
    {
        state_window[i].body_state.p = 
               s * state_window[i].body_state.p - state_window[i].body_state.R * shared_pool->extrinsics.t_imu_cam[0] 
            - (s * state_window[0].body_state.p - state_window[0].body_state.R * shared_pool->extrinsics.t_imu_cam[0]);
    }


    // align with gravity (not unique)
    Eigen::Matrix3d R0 = ImuProcessor::alignGravity(g);
    // align with the x-axis of the first frame (unique)
    double yaw = math_utils::R2ypr(R0 * state_window[0].body_state.R).x();
    R0 = math_utils::ypr2R(Eigen::Vector3d(-yaw, 0, 0)) * R0;

    std::cout << std::fixed << std::setprecision(6) << "estimated gravity = " << g.transpose() << "\n";

    g = R0 * g; // final gravity

    std::cout << std::fixed << std::setprecision(6) << "initial R0 =\n" << R0 << std::endl;
    std::cout << std::fixed << std::setprecision(6) << "align gravity with " << g.transpose() << "\n";

    // set the state of each frame in slide window with estimated scale
    Eigen::Matrix3d rot_diff = R0;
    for (int i = 0; i != WINDOW_SIZE + 1; i++)
    {
        state_window[i].body_state.p = rot_diff * state_window[i].body_state.p;
        state_window[i].body_state.R = rot_diff * state_window[i].body_state.R;
        state_window[i].body_state.q = Eigen::Quaterniond(state_window[i].body_state.R).normalized();
        state_window[i].body_state.v = rot_diff * state_window[i].body_state.v;
    }

    feature_manager->clearDepth();
    feature_manager->triangulate(state_window);

    return true;
}