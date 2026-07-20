#include "odom_utils/utils.h"
#include "estimator/paramPool.h"

#include "visual/featureManager.h"


extern std::unique_ptr<ParamPool> shared_pool;

FeatureManager::FeatureManager() : track_num(0), new_track_num(0), long_track_num(0)
{
    // nothing
}


void FeatureManager::clearState()
{
    feature_list.clear();
    outlierIndex.clear();
    
    track_num = 0;
    new_track_num = 0;
    long_track_num = 0;
}


void FeatureManager::setParameters()
{
    // nothing
}


/** \brief Insert new frame features to feature manager.
 * @param cur_frame_count the new frame id.
 * @param image_frame the new image frame with its features.
 */
void FeatureManager::insertFeature(const int cur_frame_count, const FeaturePoints &feature_points)
{
    track_num = 0;
    new_track_num = 0;
    long_track_num = 0;

    for (const std::pair<int, std::vector<FeaturePoint>> &frame_feature : feature_points)
    {
        int feature_id = frame_feature.first;
        
        auto iter = find_if (feature_list.begin(), feature_list.end(), 
            [feature_id](const FeaturePerId &feature_per_id) { return feature_per_id.feature_id == feature_id; });

        if (iter == feature_list.end())
        {
            feature_list.emplace_back(feature_id, cur_frame_count);
            feature_list.back().feature_per_frame.emplace_back(frame_feature.second, shared_pool->time_delay);
            new_track_num++;
        }
        else if (iter->feature_id == feature_id)
        {
            iter->feature_per_frame.emplace_back(frame_feature.second, shared_pool->time_delay);
            long_track_num += iter->feature_per_frame.size() >= shared_pool->reliable_track_count ? 1 : 0;
            track_num++;
        }
    }
    // std::cout << "feature_list.size = " << feature_list.size() << std::endl;
}


/** \brief Calculate the parallax between second latest frame and third latest frame.d
 * @param cur_frame_count current frame count.
 * @return the average parallax.
 */
double FeatureManager::calcParallax(const int cur_frame_count) const
{
    double parallax_num = 0, parallax_sum = 0;
    for (auto &feature_per_id : feature_list)
    {
        if (feature_per_id.start_frame < cur_frame_count - 1 && 
            feature_per_id.start_frame + static_cast<int>(feature_per_id.feature_per_frame.size()) >= cur_frame_count)
        {
            const FeaturePerFrame &frame_i = feature_per_id.feature_per_frame[cur_frame_count - 2 - feature_per_id.start_frame];
            const FeaturePerFrame &frame_j = feature_per_id.feature_per_frame[cur_frame_count - 1 - feature_per_id.start_frame];

            Eigen::Vector3d p_i = frame_i.point_0;
            Eigen::Vector3d p_j = frame_j.point_0;
            
            double depth_i = p_i(2);
            double u_i = p_i(0) / depth_i;
            double v_i = p_i(1) / depth_i;

            double depth_j = p_j(2);
            double u_j = p_j(0) / depth_j;
            double v_j = p_j(1) / depth_j;

            double du = u_i - u_j, dv = v_i - v_j;

            parallax_sum += sqrt(du * du + dv * dv);
            parallax_num++;
        }
    }

    return parallax_sum / parallax_num;
}


/** \brief Check if current frame is keyframe or not.
 * @param cur_frame_count current frame count.
 * @return true if current frame is keyframe, false is not.
 */
bool FeatureManager::isKeyframe(const int cur_frame_count) const
{
    // return (cur_frame_count < 2 || track_num < 40 || long_track_num < 20 || new_track_num > 0.5 * track_num ||
    //     calcParallax(cur_frame_count) > shared_pool->min_parallax || calcParallax(cur_frame_count) == 0);
    // return ((shared_pool->latest_state.v.norm() > 0.02 && fabs(shared_pool->latest_state.acc.norm() - G.norm()) > 0.05) &&
    //     (cur_frame_count < 2 || track_num < 20 || calcParallax(cur_frame_count) > shared_pool->min_parallax || calcParallax(cur_frame_count) == 0));
    // return (cur_frame_count < 2 || track_num < 20 || calcParallax(cur_frame_count) > shared_pool->min_parallax ||
    //     (shared_pool->latest_state.v.norm() > 0.02 && fabs(shared_pool->latest_state.acc.norm() - G.norm()) > 0.1));
    // return (cur_frame_count < 2 || track_num < 20 || calcParallax(cur_frame_count) > shared_pool->min_parallax);
    
    return (cur_frame_count < 2 || track_num < 20 || long_track_num < 40 || calcParallax(cur_frame_count) > shared_pool->min_parallax);
}


/** \brief get feature pairs appear simultaneously in two image frames
 * @param frame_i frame i
 * @param frame_j frame j
 * @return feature pairs appear simultaneously in two image frames
 */
FeaturePairs FeatureManager::getFeaturePairs(const int frame_i, const int frame_j)
{
    FeaturePairs feature_pairs;
    Eigen::Vector3d pts_i, pts_j;
    for (auto &feature_per_id : feature_list)
    {
        if (feature_per_id.start_frame <= frame_i && feature_per_id.endFrame() >= frame_j)
        {
            int idx_i = frame_i - feature_per_id.start_frame;
            int idx_j = frame_j - feature_per_id.start_frame;

            pts_i = feature_per_id.feature_per_frame[idx_i].point_0;
            pts_j = feature_per_id.feature_per_frame[idx_j].point_0;
            
            feature_pairs.emplace_back(pts_i, pts_j);
        }
    }
    return feature_pairs;
}


/** \brief Initialize the pose of frame by PnP.
 * @param cur_frame_count current frame count.
 * @param state_window[] the array of frame states in slide window.
 * @param cur_extrinsics the extrinsics between camera and imu in current frame.
 * @return true if initialize frame pose by pnp successfully, false if not.
 */
void FeatureManager::initFramePoseByPnP(const int cur_frame_count, FrameWindow &state_window)
{
    if (cur_frame_count == 0)
        return;

    std::vector<cv::Point3f> pts3D;
    std::vector<cv::Point2f> pts2D;
    const Eigen::Matrix3d &R_imu_cam0 = shared_pool->extrinsics.R_imu_cam[0];
    const Eigen::Vector3d &t_imu_cam0 = shared_pool->extrinsics.t_imu_cam[0];

    for (auto &ferture_per_id : feature_list)
    {
        if (ferture_per_id.depth <= 0) // points without depth
            continue;

        if (ferture_per_id.endFrame() < cur_frame_count)
            continue;

        size_t index = cur_frame_count - ferture_per_id.start_frame;

        Eigen::Vector3d pts_in_body = R_imu_cam0 * (ferture_per_id.feature_per_frame[0].point_0 * ferture_per_id.depth) 
                                        + t_imu_cam0;
        Eigen::Vector3d pts_in_world = state_window[ferture_per_id.start_frame].body_state.R * pts_in_body
                                        + state_window[ferture_per_id.start_frame].body_state.p;

        cv::Point3f point3d(pts_in_world.x(), pts_in_world.y(), pts_in_world.z());
        cv::Point2f point2d(ferture_per_id.feature_per_frame[index].point_0.x(), 
                            ferture_per_id.feature_per_frame[index].point_0.y());

        pts3D.push_back(point3d);
        pts2D.push_back(point2d);
    }

    if (pts2D.size() < 4)
    {
        std::cout << "feature tracking not enough, please slowly move you device!" << std::endl;
        return;
    }

    /* the camera pose of cur_frame_count - 1 frame in world frame */
    Eigen::Matrix3d R_w_cam0 = state_window[cur_frame_count - 1].body_state.R * R_imu_cam0;
    Eigen::Vector3d t_w_cam0 = state_window[cur_frame_count - 1].body_state.R * t_imu_cam0 
                             + state_window[cur_frame_count - 1].body_state.p;

    if (solvePoseByPnP(pts3D, pts2D, R_w_cam0, t_w_cam0, true))
    {
        /* the imu pose of cur_frame_count - 1 frame in world frame */
        state_window[cur_frame_count].body_state.R = R_w_cam0 * R_imu_cam0.transpose();
        state_window[cur_frame_count].body_state.q = Eigen::Quaterniond(state_window[cur_frame_count].body_state.R).normalized();
        state_window[cur_frame_count].body_state.p = R_w_cam0 * R_imu_cam0.transpose() * (-t_imu_cam0) + t_w_cam0;

        if (shared_pool->solver_flag != NONLINEAR)
        {
            std::cout << "frame count " << std::setw(2) << cur_frame_count << std::fixed << std::setprecision(6) 
                << " PnP position = " << state_window[cur_frame_count].body_state.p.transpose() << std::endl;
        }
    }
}


/** \brief Solve the pose of current frame using PnP.
 * @param pts3D point 3D position.
 * @param pts2D point 2D position.
 * @param Rotation Input & Output: input initial rotation, output rotation solved by PnP.
 * @param translation Input & Output: input initial translation, output translation solved by PnP.
 * @param inverse_flag inverse the pose if true
 * @return true if solve the pose of current by PnP successfully, false is not.
 */
bool FeatureManager::solvePoseByPnP(std::vector<cv::Point3f> &pts3D, std::vector<cv::Point2f> &pts2D, 
    Eigen::Matrix3d &Rotation, Eigen::Vector3d &translation, const bool inverse_flag)
{
    /* initial Rotation and translation for iteration */
    if (inverse_flag)
    {
        Rotation.transposeInPlace();
        translation = Rotation * (-translation);
    }

    cv::Mat rvec, tvec, tmp_r;
    cv::eigen2cv(Rotation, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(translation, tvec);
    cv::Mat D, K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);  

    if (!cv::solvePnP(pts3D, pts2D, K, D, rvec, tvec, 1))
    {
        std::cout << "pnp failed !" << std::endl;
        return false;
    }

    tmp_r.release();
    cv::Rodrigues(rvec, tmp_r);
    cv::cv2eigen(tmp_r, Rotation);
    cv::cv2eigen(tvec, translation);

    if (inverse_flag)
    {
        Rotation.transposeInPlace();
        translation = Rotation * (-translation);
    }

    return true;
}

/** \brief triangulate frame points
 * \param state_window slide window state
 * \cite https://blog.csdn.net/hltt3838/article/details/105331457
 */
void FeatureManager::triangulate(const FrameWindow &state_window)
{
    /* transform from left camera to imu */
    const Eigen::Matrix3d &R_imu_cam0 = shared_pool->extrinsics.R_imu_cam[0];
    const Eigen::Vector3d &t_imu_cam0 = shared_pool->extrinsics.t_imu_cam[0];
    /* transform from right camera to imu */
    const Eigen::Matrix3d &R_imu_cam1 = shared_pool->extrinsics.R_imu_cam[1];
    const Eigen::Vector3d &t_imu_cam1 = shared_pool->extrinsics.t_imu_cam[1];

    for (auto &feature_per_id : feature_list)
    {
        if (feature_per_id.depth > 0)
            continue;
            
        /* triangulate using left camera image and right camera image */
        if (shared_pool->is_use_stereo && feature_per_id.feature_per_frame[0].is_stereo)
        {
            int frame_id = feature_per_id.start_frame;

            /* imu pose in world frame in current frame time */
            const Eigen::Matrix3d &R_w_imu = state_window[frame_id].body_state.R;
            const Eigen::Vector3d &t_w_imu = state_window[frame_id].body_state.p;

            Eigen::Matrix<double, 3, 4> T_cam0_w;
            Eigen::Vector3d t_w_cam0 = t_w_imu + R_w_imu * t_imu_cam0;
            Eigen::Matrix3d R_w_cam0 = R_w_imu * R_imu_cam0;
            T_cam0_w.leftCols<3>() = R_w_cam0.transpose();
            T_cam0_w.rightCols<1>() = -R_w_cam0.transpose() * t_w_cam0;
            // std::cout << "left pose " << T_cam0_w << std::endl;

            Eigen::Matrix<double, 3, 4> T_cam1_w;
            Eigen::Vector3d t_w_cam1 = t_w_imu + R_w_imu * t_imu_cam1;
            Eigen::Matrix3d R_w_cam1 = R_w_imu * R_imu_cam1;
            T_cam1_w.leftCols<3>() = R_w_cam1.transpose();
            T_cam1_w.rightCols<1>() = -R_w_cam1.transpose() * t_w_cam1;
            // std::cout << "right pose " << T_cam1_w << std::endl;

            Eigen::Vector2d point0, point1;
            Eigen::Vector3d point3d;
            point0 = feature_per_id.feature_per_frame[0].point_0.head(2);
            point1 = feature_per_id.feature_per_frame[0].point_1.head(2);
            // std::cout << "point0 " << point0.transpose() << std::endl;
            // std::cout << "point1 " << point1.transpose() << std::endl;

            triangulatePoint(T_cam0_w, T_cam1_w, point0, point1, point3d);
            // std::cout << "\npoint3d = " << point3d << std::endl;

            Eigen::Vector3d localPoint;
            localPoint = T_cam0_w.leftCols<3>() * point3d + T_cam0_w.rightCols<1>();
            const double depth = localPoint.z();

            feature_per_id.depth = depth > 0 ? depth : shared_pool->init_depth;
        }
        else if (feature_per_id.feature_per_frame.size() > 1)
        {
#if 1
            /* only use front two frames to triangulate real time */
            int frame_id = feature_per_id.start_frame;

            /* imu pose in world frame in previous frame time */
            const Eigen::Matrix3d &pre_R_w_imu = state_window[frame_id].body_state.R;
            const Eigen::Vector3d &pre_t_w_imu = state_window[frame_id++].body_state.p;
            /* imu pose in world frame in current frame time */
            const Eigen::Matrix3d &cur_R_w_imu = state_window[frame_id].body_state.R;
            const Eigen::Vector3d &cur_t_w_imu = state_window[frame_id].body_state.p;
            // std::cout << "cur_t_w_imu - pre_t_w_imu" << cur_t_w_imu - pre_t_w_imu << std::endl;

            Eigen::Matrix<double, 3, 4> pre_T_cam0_w;
            Eigen::Vector3d pre_t_w_cam0 = pre_t_w_imu + pre_R_w_imu * t_imu_cam0;
            Eigen::Matrix3d pre_R_w_cam0 = pre_R_w_imu * R_imu_cam0;
            pre_T_cam0_w.leftCols<3>() = pre_R_w_cam0.transpose();
            pre_T_cam0_w.rightCols<1>() = -pre_R_w_cam0.transpose() * pre_t_w_cam0;
            // std::cout << "previous pose " << pre_T_cam0_w << std::endl;

            Eigen::Matrix<double, 3, 4> cur_T_cam0_w;
            Eigen::Vector3d cur_t_w_cam0 = cur_t_w_imu + cur_R_w_imu * t_imu_cam0;
            Eigen::Matrix3d cur_R_w_cam0 = cur_R_w_imu * R_imu_cam0;
            cur_T_cam0_w.leftCols<3>() = cur_R_w_cam0.transpose();
            cur_T_cam0_w.rightCols<1>() = -cur_R_w_cam0.transpose() * cur_t_w_cam0;
            // std::cout << "current pose " << cur_T_cam0_w << std::endl;

            Eigen::Vector2d pre_point_0, cur_point_0;
            Eigen::Vector3d point3d;
            pre_point_0 = feature_per_id.feature_per_frame[0].point_0.head(2);
            cur_point_0 = feature_per_id.feature_per_frame[1].point_0.head(2);
            // std::cout << "pre_point_0 " << pre_point_0.transpose() << std::endl;
            // std::cout << "cur_point_0 " << cur_point_0.transpose() << std::endl;

            triangulatePoint(pre_T_cam0_w, cur_T_cam0_w, pre_point_0, cur_point_0, point3d);
            Eigen::Vector3d localPoint;
            localPoint = pre_T_cam0_w.leftCols<3>() * point3d + pre_T_cam0_w.rightCols<1>();
            const double depth = localPoint.z();

            feature_per_id.depth = depth > 0 ? depth : shared_pool->init_depth;
#else
            /* use all covisible frames to triangulate for performance */
            int frame_id = feature_per_id.start_frame;

            const Eigen::Matrix3d &start_R_w_imu = state_window[frame_id].body_state.R;
            const Eigen::Vector3d &start_t_w_imu = state_window[frame_id].body_state.p;

            Eigen::Vector3d start_t_w_cam0 = start_t_w_imu + start_R_w_imu * t_imu_cam0;
            Eigen::Matrix3d start_R_w_cam0 = start_R_w_imu * R_imu_cam0;

            Eigen::MatrixXd svd_A(2 * feature_per_id.feature_per_frame.size(), 4);
            int svd_idx = 0;

            for (const auto &feature_per_frame : feature_per_id.feature_per_frame)
            {
                const Eigen::Matrix3d &tmp_R_w_imu = state_window[frame_id].body_state.R;
                const Eigen::Vector3d &tmp_t_w_imu = state_window[frame_id++].body_state.p;

                Eigen::Vector3d tmp_t_w_cam0 = tmp_t_w_imu + tmp_R_w_imu * t_imu_cam0;
                Eigen::Matrix3d tmp_R_w_cam0 = tmp_R_w_imu * R_imu_cam0;

                Eigen::Vector3d t_start_tmp = start_R_w_cam0.transpose() * (tmp_t_w_cam0 - start_t_w_cam0);
                Eigen::Matrix3d R_start_tmp = start_R_w_cam0.transpose() * tmp_R_w_cam0;

                Eigen::Matrix<double, 3, 4> T_tmp_start;
                T_tmp_start.leftCols<3>() = R_start_tmp.transpose();
                T_tmp_start.rightCols<1>() = -R_start_tmp.transpose() * t_start_tmp;

                Eigen::Vector3d point = feature_per_frame.point_0;

                svd_A.row(svd_idx++) = point[0] * T_tmp_start.row(2) - point[2] * T_tmp_start.row(0);
                svd_A.row(svd_idx++) = point[1] * T_tmp_start.row(2) - point[2] * T_tmp_start.row(1);
            }

            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();

            double depth = svd_V[2] / svd_V[3];
            feature_per_id.depth = depth > 0 ? depth : shared_pool->init_depth;
#endif
        }
    }
}


void FeatureManager::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
    Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Vector4d triangulated_point; // homogeneous coordinates
    triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}


double FeatureManager::reprojectError(
    const Eigen::Vector3d &uvi, const Eigen::Matrix3d &Ri, const Eigen::Vector3d &Pi, const Eigen::Matrix3d &rici, const Eigen::Vector3d &tici,
    const Eigen::Vector3d &uvj, const Eigen::Matrix3d &Rj, const Eigen::Vector3d &Pj, const Eigen::Matrix3d &ricj, const Eigen::Vector3d &ticj, 
    const double depth_i)
{
    Eigen::Vector3d pts_w = Ri * (rici * (depth_i * uvi) + tici) + Pi;
    Eigen::Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Eigen::Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}


void FeatureManager::detectOutliers(const FrameWindow &state_window)
{
    outlierIndex.clear();

    const Eigen::Matrix3d &R_imu_cam0 = shared_pool->extrinsics.R_imu_cam[0];
    const Eigen::Vector3d &t_imu_cam0 = shared_pool->extrinsics.t_imu_cam[0];
    const Eigen::Matrix3d &R_imu_cam1 = shared_pool->extrinsics.R_imu_cam[1];
    const Eigen::Vector3d &t_imu_cam1 = shared_pool->extrinsics.t_imu_cam[1];

    for (auto &feature_per_id : feature_list)
    {
        double err = 0;
        int err_count = 0;
        if (feature_per_id.feature_per_frame.size() < shared_pool->reliable_track_count)
            continue;
        
        int frame_i = feature_per_id.start_frame, frame_j = frame_i - 1;

        const Eigen::Matrix3d &Ri = state_window[frame_i].body_state.R;
        const Eigen::Vector3d &Pi = state_window[frame_i].body_state.p;

        Eigen::Vector3d pts_i = feature_per_id.feature_per_frame[0].point_0;
        double depth = feature_per_id.depth;
        for (auto &feature_per_frame : feature_per_id.feature_per_frame)
        {
            frame_j++;

            const Eigen::Matrix3d &Rj = state_window[frame_j].body_state.R;
            const Eigen::Vector3d &Pj = state_window[frame_j].body_state.p;

            if (frame_i != frame_j)
            {
                Eigen::Vector3d pts_j = feature_per_frame.point_0;
                double tmp_error = reprojectError(pts_i, Ri, Pi, R_imu_cam0, t_imu_cam0, 
                                                  pts_j, Rj, Pj, R_imu_cam0, t_imu_cam0,
                                                  depth);
                err += tmp_error;
                err_count++;
                // printf("tmp_error %f\n", shared_pool->focal_length / 1.5 * tmp_error);
            }
            /* need to rewrite projecton factor......... */
            if (shared_pool->is_use_stereo && feature_per_frame.is_stereo)
            {
                Eigen::Vector3d pts_j_right = feature_per_frame.point_1;
                
                if (frame_i != frame_j)
                {
                    double tmp_error = reprojectError(pts_i,        Ri, Pi, R_imu_cam0, t_imu_cam0, 
                                                      pts_j_right,  Rj, Pj, R_imu_cam1, t_imu_cam1,
                                                      depth);
                    err += tmp_error;
                    err_count++;
                    // printf("tmp_error %f\n", shared_pool->focal_length / 1.5 * tmp_error);
                }
                else
                {
                    double tmp_error = reprojectError(pts_i,        Ri, Pi, R_imu_cam0, t_imu_cam0, 
                                                      pts_j_right,  Rj, Pj, R_imu_cam1, t_imu_cam1,
                                                      depth);
                    err += tmp_error;
                    err_count++;
                    // printf("tmp_error %f\n", shared_pool->focal_length / 1.5 * tmp_error);
                }
            }
        }
        double avg_err = err / err_count;
        if (avg_err * shared_pool->focal_length > 3)
            outlierIndex.insert(feature_per_id.feature_id);
    }
    
    // std::cout << "feature number in outlierIndex: " << outlierIndex.size() << std::endl;
}


void FeatureManager::removeOutliers()
{
    FeatureList::iterator iter, iter_next;
    for (iter = feature_list.begin(); iter != feature_list.end(); iter = iter_next)
    {
        iter_next = next(iter);

        if (outlierIndex.find(iter->feature_id) != outlierIndex.end())
            feature_list.erase(iter);
    }
}


/** \brief clear all depth */
void FeatureManager::clearDepth()
{
    for (auto &feature_per_id : feature_list)
        feature_per_id.depth = -1;
}


void FeatureManager::getInvDepth(std::vector<double> &invdepth_vector)
{
    invdepth_vector.clear();
    for (auto &feature_per_id : feature_list)
    {
        if (feature_per_id.feature_per_frame.size() < shared_pool->reliable_track_count)
            continue;
            
        invdepth_vector.push_back(1.0 / feature_per_id.depth);
    }
}


void FeatureManager::setDepth(std::vector<double> &invdepth_vector)
{
    int feature_index = -1;
    for (auto &feature_per_id : feature_list)
    {
        if (feature_per_id.feature_per_frame.size() < shared_pool->reliable_track_count)
            continue;
        
        ++feature_index;

        feature_per_id.depth = 1.0 / invdepth_vector[feature_index];
        feature_per_id.solve_flag = feature_per_id.depth < 0 ? -1 : 1;
    }
    invdepth_vector.clear();
}


void FeatureManager::removeFailures()
{
    FeatureList::iterator iter, iter_next;
    for (iter = feature_list.begin(); iter != feature_list.end(); iter = iter_next)
    {
        iter_next = next(iter);

        if (iter->solve_flag == -1)
            feature_list.erase(iter);
    }
}


void FeatureManager::removeOldShiftDepth(const Eigen::Matrix3d &margR_w_cam0, const Eigen::Vector3d &margP_w_cam0,
                                         const Eigen::Matrix3d &R0_w_cam0, const Eigen::Vector3d &P0_w_cam0)
{
    FeatureList::iterator iter, iter_next;
    for (iter = feature_list.begin(); iter != feature_list.end(); iter = iter_next)
    {
        iter_next = next(iter);

        if (iter->start_frame == 0)
        {
            if (iter->feature_per_frame.size() < 2)
            {
                feature_list.erase(iter);
            }
            else
            {
                Eigen::Vector3d pts_i = iter->feature_per_frame[0].point_0 * iter->depth;
                Eigen::Vector3d w_pts_i = margR_w_cam0 * pts_i + margP_w_cam0;
                Eigen::Vector3d pts_j = R0_w_cam0.transpose() * (w_pts_i - P0_w_cam0);

                const double depth = pts_j(2);

                iter->depth = depth > 0 ? depth : shared_pool->init_depth;

                iter->feature_per_frame.erase(iter->feature_per_frame.begin());
            }
        }
        else
        {
            iter->start_frame--;
        }        
    }
}


void FeatureManager::removeOld()
{
    FeatureList::iterator iter, iter_next;
    for (iter = feature_list.begin(); iter != feature_list.end(); iter = iter_next)
    {
        iter_next = next(iter);

        if (iter->start_frame == 0)
        {
            if (iter->feature_per_frame.size() < 2)
            {
                feature_list.erase(iter);
            }
            else
            {
                iter->feature_per_frame.erase(iter->feature_per_frame.begin());
            }
        }
        else
        {
            iter->start_frame--;
        }
    }
}


void FeatureManager::removeSecondNew(int cur_frame_count)
{
    FeatureList::iterator iter, iter_next;
    for (auto iter = feature_list.begin(); iter != feature_list.end(); iter = iter_next)
    {
        iter_next = next(iter);

        if (iter->start_frame == cur_frame_count)
        {
            iter->start_frame--;
        }
        else
        {
            if (iter->endFrame() >= cur_frame_count - 1)
            {
                /* erase feature in second newest image frame */
                auto second_new_iter = iter->feature_per_frame.begin() + cur_frame_count - iter->start_frame - 1;
                iter->feature_per_frame.erase(second_new_iter);

                if (iter->feature_per_frame.size() == 0)
                    feature_list.erase(iter);
            }
        }
    }
}