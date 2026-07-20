#include "odom_utils/utils.h"
#include "estimator/paramPool.h"

#include "visual/lineFeatureManager.h"


#define USE_NORMAL_VECTOR_ERROR 0

extern std::unique_ptr<ParamPool> shared_pool;

LineFeatureManager::LineFeatureManager() : track_num(0), new_track_num(0), long_track_num(0)
{
    // nothing
}


void LineFeatureManager::clearState()
{
    linefeature_list.clear();
    outlierIndex.clear();

    track_num = 0;
    new_track_num = 0;
    long_track_num = 0;
}

void LineFeatureManager::setParameters()
{
    // nothing
}


void LineFeatureManager::insertLineFeature(const int cur_frame_count, const FeatureLines &feature_lines)
{
    track_num = 0;
    new_track_num = 0;
    long_track_num = 0;

    FeatureLines new_feature_lines; // new feature lines

    for (const std::pair<int, std::vector<FeatureLine>> &frame_linefeature : feature_lines)
    {
        int line_id = frame_linefeature.first;
        
        auto iter = find_if(linefeature_list.begin(), linefeature_list.end(), 
            [line_id](const LineFeaturePerId &linefeature_per_id) { return linefeature_per_id.line_id == line_id; });

        if (iter != linefeature_list.end())
        {
            iter->linefeature_per_frame.emplace_back(frame_linefeature.second);
            iter->track_count++;
            long_track_num += iter->linefeature_per_frame.size() >= shared_pool->reliable_line_track_count + 2 ? 1 : 0;
            track_num++;
        }
        else
        {
            new_feature_lines.insert(frame_linefeature); // ready to match
        }
    }

    utils::TicToc matchline_tictoc;
    for (const std::pair<int, std::vector<FeatureLine>> &frame_linefeature : new_feature_lines)
    {
        LineFeaturePerId *ptr_linefeature_per_id = nullptr;
#if 0
        ptr_linefeature_per_id = matchLine(frame_linefeature);
#endif
        if (ptr_linefeature_per_id != nullptr) // match successfully
        {
            ptr_linefeature_per_id->line_id = frame_linefeature.first;

            for (int i = ptr_linefeature_per_id->endFrame() + 1; i < cur_frame_count; i++)
                ptr_linefeature_per_id->linefeature_per_frame.emplace_back(); // insert empty frames
            
            ptr_linefeature_per_id->linefeature_per_frame.emplace_back(frame_linefeature.second);
            ptr_linefeature_per_id->track_count++;
            long_track_num += ptr_linefeature_per_id->linefeature_per_frame.size() >= shared_pool->reliable_line_track_count + 2 ? 1 : 0;
            track_num++;
        }
        else // new line feature
        {
            linefeature_list.emplace_back(frame_linefeature.first, cur_frame_count);
            linefeature_list.back().linefeature_per_frame.emplace_back(frame_linefeature.second);
            linefeature_list.back().track_count++;
            new_track_num++;
        }
    }

    // std::cout << "match line costs: " << matchline_tictoc.toc() << std::endl;
    // std::cout << "linefeature_list.size = " << linefeature_list.size() << std::endl;
}

LineFeaturePerId *LineFeatureManager::matchLine(const std::pair<int, std::vector<FeatureLine>> &frame_linefeature)
{
    LineFeaturePerId *matched_ptr = nullptr;

    const Eigen::Vector4d &new_line = frame_linefeature.second[0].second;

    std::vector<LineFeaturePerId *> linefeature_vec;

    for (auto &linefeature_per_id : linefeature_list)
    {
        if (linefeature_per_id.is_triangulate == false)
            continue;
        
        if (linefeature_per_id.endFrame() >= WINDOW_SIZE - 1)
            continue;
        
        const Eigen::Vector4d &old_line = linefeature_per_id.linefeature_per_frame.back().line_0;

        Eigen::Vector2d v_old = (old_line.tail(2) - old_line.head(2)).normalized();
        Eigen::Vector2d v_new = (new_line.tail(2) - new_line.head(2)).normalized();

        double angle_old = atan2(v_old.y(), v_old.x());
        double angle_new = atan2(v_new.y(), v_new.x());

        double angle_diff = math_utils::wrap2pi(angle_new - angle_old);

        if (fabs(angle_diff) < M_PI / 30)
        {
            linefeature_vec.push_back(&linefeature_per_id);
        }
    }
    

    double dist1, dist2, min_dist = 100;
    for (const auto &ptr_linefeature_per_id : linefeature_vec)
    {
        const Eigen::Vector4d &old_line = ptr_linefeature_per_id->linefeature_per_frame.back().line_0;

        dist1 = geometry_utils::Point2Line(new_line, old_line.head(2));
        dist2 = geometry_utils::Point2Line(new_line, old_line.tail(2));

        double sPoint_dist = (new_line.head(2) - old_line.head(2)).norm();
        double ePoint_dist = (new_line.tail(2) - old_line.tail(2)).norm();
        if (dist1 + dist2 < min_dist && (sPoint_dist < 0.5 || ePoint_dist < 0.5))
        {
            min_dist = dist1 + dist2;
            matched_ptr = ptr_linefeature_per_id;
        }
    }

    if (min_dist < 0.1)
    {
        // std::cout << "min_dist = " << min_dist << std::endl;
        // std::cout << "endframe = " << matched_ptr->endFrame() << std::endl;
        return matched_ptr;
    }

    return nullptr;
}


bool LineFeatureManager::isKeyframe(const int cur_frame_count) const
{
    // return (shared_pool->latest_state.v.norm() > 0.02 && fabs(shared_pool->latest_state.acc.norm() - G.norm()) > 0.1) 
    //     && (track_num < 10 || long_track_num < 10 || new_track_num > 0.5 * track_num);
    // return (shared_pool->latest_state.v.norm() > 0.02 && fabs(shared_pool->latest_state.acc.norm() - G.norm()) > 0.1) && track_num < 3;
    return false;
}


void LineFeatureManager::triangulate(const FrameWindow &state_window)
{
    /* transform from left camera to imu */
    const Eigen::Matrix3d &R_imu_cam0 = shared_pool->extrinsics.R_imu_cam[0];
    const Eigen::Vector3d &t_imu_cam0 = shared_pool->extrinsics.t_imu_cam[0];
    /* transform from right camera to imu */
    const Eigen::Matrix3d &R_imu_cam1 = shared_pool->extrinsics.R_imu_cam[1];
    const Eigen::Vector3d &t_imu_cam1 = shared_pool->extrinsics.t_imu_cam[1];

    for (auto &linefeature_per_id : linefeature_list)
    {
        if (linefeature_per_id.is_triangulate)
            continue;
        
        if (linefeature_per_id.linefeature_per_frame.size() > shared_pool->reliable_line_track_count)
        {
            if (linefeature_per_id.linefeature_per_frame[0].is_tracked == false)
                continue;

            int frame_id = linefeature_per_id.start_frame;

            /* imu pose in world frame in previous frame time */
            const Eigen::Matrix3d &start_R_w_imu = state_window[frame_id].body_state.R;
            const Eigen::Vector3d &start_t_w_imu = state_window[frame_id].body_state.p;

            /* previous frame line plane */
            const Eigen::Vector3d start_t_w_cam0 = start_t_w_imu + start_R_w_imu * t_imu_cam0;
            const Eigen::Matrix3d start_R_w_cam0 = start_R_w_imu * R_imu_cam0;

            Eigen::Vector3d start_sPoint_0, start_ePoint_0;
            start_sPoint_0 << linefeature_per_id.linefeature_per_frame[0].line_0.head(2), 1.0;
            start_ePoint_0 << linefeature_per_id.linefeature_per_frame[0].line_0.tail(2), 1.0;

            const Eigen::Vector4d &start_plane = geometry_utils::ptptpt2plane(start_sPoint_0, start_ePoint_0, Eigen::Vector3d(0, 0, 0));
            const Eigen::Vector3d &start_normal_vec = start_plane.head(3).normalized();


            double plane_angle = 1.0;
            Eigen::Vector4d best_plane;
            /* find the best plane */
            for (size_t j = 1; j != linefeature_per_id.linefeature_per_frame.size(); j++)
            {
                if (linefeature_per_id.linefeature_per_frame[j].is_tracked == false)
                    continue;

                /* imu pose in world frame in current frame time */
                const Eigen::Matrix3d &cur_R_w_imu = state_window[frame_id + j].body_state.R;
                const Eigen::Vector3d &cur_t_w_imu = state_window[frame_id + j].body_state.p;

                /* current frame line plane */
                const Eigen::Vector3d cur_t_w_cam0 = cur_t_w_imu + cur_R_w_imu * t_imu_cam0;
                const Eigen::Matrix3d cur_R_w_cam0 = cur_R_w_imu * R_imu_cam0;
                const Eigen::Vector3d t_pre_cur = start_R_w_cam0.transpose() * (cur_t_w_cam0 - start_t_w_cam0);
                const Eigen::Matrix3d R_pre_cur = start_R_w_cam0.transpose() * cur_R_w_cam0;

                Eigen::Vector3d tmp_cur_sPoint_0, tmp_cur_ePoint_0;
                tmp_cur_sPoint_0 << linefeature_per_id.linefeature_per_frame[j].line_0.head(2), 1;
                tmp_cur_ePoint_0 << linefeature_per_id.linefeature_per_frame[j].line_0.tail(2), 1;

                const Eigen::Vector3d cur_sPoint_0 = R_pre_cur * tmp_cur_sPoint_0 + t_pre_cur;
                const Eigen::Vector3d cur_ePoint_0 = R_pre_cur * tmp_cur_ePoint_0 + t_pre_cur;
                const Eigen::Vector4d cur_plane = geometry_utils::ptptpt2plane(cur_sPoint_0, cur_ePoint_0, t_pre_cur);
                const Eigen::Vector3d cur_normal_vec = cur_plane.head(3).normalized();


                const double cur_plane_angle = start_normal_vec.dot(cur_normal_vec);
                if (cur_plane_angle < plane_angle)
                {
                    plane_angle = cur_plane_angle;
                    best_plane = cur_plane;
                }
            }

            // std::cout << "plane_angle = " << plane_angle << std::endl;
            if (abs(plane_angle) > cos(M_PI / 48.0)) // two planes cannot be parallel  # 0.9958 0.998
                continue;

            /* triangulate line via intersection of planes */
            const Eigen::Matrix<double, 6, 1> &plucker_c = geometry_utils::planeplane2plucker(start_plane, best_plane);  // plucker in camera frame
            linefeature_per_id.plucker_w = geometry_utils::transformPlucker(plucker_c, start_R_w_cam0, start_t_w_cam0); // plucker in world frame
            linefeature_per_id.is_triangulate = true;
        }
        else if (shared_pool->is_use_stereo && linefeature_per_id.linefeature_per_frame[0].is_stereo &&
            linefeature_per_id.linefeature_per_frame.size() >= shared_pool->reliable_line_track_count / 2)
        {
            // continue; // stereo line is NOT used for triangulate, but for optimization
            int frame_id = linefeature_per_id.start_frame;

            /* left image line plane */
            const Eigen::Matrix3d &R_w_imu = state_window[frame_id].body_state.R;
            const Eigen::Vector3d &t_w_imu = state_window[frame_id].body_state.p;
            const Eigen::Vector3d t_w_cam0 = t_w_imu + R_w_imu * t_imu_cam0;
            const Eigen::Matrix3d R_w_cam0 = R_w_imu * R_imu_cam0;

            Eigen::Vector3d sPoint_0, ePoint_0;
            sPoint_0 << linefeature_per_id.linefeature_per_frame[0].line_0.head(2), 1.0;
            ePoint_0 << linefeature_per_id.linefeature_per_frame[0].line_0.tail(2), 1.0;

            const Eigen::Vector4d plane_0 = geometry_utils::ptptpt2plane(sPoint_0, ePoint_0, Eigen::Vector3d(0, 0, 0));
            const Eigen::Vector3d normal_vec_0 = plane_0.head(3).normalized();

            /* right image line plane */
            const Eigen::Vector3d t_w_cam1 = t_w_imu + R_w_imu * t_imu_cam1;
            const Eigen::Matrix3d R_w_cam1 = R_w_imu * R_imu_cam1;
            const Eigen::Vector3d t_cam0_cam1 = R_w_cam0.transpose() * (t_w_cam1 - t_w_cam0);
            const Eigen::Matrix3d R_cam0_cam1 = R_w_cam0.transpose() * R_w_cam1;

            Eigen::Vector3d tmp_sPoint_1, tmp_ePoint_1;
            tmp_sPoint_1 << linefeature_per_id.linefeature_per_frame[0].line_1.head(2), 1.0;
            tmp_ePoint_1 << linefeature_per_id.linefeature_per_frame[0].line_1.tail(2), 1.0;

            const Eigen::Vector3d sPoint_1 = R_cam0_cam1 * tmp_sPoint_1 + t_cam0_cam1;
            const Eigen::Vector3d ePoint_1 = R_cam0_cam1 * tmp_ePoint_1 + t_cam0_cam1;

            const Eigen::Vector4d plane_1 = geometry_utils::ptptpt2plane(sPoint_1, ePoint_1, t_cam0_cam1);
            const Eigen::Vector3d normal_vec_1 = plane_1.head(3).normalized();

            const double plane_angle = normal_vec_0.dot(normal_vec_1);
            // std::cout << "plane_angle = " << plane_angle << std::endl;;
            if (abs(plane_angle) > cos(M_PI / 60.0)) // two planes cannot be parallel # 0.995 0.999
                continue;

            /* triangulate line via intersection of planes */
            const Eigen::Matrix<double, 6, 1> &plucker_c = geometry_utils::planeplane2plucker(plane_0, plane_1);  // plucker in camera frame
            linefeature_per_id.plucker_w = geometry_utils::transformPlucker(plucker_c, R_w_cam0, t_w_cam0); // plucker in world frame
            linefeature_per_id.is_triangulate = true;
        }
    }
}


double LineFeatureManager::reprojectError(const Eigen::Vector4d &obs_line, const Eigen::Matrix<double, 6, 1> &plucker_w,
    const Eigen::Matrix3d &R_w_cam0, const Eigen::Vector3d &t_w_cam0)
{
    Eigen::Vector3d p1, p2;
    p1 << obs_line.head(2), 1.0;
    p2 << obs_line.tail(2), 1.0;

    const Eigen::Matrix<double, 6, 1> plucker_c = geometry_utils::invTransformPlucker(plucker_w, R_w_cam0, t_w_cam0);

    double error = 0;
    
#if USE_NORMAL_VECTOR_ERROR
    Eigen::Vector3d nc1 = plucker_c.head(3).normalized();
    Eigen::Vector3d nc2 = p1.cross(p2).normalized();

    error = fabs(nc1.dot(nc2));
#else
    Eigen::Vector3d nc = plucker_c.head(3).normalized();
    nc /= nc.head(2).norm();

    /* distance from endpoint to line */
    error += fabs(nc.dot(p1));
    error += fabs(nc.dot(p2));
    error /= 2;
#endif
    return error;
}


void LineFeatureManager::detectOutliers(const FrameWindow &state_window)
{
    utils::TicToc outlier_tictoc;
    outlierIndex.clear();

    const Eigen::Matrix3d &R_imu_cam0 = shared_pool->extrinsics.R_imu_cam[0];
    const Eigen::Vector3d &t_imu_cam0 = shared_pool->extrinsics.t_imu_cam[0];


    for (const auto &linefeature_per_id : linefeature_list)
    {
        if (linefeature_per_id.is_triangulate == false)
            continue;

        int frame_id = linefeature_per_id.start_frame;


        const Eigen::Matrix<double, 6, 1> &plucker_w = linefeature_per_id.plucker_w;

        const Eigen::Vector3d &t_w_imu = state_window[frame_id].body_state.p;
        const Eigen::Matrix3d &R_w_imu = state_window[frame_id].body_state.R;
        const Eigen::Vector3d t_w_cam0 = t_w_imu + R_w_imu * t_imu_cam0;
        const Eigen::Matrix3d R_w_cam0 = R_w_imu * R_imu_cam0;

        /* plucker in camera frame */
        Eigen::Matrix<double, 6, 1> plucker_c = geometry_utils::invTransformPlucker(plucker_w, R_w_cam0, t_w_cam0);


        Eigen::Vector3d sPoint3d, ePoint3d;
        planeTrimLine(plucker_c, linefeature_per_id.linefeature_per_frame[0].line_0, sPoint3d, ePoint3d);

        /* line should not behind the camera optical center */
        if(sPoint3d.z() < 0 || ePoint3d.z() < 0)
        {
            outlierIndex.insert(linefeature_per_id.line_id);
            continue;
        }

        /* line should not be to long */
        if((sPoint3d - ePoint3d).norm() > 10)
        {
            outlierIndex.insert(linefeature_per_id.line_id);
            continue;
        }

        /* line should not be normal to the camera plane */
        // if(fabs(sPoint3d.z() - ePoint3d.z()) / (sPoint3d - ePoint3d).norm() > 0.95)
        // {
        //     outlierIndex.insert(linefeature_per_id.line_id);
        //     continue;
        // }

        /* line should not be normal to the camera plane */
        if (fabs(plucker_c.tail(3).normalized().transpose() * sPoint3d.normalized()) > 0.95 ||
            fabs(plucker_c.tail(3).normalized().transpose() * ePoint3d.normalized()) > 0.95)
        {
            outlierIndex.insert(linefeature_per_id.line_id);
            continue;
        }

        double max_error = 0;

        for (const auto &linefeature_per_frame : linefeature_per_id.linefeature_per_frame)
        {
            if (linefeature_per_frame.is_tracked == false)
                continue;

            const Eigen::Vector3d &t_w_imu = state_window[frame_id].body_state.p;
            const Eigen::Matrix3d &R_w_imu = state_window[frame_id].body_state.R;
            const Eigen::Vector3d t_w_cam0 = t_w_imu + R_w_imu * t_imu_cam0;
            const Eigen::Matrix3d R_w_cam0 = R_w_imu * R_imu_cam0;

            double error = reprojectError(linefeature_per_frame.line_0, plucker_w, R_w_cam0, t_w_cam0);

            if(max_error < error)
                max_error = error;

            frame_id++;
        }

#if USE_NORMAL_VECTOR_ERROR
        if (max_error < 0.9999)
            outlierIndex.insert(linefeature_per_id.line_id);
#else
        if (max_error > 0.01)
            outlierIndex.insert(linefeature_per_id.line_id);
#endif
    }
    // std::cout << "detect outliers costs: " << outlier_tictoc.toc() << " ms" <<std::endl;
}


void LineFeatureManager::removeOutliers()
{
    LineFeatureList::iterator iter, iter_next;
    for (iter = linefeature_list.begin(); iter != linefeature_list.end(); iter = iter_next)
    {
        iter_next = next(iter);

        if (outlierIndex.find(iter->line_id) != outlierIndex.end())
            linefeature_list.erase(iter);
    }
}


void LineFeatureManager::removeFailures()
{
    LineFeatureList::iterator iter, iter_next;
    for (iter = linefeature_list.begin(); iter != linefeature_list.end(); iter = iter_next)
    {
        iter_next = next(iter);

        if (iter->solve_flag == -1)
            linefeature_list.erase(iter);
    }
}


void LineFeatureManager::planeTrimLine(const Eigen::Matrix<double, 6, 1> &plucker, 
    const Eigen::Vector4d &line2d, Eigen::Vector3d &sPoint3d, Eigen::Vector3d &ePoint3d)
{
    Eigen::Vector3d nc, vc;

    nc = plucker.head(3);
    vc = plucker.tail(3);

    Eigen::Matrix4d Lc;
    Lc << math_utils::skewSymmetric(nc), vc, -vc.transpose(), 0;

    Eigen::Vector3d sPoint, ePoint;
    sPoint << line2d.head(2), 1.0;
    ePoint << line2d.tail(2), 1.0;

    Eigen::Vector2d ln = (sPoint.cross(ePoint)).head(2);
    ln = ln / ln.norm();

    const Eigen::Vector3d sPoint_vertical = Eigen::Vector3d(sPoint(0) + ln(0), sPoint(1) + ln(1), 1.0);
    const Eigen::Vector3d ePoint_vertical = Eigen::Vector3d(ePoint(0) + ln(0), ePoint(1) + ln(1), 1.0);

    const Eigen::Vector4d cut_plane1 = geometry_utils::ptptpt2plane(Eigen::Vector3d(0, 0, 0), sPoint, sPoint_vertical);
    const Eigen::Vector4d cut_plane2 = geometry_utils::ptptpt2plane(Eigen::Vector3d(0, 0, 0), ePoint, ePoint_vertical);

    /* use plane to cut line */
    Eigen::Vector4d e1 = Lc * cut_plane1;
    Eigen::Vector4d e2 = Lc * cut_plane2;

    e1 = e1 / e1(3);
    e2 = e2 / e2(3);

    sPoint3d = e1.head(3);
    ePoint3d = e2.head(3);
}


void LineFeatureManager::getOrthoLine(std::vector<Eigen::Vector4d> &ortholine_vector, const FrameWindow &state_window)
{
    ortholine_vector.clear();

    for (auto &linefeature_per_id : linefeature_list)
    {
        if (linefeature_per_id.linefeature_per_frame.size() < shared_pool->reliable_line_track_count)
            continue;

        if (!linefeature_per_id.is_triangulate)
            continue;

        /* get orthonormal representation from plucker coordinate */
        ortholine_vector.push_back(geometry_utils::plucker2ortho(linefeature_per_id.plucker_w));
    }
}


void LineFeatureManager::setOrthoLine(std::vector<Eigen::Vector4d> &ortholine_vector, const FrameWindow &state_window)
{
    int linefeature_index = -1;
    for (auto &linefeature_per_id : linefeature_list)
    {
        if (linefeature_per_id.linefeature_per_frame.size() < shared_pool->reliable_line_track_count)
            continue;

        if (!linefeature_per_id.is_triangulate)
            continue;

        ++linefeature_index;

        /* get plucker coordinate from orthonormal representation */
        Eigen::Matrix<double, 6, 1> plucker_w = geometry_utils::ortho2plucker(ortholine_vector[linefeature_index]);

        double cos_theta = plucker_w.tail(3).normalized().dot(linefeature_per_id.plucker_w.tail(3).normalized());

        linefeature_per_id.opt_num++;
        linefeature_per_id.solve_flag = fabs(cos_theta) < 0.999 ? -1 : 1;

        linefeature_per_id.plucker_w = plucker_w;
    }
    ortholine_vector.clear();
}


void LineFeatureManager::removeOld()
{
    LineFeatureList::iterator iter, iter_next;
    for (iter = linefeature_list.begin(); iter != linefeature_list.end(); iter = iter_next)
    {
        iter_next = next(iter);

        if (iter->start_frame == 0)
        {
            if (iter->linefeature_per_frame.size() < 2)
            {
                linefeature_list.erase(iter);
            }
            else
            {
                iter->linefeature_per_frame.erase(iter->linefeature_per_frame.begin());
            }
        }
        else
        {
            iter->start_frame--;
        }
    }
}


void LineFeatureManager::removeSecondNew(int cur_frame_count)
{
    LineFeatureList::iterator iter, iter_next;
    for (auto iter = linefeature_list.begin(); iter != linefeature_list.end(); iter = iter_next)
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
                auto second_new_iter = iter->linefeature_per_frame.begin() + cur_frame_count - iter->start_frame - 1;
                iter->linefeature_per_frame.erase(second_new_iter);

                if (iter->linefeature_per_frame.size() == 0)
                    linefeature_list.erase(iter);
            }
        }
    }
}