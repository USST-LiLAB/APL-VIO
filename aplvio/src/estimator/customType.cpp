#include "estimator/paramPool.h"

#include "estimator/customType.h"


extern std::unique_ptr<ParamPool> shared_pool;


BodyState::BodyState()
    : t(0.0), p(Eigen::Vector3d::Zero()), q(Eigen::Quaterniond::Identity()),
      R(Eigen::Matrix3d::Identity()), v(Eigen::Vector3d::Zero()), 
      ba(Eigen::Vector3d::Zero()), bg(Eigen::Vector3d::Zero()),
      acc(Eigen::Vector3d::Zero()), gyr(Eigen::Vector3d::Zero())
{
    // nothing
}


void BodyState::clearState() 
{
    t = 0.0;
    p.setZero();
    q.setIdentity();
    R.setIdentity();
    v.setZero();
    ba.setZero();
    bg.setZero();
    acc.setZero();
    gyr.setZero();
}


BodyState & BodyState::operator=(const BodyState &body_state)
{
    this->t = body_state.t;
    this->p = body_state.p;
    this->R = body_state.R;
    this->q = body_state.q.normalized();

    if (shared_pool->is_use_imu)
    {
        this->v = body_state.v;
        this->ba = body_state.ba;
        this->bg = body_state.bg;
        this->acc = body_state.acc;
        this->gyr = body_state.gyr;
    }

    return *this;
}


void BodyState::swap(BodyState &body_state)
{
    std::swap(this->t, body_state.t);
    this->p.swap(body_state.p);
    this->R.swap(body_state.R);
    Eigen::Quaterniond tmp_q(this->q.normalized());
    this->q = body_state.q.normalized();
    body_state.q = tmp_q;

    if (shared_pool->is_use_imu)
    {
        this->v.swap(body_state.v);
        this->ba.swap(body_state.ba);
        this->bg.swap(body_state.bg);
        this->acc.swap(body_state.acc);
        this->gyr.swap(body_state.gyr);
    }
}



FrameState::FrameState()
    : pre_integration(nullptr)
{
    // nothing
}


void FrameState::clearState()
{
    body_state.clearState();
    pre_integration = nullptr;
}


void FrameState::swap(FrameState &frame_state)
{
    body_state.swap(frame_state.body_state);
    if (shared_pool->is_use_imu)
        std::swap(pre_integration, frame_state.pre_integration);
}



FeaturePerId::FeaturePerId(const int _feature_id, const int _start_frame) 
    : feature_id(_feature_id), start_frame(_start_frame), depth(-1.0), solve_flag(0)
{
    // nothing
}



FeaturePerFrame::FeaturePerFrame(const std::vector<FeaturePoint> &feature_points, double _td) : td(_td)
{
    if (!feature_points.empty())
    {
        point_0.x() = feature_points[0].second[0];
        point_0.y() = feature_points[0].second[1];
        point_0.z() = feature_points[0].second[2];
        uv_0.x() = feature_points[0].second[3];
        uv_0.y() = feature_points[0].second[4];
        vel_0.x() = feature_points[0].second[5];
        vel_0.y() = feature_points[0].second[6];
        is_stereo = false;

        if (feature_points.size() == 2)
        {
            point_1.x() = feature_points[1].second[0];
            point_1.y() = feature_points[1].second[1];
            point_1.z() = feature_points[1].second[2];
            uv_1.x() = feature_points[1].second[3];
            uv_1.y() = feature_points[1].second[4];
            vel_1.x() = feature_points[1].second[5];
            vel_1.y() = feature_points[1].second[6];
            is_stereo = true;
        }
    }
}

LineFeaturePerFrame::LineFeaturePerFrame() : is_stereo(false), is_tracked(false)
{
    // pass
}


LineFeaturePerFrame::LineFeaturePerFrame(const std::vector<FeatureLine> &feature_lines)
{
    if (!feature_lines.empty())
    {
        line_0 = feature_lines[0].second;
        is_stereo = false;

        if (feature_lines.size() == 2)
        {
            line_1 = feature_lines[1].second;
            is_stereo = true;
            // std::cout << "line_0 = " << line_0.transpose() << std::endl;
            // std::cout << "line_1 = " << line_1.transpose() << std::endl;
        }
        is_tracked = true;
    }
}


LineFeaturePerId::LineFeaturePerId(const int _line_id, const int _start_frame) : 
    line_id(_line_id), start_frame(_start_frame), track_count(0), 
    is_triangulate(false), is_published(false), opt_num(0), solve_flag(0)
{
    // nothing
}


SFMFeature::SFMFeature(const FeaturePerId &feature_per_id) : 
    is_triangulate(false), feature_id(feature_per_id.feature_id)
{
    int frame_i = feature_per_id.start_frame;
    for (const auto &feature_per_frame : feature_per_id.feature_per_frame)
        covisible_frames.emplace_back(frame_i++, feature_per_frame.point_0.head<2>());
}


ImageFrame::ImageFrame() : 
    t(0.0), R_w_imu(Eigen::Matrix3d::Identity()), t_w_imu(Eigen::Vector3d::Zero()), 
    pre_integration(nullptr), is_keyframe(false)
{
    // nothing
}
