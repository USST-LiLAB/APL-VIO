#include "odom_utils/utils.h"
#include "factor/integrationBase.h"
#include "estimator/paramPool.h"

#include "inertial/imuProcessor.h"


extern std::unique_ptr<ParamPool> shared_pool;

void ImuProcessor::clearState()
{
    std::lock_guard<std::mutex> imu_buffer_guard(imu_buffer_mutex);
    imu_buffer = std::queue<ImuData>();
}


void ImuProcessor::setParameters()
{
    // nothing
}


/** \brief Imu data callback. */
void ImuProcessor::imu_callback(const sensor_msgs::ImuConstPtr &pMsg)
{
    imu_data.t = pMsg->header.stamp.toSec();
    imu_data.acc.x() = pMsg->linear_acceleration.x;
    imu_data.acc.y() = pMsg->linear_acceleration.y;
    imu_data.acc.z() = pMsg->linear_acceleration.z;
    imu_data.gyr.x() = pMsg->angular_velocity.x;
    imu_data.gyr.y() = pMsg->angular_velocity.y;
    imu_data.gyr.z() = pMsg->angular_velocity.z;

    imu_buffer_mutex.lock();
    imu_buffer.push(imu_data);
    imu_buffer_mutex.unlock();

    if (shared_pool->solver_flag == NONLINEAR)
    {
        shared_pool->propagate_mutex.try_lock();
        double dt = imu_data.t - shared_pool->latest_state.t;
        fastPredict(shared_pool->latest_state, dt, imu_data);
        RosMaster::pubOdometryFast(imu_data.t, shared_pool->latest_state.p, shared_pool->latest_state.q, shared_pool->latest_state.v);
        shared_pool->propagate_mutex.unlock();
    }
}


/** \brief Check imu data in buffer is enough or not.
 * @param t cut-off time.
 * @return true if imu data is enough, false is not.
 */
bool ImuProcessor::checkImu(double t) const
{
    std::lock_guard<std::mutex> imu_buffer_guard(imu_buffer_mutex);
    
    if (imu_buffer.empty())
    {
        std::cout << "not receive imu!" << std::endl;
        return false;
    }
    if (imu_buffer.back().t < t)
    {
        std::cout << "wait for imu ..." << std::endl;
        return false;
    }

    return true;
}


/** \brief Get imu data between two frames.
 * @param t0 Input previous frame time.
 * @param t1 Input current frame time.
 * @param imu_vector Output vector of imu data between two frames, containing the first imu data after current frame time.
 * @return true if get interframe imu data, false if not.
 */
bool ImuProcessor::getInterframeImuData(const double t0, const double t1, std::vector<ImuData> &imu_vector)
{
    // ROS_ASSERT(t0 < t1);

    while (!checkImu(t1))
    {
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
    
    imu_buffer_mutex.lock();

    
    while (imu_buffer.front().t <= t0)
    {
        imu_buffer.pop();
    }
    while (imu_buffer.front().t < t1)
    {
        imu_vector.push_back(imu_buffer.front());
        imu_buffer.pop();
    }

    // ROS_ASSERT(!imu_buffer.empty());
    
    if (imu_vector.empty())
    {
        imu_vector.push_back(imu_buffer.front());
        imu_vector.back().t = t1;
    }
    else if (imu_buffer.front().t == t1)
    {
        imu_vector.push_back(imu_buffer.front());
    }
    else
    {
        ImuData last_imu_data;
        last_imu_data.t = t1;
        float prop = (t1 - imu_vector.back().t) / (imu_buffer.front().t - imu_vector.back().t);
        last_imu_data.acc = imu_vector.back().acc + (imu_buffer.front().acc - imu_vector.back().acc) * prop;
        last_imu_data.gyr = imu_vector.back().gyr + (imu_buffer.front().gyr - imu_vector.back().gyr) * prop;
        imu_vector.push_back(last_imu_data);
    }

    imu_buffer_mutex.unlock();

    return true;
}


/** \brief Get the initial imu attitude which is rotation from init imu attitude to G.
 * @param imu_vector Input measurement still imu.
 * @param init_state Output initial state.
 * @return Rotation align initial imu attitude to G.
 */
bool ImuProcessor::initImuState(const std::vector<ImuData> &imu_vector, BodyState &init_state)
{
    init_state.clearState();

    Eigen::Vector3d avg_acc(0, 0, 0);
    Eigen::Vector3d avg_gyr(0, 0, 0);
    int n = static_cast<int>(imu_vector.size());

    for (size_t i = 0; i != imu_vector.size(); i++)
    {
        avg_acc += imu_vector[i].acc;
        avg_gyr += imu_vector[i].gyr;
    }
    avg_acc /= n;
    avg_gyr /= n;

    printf("averge acc: x = %8f, y = %8f, z = %8f\n", avg_acc.x(), avg_acc.y(), avg_acc.z());
    printf("averge gyr: x = %8f, y = %8f, z = %8f\n", avg_gyr.x(), avg_gyr.y(), avg_gyr.z());

    if (abs(avg_acc.norm() - G.norm()) > 0.5 * G.norm())
    {
        ROS_WARN_STREAM("initialize gravity failed!");
        return false;
    }

    Eigen::Matrix3d R0 = alignGravity(avg_acc);

    if (shared_pool->is_use_stereo)
    {
        std::cout << std::fixed << std::setprecision(6) << "estimated gravity = " << avg_acc.transpose() << "\n";
        std::cout << std::fixed << std::setprecision(6) << "initial R0 =\n" << R0 << "\n";
        std::cout << std::fixed << std::setprecision(6) << "align gravity with " << (R0 * avg_acc).transpose() << std::endl;
        
    }

    init_state.t = imu_vector.back().t;
    init_state.acc = imu_vector.back().acc;
    init_state.gyr = imu_vector.back().gyr;
    
    init_state.R = R0;
    init_state.q = Eigen::Quaterniond(R0).normalized();

    ROS_INFO_STREAM("initialize gravity successfully!");
    return true;
}


/** \brief Get rotation required for gravity alignment.
 * @param g input measurement of gravity.
 * @return rotation align g to G.
 */
Eigen::Matrix3d ImuProcessor::alignGravity(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R_G_g; // rotation from estimated g to ground truth G
    Eigen::Vector3d normalized_g = g;
    Eigen::Vector3d normalized_G = Eigen::Vector3d(0.0, 0.0, 1.0);
    R_G_g = Eigen::Quaterniond::FromTwoVectors(normalized_g, normalized_G).toRotationMatrix();

    double yaw = math_utils::R2ypr(R_G_g).x();

    R_G_g = math_utils::ypr2R(Eigen::Vector3d(-yaw, 0, 0)) * R_G_g;

    return R_G_g;
}


/** \brief Integrate imu data between previous frame time to current frame time.
 * @param t0 previous frame time.
 * @param t1 current frame time.
 * @param imu_vector imu data between previous frame time to current frame time.
 * @param state Input & Output: input imu state at previous frame and output imu state at current frame.
 */
void ImuProcessor::interframeIntegration(FrameState &state, const double t0, const double t1, const std::vector<ImuData> &imu_vector)
{
    state.body_state.t = t0; // image_time + time_delay
    
    for (size_t i = 0; i != imu_vector.size(); i++)
    {
        double dt = imu_vector[i].t - state.body_state.t;
        state.pre_integration->push_back(dt, imu_vector[i].acc, imu_vector[i].gyr);

        fastPredict(state.body_state, dt, imu_vector[i]);
    }
}


/** \brief Integrate imu data between previous frame time to current frame time.
 * @param cur_state Input & Output imu state.
 * @param dt dt between previous imu timestamp and current imu timestamp.
 * @param cur_data imu state at current imu timestamp.
 */
void ImuProcessor::fastPredict(BodyState &cur_state, const double dt, const ImuData &cur_data)
{
    BodyState pre_state = cur_state;
    cur_state.t = cur_data.t;
    cur_state.acc = cur_data.acc;
    cur_state.gyr = cur_data.gyr;

    // gamma
    Eigen::Vector3d unbias_gyr = 0.5 * (pre_state.gyr + cur_state.gyr) - pre_state.bg;

    Eigen::Quaterniond gamma = math_utils::theta2dq(unbias_gyr * dt);
    cur_state.q = (pre_state.q * gamma).normalized();
    cur_state.R = cur_state.q.toRotationMatrix();

    Eigen::Vector3d pre_unbias_acc = pre_state.R * (pre_state.acc - pre_state.ba) - G;
    Eigen::Vector3d cur_unbias_acc = cur_state.R * (cur_state.acc - cur_state.ba) - G;
    Eigen::Vector3d avg_unbias_acc = 0.5 * (pre_unbias_acc + cur_unbias_acc);

    // beta
    Eigen::Vector3d beta = avg_unbias_acc * dt;
    cur_state.v = pre_state.v + beta;

    // alpha
    Eigen::Vector3d alpha = 0.5 * beta * dt;
    cur_state.p = pre_state.p + pre_state.v * dt + alpha;
}
