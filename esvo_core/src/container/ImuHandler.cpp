#include <esvo_core/container/ImuHandler.h>
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <yaml-cpp/yaml.h>
#pragma diagnostic pop

double stdVec(const std::vector<double> &v) {
    double sum  = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();

    std::vector<double> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(), std::bind2nd(std::minus<double>(), mean));
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    double stdev  = std::sqrt(sq_sum / v.size());

    return stdev;
}

namespace esvo_core {

const std::map<IMUTemporalStatus, std::string> imu_temporal_status_names_{
    {IMUTemporalStatus::kStationary, "Stationary"},
    {IMUTemporalStatus::kMoving, "Moving"},
    {IMUTemporalStatus::kUnkown, "Unknown"}};

ImuHandler::ImuHandler(const ImuCalibration    &imu_calib,
                       const ImuInitialization &imu_init,
                       const IMUHandlerOptions &options)
    : options_(options),
      imu_calib_(imu_calib),
      imu_init_(imu_init),
      acc_bias_(imu_init.acc_bias),
      omega_bias_(imu_init.omega_bias) {}

ImuHandler::~ImuHandler() {}
/// Get IMU measurements up to  for the exact borders.
/// Note that you have to provide the camera timestamp.
/// Internally, given the calibration it corrects the timestamps for delays.
/// Also not that this adds all the measurements since the last call, which
/// should correspond to the time of the last added frame
bool ImuHandler::getMeasurementsContainingEdges(const double     frame_timestamp, // seconds
                                                ImuMeasurements &extracted_measurements,
                                                const bool       remove_measurements) {
    ulock_t lock(measurements_mut_);
    if (measurements_.empty()) {
        LOG(WARNING) << "don't have any imu measurements!";
        return false;
    }

    // Substract camera delay to get imu timestamp.
    const double t = frame_timestamp - imu_calib_.delay_imu_cam;

    // Find the first measurement newer than frame_timestamp,
    // note that the newest measurement is at the front of the list!
    ImuMeasurements::iterator it = measurements_.begin();
    for (; it != measurements_.end(); ++it) {
        if (it->timestamp_ < t) {
            if (it == measurements_.begin()) {
                LOG(WARNING) << "need a newer measurement for interpolation!";
                return false;
            }
            // decrement iterator again to point to element >= t
            --it;
            break;
        }
    }

    // copy affected measurements
    extracted_measurements.insert(extracted_measurements.begin(), it, measurements_.end());

    // check
    if (extracted_measurements.size() < 2) {
        LOG(WARNING) << "need older imu measurements!";
        extracted_measurements.clear();
        return false;
    }

    if (remove_measurements) {
        // delete measurements that will not be used anymore (such that we keep it+1,
        // the first frame with smaller timestamp (earlier) than frame_timestamp,
        // which will be used in interpolation in next iteration
        measurements_.erase(it + 2, measurements_.end());
    }

    return true;
}

bool ImuHandler::getClosestMeasurement(const double timestamp, ImuMeasurement &measurement) const {
    ulock_t lock(measurements_mut_);
    if (measurements_.empty()) {
        LOG(WARNING) << "ImuHandler: don't have any imu measurements!";
        return false;
    }

    double dt_best                 = std::numeric_limits<double>::max();
    double img_timestamp_corrected = timestamp - imu_calib_.delay_imu_cam;
    for (const ImuMeasurement &m : measurements_) {
        const double dt = std::abs(m.timestamp_ - img_timestamp_corrected);
        if (dt < dt_best) {
            dt_best     = dt;
            measurement = m;
        }
    }

    if (dt_best > imu_calib_.max_imu_delta_t) {
        LOG(WARNING) << "ImuHandler: getClosestMeasurement: no measurement found!"
                        " closest measurement: "
                     << dt_best * 1000.0 << "ms.";
        return false;
    }
    return true;
}

bool ImuHandler::addImuMeasurement(const ImuMeasurement &m) {
    ulock_t lock(measurements_mut_);
    measurements_.push_front(m); // new measurement is at the front of the list!
    if (options_.temporal_stationary_check) {
        temporal_imu_window_.push_front(m);
    }
    return true;
}

ImuCalibration ImuHandler::loadCalibrationFromFile(const std::string &filename) {
    YAML::Node     data = YAML::LoadFile(filename);
    ImuCalibration calib;
    if (data["imu_params"].IsDefined()) {
        calib.delay_imu_cam               = data["imu_params"]["delay_imu_cam"].as<double>();
        calib.max_imu_delta_t             = data["imu_params"]["max_imu_delta_t"].as<double>();
        calib.saturation_accel_max        = data["imu_params"]["acc_max"].as<double>();
        calib.saturation_omega_max        = data["imu_params"]["omega_max"].as<double>();
        calib.gyro_noise_density          = data["imu_params"]["sigma_omega_c"].as<double>();
        calib.acc_noise_density           = data["imu_params"]["sigma_acc_c"].as<double>();
        calib.gyro_bias_random_walk_sigma = data["imu_params"]["sigma_omega_bias_c"].as<double>();
        calib.acc_bias_random_walk_sigma  = data["imu_params"]["sigma_acc_bias_c"].as<double>();
        calib.gravity_magnitude           = data["imu_params"]["g"].as<double>();
        calib.imu_rate                    = data["imu_params"]["imu_rate"].as<double>();
    } else {
        LOG(FATAL) << "Could not load IMU calibration from file";
    }
    return calib;
}

ImuInitialization ImuHandler::loadInitializationFromFile(const std::string &filename) {
    YAML::Node        data = YAML::LoadFile(filename);
    ImuInitialization init;
    if (data["imu_initialization"].IsDefined()) {
        init.velocity   = Eigen::Vector3d(data["imu_initialization"]["velocity"][0].as<double>(),
                                          data["imu_initialization"]["velocity"][1].as<double>(),
                                          data["imu_initialization"]["velocity"][2].as<double>());
        init.omega_bias = Eigen::Vector3d(data["imu_initialization"]["omega_bias"][0].as<double>(),
                                          data["imu_initialization"]["omega_bias"][1].as<double>(),
                                          data["imu_initialization"]["omega_bias"][2].as<double>());
        init.acc_bias   = Eigen::Vector3d(data["imu_initialization"]["acc_bias"][0].as<double>(),
                                          data["imu_initialization"]["acc_bias"][1].as<double>(),
                                          data["imu_initialization"]["acc_bias"][2].as<double>());
        init.velocity_sigma   = data["imu_initialization"]["velocity_sigma"].as<double>();
        init.omega_bias_sigma = data["imu_initialization"]["omega_bias_sigma"].as<double>();
        init.acc_bias_sigma   = data["imu_initialization"]["acc_bias_sigma"].as<double>();
    } else {
        LOG(FATAL) << "Could not load IMU initialization from file";
    }
    return init;
}

bool ImuHandler::getInitialAttitude(double timestamp, tools::Quaternion &R_imu_world) const {
    ImuMeasurement m;
    if (!getClosestMeasurement(timestamp, m)) {
        LOG(WARNING) << "ImuHandler: Could not get initial attitude. No measurements!";
        return false;
    }

    // Set initial coordinate frame based on gravity direction.
    const Eigen::Vector3d &g = m.linear_acceleration_;
    const Eigen::Vector3d  z = g.normalized(); // imu measures positive-z when static
    // TODO: make sure z != -1,0,0
    Eigen::Vector3d p(1, 0, 0);
    Eigen::Vector3d p_alternative(0, 1, 0);
    if (std::fabs(z.dot(p)) > std::fabs(z.dot(p_alternative)))
        p = p_alternative;
    Eigen::Vector3d y = z.cross(p); // make sure gravity is not in x direction
    y.normalize();
    const Eigen::Vector3d x = y.cross(z);
    Eigen::Matrix3d       C_imu_world; // world unit vectors in imu coordinates
    C_imu_world.col(0) = x;
    C_imu_world.col(1) = y;
    C_imu_world.col(2) = z;

    VLOG(3) << "Initial Rotation = " << C_imu_world;

    R_imu_world = tools::Quaternion(C_imu_world);
    return true;
}

void ImuHandler::reset() {
    ulock_t lock(measurements_mut_);
    measurements_.clear();
    temporal_imu_window_.clear();
}

IMUTemporalStatus ImuHandler::checkTemporalStatus(const double time_sec) {
    IMUTemporalStatus res = IMUTemporalStatus::kMoving;

    if (!options_.temporal_stationary_check) {
        CHECK_EQ(temporal_imu_window_.size(), 0u);
        LOG(WARNING) << "Stationary check is not enabled. Will assume moving.";
        return res;
    }

    // do we have the IMU up to this time?
    if (temporal_imu_window_.empty() || temporal_imu_window_.front().timestamp_ < time_sec) {
        return IMUTemporalStatus::kUnkown;
    }
    // check whether we have enough IMU measurements before
    int start_idx = -1;
    int end_idx   = -1;
    for (size_t idx = 0; idx < temporal_imu_window_.size(); idx++) {
        if (start_idx == -1 && temporal_imu_window_[idx].timestamp_ < time_sec) {
            // we know the first is not the start point
            CHECK_GT(idx, 0u);
            start_idx = idx - 1;
            continue;
        }

        if (start_idx != 1 && (time_sec - temporal_imu_window_[idx].timestamp_ >
                               options_.temporal_window_length_sec_)) {
            end_idx = idx;
            break;
        }
    }
    if (end_idx == -1 || start_idx == -1) {
        return IMUTemporalStatus::kUnkown;
    }
    // check the status by standard deviation
    const double        sqrt_dt = std::sqrt(1.0 / imu_calib_.imu_rate);
    std::vector<double> gyr_x(end_idx - start_idx + 1);
    std::vector<double> gyr_y(end_idx - start_idx + 1);
    std::vector<double> gyr_z(end_idx - start_idx + 1);
    std::vector<double> acc_x(end_idx - start_idx + 1);
    std::vector<double> acc_y(end_idx - start_idx + 1);
    std::vector<double> acc_z(end_idx - start_idx + 1);
    size_t              idx = 0;
    for (size_t midx = start_idx; midx <= static_cast<size_t>(end_idx); midx++) {
        const ImuMeasurement &m = temporal_imu_window_[midx];
        gyr_x[idx]              = m.angular_velocity_.x();
        gyr_y[idx]              = m.angular_velocity_.y();
        gyr_z[idx]              = m.angular_velocity_.z();
        acc_x[idx]              = m.linear_acceleration_.x();
        acc_y[idx]              = m.linear_acceleration_.y();
        acc_z[idx]              = m.linear_acceleration_.z();
        idx++;
    }
    std::array<double, 3> gyr_std{0.0, 0.0, 0.0};
    std::array<double, 3> acc_std{0.0, 0.0, 0.0};
    gyr_std[0] = stdVec(gyr_x) * sqrt_dt;
    gyr_std[1] = stdVec(gyr_y) * sqrt_dt;
    gyr_std[2] = stdVec(gyr_z) * sqrt_dt;
    acc_std[0] = stdVec(acc_x) * sqrt_dt;
    acc_std[1] = stdVec(acc_y) * sqrt_dt;
    acc_std[2] = stdVec(acc_z) * sqrt_dt;

    bool stationary = true;
    for (size_t idx = 0; idx < 3; idx++) {
        stationary &= (gyr_std[idx] < options_.stationary_gyr_sigma_thresh_);
        stationary &= (acc_std[idx] < options_.stationary_acc_sigma_thresh_);
    }

    // remove up to the used ones to make sure we still have enough
    temporal_imu_window_.erase(temporal_imu_window_.begin() + end_idx, temporal_imu_window_.end());

    return stationary ? IMUTemporalStatus::kStationary : IMUTemporalStatus::kMoving;
}

} // namespace esvo_core