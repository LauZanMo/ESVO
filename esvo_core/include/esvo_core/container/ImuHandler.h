#pragma once

#include <esvo_core/container/ImuCalibration.h>
#include <esvo_core/tools/utils.h>

namespace esvo_core {

struct IMUHandlerOptions {
    bool   temporal_stationary_check    = false;
    double temporal_window_length_sec_  = 0.5;
    double stationary_acc_sigma_thresh_ = 10e-4;
    double stationary_gyr_sigma_thresh_ = 6e-5;
};

enum class IMUTemporalStatus {
    kStationary,
    kMoving,
    kUnkown
};

extern const std::map<IMUTemporalStatus, std::string> imu_temporal_status_names_;

class ImuHandler {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<ImuHandler> Ptr;
    typedef std::mutex                  mutex_t;
    typedef std::unique_lock<mutex_t>   ulock_t;

    IMUHandlerOptions options_;
    ImuCalibration    imu_calib_;
    ImuInitialization imu_init_;

    // TODO: make private
    mutable mutex_t bias_mut_;
    Eigen::Vector3d acc_bias_;   //!< Accleration bias used during preintegration
    Eigen::Vector3d omega_bias_; //!< Angular rate bias values used during preintegration

    ImuHandler(const ImuCalibration    &imu_calib,
               const ImuInitialization &imu_init,
               const IMUHandlerOptions &options);
    ~ImuHandler();

    const Eigen::Vector3d &getAccelerometerBias() const {
        ulock_t lock(bias_mut_);
        return acc_bias_;
    }

    const Eigen::Vector3d &getGyroscopeBias() const {
        ulock_t lock(bias_mut_);
        return omega_bias_;
    }

    void setAccelerometerBias(const Eigen::Vector3d &acc_bias) {
        ulock_t lock(bias_mut_);
        acc_bias_ = acc_bias;
    }

    void setGyroscopeBias(const Eigen::Vector3d &omega_bias) {
        ulock_t lock(bias_mut_);
        omega_bias_ = omega_bias;
    }

    ImuMeasurements getMeasurementsCopy() const {
        ulock_t lock(measurements_mut_);
        return measurements_;
    }

    /// Get IMU measurements up to  for the exact borders.
    /// Note that you have to provide the camera timestamp.
    /// Internally, given the calibration it corrects the timestamps for delays.
    /// The returned measurement will cover the full timeinterval
    /// (getMeasurements only gives newest measurement smaller than new_cam_timestamp
    bool getMeasurementsContainingEdges(const double     frame_timestamp, // seoncds
                                        ImuMeasurements &measurements,
                                        const bool       remove_measurements);

    bool getClosestMeasurement(const double timestamp, ImuMeasurement &measurement) const;

    /// Assumes we are in hover condition and estimates the inital orientation by
    /// estimating the gravity direction. The yaw direction is not deterministic.
    bool getInitialAttitude(double timestamp, tools::Quaternion &R_imu_world) const;

    bool addImuMeasurement(const ImuMeasurement &measurement);

    static ImuCalibration    loadCalibrationFromFile(const std::string &filename);
    static ImuInitialization loadInitializationFromFile(const std::string &filename);

    void reset();

    double getLatestTimestamp() const {
        ulock_t lock(measurements_mut_);
        return measurements_.front().timestamp_;
    }

    IMUTemporalStatus checkTemporalStatus(const double time_sec);

private:
    mutable mutex_t measurements_mut_;
    ImuMeasurements measurements_; ///< Newest measurement is at the front of the list
    ImuMeasurements temporal_imu_window_;
};

} // namespace esvo_core
