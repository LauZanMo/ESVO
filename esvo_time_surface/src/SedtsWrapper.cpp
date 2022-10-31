#include "esvo_time_surface/SedtsWrapper.h"

#include <Eigen/Core>
#include <common/Camera.h>
#include <opencv2/calib3d/calib3d.hpp>

#include <cv_bridge/cv_bridge.h>

namespace EVIO {

SedtsWrapper::SedtsWrapper(ros::NodeHandle &nh, ros::NodeHandle nh_private) : nh_(nh) {
    nh_private.param<std::string>("cam_config_file", cam_config_file_, "");
    nh_private.param<int>("cam_idx", cam_idx_, 0);
    nh_private.param<bool>("use_sim_time", use_sim_time_, true);

    std::vector<double> gyro_param;
    nh_private.param<std::vector<double>>("gyro_param", gyro_param, std::vector<double>(3, 0.0));
    gyro_bias_ = Eigen::Vector3d(gyro_param[0], gyro_param[1], gyro_param[2]);

    sync_sub_        = nh_.subscribe("sync", 1, &SedtsWrapper::syncCallback, this);
    camera_info_sub_ = nh_.subscribe("camera_info", 1, &SedtsWrapper::cameraInfoCallback, this);
    event_sub_       = nh_.subscribe("events", 0, &SedtsWrapper::eventsCallback, this);
    imu_sub_         = nh_.subscribe("imu", 0, &SedtsWrapper::imuCallback, this);
    image_transport::ImageTransport it_(nh_);
    time_surface_pub_ = it_.advertise("time_surface", 1);

    NCamera::Ptr ncam = NCamera::loadFromYaml(cam_config_file_);
    cam_              = ncam->getCameraShared(cam_idx_);

    TimeSurfaceOptions ts_options;
    ts_options.ts_type         = TimeSurfaceType::SEDTS;
    ts_options.tau_scale       = 8;
    ts_options.min_tau         = 50;
    ts_options.blur_type       = EVIO::TimeSurfaceBlurType::Median;
    ts_options.blur_size       = 2;
    ts_options.ignore_polarity = true;
    ts_                        = ts_utils::makeTimeSurface(ts_options, cam_);

    cam_info_available_ = false;
}

SedtsWrapper::~SedtsWrapper() {}

void SedtsWrapper::syncCallback(const std_msgs::TimeConstPtr &msg) {
    if (!cam_info_available_)
        return;

    ros::Time sync_time;
    if (use_sim_time_)
        sync_time = ros::Time::now();
    else
        sync_time = msg->data;

    ts_->updateTau(imu_data_, gyro_bias_);
    int64_t sync_ts;
    cv::Mat ts_map, events_map;
    ts_->createTimeSurfaceMap(sync_ts, ts_map, events_map);

    cv_bridge::CvImage cv_image;
    cv_image.encoding     = "mono8";
    cv_image.header.stamp = sync_time;
    cv::remap(ts_map, cv_image.image, map1_, map2_, CV_INTER_LINEAR);
    time_surface_pub_.publish(cv_image.toImageMsg());
}

void SedtsWrapper::cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr &msg) {
    if (cam_info_available_)
        return;

    std::string distortion_model = msg->distortion_model;

    K_ = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            K_.at<double>(cv::Point(i, j)) = msg->K[i + j * 3];

    D_ = cv::Mat(msg->D.size(), 1, CV_64F);
    for (int i = 0; i < msg->D.size(); i++)
        D_.at<double>(i) = msg->D[i];

    R_ = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R_.at<double>(cv::Point(i, j)) = msg->R[i + j * 3];

    P_ = cv::Mat(3, 4, CV_64F);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
            P_.at<double>(cv::Point(i, j)) = msg->P[i + j * 4];

    if (distortion_model == "equidistant") {
        cv::fisheye::initUndistortRectifyMap(
            K_, D_, R_, P_, cv::Size(cam_->width(), cam_->height()), CV_32FC1, map1_, map2_);
        cam_info_available_ = true;
        ROS_INFO("Camera information is loaded (Distortion model %s).", distortion_model.c_str());
    } else if (distortion_model == "plumb_bob") {
        cv::initUndistortRectifyMap(K_, D_, R_, P_, cv::Size(cam_->width(), cam_->height()),
                                    CV_32FC1, map1_, map2_);
        cam_info_available_ = true;
        ROS_INFO("Camera information is loaded (Distortion model %s).", distortion_model.c_str());
    } else {
        ROS_ERROR_ONCE("Distortion model %s is not supported.", distortion_model.c_str());
        cam_info_available_ = false;
        return;
    }
}

void SedtsWrapper::eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg) {
    int        size = msg->events.size();
    EventsData events =
        std::make_shared<Eigen::Matrix<int64_t, 4, Eigen::Dynamic, Eigen::RowMajor>>(4, size);
    for (int i = 0; i < size; i++) {
        (*events)(0, i) = msg->events[i].ts.toSec() * 1e6;
        (*events)(1, i) = msg->events[i].x;
        (*events)(2, i) = msg->events[i].y;
        (*events)(3, i) = msg->events[i].polarity ? 1 : 0;
    }
    ts_->insertEvents(events);
}

void SedtsWrapper::imuCallback(const sensor_msgs::Imu::ConstPtr &msg) {
    ImuData imu_data;
    imu_data(0) = msg->header.stamp.toSec() * 1e6;
    imu_data(1) = msg->angular_velocity.x;
    imu_data(2) = msg->angular_velocity.y;
    imu_data(3) = msg->angular_velocity.z;
    imu_data(4) = msg->linear_acceleration.x;
    imu_data(5) = msg->linear_acceleration.y;
    imu_data(6) = msg->linear_acceleration.z;
    imu_data(7) = 0;
    imu_data_.push_back(std::move(imu_data));
    if (imu_data_.size() > 2)
        if (imu_data_.back()(0) - imu_data_[1](0) > 20e3)
            imu_data_.pop_front();
}

} // namespace EVIO
