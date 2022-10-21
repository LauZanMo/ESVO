#pragma once

#include <common/CameraFwd.h>
#include <opencv2/core/core.hpp>
#include <string>
#include <time_surface/AbstractTimeSurface.h>

#include <dvs_msgs/EventArray.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Time.h>

namespace EVIO {

class SedtsWrapper {
public:
    SedtsWrapper(ros::NodeHandle &nh, ros::NodeHandle nh_private);
    ~SedtsWrapper();

    void syncCallback(const std_msgs::TimeConstPtr &msg);
    void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr &msg);
    void eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg);
    void imuCallback(const sensor_msgs::Imu::ConstPtr &msg);

    ros::NodeHandle nh_;

    std::string cam_config_file_;
    int         cam_idx_;
    CameraPtr   cam_;

    AbstractTimeSurface::Ptr ts_;
    Eigen::Vector3d          gyro_bias_;
    ImuDatas                 imu_data_;

    cv::Mat D_, K_, R_, P_;
    cv::Mat map1_, map2_;

    ros::Subscriber            sync_sub_;
    ros::Subscriber            camera_info_sub_;
    ros::Subscriber            event_sub_;
    ros::Subscriber            imu_sub_;
    image_transport::Publisher time_surface_pub_;

    bool cam_info_available_;
    bool use_sim_time_;
};

} // namespace EVIO