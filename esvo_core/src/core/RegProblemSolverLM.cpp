#include <esvo_core/core/RegProblemSolverLM.h>
#include <esvo_core/tools/cayley.h>

#ifdef SAVE_DUBUG_INFO
#include <evaluation/DebugValueSaver.h>
int track_idx = 0;
#endif

namespace esvo_core {
namespace core {
RegProblemSolverLM::RegProblemSolverLM(esvo_core::CameraSystem::Ptr   &camSysPtr,
                                       shared_ptr<RegProblemConfig>   &rpConfigPtr,
                                       esvo_core::core::RegProblemType rpType,
                                       size_t                          numThread,
                                       ImuHandler::Ptr                 imu_handler)
    : camSysPtr_(camSysPtr),
      rpConfigPtr_(rpConfigPtr),
      rpType_(rpType),
      NUM_THREAD_(numThread),
      imu_handler_(imu_handler),
      bPrint_(false),
      bVisualize_(true) {
    if (rpType_ == REG_NUMERICAL) {
        LOG(ERROR) << "Not support numerical!!!";
        exit(-1);
        // numDiff_regProblemPtr_ = std::make_shared<Eigen::NumericalDiff<RegProblemLM>>(
        //     camSysPtr_, rpConfigPtr_, NUM_THREAD_, imu_handler_);
    } else if (rpType_ == REG_ANALYTICAL) {
        regProblemPtr_ =
            std::make_shared<RegProblemLM>(camSysPtr_, rpConfigPtr_, NUM_THREAD_, imu_handler_);
    } else {
        LOG(ERROR) << "Wrong Registration Problem Type is assigned!!!";
        exit(-1);
    }
    z_min_ = 1.0 / rpConfigPtr_->invDepth_max_range_;
    z_max_ = 1.0 / rpConfigPtr_->invDepth_min_range_;

    lmStatics_.nPoints_ = 0;
    lmStatics_.nfev_    = 0;
    lmStatics_.nIter_   = 0;
}

RegProblemSolverLM::~RegProblemSolverLM() {}

bool RegProblemSolverLM::resetRegProblem(RefFrame        *ref,
                                         CurFrame        *cur,
                                         Eigen::Vector3d *gyro_bias,
                                         ImuMeasurements *ms_ref_cur) {
    if (cur->numEventsSinceLastObs_ < rpConfigPtr_->MIN_NUM_EVENTS_) {
        LOG(INFO) << "resetRegProblem RESET fails for no enough events coming in.";
        LOG(INFO) << "However, the system remains to work.";
    }
    if (ref->vPointXYZPtr_.size() < rpConfigPtr_->BATCH_SIZE_) {
        LOG(INFO) << "resetRegProblem RESET fails for no enough point cloud in the local map.";
        LOG(INFO) << "The system will be re-initialized";
        return false;
    }
    //  LOG(INFO) << "resetRegProblem RESET succeeds.";
    if (rpType_ == REG_NUMERICAL) {
        numDiff_regProblemPtr_->setProblem(ref, cur, false);
        //    LOG(INFO) << "numDiff_regProblemPtr_->setProblem(ref, cur, false) -----------------";
    }
    if (rpType_ == REG_ANALYTICAL) {
        regProblemPtr_->setProblem(ref, cur, true, gyro_bias, ms_ref_cur);
        //    LOG(INFO) << "regProblemPtr_->setProblem(ref, cur, true) -----------------";
    }

    lmStatics_.nPoints_ = 0;
    lmStatics_.nfev_    = 0;
    lmStatics_.nIter_   = 0;
    return true;
}

bool RegProblemSolverLM::solve_numerical() {
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<RegProblemLM>, double> lm(
        *numDiff_regProblemPtr_.get());
    lm.resetParameters();
    lm.parameters.ftol   = 1e-3;
    lm.parameters.xtol   = 1e-3;
    lm.parameters.maxfev = rpConfigPtr_->MAX_ITERATION_ * 8;

    size_t iteration = 0;
    size_t nfev      = 0;
    while (true) {
        if (iteration >= rpConfigPtr_->MAX_ITERATION_)
            break;
        numDiff_regProblemPtr_->setStochasticSampling(
            (iteration % numDiff_regProblemPtr_->numBatches_) * rpConfigPtr_->BATCH_SIZE_,
            rpConfigPtr_->BATCH_SIZE_);
        Eigen::VectorXd x(6);
        x.fill(0.0);
        if (lm.minimizeInit(x) == Eigen::LevenbergMarquardtSpace::ImproperInputParameters) {
            LOG(ERROR) << "ImproperInputParameters for LM (Tracking)." << std::endl;
            return false;
        }

        Eigen::LevenbergMarquardtSpace::Status status = lm.minimizeOneStep(x);
        numDiff_regProblemPtr_->addMotionUpdate(x);

        iteration++;
        nfev += lm.nfev;

        /*************************** Visualization ************************/
        if (bVisualize_) // will slow down the tracker's performance a little bit
        {
            size_t  width          = camSysPtr_->cam_left_ptr_->width_;
            size_t  height         = camSysPtr_->cam_left_ptr_->height_;
            cv::Mat reprojMap_left = cv::Mat(cv::Size(width, height), CV_8UC1, cv::Scalar(0));
            cv::eigen2cv(numDiff_regProblemPtr_->cur_->pTsObs_->TS_negative_left_, reprojMap_left);
            reprojMap_left.convertTo(reprojMap_left, CV_8UC1);
            cv::cvtColor(reprojMap_left, reprojMap_left, CV_GRAY2BGR);

            // project 3D points to current frame
            Eigen::Matrix3d R_cur_ref = numDiff_regProblemPtr_->R_.transpose();
            Eigen::Vector3d t_cur_ref =
                -numDiff_regProblemPtr_->R_.transpose() * numDiff_regProblemPtr_->t_;

            size_t numVisualization =
                std::min(numDiff_regProblemPtr_->ResItems_.size(), (size_t)2000);
            for (size_t i = 0; i < numVisualization; i++) {
                ResidualItem   &ri   = numDiff_regProblemPtr_->ResItems_[i];
                Eigen::Vector3d p_3D = R_cur_ref * ri.p_ + t_cur_ref;
                Eigen::Vector2d p_img_left;
                camSysPtr_->cam_left_ptr_->world2Cam(p_3D, p_img_left);
                double z = ri.p_[2];
                visualizor_.DrawPoint(1.0 / z, 1.0 / z_min_, 1.0 / z_max_,
                                      Eigen::Vector2d(p_img_left(0), p_img_left(1)),
                                      reprojMap_left);
            }
            std_msgs::Header header;
            header.stamp = numDiff_regProblemPtr_->cur_->t_;
            sensor_msgs::ImagePtr msg =
                cv_bridge::CvImage(header, "bgr8", reprojMap_left).toImageMsg();
            reprojMap_pub_->publish(msg);
        }
        /*************************** Visualization ************************/
        if (status == 2 || status == 3)
            break;
    }
    //  LOG(INFO) << "LM Finished ...................";
    numDiff_regProblemPtr_->setPose();
    lmStatics_.nPoints_ = numDiff_regProblemPtr_->numPoints_;
    lmStatics_.nfev_    = nfev;
    lmStatics_.nIter_   = iteration;
    return 0;
}

bool RegProblemSolverLM::solve_analytical() {
    Eigen::LevenbergMarquardt<RegProblemLM, double> lm(*regProblemPtr_.get());
    lm.resetParameters();
    lm.parameters.ftol   = 1e-3;
    lm.parameters.xtol   = 1e-3;
    lm.parameters.maxfev = rpConfigPtr_->MAX_ITERATION_ * 8;

    size_t iteration = 0;
    size_t nfev      = 0;

    std::vector<Eigen::Vector<double, 9>> iter_values;
    while (true) {
        if (iteration >= rpConfigPtr_->MAX_ITERATION_)
            break;
        // 一次只取一小部分数据用于计算
        regProblemPtr_->setStochasticSampling((iteration % regProblemPtr_->numBatches_) *
                                                  rpConfigPtr_->BATCH_SIZE_,
                                              rpConfigPtr_->BATCH_SIZE_);
        Eigen::VectorXd x(9);
        x.fill(0.0);
        if (lm.minimizeInit(x) == Eigen::LevenbergMarquardtSpace::ImproperInputParameters) {
            LOG(ERROR) << "ImproperInputParameters for LM (Tracking)." << std::endl;
            return false;
        }
        Eigen::LevenbergMarquardtSpace::Status status = lm.minimizeOneStep(x);
        regProblemPtr_->addMotionUpdate(x);
        iter_values.push_back(x);

        iteration++;
        nfev += lm.nfev;
        if (status == 2 || status == 3)
            break;
    }

    if (!regProblemPtr_->gyro_bias_initialized_ && regProblemPtr_->imu_handler_ != nullptr) {
        GyroSolverStruct gss(regProblemPtr_->delta_q_, Eigen::Quaterniond(regProblemPtr_->R_bi_bj_),
                             regProblemPtr_->jacobian_.block<3, 3>(0, 3));
        gss_vec_.push_back(gss);
        // LOG(INFO) << "Solve: " << Eigen::AngleAxisd(regProblemPtr_->R_bi_bj_).angle() * 180 /
        // M_PI
        //           << "deg, " << Eigen::AngleAxisd(regProblemPtr_->R_bi_bj_).axis().transpose();
        // LOG(INFO) << "Integrate: "
        //           << Eigen::AngleAxisd(regProblemPtr_->delta_q_).angle() * 180 / M_PI << "deg, "
        //           << Eigen::AngleAxisd(regProblemPtr_->delta_q_).axis().transpose();
        // LOG(INFO) << "---";
        if (gss_vec_.size() > 100) {
            regProblemPtr_->solveGyroBias(gss_vec_);
            LOG(INFO) << "Gyro bias initialized: " << regProblemPtr_->gyro_bias_->transpose();
        }
    } else {
        // LOG(INFO) << "Solve: " << Eigen::AngleAxisd(regProblemPtr_->R_bi_bj_).angle() * 180 /
        // M_PI
        //           << "deg, " << Eigen::AngleAxisd(regProblemPtr_->R_bi_bj_).axis().transpose();
        // LOG(INFO) << "Integrate: "
        //           << Eigen::AngleAxisd(regProblemPtr_->delta_q_).angle() * 180 / M_PI << "deg, "
        //           << Eigen::AngleAxisd(regProblemPtr_->delta_q_).axis().transpose();
        // LOG(INFO) << "jacobian: " << std::endl << regProblemPtr_->jacobian_;
        // LOG(INFO) << "covariance: " << std::endl << regProblemPtr_->covariance_;
        // LOG(INFO) << "---";
    }

    cv::Mat_<uint8_t> proj_img, proj_img_init;
    /*************************** Visualization ************************/
    if (bVisualize_) // will slow down the tracker a little bit
    {
        size_t  width          = camSysPtr_->cam_left_ptr_->width_;
        size_t  height         = camSysPtr_->cam_left_ptr_->height_;
        cv::Mat reprojMap_left = cv::Mat(cv::Size(width, height), CV_8UC1, cv::Scalar(0));
        cv::eigen2cv(regProblemPtr_->cur_->pTsObs_->TS_negative_left_, reprojMap_left);
        reprojMap_left.convertTo(reprojMap_left, CV_8UC1);
        cv::cvtColor(reprojMap_left, reprojMap_left, CV_GRAY2BGR);
        cv::Mat reproj_map_left_init = reprojMap_left.clone();

        // project 3D points to current frame
        Eigen::Matrix3d R_cur_ref = regProblemPtr_->R_.transpose();
        Eigen::Vector3d t_cur_ref = -regProblemPtr_->R_.transpose() * regProblemPtr_->t_;

        size_t numVisualization = std::min(regProblemPtr_->ResItems_.size(), (size_t)2000);
        for (size_t i = 0; i < numVisualization; i++) {
            ResidualItem   &ri   = regProblemPtr_->ResItems_[i];
            Eigen::Vector3d p_3D = R_cur_ref * ri.p_ + t_cur_ref;
            Eigen::Vector2d p_img_left;
            camSysPtr_->cam_left_ptr_->world2Cam(p_3D, p_img_left);
            double z = ri.p_[2];
            visualizor_.DrawPoint(1.0 / z, 1.0 / z_min_, 1.0 / z_max_,
                                  Eigen::Vector2d(p_img_left(0), p_img_left(1)), reprojMap_left);
#ifdef SAVE_DUBUG_INFO
            camSysPtr_->cam_left_ptr_->world2Cam(ri.p_, p_img_left);
            visualizor_.DrawPoint(1.0 / z, 1.0 / z_min_, 1.0 / z_max_,
                                  Eigen::Vector2d(p_img_left(0), p_img_left(1)),
                                  reproj_map_left_init);
#endif
        }
        std_msgs::Header header;
        header.stamp              = regProblemPtr_->cur_->t_;
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "bgr8", reprojMap_left).toImageMsg();
        reprojMap_pub_->publish(msg);
        proj_img      = reprojMap_left;
        proj_img_init = reproj_map_left_init;
    }
    /*************************** Visualization ************************/

#ifdef SAVE_DUBUG_INFO
    Eigen::Vector<double, 6> final_values;
    final_values.segment<3>(0) = tools::rot2cayley(regProblemPtr_->R_);
    final_values.segment<3>(3) = regProblemPtr_->t_;
    H5Easy::dump(EVIO::DebugValueSaver::saver(), "track/delta_" + std::to_string(track_idx),
                 iter_values);
    H5Easy::dump(EVIO::DebugValueSaver::saver(), "track/final_" + std::to_string(track_idx),
                 final_values);
    H5Easy::dump(EVIO::DebugValueSaver::saver(), "track/proj_map_" + std::to_string(track_idx),
                 proj_img);
    H5Easy::dump(EVIO::DebugValueSaver::saver(), "track/proj_map_init_" + std::to_string(track_idx),
                 proj_img_init);
    H5Easy::dump(EVIO::DebugValueSaver::saver(), "track/size", ++track_idx,
                 H5Easy::DumpMode::Overwrite);
#endif

    regProblemPtr_->setPose();
    lmStatics_.nPoints_ = regProblemPtr_->numPoints_;
    lmStatics_.nfev_    = nfev;
    lmStatics_.nIter_   = iteration;
    return 0;
}

void RegProblemSolverLM::setRegPublisher(image_transport::Publisher *reprojMap_pub) {
    reprojMap_pub_ = reprojMap_pub;
}

} // namespace core
} // namespace esvo_core
