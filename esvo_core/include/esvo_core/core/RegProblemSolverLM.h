#ifndef ESVO_CORE_CORE_REGPROBLEMSOLVERLM_H
#define ESVO_CORE_CORE_REGPROBLEMSOLVERLM_H

#include <esvo_core/container/ImuHandler.h>
#include <esvo_core/core/RegProblemLM.h>
#include <esvo_core/optimization/OptimizationFunctor.h>
#include <esvo_core/tools/Visualization.h>
#include <memory>

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace esvo_core {
namespace core {
enum RegProblemType {
    REG_NUMERICAL,
    REG_ANALYTICAL
};

struct LM_statics {
    size_t nPoints_;
    size_t nfev_;
    size_t nIter_;
};

class RegProblemSolverLM {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<RegProblemSolverLM> Ptr;

    RegProblemSolverLM(CameraSystem::Ptr                 &camSysPtr,
                       std::shared_ptr<RegProblemConfig> &rpConfigPtr,
                       RegProblemType                     rpType      = REG_NUMERICAL,
                       size_t                             numThread   = 1,
                       std::shared_ptr<ImuHandler>        imu_handler = nullptr);
    virtual ~RegProblemSolverLM();

    bool resetRegProblem(RefFrame        *ref,
                         CurFrame        *cur,
                         Eigen::Vector3d *gyro_bias,
                         ImuMeasurements *ms_ref_cur);
    bool solve_numerical();  // relatively slower
    bool solve_analytical(); // faster

    // For test and visualization
    void       setRegPublisher(image_transport::Publisher *reprojMap_pub);
    LM_statics lmStatics_; // record LevenburgMarquardt log.

    // variables

private:
    CameraSystem::Ptr                &camSysPtr_;
    ImuHandler::Ptr                   imu_handler_;
    std::shared_ptr<RegProblemConfig> rpConfigPtr_;
    size_t                            NUM_THREAD_;
    RegProblemType                    rpType_;

    std::shared_ptr<RegProblemLM>                       regProblemPtr_;
    std::shared_ptr<Eigen::NumericalDiff<RegProblemLM>> numDiff_regProblemPtr_;

    std::vector<GyroSolverStruct> gss_vec_;

    // For test
    double                      z_min_, z_max_;
    image_transport::Publisher *reprojMap_pub_;
    Visualization               visualizor_;
    bool                        bPrint_, bVisualize_;
};
} // namespace core
} // namespace esvo_core
#endif // ESVO_CORE_CORE_REGPROBLEMSOLVER2_H