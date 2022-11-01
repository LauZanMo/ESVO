#include <esvo_core/tools/Visualization.h>

namespace esvo_core {
namespace tools {

Visualization::Visualization() {}

Visualization::~Visualization() {}

void Visualization::plot_map(DepthMap::Ptr &depthMapPtr,
                             VisMapType     vmType,
                             cv::Mat       &img,
                             double         max_range,
                             double         min_range,
                             double         visualization_threshold1,
                             double         visualization_threshold2) {
    size_t height = depthMapPtr->rows();
    size_t width  = depthMapPtr->cols();
    img           = cv::Mat(cv::Size(width, height), CV_8UC1, cv::Scalar(0));
    cv::cvtColor(img, img, CV_GRAY2BGR);

    switch (vmType) {
    case InvDepthMap: {
        std::vector<double> vDepth;
        vDepth.reserve(10000);
        for (auto it = depthMapPtr->begin(); it != depthMapPtr->end(); it++) {
            if (it->valid() && it->variance() < pow(visualization_threshold1, 2) &&
                it->age() >= (int)visualization_threshold2) {
                vDepth.emplace_back(1.0 / it->invDepth());
                DrawPoint(it->invDepth(), max_range, min_range, it->x(), img);
            }
        }
        break;
    }
    case StdVarMap: {
        for (auto it = depthMapPtr->begin(); it != depthMapPtr->end(); it++) {
            if (it->valid() && it->variance() < pow(visualization_threshold1, 2))
                DrawPoint(sqrt(it->variance()), max_range, min_range, it->x(), img);
        }
        break;
    }
    case CostMap: {
        for (auto it = depthMapPtr->begin(); it != depthMapPtr->end(); it++) {
            if (it->valid() && it->residual() < visualization_threshold1)
                DrawPoint(it->residual(), max_range, min_range, it->x(), img);
        }
        break;
    }
    case AgeMap: {
        for (auto it = depthMapPtr->begin(); it != depthMapPtr->end(); it++) {
            if (it->valid() && it->age() >= (int)visualization_threshold1)
                DrawPoint(it->age(), max_range, min_range, it->x(), img);
        }
        break;
    }
    default: {
        LOG(INFO) << "Wrong type chosed------------------";
        exit(0);
    }
    }
}

void Visualization::DrawPoint(
    double val, double max_range, double min_range, const Eigen::Vector2d &location, cv::Mat &img) {
    // define the color based on the inverse depth
    int index = floor((val - min_range) / (max_range - min_range) * 255.0f);
    if (index > 255)
        index = 255;
    if (index < 0)
        index = 0;

    cv::Scalar color = CV_RGB(255.0f * r[index], 255.0f * g[index], 255.0f * b[index]);
    // draw the point
    cv::Point point;
    point.x = location[0];
    point.y = location[1];
    cv::circle(img, point, 1, color, cv::FILLED);
}

void Visualization::plot_eventMap(std::vector<dvs_msgs::Event *> &vEventPtr,
                                  cv::Mat                        &eventMap,
                                  size_t                          row,
                                  size_t                          col) {
    eventMap = cv::Mat(cv::Size(col, row), CV_8UC1, cv::Scalar(0));
    for (size_t i = 0; i < vEventPtr.size(); i++)
        eventMap.at<uchar>(vEventPtr[i]->y, vEventPtr[i]->x) = 255;
}

void Visualization::plot_events(
    std::vector<Eigen::Matrix<double, 2, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 2, 1>>>
            &vEvents,
    cv::Mat &event_img,
    size_t   row,
    size_t   col) {
    event_img = cv::Mat(cv::Size(col, row), CV_8UC1, cv::Scalar(0));
    auto it   = vEvents.begin();
    while (it != vEvents.end()) {
        int xcoor = std::floor((*it)(0));
        int ycoor = std::floor((*it)(1));
        if (xcoor < 0 || xcoor >= col || ycoor < 0 || ycoor >= row) {
        } else
            event_img.at<uchar>(ycoor, xcoor) = 255;
        it++;
        //    LOG(INFO) << xcoor << " " << ycoor;
    }
}

const float Visualization::r[] = {0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0.00588235294117645,
                                  0.02156862745098032,
                                  0.03725490196078418,
                                  0.05294117647058827,
                                  0.06862745098039214,
                                  0.084313725490196,
                                  0.1000000000000001,
                                  0.115686274509804,
                                  0.1313725490196078,
                                  0.1470588235294117,
                                  0.1627450980392156,
                                  0.1784313725490196,
                                  0.1941176470588235,
                                  0.2098039215686274,
                                  0.2254901960784315,
                                  0.2411764705882353,
                                  0.2568627450980392,
                                  0.2725490196078431,
                                  0.2882352941176469,
                                  0.303921568627451,
                                  0.3196078431372549,
                                  0.3352941176470587,
                                  0.3509803921568628,
                                  0.3666666666666667,
                                  0.3823529411764706,
                                  0.3980392156862744,
                                  0.4137254901960783,
                                  0.4294117647058824,
                                  0.4450980392156862,
                                  0.4607843137254901,
                                  0.4764705882352942,
                                  0.4921568627450981,
                                  0.5078431372549019,
                                  0.5235294117647058,
                                  0.5392156862745097,
                                  0.5549019607843135,
                                  0.5705882352941174,
                                  0.5862745098039217,
                                  0.6019607843137256,
                                  0.6176470588235294,
                                  0.6333333333333333,
                                  0.6490196078431372,
                                  0.664705882352941,
                                  0.6803921568627449,
                                  0.6960784313725492,
                                  0.7117647058823531,
                                  0.7274509803921569,
                                  0.7431372549019608,
                                  0.7588235294117647,
                                  0.7745098039215685,
                                  0.7901960784313724,
                                  0.8058823529411763,
                                  0.8215686274509801,
                                  0.8372549019607844,
                                  0.8529411764705883,
                                  0.8686274509803922,
                                  0.884313725490196,
                                  0.8999999999999999,
                                  0.9156862745098038,
                                  0.9313725490196076,
                                  0.947058823529412,
                                  0.9627450980392158,
                                  0.9784313725490197,
                                  0.9941176470588236,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  0.9862745098039216,
                                  0.9705882352941178,
                                  0.9549019607843139,
                                  0.93921568627451,
                                  0.9235294117647062,
                                  0.9078431372549018,
                                  0.892156862745098,
                                  0.8764705882352941,
                                  0.8607843137254902,
                                  0.8450980392156864,
                                  0.8294117647058825,
                                  0.8137254901960786,
                                  0.7980392156862743,
                                  0.7823529411764705,
                                  0.7666666666666666,
                                  0.7509803921568627,
                                  0.7352941176470589,
                                  0.719607843137255,
                                  0.7039215686274511,
                                  0.6882352941176473,
                                  0.6725490196078434,
                                  0.6568627450980391,
                                  0.6411764705882352,
                                  0.6254901960784314,
                                  0.6098039215686275,
                                  0.5941176470588236,
                                  0.5784313725490198,
                                  0.5627450980392159,
                                  0.5470588235294116,
                                  0.5313725490196077,
                                  0.5156862745098039,
                                  0.5};
const float Visualization::g[] = {0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0.001960784313725483,
                                  0.01764705882352935,
                                  0.03333333333333333,
                                  0.0490196078431373,
                                  0.06470588235294117,
                                  0.08039215686274503,
                                  0.09607843137254901,
                                  0.111764705882353,
                                  0.1274509803921569,
                                  0.1431372549019607,
                                  0.1588235294117647,
                                  0.1745098039215687,
                                  0.1901960784313725,
                                  0.2058823529411764,
                                  0.2215686274509804,
                                  0.2372549019607844,
                                  0.2529411764705882,
                                  0.2686274509803921,
                                  0.2843137254901961,
                                  0.3,
                                  0.3156862745098039,
                                  0.3313725490196078,
                                  0.3470588235294118,
                                  0.3627450980392157,
                                  0.3784313725490196,
                                  0.3941176470588235,
                                  0.4098039215686274,
                                  0.4254901960784314,
                                  0.4411764705882353,
                                  0.4568627450980391,
                                  0.4725490196078431,
                                  0.4882352941176471,
                                  0.503921568627451,
                                  0.5196078431372548,
                                  0.5352941176470587,
                                  0.5509803921568628,
                                  0.5666666666666667,
                                  0.5823529411764705,
                                  0.5980392156862746,
                                  0.6137254901960785,
                                  0.6294117647058823,
                                  0.6450980392156862,
                                  0.6607843137254901,
                                  0.6764705882352942,
                                  0.692156862745098,
                                  0.7078431372549019,
                                  0.723529411764706,
                                  0.7392156862745098,
                                  0.7549019607843137,
                                  0.7705882352941176,
                                  0.7862745098039214,
                                  0.8019607843137255,
                                  0.8176470588235294,
                                  0.8333333333333333,
                                  0.8490196078431373,
                                  0.8647058823529412,
                                  0.8803921568627451,
                                  0.8960784313725489,
                                  0.9117647058823528,
                                  0.9274509803921569,
                                  0.9431372549019608,
                                  0.9588235294117646,
                                  0.9745098039215687,
                                  0.9901960784313726,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  0.9901960784313726,
                                  0.9745098039215687,
                                  0.9588235294117649,
                                  0.943137254901961,
                                  0.9274509803921571,
                                  0.9117647058823528,
                                  0.8960784313725489,
                                  0.8803921568627451,
                                  0.8647058823529412,
                                  0.8490196078431373,
                                  0.8333333333333335,
                                  0.8176470588235296,
                                  0.8019607843137253,
                                  0.7862745098039214,
                                  0.7705882352941176,
                                  0.7549019607843137,
                                  0.7392156862745098,
                                  0.723529411764706,
                                  0.7078431372549021,
                                  0.6921568627450982,
                                  0.6764705882352944,
                                  0.6607843137254901,
                                  0.6450980392156862,
                                  0.6294117647058823,
                                  0.6137254901960785,
                                  0.5980392156862746,
                                  0.5823529411764707,
                                  0.5666666666666669,
                                  0.5509803921568626,
                                  0.5352941176470587,
                                  0.5196078431372548,
                                  0.503921568627451,
                                  0.4882352941176471,
                                  0.4725490196078432,
                                  0.4568627450980394,
                                  0.4411764705882355,
                                  0.4254901960784316,
                                  0.4098039215686273,
                                  0.3941176470588235,
                                  0.3784313725490196,
                                  0.3627450980392157,
                                  0.3470588235294119,
                                  0.331372549019608,
                                  0.3156862745098041,
                                  0.2999999999999998,
                                  0.284313725490196,
                                  0.2686274509803921,
                                  0.2529411764705882,
                                  0.2372549019607844,
                                  0.2215686274509805,
                                  0.2058823529411766,
                                  0.1901960784313728,
                                  0.1745098039215689,
                                  0.1588235294117646,
                                  0.1431372549019607,
                                  0.1274509803921569,
                                  0.111764705882353,
                                  0.09607843137254912,
                                  0.08039215686274526,
                                  0.06470588235294139,
                                  0.04901960784313708,
                                  0.03333333333333321,
                                  0.01764705882352935,
                                  0.001960784313725483,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0};
const float Visualization::b[] = {0.5,
                                  0.5156862745098039,
                                  0.5313725490196078,
                                  0.5470588235294118,
                                  0.5627450980392157,
                                  0.5784313725490196,
                                  0.5941176470588235,
                                  0.6098039215686275,
                                  0.6254901960784314,
                                  0.6411764705882352,
                                  0.6568627450980392,
                                  0.6725490196078432,
                                  0.6882352941176471,
                                  0.7039215686274509,
                                  0.7196078431372549,
                                  0.7352941176470589,
                                  0.7509803921568627,
                                  0.7666666666666666,
                                  0.7823529411764706,
                                  0.7980392156862746,
                                  0.8137254901960784,
                                  0.8294117647058823,
                                  0.8450980392156863,
                                  0.8607843137254902,
                                  0.8764705882352941,
                                  0.892156862745098,
                                  0.907843137254902,
                                  0.9235294117647059,
                                  0.9392156862745098,
                                  0.9549019607843137,
                                  0.9705882352941176,
                                  0.9862745098039216,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  0.9941176470588236,
                                  0.9784313725490197,
                                  0.9627450980392158,
                                  0.9470588235294117,
                                  0.9313725490196079,
                                  0.915686274509804,
                                  0.8999999999999999,
                                  0.884313725490196,
                                  0.8686274509803922,
                                  0.8529411764705883,
                                  0.8372549019607844,
                                  0.8215686274509804,
                                  0.8058823529411765,
                                  0.7901960784313726,
                                  0.7745098039215685,
                                  0.7588235294117647,
                                  0.7431372549019608,
                                  0.7274509803921569,
                                  0.7117647058823531,
                                  0.696078431372549,
                                  0.6803921568627451,
                                  0.6647058823529413,
                                  0.6490196078431372,
                                  0.6333333333333333,
                                  0.6176470588235294,
                                  0.6019607843137256,
                                  0.5862745098039217,
                                  0.5705882352941176,
                                  0.5549019607843138,
                                  0.5392156862745099,
                                  0.5235294117647058,
                                  0.5078431372549019,
                                  0.4921568627450981,
                                  0.4764705882352942,
                                  0.4607843137254903,
                                  0.4450980392156865,
                                  0.4294117647058826,
                                  0.4137254901960783,
                                  0.3980392156862744,
                                  0.3823529411764706,
                                  0.3666666666666667,
                                  0.3509803921568628,
                                  0.335294117647059,
                                  0.3196078431372551,
                                  0.3039215686274508,
                                  0.2882352941176469,
                                  0.2725490196078431,
                                  0.2568627450980392,
                                  0.2411764705882353,
                                  0.2254901960784315,
                                  0.2098039215686276,
                                  0.1941176470588237,
                                  0.1784313725490199,
                                  0.1627450980392156,
                                  0.1470588235294117,
                                  0.1313725490196078,
                                  0.115686274509804,
                                  0.1000000000000001,
                                  0.08431372549019622,
                                  0.06862745098039236,
                                  0.05294117647058805,
                                  0.03725490196078418,
                                  0.02156862745098032,
                                  0.00588235294117645,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0};
} // namespace tools
} // namespace esvo_core
