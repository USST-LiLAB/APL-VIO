#include "odom_utils/utils.h"
#include "estimator/paramPool.h"
#include "rosMaster/rosMaster.h"

#include "visual/lineOpticalFlowTracker.h"
#include "visual/lineFeatureTracker_LOFT.h"


extern std::shared_ptr<ParamPool> shared_pool;

std::vector<std::vector<double>> LineOpticalFlowTracker::param;

void LineOpticalFlowTracker::OpticalFlowMultiLevel(
    const cv::Mat &image1,
    const cv::Mat &image2,
    const std::vector<Line> &kls1,
    std::vector<Line> &kls2,
    std::vector<uchar> &status,
    bool inverse,
    double graydoor)
{
    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};
    param = std::vector<std::vector<double>> (kls1.size(), std::vector<double>(3));

    // create pyramids
    std::vector<cv::Mat> img1_pyr;
    std::vector<cv::Mat> img2_pyr;
    img1_pyr.push_back(image1);
    img2_pyr.push_back(image2);

    for (int i = 1; i < pyramids; i++)
    {
        cv::Mat img1, img2;
        cv::resize(img1_pyr[i - 1], img1, cv::Size(img1_pyr[i - 1].cols * pyramid_scale, img1_pyr[i - 1].rows * pyramid_scale));
        cv::resize(img2_pyr[i - 1], img2, cv::Size(img2_pyr[i - 1].cols * pyramid_scale, img2_pyr[i - 1].rows * pyramid_scale));
        img1_pyr.push_back(img1);
        img2_pyr.push_back(img2);
    }
    // coarse-to-fine LK tracking in pyramids
    std::vector<Line> kl1_pyr, kl2_pyr;
    for (auto kl : kls1)
    {
        for (size_t i = 0; i < kl.key_points.size(); i++)
        {
            kl.key_points[i] *= scales[pyramids - 1];
        }
        kl1_pyr.push_back(kl);
        kl2_pyr.push_back(kl);
    }

    status = std::vector<uchar>(kl2_pyr.size(), 1);
    for (int level = pyramids - 1; level >= 0; level--)
    {
        // from coarse to fine
        // status.assign(kl2_pyr.size(), 1);

        OpticalFlowSingleLevel(img1_pyr[level], img2_pyr[level], kl1_pyr, kl2_pyr, status, false, level, graydoor);

        if (level > 0)
        {
            for (auto &kl : kl1_pyr)
                for (auto &line_pt : kl.key_points)
                    line_pt /= pyramid_scale;

            for (auto &kl : kl2_pyr)
                for (auto &line_pt : kl.key_points)
                    line_pt /= pyramid_scale;

            for (auto &c : param)
                for (int i = 0; i < 2; i++)
                    c[i] /= pyramid_scale;
        }
    }

    /* save the succssfully tracked lines   */
    kls2 = kl2_pyr;
}


void LineOpticalFlowTracker::OpticalFlowSingleLevel(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const std::vector<Line> &kls1,
    std::vector<Line> &kls2,
    std::vector<uchar> &status,
    bool inverse, 
    int layer,
    double graydoor)
{
    kls2.resize(kls1.size());
    status.resize(kls1.size());
    LineOpticalFlowTracker tracker(img1, img2, kls1, kls2, status, inverse, layer, graydoor);
    cv::parallel_for_(cv::Range(0, kls1.size()),
        std::bind(&LineOpticalFlowTracker::calculateOpticalFlow, &tracker, std::placeholders::_1));
}


void LineOpticalFlowTracker::calculateOpticalFlow(const cv::Range &range)
{

    /**********************************************************************************************************************************************
        u'_n = u_n + g1 + l_n * cos(\alpha + g3) - l_n * cos(\alpha)
        v'_n = v_n + g2 + l_n * sin(\alpha + g3) - l_n * sin(\alpha)
        ==>
        u'_n = u_n + g1 + l_n * [cos(\alpha) * cos(g3) - sin(\alpha) * sin(g3)] - l_n * cos(\alpha)
        v'_n = v_n + g2 + l_n * [sin(\alpha) * cos(g3) + cos(\alpha) * sin(g3)] - l_n * sin(\alpha)
        let g3 --> 0, get
        u'_n = u_n + g1 + l_n * [cos(\alpha) - sin(\alpha) * g3] - l_n * cos(\alpha)
        v'_n = v_n + g2 + l_n * [sin(\alpha) + cos(\alpha) * g3] - l_n * sin(\alpha)
        ==>
        u'_n = u_n + g1 - l_n * sin(\alpha) * g3
        v'_n = v_n + g2 + l_n * cos(\alpha) * g3
        ==>
        u'_n = u_n + g1 - (v_n - v_1) * g3
        v'_n = v_n + g2 + (u_n - u_1) * g3

        Besides, the jacobian can be calculated
        d(u) / d(g1) = 1, d(u) / d(g2) = 0,
        d(v) / d(g1) = 0, d(v) / d(g2) = 1,
        d(u) / d(g3) = - l_n * cos(\alpha) * sin(g3) - l_n * sin(\alpha) * cos(g3) = - (u_n - u_1) * sin(g3) - (v_n - v_1) * cos(g3)
        d(u) / d(g3) = - l_n * sin(\alpha) * sin(g3) + l_n * cos(\alpha) * cos(g3) = - (v_n - v_1) * sin(g3) + (u_n - u_1) * cos(g3)
        let g3 --> 0, get
        d(u_n) / d(g3) = - (u_n - u_1) * g3 - (v_n - v_1)
        d(u_n) / d(g3) = - (v_n - v_1) * g3 + (u_n - u_1)
    **********************************************************************************************************************************************/

    int half_patch_size = 3; // 4
    int iterations = 10; // 5

    // for each line
    for (int i = range.start; i < range.end; i++)
    {
        /* only perform LOFT for the successfully tracked lines */
        if (status[i] == 0)
            continue;

        std::vector<cv::Point2f> kps = kls1[i].key_points;

        double &g1 = param[i][0], &g2 = param[i][1], &g3 = param[i][2];

        double cost = 0, lastCost = 0;
        int succ = 1; // indicate if this point succeeded
        double errorGray = 0;

        const cv::Point2f &anchor_pt = kps[0];

        /* Gauss-Newton iterations */
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero(); // hessian
        Eigen::Vector3d b = Eigen::Vector3d::Zero(); // bias
        Eigen::Vector3d J;
        for (int iter = 0; iter < iterations; iter++)
        {
            if (inverse == false)
            {
                H = Eigen::Matrix3d::Zero();
                b = Eigen::Vector3d::Zero();
            }
            else
            {
                /* only reset b */
                b = Eigen::Vector3d::Zero();
            }

            cost = 0;
            errorGray = 0;

            /* compute cost and jacobian for each point in this line */
            for (size_t j = 0; j < kps.size(); j++)
            {
                double gray1 = 0, gray2 = 0;

                double dx = g1 - (kps[j].y - anchor_pt.y) * g3;
                double dy = g2 + (kps[j].x - anchor_pt.x) * g3;


                for (int x = -half_patch_size; x <= half_patch_size; x++)
                {
                    for (int y = -half_patch_size; y <= half_patch_size; y++)
                    {
                        double gray01 = GetPixelValue(img1, kps[j].x + x, kps[j].y + y);
                        double gray02 = GetPixelValue(img2, kps[j].x + x + dx, kps[j].y + y + dy);
                        double error = gray01 - gray02;

                        gray1 += gray01;
                        gray2 += gray02;

                        /* Jacobian */
                        if (inverse == false)
                        {
                            Eigen::Vector2d J0 = -1.0 * Eigen::Vector2d(
                                0.5 * (GetPixelValue(img2, kps[j].x + dx + x + 1, kps[j].y + dy + y) -
                                        GetPixelValue(img2, kps[j].x + dx + x - 1, kps[j].y + dy + y)),
                                0.5 * (GetPixelValue(img2, kps[j].x + dx + x, kps[j].y + dy + y + 1) -
                                        GetPixelValue(img2, kps[j].x + dx + x, kps[j].y + dy + y - 1)));
                            Eigen::Matrix<double, 2, 3> J1;
                            J1 << 
                                1.0, 0.0, - (kps[j].x - anchor_pt.x) * g3 - (kps[j].y - anchor_pt.y),
                                0.0, 1.0, - (kps[j].y - anchor_pt.y) * g3 + (kps[j].x - anchor_pt.x);
                            J = J0.transpose() * J1;
                        }
                        else if (iter == 0)
                        {
                            Eigen::Vector2d J0 = -1.0 * Eigen::Vector2d(
                                0.5 * (GetPixelValue(img2, kps[j].x + dx + x + 1, kps[j].y + dy + y) -
                                        GetPixelValue(img2, kps[j].x + dx + x - 1, kps[j].y + dy + y)),
                                0.5 * (GetPixelValue(img2, kps[j].x + dx + x, kps[j].y + dy + y + 1) -
                                        GetPixelValue(img2, kps[j].x + dx + x, kps[j].y + dy + y - 1)));
                            Eigen::Matrix<double, 2, 3> J1;
                            J1 << 
                                1.0, 0.0, - (kps[j].x - anchor_pt.x) * g3 - (kps[j].y - anchor_pt.y),
                                0.0, 1.0, - (kps[j].y - anchor_pt.y) * g3 + (kps[j].x - anchor_pt.x);
                            J = J0.transpose() * J1;
                        }

                        /* compute H, b and set cost */
                        b += -error * J;
                        // cost += huberloss(error, 1) * 2;
                        cost += error * error;
                        
                        if (inverse == false || iter == 0)
                        {
                            /* also update H */
                            H += J * J.transpose();
                        }
                    }
                }
                errorGray += abs(gray1 - gray2) / ((2 * half_patch_size + 1) * (2 * half_patch_size + 1));
            }

            /* compute update */
            Eigen::Vector3d update = H.ldlt().solve(b);

            if (std::isnan(update[0]) || std::isnan(update[1]) || std::isnan(update[2]) ||
                std::isinf(update[0]) || std::isinf(update[1]) || std::isinf(update[2]))
            {
                succ = 0;
                ROS_WARN("fail line \n");
                break;
            }

            if (iter > 0 && cost > lastCost)
                break;

            /* update dx, dy */
            g1 += update[0];
            g2 += update[1];
            g3 += update[2];

            lastCost = cost;
            succ = 1;
            /* converge */
            if (update.norm() < 1e-2)
                break;
        }
        cost /= ((2 * half_patch_size + 1) * (2 * half_patch_size + 1));

        // ROS_WARN("Error:%f  AvrError:%f", cost, errorGray);
        if (layer == 0 && errorGray >= graydoor * kps.size())
            succ = 0;

        // set keylines2
        for (size_t j = 0; j < kps.size() && succ != 0; j++)
        {
            cv::Point2f dp;
            dp.x = g1 - (kps[j].y - anchor_pt.y) * g3;
            dp.y = g2 + (kps[j].x - anchor_pt.x) * g3;
            cv::Point2f pt = kps[j] + dp;

            if (j == 0 && !utils::inside(img1, pt))
                succ = 0;

            kls2[i].key_points[j] = pt;
        }

        status[i] = succ;
    }
}


/** \brief Utilizing the bilinear interpolation to get the pixel value
 * @param img the raw image
 * @param x the x coordinate
 * @param y the y coordinate
 * \return the pixel value
 */
float LineOpticalFlowTracker::GetPixelValue(const cv::Mat &img, float x, float y)
{
    // boundary check
    if (x < 0)
        x = 0;
    if (y < 0)
        y = 0;
    if (x >= img.cols - 1)
        x = img.cols - 2;
    if (y >= img.rows - 1)
        y = img.rows - 2;

    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);

    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x) + xx * (1 - yy) * img.at<uchar>(y, x_a1) + (1 - xx) * yy * img.at<uchar>(y_a1, x) + xx * yy * img.at<uchar>(y_a1, x_a1);
}

