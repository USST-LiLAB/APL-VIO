#ifndef __LINE_OPTICAL_FLOW_TRACKER_H
#define __LINE_OPTICAL_FLOW_TRACKER_H

#include <vector>
#include <opencv2/opencv.hpp>


struct Line;

class LineOpticalFlowTracker
{
public:
    LineOpticalFlowTracker(
        const cv::Mat &img1_,
        const cv::Mat &img2_,
        const std::vector<Line> &kls1_,
        std::vector<Line> &kls2_,
        std::vector<uchar> &status_,
        bool inverse_ = true, 
        int layer_ = 1,
        double graydoor_ = 5.0) : 
        img1(img1_), img2(img2_), kls1(kls1_), kls2(kls2_), 
        status(status_), inverse(inverse_), layer(layer_), graydoor(graydoor_) {}

    void calculateOpticalFlow(const cv::Range &range);
    float GetPixelValue(const cv::Mat &img, float x, float y);

    static void OpticalFlowSingleLevel(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const std::vector<Line> &kls1,
        std::vector<Line> &kls2,
        std::vector<uchar> &success,
        bool inverse = false,
        int layer_ = 0,
        double graydoor = 5.0);

    static void OpticalFlowMultiLevel(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const std::vector<Line> &kls1,
        std::vector<Line> &kls2,
        std::vector<uchar> &success,
        bool inverse = false,
        double graydoor = 5.0);


private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const std::vector<Line> &kls1;
    std::vector<Line> &kls2;
    std::vector<uchar> &status;
    bool inverse;
    int layer;
    double graydoor;
    static std::vector<std::vector<double>> param;
};

#endif
