#include "../connect/Preprocessing.h"

std::vector<float> preprocessImage(const cv::Mat& input) {

    cv::Mat resized, floatImage, rgbImage;

    // 1. Resize to 224x224
    cv::resize(input, resized, cv::Size(224, 224));

    // 2. Convert BGR to RGB
    cv::cvtColor(resized, rgbImage, cv::COLOR_BGR2RGB);

    // 3. Convert to float32 and normalize
    rgbImage.convertTo(floatImage, CV_32F, 1.0 / 255.0);

    // 4. Mean/Std normalization
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};

    std::vector<float> tensor;
    tensor.reserve(3 * 224 * 224);

    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < 224; ++y) {
            for (int x = 0; x < 224; ++x) {
                float val = floatImage.at<cv::Vec3f>(y, x)[c];
                val = (val - mean[c]) / std[c];
                tensor.push_back(val);
            }
        }
    }

    return tensor;
}