// FrameSaver.h
#ifndef FRAME_SAVER_H
#define FRAME_SAVER_H

#include <string>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>

class FrameSaver {
public:
    explicit FrameSaver(std::string baseDir);
    bool save(const cv::Mat& frame, const cv::Point& gazePoint);

private:
    std::string baseDirectory;
    std::string imagesPath;
    std::string coordsPath;
    std::string labelsPath;
    int imageCounter;

    void init();
    int getLastImageIndex();
};

#endif // FRAME_SAVER_H

