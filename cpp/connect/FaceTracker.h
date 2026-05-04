// FaceTracker.h
#ifndef FACE_TRACKER_H
#define FACE_TRACKER_H

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>

class FrameSaver; // forward declaration

class FaceTracker {
public:
    FaceTracker();
    ~FaceTracker();

    bool init(const std::string& prototxtPath,
              const std::string& modelPath,
              const std::string& faceMarkModel);

    bool initDefault();

    void startTracking();
    void stopTracking();

    cv::Mat getLatestFrame();
    [[nodiscard]] bool isFaceDetected() const;

    void saveCurrentFrame(FrameSaver& saver, const cv::Point& gazePoint);

    mutable std::mutex faceMutex;
    cv::Rect lastFaceBox;

    bool getProcessedFrame(cv::Mat& result);

private:
    cv::VideoCapture cap;
    cv::Ptr<cv::face::Facemark> facemark;
    cv::dnn::Net faceNet;

    int videoWidth = 640;
    int videoHeight = 360;
    bool trackingStarted = false;
    bool faceDetected = false;

    cv::Mat latestFrame;

    std::vector<std::vector<cv::Point2f>> landmarks;
    std::vector<cv::Rect> detectFacesDNN(cv::Mat& frame);

    static float getHeadAngle(const std::vector<cv::Point2f>& landmarks);
    static std::vector<cv::Point2f> rotateLandmarks(const std::vector<cv::Point2f>& landmarks, const cv::Mat& rotMat);
    static cv::Rect getEyesRect(const std::vector<cv::Point2f>& landmarks, const cv::Size& frameSize);

    void trackingLoop();

    std::thread trackingThread;
    mutable std::mutex frameMutex;

    cv::Rect detectMainFace(cv::Mat &frame);
};

#endif
