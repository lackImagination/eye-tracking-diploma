// FaceTracker.cpp
#include "../connect/FaceTracker.h"
#include "../connect/FrameSaver.h"
#include "../connect/Preprocessing.h"

using namespace cv;
using namespace cv::face;
using namespace cv::dnn;
namespace fs = std::filesystem;


FaceTracker::FaceTracker() = default;

FaceTracker::~FaceTracker() {
    stopTracking();
    if (cap.isOpened()) {
        cap.release();
    }
}

bool FaceTracker::initDefault() {
    std::string proto = "data/models/face_detector/deploy.prototxt";
    std::string model = "data/models/face_detector/res10_300x300_ssd_iter_140000.caffemodel";
    std::string facemark = "opencv/data/lbfmodel.yaml";

    // Проверим, что файлы существуют
    if (!fs::exists(proto)) {
        std::cerr << "[ERROR] Prototxt file not found: " << proto << std::endl;
        return false;
    }
    if (!fs::exists(model)) {
        std::cerr << "[ERROR] Caffe model not found: " << model << std::endl;
        return false;
    }
    if (!fs::exists(facemark)) {
        std::cerr << "[ERROR] Facemark model not found: " << facemark << std::endl;
        return false;
    }

    std::cout << "[INFO] Initializing FaceTracker with default models..." << std::endl;
    return init(proto, model, facemark);
}


bool FaceTracker::init(const std::string& prototxtPath, const std::string& modelPath, const std::string& facemarkModel) {
    try {
        faceNet = readNetFromCaffe(prototxtPath, modelPath);
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading DNN model: " << e.what() << std::endl;
        return false;
    }

    auto facemarkLBF = FacemarkLBF::create();
    try {
        facemarkLBF->loadModel(facemarkModel);
        facemarkLBF->setFaceDetector([](cv::InputArray, cv::OutputArray, void*) { return true; });
        facemark = facemarkLBF;
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading face mark: " << e.what() << std::endl;
        return false;
    }

    return true;
}

std::vector<Rect> FaceTracker::detectFacesDNN(Mat& frame) {
    std::vector<Rect> faces;
    if (frame.empty()) return faces;

    Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 177, 123));
    faceNet.setInput(blob);
    Mat detections = faceNet.forward();

    Mat detMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());
    for (int i = 0; i < detMat.rows; ++i) {
        float confidence = detMat.at<float>(i, 2);
        if (confidence > 0.3) {
            int x1 = static_cast<int>(detMat.at<float>(i, 3) * frame.cols);
            int y1 = static_cast<int>(detMat.at<float>(i, 4) * frame.rows);
            int x2 = static_cast<int>(detMat.at<float>(i, 5) * frame.cols);
            int y2 = static_cast<int>(detMat.at<float>(i, 6) * frame.rows);
            Rect faceBox(Point(x1, y1), Point(x2, y2));
            faces.push_back(faceBox & Rect(0, 0, frame.cols, frame.rows));
        }
    }

    return faces;
}

float FaceTracker::getHeadAngle(const std::vector<Point2f>& landmarks) {
    Point2f l = (landmarks[36] + landmarks[39]) * 0.5f;
    Point2f r = (landmarks[42] + landmarks[45]) * 0.5f;
    return atan2(r.y - l.y, r.x - l.x) * 180.0f / CV_PI;
}

std::vector<Point2f> FaceTracker::rotateLandmarks(const std::vector<Point2f>& landmarks, const Mat& rotMat) {
    std::vector<Point2f> rotated;
    transform(landmarks, rotated, rotMat);
    return rotated;
}

Rect FaceTracker::getEyesRect(const std::vector<Point2f>& landmarks, const Size& size) {
    Point2f l = (landmarks[36] + landmarks[39]) * 0.5f;
    Point2f r = (landmarks[42] + landmarks[45]) * 0.5f;
    Point2f mid = (l + r) * 0.5f;
    float dist = norm(l - r);
    Size roiSize(dist * 1.5f, dist);
    Point2f tl = mid - Point2f(roiSize.width / 2, roiSize.height * 0.6f);
    Rect rect(cvRound(tl.x), cvRound(tl.y), roiSize.width, roiSize.height);
    return rect & Rect(0, 0, size.width, size.height);
}

cv::Rect FaceTracker::detectMainFace(cv::Mat& frame) {
    std::vector<cv::Rect> faces;
    if (frame.empty()) return {};

    cv::Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 177, 123));
    faceNet.setInput(blob);
    cv::Mat detections = faceNet.forward();

    cv::Mat detMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());
    std::vector<std::pair<float, cv::Rect>> candidates;

    for (int i = 0; i < detMat.rows; ++i) {
        float confidence = detMat.at<float>(i, 2);
        if (confidence > 0.3f) {
            int x1 = static_cast<int>(detMat.at<float>(i, 3) * frame.cols);
            int y1 = static_cast<int>(detMat.at<float>(i, 4) * frame.rows);
            int x2 = static_cast<int>(detMat.at<float>(i, 5) * frame.cols);
            int y2 = static_cast<int>(detMat.at<float>(i, 6) * frame.rows);
            cv::Rect box(cv::Point(x1, y1), cv::Point(x2, y2));
            box &= cv::Rect(0, 0, frame.cols, frame.rows);
            if (box.area() > 10000) {
                candidates.emplace_back(confidence, box);
            }
        }
    }

    if (candidates.empty()) {
        std::lock_guard<std::mutex> lock(faceMutex);
        lastFaceBox = {}; // сбрасываем кеш
        return {};
    }

    // Выбираем лицо ближе к центру
    cv::Point2f center(frame.cols / 2.0f, frame.rows / 2.0f);
    std::sort(candidates.begin(), candidates.end(), [&](auto& a, auto& b) {
        cv::Point2f ac = 0.5f * (cv::Point2f(a.second.tl()) + cv::Point2f(a.second.br()));
        cv::Point2f bc = 0.5f * (cv::Point2f(b.second.tl()) + cv::Point2f(b.second.br()));
        return cv::norm(ac - center) < cv::norm(bc - center);
    });

    cv::Rect best = candidates.front().second;

    // Проверка на устойчивость (если предыдущая рамка похожа — оставляем её)
    {
        std::lock_guard<std::mutex> lock(faceMutex);
        if (!lastFaceBox.empty()) {
            float iou = (best & lastFaceBox).area() / float((best | lastFaceBox).area());
            if (iou > 0.5f) {
                best = lastFaceBox; // сохраняем стабильность
            } else {
                lastFaceBox = best;
            }
        } else {
            lastFaceBox = best;
        }
    }

    return best;
}


bool FaceTracker::getProcessedFrame(Mat& result) {
    if (!cap.isOpened()) return false;
    Mat frame;
    cap >> frame;
    if (frame.empty()) return false;

    flip(frame, frame, 1);
    auto face = detectMainFace(frame);
    std::vector<cv::Rect> faces;
    if (!face.empty()) faces.push_back(face);
    faceDetected = false;

    std::vector<std::vector<Point2f>> landmarks;
    if (faces.empty() || !facemark->fit(frame, faces, landmarks)) return false;
    if (landmarks[0].size() < 46) return false;

    faceDetected = true;

    float angle = getHeadAngle(landmarks[0]);
    Mat rotMat = getRotationMatrix2D(Point2f(frame.cols / 2.0f, frame.rows / 2.0f), angle, 1.0);
    Mat aligned;
    warpAffine(frame, aligned, rotMat, frame.size());

    auto rotated = rotateLandmarks(landmarks[0], rotMat);
    Rect roi = getEyesRect(rotated, aligned.size());

    if (roi.area() > 0) {
        result = aligned(roi).clone();
        return !result.empty();
    }

    return false;
}

cv::Mat FaceTracker::getLatestFrame() {
    std::lock_guard<std::mutex> lock(frameMutex);
    return latestFrame.clone();
}

bool FaceTracker::isFaceDetected() const {
    std::lock_guard<std::mutex> lock(frameMutex);
    return faceDetected;
}

void FaceTracker::startTracking() {
    if (trackingStarted) return;

    if (!cap.isOpened()) {
        cap.open(0);
        cap.set(CAP_PROP_FRAME_WIDTH, videoWidth);
        cap.set(CAP_PROP_FRAME_HEIGHT, videoHeight);
    }
    trackingStarted = true;
    trackingThread = std::thread(&FaceTracker::trackingLoop, this);
}

void FaceTracker::stopTracking() {
    trackingStarted = false;
    if (trackingThread.joinable()) {
        trackingThread.join();
    }
    cap.release();
}

void FaceTracker::trackingLoop() {
    while (trackingStarted) {
        Mat processed;
        if (getProcessedFrame(processed)) {
            std::lock_guard<std::mutex> lock(frameMutex);
            latestFrame = processed.clone();
            faceDetected = !processed.empty();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
}

void FaceTracker::saveCurrentFrame(FrameSaver& saver, const Point& gazePoint) {
    // std::cout << "[INFO] FaceTracker::saveCurrentFrame.\n";
    std::lock_guard<std::mutex> lock(frameMutex);
    if (faceDetected && !latestFrame.empty()) {
        saver.save(latestFrame, gazePoint);
    } else {
        std::cout << "[WARNING] Frame not saved — no face detected.\n";
    }
}
