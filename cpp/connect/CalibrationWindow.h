// CalibrationWindow.h
#pragma once

#include <FL/Fl.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Double_Window.H>
#include <opencv2/opencv.hpp>
#include <memory>
#include "FaceTracker.h"
#include "FrameSaver.h"
#include "../ui/HeatmapOverlay.h"
#include "../ui/HeatmapUtils.h"
#include "../ui/MouseOnlyButton.h"
#include "../ui/PredictionPointOverlay.h"
#include "../zmqConnect/GazeData.h"
#include "../zmqConnect/CommandType.h"

class CalibrationWindow {
public:
    explicit CalibrationWindow(FaceTracker& tracker, GazeData* gazeData);
    void run();

private:
    FaceTracker& tracker;
    FrameSaver saver;
    GazeData* gazeData;

    class CameraBox : public Fl_Box {
    public:
        cv::Mat frameRGB;
        FrameSaver& saver;
        FaceTracker& tracker;
        GazeData* gazeData;

        CameraBox(int X, int Y, int W, int H, FrameSaver& s, FaceTracker& t, GazeData* g);
        void updateImage(const cv::Mat& img);
        void draw() override;
        int handle(int event) override;
    };


    static void update_frame(void* data);
};

