//HeatmapOverlay.h
#pragma once
#include <FL/Fl_Box.H>
#include <opencv2/core.hpp>
#include <vector>
#include "../zmqConnect/GazeData.h"

class HeatmapOverlay : public Fl_Box {
public:
    std::vector<cv::Point> points;
    bool visible = false;
    int imageWidth = 2560;
    int imageHeight = 1600;

    GazeData* gazeData = nullptr;

    HeatmapOverlay(int X, int Y, int W, int H);
    void draw() override;
};

