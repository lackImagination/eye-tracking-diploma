// GazeData.h
#pragma once
#include <atomic>

struct GazeData {
    std::atomic<float> x;
    std::atomic<float> y;
};
