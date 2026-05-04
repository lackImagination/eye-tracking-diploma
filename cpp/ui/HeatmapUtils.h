//HeatmapUtils.h
#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <string>

std::vector<cv::Point> loadPointsFromCSV(const std::string& filename);
