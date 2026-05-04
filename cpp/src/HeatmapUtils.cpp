//HeatmapUtils.cpp
#include "../ui/HeatmapUtils.h"
#include <fstream>
#include <sstream>
#include <iostream>

std::vector<cv::Point> loadPointsFromCSV(const std::string& filename) {
    std::vector<cv::Point> points;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Не удалось открыть: " << filename << "\n";
        return points;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string img, x_str, y_str;

        if (std::getline(ss, img, ',') &&
            std::getline(ss, x_str, ',') &&
            std::getline(ss, y_str)) {
            try {
                points.emplace_back(std::stoi(x_str), std::stoi(y_str));
            } catch (...) {
                std::cerr << "[ERROR] Невозможно преобразовать координаты: " << line << "\n";
            }
        }
    }

    return points;
}
