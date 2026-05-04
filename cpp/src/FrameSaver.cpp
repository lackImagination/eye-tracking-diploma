// FrameSaver.cpp
#include "../connect/FrameSaver.h"
#include <sstream>
#include <iostream>

namespace fs = std::filesystem;

FrameSaver::FrameSaver(std::string baseDir)
        : baseDirectory(std::move(baseDir)), imageCounter(0) {
    init();
}

void FrameSaver::init() {
    imagesPath = baseDirectory + "/images";
    coordsPath = baseDirectory + "/coords.csv";

    try {
        fs::create_directories(imagesPath);
    } catch (const fs::filesystem_error& e) {
        std::cerr << "[ERROR] Не удалось создать директории: " << e.what() << std::endl;
    }

    imageCounter = getLastImageIndex();
}

int FrameSaver::getLastImageIndex() {
    int maxIndex = 0;
    if (!fs::exists(imagesPath)) return 0;

    for (const auto& entry : fs::directory_iterator(imagesPath)) {
        std::string filename = entry.path().filename().string();
        if (filename.rfind("eye_", 0) == 0 && filename.find(".png") != std::string::npos) {
            try {
                int idx = std::stoi(filename.substr(4, filename.find(".png") - 4));
                if (idx >= maxIndex) maxIndex = idx + 1;
            } catch (...) {
                // Просто пропускаем файл, если не парсится
            }
        }
    }
    return maxIndex;
}

bool FrameSaver::save(const cv::Mat& frame, const cv::Point& gazePoint) {
    if (frame.empty()) {
        std::cerr << "[ERROR] Пустой кадр, не сохраняем\n";
        return false;
    }

    std::ostringstream filename;
    filename << "eye_" << std::setfill('0') << std::setw(4) << imageCounter << ".png";
    std::string imagePath = imagesPath + "/" + filename.str();

    if (!cv::haveImageWriter(".png")) {
        std::cerr << "[ERROR] OpenCV не поддерживает сохранение .png!\n";
        return false;
    }

    if (cv::imwrite(imagePath, frame)) {
        std::ofstream out(this->coordsPath, std::ios::app);
        if (out.is_open()) {
            out << filename.str() << "," << gazePoint.x << "," << gazePoint.y << "\n";
            std::cout << "[INFO] Сохранили: " << filename.str()
                      << " (" << gazePoint.x << "," << gazePoint.y << ")\n";
            imageCounter++;
            return true;
        } else {
            std::cerr << "[ERROR] Не удалось открыть файл меток: " << this->labelsPath << std::endl;
            std::cerr << "[HINT] Убедись, что путь существует и доступен для записи.\n";
        }
    } else {
        std::cerr << "[ERROR] Не удалось сохранить изображение!\n";
        std::cerr << "[HINT] Возможно, неверный путь или проблема с правами доступа.\n";
    }

    return false;
}
