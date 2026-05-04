// CalibrationWindow.cpp
#include "../connect/CalibrationWindow.h"
#include <FL/fl_draw.H>
#include <iostream>
#include <zmq.hpp>

bool space_pressed = false;

// В main.cpp передавай путь как аргумент или читай из конфига
// Для простоты — путь относительно рабочей директории:
CalibrationWindow::CalibrationWindow(FaceTracker& tracker, GazeData* gazeData, const std::string& datasetPath)
        : tracker(tracker), saver(datasetPath), gazeData(gazeData) {}


CalibrationWindow::CameraBox::CameraBox(int X, int Y, int W, int H, FrameSaver& s, FaceTracker& t, GazeData* g)
        : Fl_Box(X, Y, W, H), saver(s), tracker(t), gazeData(g) {
    box(FL_FLAT_BOX);
    color(FL_BLACK);
    take_focus();
}

void CalibrationWindow::CameraBox::updateImage(const cv::Mat& img) {
    if (img.empty()) return;
    frameRGB = img.clone();
    Fl::lock();
    redraw();
    Fl::unlock();
}

void CalibrationWindow::CameraBox::draw() {
    fl_color(FL_BLACK);
    fl_rectf(x(), y(), w(), h());

    if (!frameRGB.empty()) {
        cv::Mat resized;
        cv::resize(frameRGB, resized, cv::Size(w(), h()));
        if (!resized.isContinuous()) resized = resized.clone();
        // fl_draw_image(resized.data, x(), y(), resized.cols, resized.rows, 3);
        static std::unique_ptr<Fl_RGB_Image> image;
        image = std::make_unique<Fl_RGB_Image>(resized.data, resized.cols, resized.rows, 3);
        image->draw(x(), y());
    }
}

int CalibrationWindow::CameraBox::handle(int event) {
    if (event == FL_FOCUS) return 1;

    if (event == FL_KEYDOWN && Fl::event_key() == ' ') {
        if (!space_pressed) {
            space_pressed = true;
            int mx = Fl::event_x();
            int my = Fl::event_y();
            float gx = gazeData->x.load();
            float gy = gazeData->y.load();

            double distance = std::hypot(mx - gx, my - gy);
            if (distance > 300) {
                std::cout << "[WARN] Пользователь смотрит в другую сторону. Координаты не сохраняются.\n";
                return 1;
            }
            tracker.saveCurrentFrame(saver, cv::Point(mx, my));
            std::cout << "[INFO] Сохранили точку: (" << mx << ", " << my << ")\n";
        }
        return 1;
    }

    if (event == FL_KEYUP && Fl::event_key() == ' ') {
        space_pressed = false;
        return 1;
    }

    return Fl_Box::handle(event);
}

void CalibrationWindow::update_frame(void* data) {
    auto* box = static_cast<CameraBox*>(data);
    static cv::Mat lastValidFrame;

    cv::Mat frame;
    if (box->tracker.isFaceDetected()) {
        frame = box->tracker.getLatestFrame();
        if (!frame.empty()) lastValidFrame = frame.clone();
    } else {
        frame = lastValidFrame.empty() ? cv::Mat::zeros(250, 250, CV_8UC3) : lastValidFrame.clone();
    }

    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    box->updateImage(rgb);

    Fl::repeat_timeout(1.0 / 60.0, update_frame, box);
}

void update_overlay_frame(void* data) {
    auto* overlay = static_cast<HeatmapOverlay*>(data);
    overlay->damage(FL_DAMAGE_ALL);
    overlay->redraw();
    Fl::repeat_timeout(1.0 / 60.0, update_overlay_frame, data);
}

void sendExitCommand() {
    try {
        zmq::context_t context(1);
        zmq::socket_t socket(context, zmq::socket_type::push);
        socket.connect("tcp://localhost:6000");

        uint8_t cmd = static_cast<uint8_t>(CommandType::STOP_CALIBRATION);
        zmq::message_t msg(&cmd, 1);
        socket.send(msg, zmq::send_flags::none);

    } catch (const zmq::error_t& e) {
        std::cerr << "[ZMQ] Ошибка отправки STOP_CALIBRATION: " << e.what() << std::endl;
    }
}

void CalibrationWindow::run() {
    Fl::background(35, 35, 34);
    int x, y, w, h;
    Fl::screen_work_area(x, y, w, h);

    auto* window = new Fl_Double_Window(x, y, w, h, "Калибровка");
    auto* predictionOverlay = new PredictionPointOverlay(0, 0, w, h, gazeData);

    auto* box = new CameraBox(0, 0, 200, 200, saver, tracker, gazeData);
    auto* filler = new Fl_Box(box->w(), 0, w - box->w(), h);
    filler->box(FL_FLAT_BOX);

    auto* heatmapOverlay = new HeatmapOverlay(0, 0, w, h);
    heatmapOverlay->gazeData = gazeData;
    heatmapOverlay->imageWidth = 1920;
    heatmapOverlay->imageHeight = 1080;

    auto* drawHeatmapButton = new MouseOnlyButton(0, box->h(), box->w(), 40, "Нарисовать тепловую карту");
    drawHeatmapButton->color(fl_rgb_color(128, 128, 128));
    drawHeatmapButton->labelcolor(fl_rgb_color(0, 0, 0));
    drawHeatmapButton->callback([](Fl_Widget*, void* userData) {
        auto* overlay = static_cast<HeatmapOverlay*>(userData);
        std::string filePath = "./dataset/coords.csv";
        overlay->points = loadPointsFromCSV(filePath);
        overlay->visible = !overlay->points.empty();
        overlay->redraw();
    }, heatmapOverlay);

    auto* clearHeatmapButton = new MouseOnlyButton(0, box->h() + 40, box->w(), 40, "Очистить тепловую карту");
    clearHeatmapButton->color(fl_rgb_color(128, 128, 128));
    clearHeatmapButton->labelcolor(fl_rgb_color(0, 0, 0));
    clearHeatmapButton->callback([](Fl_Widget*, void* userData) {
        auto* overlay = static_cast<HeatmapOverlay*>(userData);
        overlay->points.clear();
        overlay->visible = false;
        overlay->damage(FL_DAMAGE_ALL);
        if (overlay->window()) overlay->window()->redraw();
    }, heatmapOverlay);

    auto* trainModelButton = new MouseOnlyButton(0, box->h() + 80, box->w(), 40, "Обучить модель");
    trainModelButton->color(fl_rgb_color(128, 128, 128));
    trainModelButton->labelcolor(fl_rgb_color(0, 0, 0));
    trainModelButton->callback([](Fl_Widget*, void*) {
        std::cout << "[INFO] Нажата кнопка 'Обучить модель' (пока без действий)\n";
    });

    auto* exitButton = new MouseOnlyButton(0, box->h() + 120, box->w(), 40, "Завершить");
    exitButton->color(fl_rgb_color(200, 80, 80));
    exitButton->labelcolor(FL_WHITE);
    exitButton->callback([](Fl_Widget*, void* userData) {
        auto* win = static_cast<Fl_Window*>(userData);
        sendExitCommand();
        std::cout << "[INFO] Закрытие окна калибровки\n";
        win->hide();
    }, window);


    window->resizable(filler);

    window->begin();
    window->add(predictionOverlay);
    window->add(box);                  // Камера
    window->add(filler);
    window->add(heatmapOverlay);      // Тепловая карта
    window->add(drawHeatmapButton);
    window->add(clearHeatmapButton);
    window->add(trainModelButton);
    window->add(exitButton);
    window->end();

    window->show();
    box->take_focus();

    Fl::add_timeout(1.0 / 60.0, update_frame, box);
    Fl::add_timeout(1.0 / 60.0, update_overlay_frame, heatmapOverlay);
    Fl::add_timeout(1.0 / 60.0, update_overlay_frame, drawHeatmapButton);
    Fl::add_timeout(1.0 / 60.0, update_overlay_frame, clearHeatmapButton);
    Fl::add_timeout(1.0 / 60.0, update_overlay_frame, trainModelButton);
    Fl::add_timeout(1.0 / 60.0, update_overlay_frame, exitButton);
    predictionOverlay->start();
    Fl::run();
}

