//ZmqStreamer.cpp
#include "../zmqConnect/ZmqStreamer.h"
#include "../zmqConnect/ZmqSender.h"
#include "../connect/FaceTracker.h"
#include <opencv2/imgcodecs.hpp> // imencode
#include <thread>
#include <chrono>

ZmqStreamer::ZmqStreamer(FaceTracker& tracker)
        : tracker(tracker), context(1), socket(context, ZMQ_PUB), running(false)
{
    socket.bind("tcp://*:5555");  // Локальный сокет
}

ZmqStreamer::~ZmqStreamer() {
    stop();
}

void ZmqStreamer::start() {
    if (running) return;
    running = true;
    streamingThread = std::thread(&ZmqStreamer::streamingLoop, this);
}

void ZmqStreamer::stop() {
    running = false;
    if (streamingThread.joinable()) {
        streamingThread.join();
    }
}

void ZmqStreamer::streamingLoop() {
    while (running) {
        if (!tracker.isFaceDetected()) {
            // Отправляем 1 байт, сигнализирующий, что лицо не найдено
            uint8_t flag = 0x00;
            zmq::message_t msg(&flag, 1);
            socket.send(msg, zmq::send_flags::none);
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }

        cv::Mat frame = tracker.getLatestFrame();

        if (!frame.empty()) {
            std::vector<uchar> buffer;
            if (cv::imencode(".jpg", frame, buffer)) {
                zmq::message_t msg(buffer.size());
                memcpy(msg.data(), buffer.data(), buffer.size());
                socket.send(msg, zmq::send_flags::none);
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(30)); // ~30 fps
    }
}

void someScenarioTrigger() {
    ZmqSender sender;
    sender.sendCommand(CommandType::START_CALIBRATION);
}