//ZmqCommandReceiver.cpp
#include "../zmqConnect/ZmqCommandReceiver.h"
#include <iostream>

ZmqCommandReceiver::ZmqCommandReceiver(FaceTracker& tracker, ZmqStreamer& streamer)
        : tracker(tracker), streamer(streamer), context(1), socket(context, zmq::socket_type::pull), running(false) {
    socket.bind("tcp://*:6000");
}

ZmqCommandReceiver::~ZmqCommandReceiver() {
    stop();
}

void ZmqCommandReceiver::start() {
    if (running) return;
    running = true;
    listenerThread = std::thread(&ZmqCommandReceiver::listenLoop, this);
}

void ZmqCommandReceiver::stop() {
    running = false;
    if (listenerThread.joinable()) {
        listenerThread.join();
    }
    socket.close();
}

void ZmqCommandReceiver::listenLoop() {
    try {
        while (running) {
            zmq::message_t msg;
            if (!socket.recv(msg, zmq::recv_flags::none)) continue;

            if (msg.size() != 1) continue; // ожидаем 1 байт
            CommandType cmd = static_cast<CommandType>(*static_cast<uint8_t*>(msg.data()));

            switch (cmd) {
                case CommandType::START_TRACKING:
                    std::cout << "[ZMQ] START_TRACKING\n";
                    tracker.startTracking();
                    streamer.start();
                    break;

                case CommandType::STOP_TRACKING:
                    std::cout << "[ZMQ] STOP_TRACKING\n";
                    tracker.stopTracking();
                    streamer.stop();
                    break;

                case CommandType::START_CALIBRATION:
                    std::cout << "[ZMQ] START_CALIBRATION\n";
                    tracker.startTracking();
                    streamer.start();
                    startCalibrationFlag = true;
                    break;

                case CommandType::STOP_CALIBRATION:
                    std::cout << "[ZMQ] STOP_CALIBRATION\n";
                    tracker.stopTracking();
                    streamer.stop();
                    break;

                case CommandType::SAVE_FRAME:
                    std::cout << "[ZMQ] SAVE_FRAME\n";
                    break;

                case CommandType::EXIT:
                    std::cout << "[ZMQ] EXIT received\n";
                    exitFlag = true;
                    running = false;        // остановить цикл
                    socket.close();         // прервать блокирующий recv
                    break;
            }
        }
    } catch (const zmq::error_t& e) {
        std::cerr << "[ZMQ] Exception: " << e.what() << std::endl;
    }
}

