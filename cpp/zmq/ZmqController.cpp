//ZmqController.cpp
#include "../zmqConnect/ZmqController.h"
#include <iostream>

ZmqController::ZmqController(FaceTracker& tracker)
        : tracker(tracker), context(1), socket(context, zmq::socket_type::pull), running(false) {
    socket.bind("tcp://*:6005");
}

ZmqController::~ZmqController() {
    stop();
}

void ZmqController::start() {
    running = true;
    listenerThread = std::thread(&ZmqController::listenLoop, this);
}

void ZmqController::stop() {
    running = false;
    if (listenerThread.joinable()) {
        listenerThread.join();
    }
}

void ZmqController::listenLoop() {
    std::cout << "ZMQ controller started. Waiting for commands...\n";
    while (running) {
        zmq::message_t message;
        if (socket.recv(message, zmq::recv_flags::none)) {
            if (message.size() == 1) {
                CommandType cmd = static_cast<CommandType>(*static_cast<uint8_t*>(message.data()));
                switch (cmd) {
                    case CommandType::START_CALIBRATION:
                        std::cout << "[ZMQ] START_CALIBRATION 111\n";
                        startCalibrationFlag = true;  // просто ставим флаг
                        break;

                    case CommandType::EXIT:
                        std::cout << "[ZMQ] EXIT received 111 \n";
                        exitFlag = true; // сигнал для выхода из main()
                        break;
                    default:
                        std::cout << "[ZMQ] Unknown command: " << static_cast<int>(cmd) << "\n";
                        break;
                }
            }
        }
    }
}
