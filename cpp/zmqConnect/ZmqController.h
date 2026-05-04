// ZmqController.h
#pragma once
#include "CommandType.h"
#include "../connect/FaceTracker.h"
#include "../connect/CalibrationWindow.h"
#include "zmq.hpp"
#include <thread>
#include <atomic>

class ZmqController {
public:
    explicit ZmqController(FaceTracker& tracker);
    ~ZmqController();

    std::atomic<bool> startCalibrationFlag{false};
    std::atomic<bool> exitFlag{false};

    void start();
    void stop();

private:
    void listenLoop();

    FaceTracker& tracker;
    zmq::context_t context;
    zmq::socket_t socket;
    std::thread listenerThread;
    std::atomic<bool> running;
};
