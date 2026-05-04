//ZmqCommandReceiver.h
#pragma once
#include "zmq.hpp"
#include <thread>
#include <atomic>
#include "CommandType.h"
#include "../connect/FaceTracker.h"
#include "ZmqStreamer.h"

class ZmqCommandReceiver {
public:
    ZmqCommandReceiver(FaceTracker& tracker, ZmqStreamer& streamer);
    ~ZmqCommandReceiver();

    std::atomic<bool> startCalibrationFlag{false};
    std::atomic<bool> exitFlag{false};

    void start();
    void stop();

private:
    void listenLoop();

    FaceTracker& tracker;
    ZmqStreamer& streamer;

    zmq::context_t context;
    zmq::socket_t socket;
    std::atomic<bool> running;
    std::thread listenerThread;
};
