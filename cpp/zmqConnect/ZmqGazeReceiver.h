// ZmqGazeReceiver.h
#pragma once
#include "zmq.hpp"
#include <thread>
#include <atomic>
#include <functional>

class ZmqGazeReceiver {
public:
    ZmqGazeReceiver();
    ~ZmqGazeReceiver();

    void start();
    void stop();

    std::function<void(float x, float y)> onGazeReceived;

private:
    void listenLoop();

    zmq::context_t context;
    zmq::socket_t socket;
    std::atomic<bool> running;
    std::thread listenerThread;
};
