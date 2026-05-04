// GazeForwarder.h
#pragma once
#include "zmq.hpp"
#include <thread>
#include <atomic>

class GazeForwarder {
public:
    GazeForwarder();
    ~GazeForwarder();
    void forward(float x, float y);

private:
    zmq::context_t context;
    zmq::socket_t socket;
};
