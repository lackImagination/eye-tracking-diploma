// GazeForwarder.cpp
#include "../zmqConnect/GazeForwarder.h"
#include <cstring>
#include <iostream>

GazeForwarder::GazeForwarder()
        : context(1), socket(context, zmq::socket_type::push) {
    try {
        socket.bind("tcp://*:6003");  // Java будет подключаться сюда
        // std::cout << "[GazeForwarder] Bound to tcp://*:6003\n";
    } catch (const zmq::error_t& e) {
        std::cerr << "[GazeForwarder] Failed to bind: " << e.what() << "\n";
    }
}

GazeForwarder::~GazeForwarder() {
    socket.close();
    context.close();
}

void GazeForwarder::forward(float x, float y) {
    float data[2] = { x, y };
    zmq::message_t msg(sizeof(data));
    std::memcpy(msg.data(), data, sizeof(data));
    try {
        socket.send(msg, zmq::send_flags::dontwait);
    } catch (const zmq::error_t& e) {
        std::cerr << "[GazeForwarder] Send failed: " << e.what() << "\n";
    }
}
