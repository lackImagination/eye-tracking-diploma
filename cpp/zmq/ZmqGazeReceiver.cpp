// ZmqGazeReceiver.cpp
#include "../zmqConnect/ZmqGazeReceiver.h"
#include <iostream>
#include <cstring>

ZmqGazeReceiver::ZmqGazeReceiver()
        : context(1), socket(context, zmq::socket_type::pull), running(false) {
    socket.set(zmq::sockopt::rcvtimeo, 100);
    socket.bind("tcp://*:6001");
}

ZmqGazeReceiver::~ZmqGazeReceiver() {
    stop();
}

void ZmqGazeReceiver::start() {
    if (running) return;
    running = true;
    listenerThread = std::thread(&ZmqGazeReceiver::listenLoop, this);
}

void ZmqGazeReceiver::stop() {
    running = false;
    socket.close();
    if (listenerThread.joinable()) {
        listenerThread.join();
    }
}

void ZmqGazeReceiver::listenLoop() {
    while (running) {
        zmq::message_t msg;
        if (!socket.recv(msg, zmq::recv_flags::none)) continue;

        if (msg.size() == sizeof(float) * 2) {
            float coords[2];
            std::memcpy(coords, msg.data(), sizeof(coords));
            if (onGazeReceived)
                onGazeReceived(coords[0], coords[1]);
        } else {
            std::cerr << "[ZMQ] Invalid gaze data size: " << msg.size() << std::endl;
        }
    }
}
