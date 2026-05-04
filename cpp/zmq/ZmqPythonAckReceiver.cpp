// ZmqPythonAckReceiver.cpp
#include "../zmqConnect/ZmqPythonAckReceiver.h"
#include <iostream>
#include <thread>

ZmqPythonAckReceiver::ZmqPythonAckReceiver()
        : context(1), socket(context, zmq::socket_type::pull) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    socket.bind("tcp://*:6004");  // отдельный порт для ACK
}

bool ZmqPythonAckReceiver::waitForAck(int timeoutMs) {
    socket.set(zmq::sockopt::rcvtimeo, timeoutMs);
    zmq::message_t msg;
    if (socket.recv(msg)) {
        std::string ack(static_cast<char*>(msg.data()), msg.size());
        if (ack == "ACK_EXIT") {
            std::cout << "[C++] Received ACK_EXIT from Python\n";
            return true;
        } else {
            std::cout << "[C++] Unexpected message: " << ack << "\n";
        }
    } else {
        std::cerr << "[C++] Timeout waiting for ACK_EXIT\n";
    }
    return false;
}
