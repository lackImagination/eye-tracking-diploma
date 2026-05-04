// ZmqPythonCommander.cpp
#include "../zmqConnect/ZmqPythonCommander.h"
#include <iostream>

ZmqPythonCommander::ZmqPythonCommander()
        : context(1), socket(context, zmq::socket_type::pub) {
    socket.bind("tcp://*:6002"); // Python SUB слушает
}

void ZmqPythonCommander::sendExitSignal() {
    zmq::message_t msg("EXIT", 4); // читаемый текст
    socket.send(msg, zmq::send_flags::none);
    std::cout << "[C++] Sent EXIT to Python\n";
}
