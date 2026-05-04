//ZmqSender.h
#pragma once

#include "zmq.hpp"
#include <string>
#include "CommandType.h"

class ZmqSender {
public:
    ZmqSender(const std::string& address = "tcp://127.0.0.1:6000");
    void sendCommand(CommandType cmd);

private:
    zmq::context_t context;
    zmq::socket_t socket;
};
