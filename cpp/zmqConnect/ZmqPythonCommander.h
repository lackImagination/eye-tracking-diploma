// ZmqPythonCommander.h
#pragma once
#include "zmq.hpp"

class ZmqPythonCommander {
public:
    ZmqPythonCommander();
    void sendExitSignal();
private:
    zmq::context_t context;
    zmq::socket_t socket;
};
