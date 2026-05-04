// ZmqPythonAckReceiver.h
#pragma once
#include "zmq.hpp"

class ZmqPythonAckReceiver {
public:
    ZmqPythonAckReceiver();
    bool waitForAck(int timeoutMs = 2000);  // ждём ACK с таймаутом
private:
    zmq::context_t context;
    zmq::socket_t socket;
};
