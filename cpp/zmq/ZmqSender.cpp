//ZmqSender.cpp
#include "../zmqConnect/ZmqSender.h"

ZmqSender::ZmqSender(const std::string& address)
        : context(1), socket(context, zmq::socket_type::push) {
    socket.connect(address);
}

void ZmqSender::sendCommand(CommandType cmd) {
    uint8_t code = static_cast<uint8_t>(cmd);
    zmq::message_t message(&code, sizeof(code));
    socket.send(message, zmq::send_flags::none);
}
