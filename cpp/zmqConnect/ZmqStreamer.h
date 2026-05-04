//ZmqStreamer.h
#ifndef ZMQ_STREAMER_H
#define ZMQ_STREAMER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <atomic>
#include "zmq.hpp"

class FaceTracker;

class ZmqStreamer {
public:
    ZmqStreamer(FaceTracker& tracker);
    ~ZmqStreamer();

    void start();
    void stop();

private:
    void streamingLoop();

    FaceTracker& tracker;
    std::thread streamingThread;
    std::atomic<bool> running;
    zmq::context_t context;
    zmq::socket_t socket;
};

#endif
