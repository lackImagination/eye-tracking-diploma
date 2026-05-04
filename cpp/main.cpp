// main.cpp
#include "connect/FaceTracker.h"
#include "connect/CalibrationWindow.h"
#include "zmqConnect/ZmqStreamer.h"
#include "zmqConnect/ZmqCommandReceiver.h"
#include "zmqConnect/ZmqGazeReceiver.h"
#include "zmqConnect/ZmqPythonCommander.h"
#include "zmqConnect/ZmqPythonAckReceiver.h"
#include "zmqConnect/GazeForwarder.h"
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    FaceTracker tracker;
    if (!tracker.initDefault()) {
        std::cerr << "[FATAL] FaceTracker failed to initialize.\n";
        return -1;
    }

    ZmqStreamer streamer(tracker);
    ZmqCommandReceiver receiver(tracker, streamer);
    receiver.start();

    ZmqGazeReceiver gazeReceiver;
    auto* gazeData = new GazeData();
    gazeReceiver.onGazeReceived = [gazeData](float gx, float gy) {
        gazeData->x.store(gx);
        gazeData->y.store(gy);
    };
    gazeReceiver.start();

    GazeForwarder forwarder;
    gazeReceiver.onGazeReceived = [gazeData, &forwarder](float gx, float gy) {
        gazeData->x.store(gx);
        gazeData->y.store(gy);
        forwarder.forward(gx, gy);
    };

    std::cout << "System running. Waiting for commands...\n";

    // Главный цикл
    while (!receiver.exitFlag.load()) {
        if (receiver.startCalibrationFlag.exchange(false)) {
            CalibrationWindow calibWindow(tracker, gazeData);
            calibWindow.run();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Завершение Python
    ZmqPythonCommander pythonCommander;
    ZmqPythonAckReceiver ackReceiver;
    pythonCommander.sendExitSignal();
    bool ack = ackReceiver.waitForAck();
    if (!ack) {
        std::cerr << "[MAIN] Python не прислал ACK_EXIT\n";
    } else {
        std::cout << "[MAIN] Python завершился корректно.\n";
    }

    // Завершение компонентов
    receiver.stop();
    streamer.stop();
    gazeReceiver.stop();
    tracker.stopTracking();

    std::cout << "Program exiting...\n";

    return 0;
}

