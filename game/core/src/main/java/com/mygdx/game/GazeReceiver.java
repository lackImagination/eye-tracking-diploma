// GazeReceiver.java
package com.mygdx.game;

import org.zeromq.ZMQ;

public class GazeReceiver implements Runnable {
    private volatile float gazeX = -1;
    private volatile float gazeY = -1;
    private boolean running = true;

    public float getGazeX() {
        return gazeX;
    }

    public float getGazeY() {
        return gazeY;
    }

    public void stop() {
        running = false;
    }

    @Override
    public void run() {
        ZMQ.Context context = ZMQ.context(1);
        ZMQ.Socket socket = context.socket(ZMQ.PULL);
        socket.connect("tcp://127.0.0.1:6003");

        while (running) {
            byte[] data = socket.recv(ZMQ.DONTWAIT);
            if (data != null && data.length == 8) {
                // System.out.println("successfully");
                int intBitsX = java.nio.ByteBuffer.wrap(data, 0, 4).order(java.nio.ByteOrder.LITTLE_ENDIAN).getInt();
                int intBitsY = java.nio.ByteBuffer.wrap(data, 4, 4).order(java.nio.ByteOrder.LITTLE_ENDIAN).getInt();
                gazeX = Float.intBitsToFloat(intBitsX);
                gazeY = Float.intBitsToFloat(intBitsY);
            }

            try {
                Thread.sleep(10); // не грузим CPU
            } catch (InterruptedException ignored) {}
        }

        socket.close();
        context.close();
    }
}
