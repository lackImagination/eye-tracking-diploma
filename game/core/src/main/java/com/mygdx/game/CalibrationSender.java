//CalibrationSender.java
package com.mygdx.game;
import com.badlogic.gdx.Gdx;
import org.zeromq.ZMQ;

public class CalibrationSender {
    private static final byte START_TRACKING = 1;
    private static final byte STOP_TRACKING = 2;
    private static final byte START_CALIBRATION = 4;
    private static final byte EXIT = 6;
    private static final String ADDRESS = "tcp://127.0.0.1:6000";

    public static void sendStartTrackingCommand() {
        try (ZMQ.Context context = ZMQ.context(1);
             ZMQ.Socket socket = context.socket(ZMQ.PUSH)) {

            socket.connect(ADDRESS);
            socket.send(new byte[]{START_TRACKING});
            System.out.println("Sent command: START_TRACKING");

        } catch (Exception e) {
            System.err.println("Failed to send START_TRACKING command: " + e.getMessage());
        }
    }

    public static void sendStopTrackingCommand() {
        try (ZMQ.Context context = ZMQ.context(1);
             ZMQ.Socket socket = context.socket(ZMQ.PUSH)) {
            socket.connect(ADDRESS);
            socket.send(new byte[]{STOP_TRACKING});
            System.out.println("Sent command: STOP_TRACKING");
        } catch (Exception e) {
            System.err.println("Failed to send STOP_TRACKING command: " + e.getMessage());
        }
    }

    public static void sendCalibrationCommand() {
        try (ZMQ.Context context = ZMQ.context(1);
             ZMQ.Socket socket = context.socket(ZMQ.PUSH)) {

            socket.connect(ADDRESS);
            socket.send(new byte[]{START_CALIBRATION});
            System.out.println("Sent command: START_CALIBRATION");

        } catch (Exception e) {
            System.err.println("Failed to send calibration command: " + e.getMessage());
        }
    }
    public static void sendExitCommand() {
        try (ZMQ.Context context = ZMQ.context(1);
             ZMQ.Socket socket = context.socket(ZMQ.PUSH)) {
            socket.connect(ADDRESS);
            socket.send(new byte[]{EXIT});
            System.out.println("Sent command: EXIT");
            Thread.sleep(300);

            Gdx.app.exit();
            System.exit(0);

        } catch (Exception e) {
            System.err.println("Failed to send EXIT command: " + e.getMessage());
        }
    }
}
