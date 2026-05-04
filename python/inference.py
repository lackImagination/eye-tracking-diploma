import os
import zmq
import numpy as np
import cv2
import torch
from PIL import Image
from visualize_predictions import screen_info
from experiments.experiment_6 import get_transform

# ==== Настройки ====
# Путь к модели: задай переменную окружения GAZE_MODEL_PATH
# Пример: export GAZE_MODEL_PATH=/path/to/model_6.pt
model_path = os.environ.get("GAZE_MODEL_PATH")
if not model_path:
    raise EnvironmentError(
        "Не задана переменная окружения GAZE_MODEL_PATH.\n"
        "Пример: export GAZE_MODEL_PATH=/path/to/model_6.pt\n"
        "Модель можно скачать с Hugging Face: [ссылка]"
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Загрузка модели ====
model = torch.jit.load(model_path, map_location=device)
model.eval()

transform = get_transform()

# ==== Настройка ZeroMQ ====
context = zmq.Context()

frame_socket = context.socket(zmq.SUB)
frame_socket.connect("tcp://127.0.0.1:5555")
frame_socket.setsockopt(zmq.SUBSCRIBE, b"")
frame_socket.setsockopt(zmq.CONFLATE, 1)

gaze_socket = context.socket(zmq.PUSH)
gaze_socket.connect("tcp://127.0.0.1:6001")

command_socket = context.socket(zmq.SUB)
command_socket.connect("tcp://127.0.0.1:6002")
command_socket.setsockopt(zmq.SUBSCRIBE, b"")
command_socket.setsockopt(zmq.RCVTIMEO, 100)

ack_socket = context.socket(zmq.PUSH)
ack_socket.connect("tcp://127.0.0.1:6004")

MAX_X, MAX_Y = screen_info.width, screen_info.height

print("Ready to receive frames and predict gaze coordinates...")

running = True
try:
    while running:
        try:
            command = command_socket.recv(flags=zmq.NOBLOCK)
            if command == b'EXIT':
                print("[PY] Received EXIT signal from C++")
                running = False
                break
        except zmq.Again:
            pass

        if frame_socket.poll(100):
            frame_data = frame_socket.recv()
            if len(frame_data) == 1 and frame_data[0] == 0x00:
                continue

            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                print("Invalid frame")
                continue

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = model(input_tensor).cpu().squeeze().numpy()
            x, y = np.clip(prediction, [0, 0], [MAX_X, MAX_Y])
            gaze_socket.send(np.array([x, y], dtype=np.float32).tobytes())

except Exception as e:
    print(f"[PY] Outer exception: {e}")
except KeyboardInterrupt:
    print("[PY] Interrupted manually")
finally:
    print("[PY] In finally block — preparing to send ACK_EXIT")
    try:
        ack_socket.send(b"ACK_EXIT")
        print("[PY] Sent ACK_EXIT to C++")
    except Exception as e:
        print(f"[PY] Failed to send ACK_EXIT: {e}")

    frame_socket.close()
    gaze_socket.close()
    command_socket.close()
    ack_socket.close()
    context.term()
    cv2.destroyAllWindows()
    print("[PY] Shutdown complete.")
