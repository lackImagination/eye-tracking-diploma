# Eye Tracking System

Дипломный проект — система отслеживания взгляда в реальном времени с калибровкой, нейросетевым предсказанием координат и демо-игрой с управлением взглядом.

## Архитектура системы

```
┌─────────────────┐     ZMQ:6000      ┌──────────────────┐
│   LibGDX Game   │ ───────────────►  │                  │
│   (Java)        │                   │   C++ Core       │
│                 │ ◄─────────────── │   (FaceTracker + │
│  управление     │     ZMQ:6003      │    ZmqStreamer)  │
│  взглядом       │                   │                  │
└─────────────────┘                   └────────┬─────────┘
                                               │ ZMQ:5555
                                               │ (видеопоток)
                                               ▼
                                      ┌──────────────────┐
                                      │  Python Inference│
                                      │  (PyTorch model) │
                                      └────────┬─────────┘
                                               │ ZMQ:6001
                                               │ (координаты взгляда)
                                               ▼
                                      ┌──────────────────┐
                                      │   C++ Core       │
                                      │   (GazeReceiver) │
                                      └──────────────────┘
```

**Как это работает:**
1. C++ захватывает видео с камеры, детектирует лицо и вырезает область глаз
2. Кадры передаются в Python по ZeroMQ
3. PyTorch модель предсказывает координаты взгляда на экране
4. Координаты возвращаются в C++ и пробрасываются в Java-игру
5. В игре персонаж движется за взглядом игрока

## Структура репозитория

```
eye-tracking-diploma/
├── cpp/                    # C++ ядро системы
│   ├── connect/            # FaceTracker, CalibrationWindow, FrameSaver
│   ├── ui/                 # FLTK виджеты (HeatmapOverlay, PredictionPoint...)
│   ├── zmq/                # ZeroMQ компоненты (.cpp)
│   ├── zmqConnect/         # ZeroMQ заголовки (.h)
│   ├── main.cpp
│   └── CMakeLists.txt
│
├── python/                 # Python: обучение и инференс
│   ├── inference.py        # Запуск предсказания в реальном времени
│   ├── experiments/        # Архитектуры моделей (experiment_N.py)
│   ├── training_logs/      # Метрики обучения (CSV, графики)
│   └── requirements.txt
│
└── game/                   # LibGDX игра (Java)
    └── core/src/com/mygdx/game/
        ├── GazeGame.java
        ├── GameScreen.java
        ├── MainMenuScreen.java
        ├── GazeReceiver.java
        └── CalibrationSender.java
```

## Зависимости

### C++
- CMake ≥ 3.16
- OpenCV ≥ 4.x (с модулями `face`, `dnn`)
- FLTK 1.4.x
- ZeroMQ (libzmq)
- C++20

### Python
```bash
pip install -r python/requirements.txt
```

### Java / LibGDX
- JDK 11+
- LibGDX 1.12.x
- jeromq (ZeroMQ для Java)
- Gradle

## Сборка и запуск

### 1. C++ ядро

```bash
cd cpp
cmake -B build \
  -DOpenCV_DIR=/path/to/opencv/build
cmake --build build
```

Или через переменные окружения:
```bash
export OPENCV_DIR=/path/to/opencv/build
export FLTK_ROOT=/usr/local/opt/fltk   # brew install fltk
cmake -B build && cmake --build build
```

### 2. Python инференс

```bash
cd python
pip install -r requirements.txt

# Укажи путь к модели через переменную окружения:
export GAZE_MODEL_PATH=/path/to/model.pt
python inference.py
```

> 💾 Веса модели доступны на [Hugging Face](#) *(добавь ссылку)*

### 3. Java игра

```bash
cd game
./gradlew desktop:run
```

### Порядок запуска

1. Запусти C++ ядро (`./build/eye_tracking`)
2. Запусти Python инференс (`python inference.py`)
3. Запусти игру (`./gradlew desktop:run`)
4. В меню игры: **Starting Calibration** → откроется окно калибровки в C++
5. После калибровки: **New Game** → персонаж управляется взглядом

## Калибровка

В окне калибровки:
- Смотри на точку на экране, нажимай **пробел** — кадр сохраняется
- Кнопка **«Нарисовать тепловую карту»** — визуализация собранных точек
- Кнопка **«Обучить модель»** — запуск дообучения *(в разработке)*

## ZeroMQ порты

| Порт | Тип | Назначение |
|------|-----|------------|
| 5555 | PUB/SUB | Видеопоток (C++ → Python) |
| 6000 | PUSH/PULL | Команды (Java → C++) |
| 6001 | PUSH/PULL | Координаты взгляда (Python → C++) |
| 6002 | PUB/SUB | Команды Python (C++ → Python) |
| 6003 | PUSH/PULL | Координаты взгляда (C++ → Java) |
| 6004 | PUSH/PULL | ACK подтверждения (Python → C++) |

## Датасет

Датасет с изображениями глаз для обучения доступен на [Hugging Face](#) *(добавь ссылку)*.

Формат: папка `images/` с файлами `eye_XXXX.png` + `coords.csv`:
```
eye_0001.png,960,540
eye_0002.png,1280,400
...
```
