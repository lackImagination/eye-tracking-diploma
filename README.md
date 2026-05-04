# EyeControl — управление игровым персонажем взглядом

Дипломный проект. Система отслеживания взгляда в реальном времени: C++ захватывает и обрабатывает видеопоток, PyTorch-модель предсказывает координаты взгляда, Java-игра на LibGDX управляет персонажем — всё связано через ZeroMQ.

## Как это работает

```
Камера → C++ (детекция лица + область глаз) → Python (предсказание X,Y) → Java (движение персонажа)
              ↑                                                                        |
              └────────────────── ZMQ команды (старт/стоп/калибровка) ───────────────┘
```

1. C++ захватывает видео, детектирует лицо через DNN (ResNet SSD), извлекает ключевые точки глаз, компенсирует поворот головы и вырезает область глаз 224×224
2. Кадры передаются в Python по ZMQ (PUB/SUB)
3. PyTorch-модель предсказывает координаты взгляда на экране
4. Координаты возвращаются в C++ и пробрасываются в Java-игру (PUSH/PULL)
5. Персонаж движется в сторону взгляда, предметы собираются фиксацией взгляда на 1.5 секунды

## Архитектурные диаграммы

| Диаграмма | Файл |
|-----------|------|
| Варианты использования | [use case diagram.png](doc/diagrams/use%20case%20diagram.png) |
| Модули системы | [module diagram.png](doc/diagrams/module%20diagram.png) |
| DFD нулевой уровень | [DFD Zero level of detail.png](doc/diagrams/DFD%20Zero%20level%20of%20detail.png) |
| DFD первый уровень | [DFD First level of detail.png](doc/diagrams/DFD%20First%20level%20of%20detail.png) |
| DFD второй уровень | [DFD Second Level of Detail.png](doc/diagrams/DFD%20Second%20Level%20of%20Detail.png) |
| Диаграмма последовательности | [sequence diagram.png](doc/diagrams/sequence%20diagram.png) |

## Структура репозитория

```
eye-tracking-diploma/
├── cpp/                        # C++ ядро: захват видео, обработка, ZMQ
│   ├── connect/                # FaceTracker, CalibrationWindow, FrameSaver
│   ├── ui/                     # FLTK-виджеты (тепловая карта, оверлей взгляда)
│   ├── zmq/                    # ZMQ-компоненты (.cpp)
│   ├── zmqConnect/             # ZMQ-заголовки (.h)
│   ├── data/models/            # Предобученные модели детекции лица
│   ├── main.cpp
│   └── CMakeLists.txt
│
├── python/                     # Обучение и инференс
│   ├── experiments/            # Архитектуры моделей (experiment_1..6)
│   ├── training_logs/          # Метрики и графики по каждому эксперименту
│   ├── inference.py            # Инференс в реальном времени через ZMQ
│   ├── main.py                 # Обучение модели
│   ├── data.py                 # Датасет и базовые утилиты
│   ├── visualize_predictions.py
│   ├── plot_error_distribution.py
│   ├── predictions_one_example.py
│   ├── worst-case.py
│   └── requirements.txt
│
├── game/                       # LibGDX-игра (Java)
│   ├── core/src/               # Игровая логика, GazeReceiver, CalibrationSender
│   ├── lwjgl3/                 # Десктопный лаунчер
│   └── assets/                 # Текстуры, UI-скин
│
└── doc/diagrams/               # Архитектурные диаграммы
```

## Зависимости

### C++
- CMake ≥ 3.16, C++20
- OpenCV ≥ 4.x (с модулями `face`, `dnn`)
- FLTK 1.4.x — `brew install fltk`
- ZeroMQ — `brew install zeromq` / `apt install libzmq3-dev`
- Модели детекции лица уже включены в `cpp/data/models/`
- Модель ключевых точек лица `lbfmodel.yaml` — скачать отдельно:
  ```bash
  wget https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml
  # положить в cpp/opencv/data/lbfmodel.yaml
  ```

### Python
```bash
cd python
pip install -r requirements.txt
```

### Java / LibGDX
- JDK 11+, Gradle (обёртка `gradlew` включена в репозиторий)
- jeromq подтянется через Gradle автоматически

## Сборка

### C++

```bash
cd cpp
cmake -B build -DOpenCV_DIR=/path/to/opencv/build
cmake --build build
```

Или через переменные окружения:
```bash
export OPENCV_DIR=/path/to/opencv/build
export FLTK_ROOT=$(brew --prefix fltk)   # macOS
cmake -B build && cmake --build build
```

### Java-игра

```bash
cd game
./gradlew lwjgl3:run
```

## Запуск

Запускать в таком порядке:

```bash
# 1. C++ ядро
./cpp/build/eye_tracking

# 2. Python инференс (в отдельном терминале)
export GAZE_MODEL_PATH=/path/to/model.pt
cd python && python inference.py

# 3. Игра (в отдельном терминале)
cd game && ./gradlew lwjgl3:run
```

В меню игры:
- **Starting Calibration** — открывает окно калибровки в C++. Смотри на точки на экране, нажимай пробел для сохранения. Чем больше точек — тем точнее модель
- **New Game** — запускает игру с управлением взглядом

## ZeroMQ порты

| Порт | Паттерн | Назначение |
|------|---------|------------|
| 5555 | PUB/SUB | Видеопоток с области глаз (C++ → Python) |
| 6000 | PUSH/PULL | Команды управления (Java → C++) |
| 6001 | PUSH/PULL | Координаты взгляда (Python → C++) |
| 6002 | PUB/SUB | Команды завершения (C++ → Python) |
| 6003 | PUSH/PULL | Координаты взгляда (C++ → Java) |
| 6004 | PUSH/PULL | ACK подтверждения завершения (Python → C++) |

## Модель

Финальная архитектура — `experiments/experiment_6V2.py`: кастомная CNN (~10.5M параметров). Вход: область глаз 224×224, выход: координаты (X, Y) на экране в пикселях.

Предобработка: детекция лица → ключевые точки → компенсация поворота головы → вырезание области глаз → паддинг до квадрата → resize 224×224.

Веса модели скачать с Hugging Face *(ссылка будет добавлена)*.

## Датасет

Собирался вручную в процессе калибровки: изображения области глаз + координаты точки взгляда на экране.

Доступен на [Hugging Face](https://huggingface.co/datasets/lackOfImagination/eye-gaze-rgb-dataset).

Формат:
```
dataset/
├── images/
│   ├── eye_0001.png
│   └── ...
└── coords.csv   # image_name,x,y
```

## Эксперименты

В `python/training_logs/` хранятся результаты всех экспериментов — конфиги, графики и метрики (train loss, val loss, pixel error).

| Эксперимент | Описание |
|-------------|----------|
| experiment_1 | Базовая CNN с нуля |
| experiment_2–3 | ResNet18 pretrained, нормализованные координаты |
| experiment_4–5 | Аугментации, TTA (test-time augmentation) |
| experiment_6 | Финальная кастомная архитектура, пиксельные координаты |
| experiment_6V2 | Fine-tune финальной модели |

## Лицензия

[GNU GPL v3](LICENSE) — можно использовать и модифицировать, но производные проекты обязаны оставаться открытыми под той же лицензией.

© 2025 Ariana
