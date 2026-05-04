cmake_minimum_required(VERSION 3.16)
project(eye_tracking)

set(CMAKE_CXX_STANDARD 20)

# ---------------------------------------------------------------------------
# Dependencies — настрой пути под свою систему одним из способов:
#
# Способ 1 (рекомендуется): передай при сборке через cmake -D...
#   cmake -B build -DOpenCV_DIR=/path/to/opencv/build
#
# Способ 2: задай переменные окружения OPENCV_DIR, FLTK_ROOT
#
# Способ 3: vcpkg (уже настроен ниже через VCPKG_ROOT)
# ---------------------------------------------------------------------------

# vcpkg — если используешь, передай путь через переменную окружения или -D при cmake
if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
    # vcpkg лежит внутри проекта (не рекомендуется коммитить, но если есть)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
endif()

# OpenCV
if(DEFINED ENV{OPENCV_DIR})
    set(OpenCV_DIR "$ENV{OPENCV_DIR}")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/opencv/build")
    set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/opencv/build")
endif()

# FLTK — homebrew или системная установка
if(DEFINED ENV{FLTK_ROOT})
    set(FLTK_INCLUDE_DIR "$ENV{FLTK_ROOT}/include")
    set(FLTK_LIB_DIR     "$ENV{FLTK_ROOT}/lib")
else()
    # Попробуем найти через brew --prefix автоматически
    execute_process(
        COMMAND brew --prefix fltk
        OUTPUT_VARIABLE FLTK_BREW_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(FLTK_BREW_PREFIX)
        set(FLTK_INCLUDE_DIR "${FLTK_BREW_PREFIX}/include")
        set(FLTK_LIB_DIR     "${FLTK_BREW_PREFIX}/lib")
    endif()
endif()

# ---------------------------------------------------------------------------
# Найдём пакеты
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED)
find_package(FLTK REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ZeroMQ REQUIRED libzmq)

# ---------------------------------------------------------------------------
# Include и link директории
# ---------------------------------------------------------------------------
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${FLTK_INCLUDE_DIR}
    ${ZeroMQ_INCLUDE_DIRS}
)

link_directories(
    ${FLTK_LIB_DIR}
    ${ZeroMQ_LIBRARY_DIRS}
)

# vcpkg include (x64-osx или другая платформа — определяем автоматически)
if(DEFINED CMAKE_TOOLCHAIN_FILE AND EXISTS "${CMAKE_SOURCE_DIR}/vcpkg/installed")
    # Определяем triplet автоматически
    if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        set(VCPKG_TRIPLET "arm64-osx")
    elseif(APPLE)
        set(VCPKG_TRIPLET "x64-osx")
    elseif(UNIX)
        set(VCPKG_TRIPLET "x64-linux")
    endif()
    include_directories("${CMAKE_SOURCE_DIR}/vcpkg/installed/${VCPKG_TRIPLET}/include")
endif()

# ---------------------------------------------------------------------------
# Исполняемый файл
# ---------------------------------------------------------------------------
add_executable(eye_tracking
    main.cpp
    src/FaceTracker.cpp
    src/FrameSaver.cpp
    src/HeatmapOverlay.cpp
    src/HeatmapUtils.cpp
    src/Preprocessing.cpp
    src/CalibrationWindow.cpp
    zmq/ZmqSender.cpp
    zmq/ZmqStreamer.cpp
    zmq/ZmqCommandReceiver.cpp
    zmq/ZmqController.cpp
    zmq/ZmqGazeReceiver.cpp
    zmq/ZmqPythonCommander.cpp
    zmq/ZmqPythonAckReceiver.cpp
    zmq/GazeForwarder.cpp
    # Headers перечислять не обязательно, но можно для IDE
    connect/FaceTracker.h
    connect/FrameSaver.h
    connect/Preprocessing.h
    connect/CalibrationWindow.h
    ui/HeatmapOverlay.h
    ui/HeatmapUtils.h
    ui/MouseOnlyButton.h
    ui/PredictionPointOverlay.h
    zmqConnect/CommandType.h
    zmqConnect/GazeData.h
    zmqConnect/GazeForwarder.h
    zmqConnect/ZmqSender.h
    zmqConnect/ZmqStreamer.h
    zmqConnect/ZmqCommandReceiver.h
    zmqConnect/ZmqController.h
    zmqConnect/ZmqGazeReceiver.h
    zmqConnect/ZmqPythonCommander.h
    zmqConnect/ZmqPythonAckReceiver.h
    zmqConnect/sendExitCalibration.h
)

target_link_libraries(eye_tracking
    ${OpenCV_LIBS}
    fltk
    fltk_images
    ${ZeroMQ_LIBRARIES}
)
