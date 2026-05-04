//CommandType.h
#pragma once
#include <cstdint>

enum class CommandType : uint8_t {
    START_TRACKING = 1,
    STOP_TRACKING,
    SAVE_FRAME,
    START_CALIBRATION,
    STOP_CALIBRATION,
    EXIT
};
