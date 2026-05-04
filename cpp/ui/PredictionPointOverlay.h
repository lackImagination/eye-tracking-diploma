// PredictionPointOverlay.h
#pragma once
#include <FL/Fl_Box.H>
#include <FL/Fl.H>
#include <FL/Fl_Box.H>
#include <FL/fl_draw.H>
#include <FL/Fl_Double_Window.H>
#include "../zmqConnect/GazeData.h"

class PredictionPointOverlay : public Fl_Box {
public:
    PredictionPointOverlay(int X, int Y, int W, int H, GazeData* gazeData)
            : Fl_Box(X, Y, W, H), gazeData(gazeData) {
        box(FL_NO_BOX);
    }

    int handle(int) override {
        return 0;
    }

    void draw() override {
        fl_push_clip(x(), y(), w(), h());
        fl_color(fl_rgb_color(35, 35, 35));
        fl_rectf(x(), y(), w(), h());
        fl_pop_clip();


        if (gazeData) {
            float gx = gazeData->x.load();
            float gy = gazeData->y.load();

            fl_color(FL_GREEN);
            int radius = 50;
            fl_pie(static_cast<int>(gx) - radius, static_cast<int>(gy) - radius,
                   radius * 2, radius * 2, 0, 360);
        }
    }

    static void update_cb(void* data) {
        auto* overlay = static_cast<PredictionPointOverlay*>(data);
        overlay->damage(FL_DAMAGE_ALL);
        Fl::repeat_timeout(1.0 / 60.0, update_cb, data);
    }

    void start() {
        Fl::add_timeout(1.0 / 60.0, update_cb, this);
    }

private:
    GazeData* gazeData;
};
