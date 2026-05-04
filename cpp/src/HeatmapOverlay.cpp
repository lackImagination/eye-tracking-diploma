//HeatmapOverlay.cpp
#include "../ui/HeatmapOverlay.h"
#include <FL/fl_draw.H>

HeatmapOverlay::HeatmapOverlay(int X, int Y, int W, int H)
        : Fl_Box(X, Y, W, H) {
    box(FL_NO_BOX);
}

void HeatmapOverlay::draw() {
    Fl_Box::draw();

    if (!visible || points.empty()) return;

    int offsetX = x();
    int offsetY = y();

    fl_color(FL_RED);
    for (const auto& pt : points) {
        int drawX = offsetX + pt.x;
        int drawY = offsetY + pt.y;

        int radius = 15;
        fl_pie(drawX - radius, drawY - radius, radius * 2, radius * 2, 0, 360);
    }
}
