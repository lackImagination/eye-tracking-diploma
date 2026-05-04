//MouseOnlyButton.h
#pragma once
#include <FL/Fl_Button.H>

class MouseOnlyButton : public Fl_Button {
public:
    using Fl_Button::Fl_Button;

    int handle(int event) override {
        if (event == FL_PUSH) {
            do_callback();
            return Fl_Button::handle(event);
        }
        return 0;
    }
};

