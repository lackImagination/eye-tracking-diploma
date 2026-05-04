// GazeGame.java
package com.mygdx.game;

import com.badlogic.gdx.Game;

public class GazeGame extends Game {

    @Override
    public void create() {this.setScreen(new MainMenuScreen(this));}

    @Override
    public void render() {
        super.render();
    }

    @Override
    public void dispose() {
        super.dispose();
    }
}
