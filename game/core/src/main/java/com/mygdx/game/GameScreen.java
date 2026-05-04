// GameScreen.java
package com.mygdx.game;

import com.badlogic.gdx.graphics.OrthographicCamera;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.Screen;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.graphics.g2d.TextureRegion;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer;
import com.badlogic.gdx.scenes.scene2d.InputEvent;
import com.badlogic.gdx.scenes.scene2d.Stage;
import com.badlogic.gdx.scenes.scene2d.ui.Skin;
import com.badlogic.gdx.scenes.scene2d.ui.Table;
import com.badlogic.gdx.scenes.scene2d.ui.TextButton;
import com.badlogic.gdx.scenes.scene2d.utils.ClickListener;
import com.badlogic.gdx.utils.viewport.ScreenViewport;

public class GameScreen implements Screen {
    private OrthographicCamera camera;
    private SpriteBatch batch;
    private ShapeRenderer shapeRenderer;
    private Texture playerTexture;
    private Texture backgroundTexture;
    private float playerX, playerY;
    private Texture itemTexture;
    private float itemX, itemY;
    private boolean itemCollected = false;
    private Stage hudStage;
    private Skin skin;
    private TextButton menuButton;
    private GazeGame game;
    private GazeReceiver gazeReceiver;
    private Thread gazeThread;
    private float smoothedX = -1;
    private float smoothedY = -1;
    private static final float SMOOTHING_ALPHA = 0.3f;
    private float gazeOnItemTime = 0f;
    private static final float GAZE_THRESHOLD = 1.5f;
    private boolean isGazingAtItem = false;

    private static final float PLAYER_SPEED = 200f; // скорость персонажа в пикселях в секунду

    public GameScreen(GazeGame game) {
        camera = new OrthographicCamera();
        camera.setToOrtho(true, Gdx.graphics.getWidth(), Gdx.graphics.getHeight());
        batch = new SpriteBatch();
        shapeRenderer = new ShapeRenderer();
        shapeRenderer.setProjectionMatrix(camera.combined);
        batch.setProjectionMatrix(camera.combined);

        try {
            playerTexture = new Texture(Gdx.files.internal("player.png"));
            System.out.println("Image loaded successfully.");
        } catch (Exception e) {
            System.out.println("Ошибка загрузки изображения: " + e.getMessage());
        }
        try {
            backgroundTexture = new Texture(Gdx.files.internal("background.png"));
            System.out.println("Background loaded.");
        } catch (Exception e) {
            System.out.println("Ошибка загрузки фона: " + e.getMessage());
        }
        this.game = game;

        itemTexture = new Texture(Gdx.files.internal("item.png"));
        itemX = 180;
        itemY = 180;

        playerX = Gdx.graphics.getWidth() / 2; // начальная позиция по X
        playerY = Gdx.graphics.getHeight(); // начальная позиция по Y
        // UI: кнопка меню
        hudStage = new Stage(new ScreenViewport());
        Gdx.input.setInputProcessor(hudStage); // обрабатывает клики по UI

        skin = new Skin(Gdx.files.internal("uiskin.json"));
        menuButton = new TextButton("Menu", skin);

        menuButton.addListener(new ClickListener() {
            @Override
            public void clicked(InputEvent event, float x, float y) {
                CalibrationSender.sendStopTrackingCommand();
                game.setScreen(new MainMenuScreen(game));
            }
        });
        // размещение кнопки в левом верхнем углу
        Table table = new Table();
        table.top().left();
        table.setFillParent(true);
        table.add(menuButton).pad(10);
        hudStage.addActor(table);

        gazeReceiver = new GazeReceiver();
        gazeThread = new Thread(gazeReceiver);
        gazeThread.start();
    }

    @Override
    public void show() {
    }

    @Override
    public void render(float delta) {
        // Очистка экрана
        Gdx.gl.glClearColor(1, 1, 1, 1); // RGB для белого цвета
        Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);

        handleInput(delta);

        float gazeX = smoothedX;
        float gazeY = smoothedY;
        float itemScale = 2.0f;
        float itemWidth = itemTexture.getWidth() * itemScale;
        float itemHeight = itemTexture.getHeight() * itemScale;
        float margin = 30f;
        boolean isInBounds = !itemCollected &&
            gazeX >= itemX - margin && gazeX <= itemX + itemWidth + margin &&
            gazeY >= itemY - margin && gazeY <= itemY + itemHeight + margin;

        if (isInBounds) {
            gazeOnItemTime += delta;
            if (gazeOnItemTime >= GAZE_THRESHOLD) {
                itemCollected = true;
                System.out.println("Предмет собран!");
            }
        } else {
            gazeOnItemTime = 0f;
        }

        batch.begin();
        TextureRegion backgroundRegion = new TextureRegion(backgroundTexture);
        backgroundRegion.flip(false, true);
        batch.draw(backgroundRegion, 0, 0, Gdx.graphics.getWidth(), Gdx.graphics.getHeight());

        if (!itemCollected) {
            TextureRegion itemRegion = new TextureRegion(itemTexture);
            itemRegion.flip(false, true);
            batch.draw(itemRegion, itemX, itemY, itemTexture.getWidth() * itemScale, itemTexture.getHeight() * itemScale);
        }

        // Рисуем персонажа
        TextureRegion playerRegion = new TextureRegion(playerTexture);
        playerRegion.flip(false, true);
        float scale = 0.2f;
        batch.draw(playerRegion, playerX, playerY,
            playerTexture.getWidth() * scale,
            playerTexture.getHeight() * scale);

        batch.end();
        // Отрисовка круга сбора (визуальный индикатор внимания)
        if (!itemCollected && gazeOnItemTime > 0f) {
            float progress = gazeOnItemTime / GAZE_THRESHOLD;
            if (progress > 1f) progress = 1f;

            float centerX = itemX + itemWidth / 2;
            float centerY = itemY + itemHeight / 2;
            float radius = 30f;

            shapeRenderer.begin(ShapeRenderer.ShapeType.Filled);
            shapeRenderer.setColor(0f, 0.5f, 1f, 0.5f); // полупрозрачный синий
            shapeRenderer.arc(centerX, centerY, radius, 90f, 360f * progress);
            shapeRenderer.end();
        }
        hudStage.act(delta);
        hudStage.draw();
    }

    private void handleInput(float delta) {
        float targetX = gazeReceiver.getGazeX();
        float targetY = gazeReceiver.getGazeY();

        if (targetX >= 0 && targetY >= 0) {
            if (smoothedX < 0) smoothedX = targetX;
            if (smoothedY < 0) smoothedY = targetY;

            smoothedX = SMOOTHING_ALPHA * targetX + (1 - SMOOTHING_ALPHA) * smoothedX;
            smoothedY = SMOOTHING_ALPHA * targetY + (1 - SMOOTHING_ALPHA) * smoothedY;

            float dx = smoothedX - playerX;
            float dy = smoothedY - playerY;
            float distance = (float) Math.sqrt(dx * dx + dy * dy);
            float speed = PLAYER_SPEED * delta;
            if (distance > 1f) {
                playerX += dx / distance * speed;
                playerY += dy / distance * speed;
            }
        }
        // Ограничиваем движение персонажа в пределах экрана
        float scale = 0.2f;
        float playerWidth = playerTexture.getWidth() * scale;
        float playerHeight = playerTexture.getHeight() * scale;

        if (playerX > Gdx.graphics.getWidth() - playerWidth) playerX = Gdx.graphics.getWidth() - playerWidth;
        if (playerY > Gdx.graphics.getHeight() - playerHeight) playerY = Gdx.graphics.getHeight() - playerHeight;
    }

    @Override
    public void resize(int width, int height) {
        hudStage.getViewport().update(width, height, true);
    }
    @Override
    public void hide() {}
    @Override
    public void pause() {}
    @Override
    public void resume() {}
    @Override
    public void dispose() {
        batch.dispose();
        shapeRenderer.dispose();
        playerTexture.dispose();
        backgroundTexture.dispose();
        itemTexture.dispose();

        gazeReceiver.stop();
        try {gazeThread.join();} catch (InterruptedException e) {e.printStackTrace();}
    }
}
