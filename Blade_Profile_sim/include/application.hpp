#pragma once

#include <GLFW/glfw3.h>
#include <memory>
#include <string>

class Renderer;
class Camera;
class WindSimulation;
class BladePhysics;
class StatsOverlay;
class CADLoader;
class Mesh;

class Application {
public:
    Application(int width, int height, const std::string& title);
    ~Application();
    
    void run();
    void loadBlade(const std::string& filepath);
    
    // Input callbacks
    void onKeyPress(int key, int action, int mods);
    void onMouseMove(double xpos, double ypos);
    void onMouseButton(int button, int action, int mods);
    void onScroll(double xoffset, double yoffset);
    void onResize(int width, int height);

private:
    void init();
    void update(float deltaTime);
    void render();
    void processInput(float deltaTime);
    void updateStats();
    
    GLFWwindow* m_window;
    int m_width;
    int m_height;
    std::string m_title;
    
    std::unique_ptr<Renderer> m_renderer;
    std::unique_ptr<Camera> m_camera;
    std::unique_ptr<WindSimulation> m_windSim;
    std::unique_ptr<BladePhysics> m_bladePhysics;
    std::unique_ptr<StatsOverlay> m_statsOverlay;
    std::unique_ptr<CADLoader> m_cadLoader;
    std::shared_ptr<Mesh> m_bladeMesh;
    
    // Input state
    bool m_firstMouse = true;
    double m_lastX = 0.0;
    double m_lastY = 0.0;
    bool m_mousePressed = false;
    
    // Simulation parameters
    float m_windSpeed = 10.0f;
    float m_windAngle = 0.0f;
    float m_rotationSpeed = 0.0f;
    bool m_simulationRunning = true;
    bool m_showParticles = true;
    bool m_showStreamlines = true;
};
