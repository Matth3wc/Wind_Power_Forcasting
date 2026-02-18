#include "application.hpp"
#include "renderer.hpp"
#include "camera.hpp"
#include "wind_simulation.hpp"
#include "blade_physics.hpp"
#include "stats_overlay.hpp"
#include "cad_loader.hpp"
#include "mesh.hpp"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <stdexcept>

// Global application pointer for callbacks
static Application* g_app = nullptr;

// Callback wrappers
static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (g_app) g_app->onKeyPress(key, action, mods);
}

static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    if (g_app) g_app->onMouseMove(xpos, ypos);
}

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (g_app) g_app->onMouseButton(button, action, mods);
}

static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    if (g_app) g_app->onScroll(xoffset, yoffset);
}

static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    if (g_app) g_app->onResize(width, height);
}

Application::Application(int width, int height, const std::string& title)
    : m_width(width)
    , m_height(height)
    , m_title(title)
    , m_window(nullptr)
{
    g_app = this;
    init();
}

Application::~Application() {
    g_app = nullptr;
    if (m_window) {
        glfwDestroyWindow(m_window);
    }
    glfwTerminate();
}

void Application::init() {
    // Initialize GLFW
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    
    // Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);  // Anti-aliasing
    
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    
    // Create window
    m_window = glfwCreateWindow(m_width, m_height, m_title.c_str(), nullptr, nullptr);
    if (!m_window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    
    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1);  // V-sync
    
    // Set callbacks
    glfwSetKeyCallback(m_window, keyCallback);
    glfwSetCursorPosCallback(m_window, cursorPosCallback);
    glfwSetMouseButtonCallback(m_window, mouseButtonCallback);
    glfwSetScrollCallback(m_window, scrollCallback);
    glfwSetFramebufferSizeCallback(m_window, framebufferSizeCallback);
    
    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        throw std::runtime_error("Failed to initialize GLAD");
    }
    
    // Enable OpenGL features
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Initialize components
    m_renderer = std::make_unique<Renderer>(m_width, m_height);
    m_renderer->init();
    
    m_camera = std::make_unique<Camera>(45.0f, static_cast<float>(m_width) / m_height);
    m_camera->setPosition(glm::vec3(5.0f, 3.0f, 5.0f));
    m_camera->setTarget(glm::vec3(0.0f));
    
    m_windSim = std::make_unique<WindSimulation>();
    m_bladePhysics = std::make_unique<BladePhysics>();
    m_statsOverlay = std::make_unique<StatsOverlay>(m_width, m_height);
    m_statsOverlay->init();
    
    m_cadLoader = std::make_unique<CADLoader>();
    
    std::cout << "Wind Turbine Blade Simulator initialized\n";
    std::cout << "Controls:\n";
    std::cout << "  Mouse drag: Rotate view\n";
    std::cout << "  Scroll: Zoom\n";
    std::cout << "  WASD: Pan camera\n";
    std::cout << "  +/-: Adjust wind speed\n";
    std::cout << "  Left/Right arrows: Adjust wind angle\n";
    std::cout << "  Space: Toggle simulation\n";
    std::cout << "  P: Toggle particles\n";
    std::cout << "  L: Toggle streamlines\n";
    std::cout << "  H: Toggle stats overlay\n";
    std::cout << "  R: Reset view\n";
    std::cout << "  Esc: Quit\n";
}

void Application::loadBlade(const std::string& filepath) {
    std::cout << "Loading blade from: " << filepath << std::endl;
    
    m_bladeMesh = m_cadLoader->loadFile(filepath);
    
    if (m_bladeMesh) {
        std::cout << "Blade loaded successfully!" << std::endl;
        std::cout << "  Vertices: " << m_bladeMesh->getVertexCount() << std::endl;
        std::cout << "  Triangles: " << m_bladeMesh->getTriangleCount() << std::endl;
        std::cout << "  Surface Area: " << m_bladeMesh->getSurfaceArea() << " m²" << std::endl;
        
        // Center camera on blade
        auto bbox = m_bladeMesh->getBoundingBox();
        m_camera->fitToBoundingBox(bbox.min, bbox.max);
        
        // Set up simulation
        m_windSim->setMesh(m_bladeMesh.get());
        m_windSim->setDomainSize(bbox.size() * 3.0f);
        m_windSim->reset();
        
        m_bladePhysics->setMesh(m_bladeMesh.get());
        m_bladePhysics->setWindSimulation(m_windSim.get());
        m_bladePhysics->calculateStaticProperties();
    } else {
        std::cerr << "Failed to load blade: " << m_cadLoader->getLastError() << std::endl;
    }
}

void Application::run() {
    float lastTime = static_cast<float>(glfwGetTime());
    
    while (!glfwWindowShouldClose(m_window)) {
        float currentTime = static_cast<float>(glfwGetTime());
        float deltaTime = currentTime - lastTime;
        lastTime = currentTime;
        
        glfwPollEvents();
        processInput(deltaTime);
        
        if (m_simulationRunning) {
            update(deltaTime);
        }
        
        render();
        
        glfwSwapBuffers(m_window);
    }
}

void Application::update(float deltaTime) {
    // Update wind simulation
    m_windSim->setWindDirection(glm::vec3(
        std::cos(glm::radians(m_windAngle)),
        0.0f,
        std::sin(glm::radians(m_windAngle))
    ));
    m_windSim->setWindSpeed(m_windSpeed);
    m_windSim->update(deltaTime);
    
    // Update blade physics
    m_bladePhysics->update(deltaTime);
}

void Application::render() {
    m_renderer->clear();
    
    // Render grid and axes
    m_renderer->renderGrid(*m_camera);
    m_renderer->renderAxes(*m_camera);
    
    // Render blade mesh
    if (m_bladeMesh) {
        glm::mat4 model = glm::mat4(1.0f);
        m_renderer->renderMesh(*m_bladeMesh, *m_camera, model);
    }
    
    // Render wind visualization
    if (m_showParticles) {
        m_renderer->renderParticles(m_windSim->getParticles(), *m_camera);
    }
    
    if (m_showStreamlines) {
        m_renderer->renderStreamlines(m_windSim->getStreamlines(), *m_camera);
    }
    
    // Render wind direction arrows
    m_renderer->renderWindArrows(m_windSim->getWindDirection(), m_windSpeed, *m_camera);
    
    // Render stats overlay (top-right corner)
    m_statsOverlay->render(m_bladePhysics->getStats(), m_windSpeed, m_windAngle);
}

void Application::processInput(float deltaTime) {
    // Camera pan with WASD
    float panSpeed = 5.0f * deltaTime;
    
    if (glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS) {
        m_camera->pan(0.0f, panSpeed);
    }
    if (glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS) {
        m_camera->pan(0.0f, -panSpeed);
    }
    if (glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS) {
        m_camera->pan(-panSpeed, 0.0f);
    }
    if (glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS) {
        m_camera->pan(panSpeed, 0.0f);
    }
}

void Application::onKeyPress(int key, int action, int mods) {
    if (action != GLFW_PRESS) return;
    
    switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(m_window, true);
            break;
        case GLFW_KEY_SPACE:
            m_simulationRunning = !m_simulationRunning;
            std::cout << "Simulation " << (m_simulationRunning ? "running" : "paused") << std::endl;
            break;
        case GLFW_KEY_P:
            m_showParticles = !m_showParticles;
            break;
        case GLFW_KEY_L:
            m_showStreamlines = !m_showStreamlines;
            break;
        case GLFW_KEY_H:
            m_statsOverlay->setVisible(!m_statsOverlay->isVisible());
            break;
        case GLFW_KEY_R:
            m_camera->reset();
            if (m_bladeMesh) {
                auto bbox = m_bladeMesh->getBoundingBox();
                m_camera->fitToBoundingBox(bbox.min, bbox.max);
            }
            break;
        case GLFW_KEY_EQUAL:
        case GLFW_KEY_KP_ADD:
            m_windSpeed = std::min(m_windSpeed + 1.0f, 50.0f);
            std::cout << "Wind speed: " << m_windSpeed << " m/s" << std::endl;
            break;
        case GLFW_KEY_MINUS:
        case GLFW_KEY_KP_SUBTRACT:
            m_windSpeed = std::max(m_windSpeed - 1.0f, 1.0f);
            std::cout << "Wind speed: " << m_windSpeed << " m/s" << std::endl;
            break;
        case GLFW_KEY_LEFT:
            m_windAngle -= 10.0f;
            if (m_windAngle < -180.0f) m_windAngle += 360.0f;
            std::cout << "Wind angle: " << m_windAngle << "°" << std::endl;
            break;
        case GLFW_KEY_RIGHT:
            m_windAngle += 10.0f;
            if (m_windAngle > 180.0f) m_windAngle -= 360.0f;
            std::cout << "Wind angle: " << m_windAngle << "°" << std::endl;
            break;
    }
}

void Application::onMouseMove(double xpos, double ypos) {
    if (m_firstMouse) {
        m_lastX = xpos;
        m_lastY = ypos;
        m_firstMouse = false;
    }
    
    float xoffset = static_cast<float>(xpos - m_lastX);
    float yoffset = static_cast<float>(m_lastY - ypos);
    
    m_lastX = xpos;
    m_lastY = ypos;
    
    if (m_mousePressed) {
        m_camera->orbit(xoffset * 0.5f, yoffset * 0.5f);
    }
}

void Application::onMouseButton(int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        m_mousePressed = (action == GLFW_PRESS);
    }
}

void Application::onScroll(double xoffset, double yoffset) {
    m_camera->zoom(static_cast<float>(yoffset) * -0.5f);
}

void Application::onResize(int width, int height) {
    m_width = width;
    m_height = height;
    
    glViewport(0, 0, width, height);
    m_camera->setAspectRatio(static_cast<float>(width) / height);
    m_renderer->resize(width, height);
    m_statsOverlay->resize(width, height);
}
