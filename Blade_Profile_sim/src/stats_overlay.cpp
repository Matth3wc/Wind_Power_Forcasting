#include "stats_overlay.hpp"
#include "text_renderer.hpp"
#include "blade_physics.hpp"

#include <glad/glad.h>
#include <sstream>
#include <iomanip>
#include <cmath>

StatsOverlay::StatsOverlay(int screenWidth, int screenHeight)
    : m_screenWidth(screenWidth)
    , m_screenHeight(screenHeight)
{
    // Position in top-right corner
    m_boxWidth = 320;
    m_boxHeight = 480;
    m_boxX = screenWidth - m_boxWidth - 20;
    m_boxY = 20;
}

StatsOverlay::~StatsOverlay() {
    if (m_bgVAO) {
        glDeleteVertexArrays(1, &m_bgVAO);
        glDeleteBuffers(1, &m_bgVBO);
    }
    if (m_bgShader) {
        glDeleteProgram(m_bgShader);
    }
}

void StatsOverlay::init() {
    m_textRenderer = std::make_unique<TextRenderer>(m_screenWidth, m_screenHeight);
    m_textRenderer->init();
    
    // Create background shader
    const char* bgVertSrc = R"(
        #version 330 core
        layout (location = 0) in vec2 aPos;
        
        uniform mat4 projection;
        
        void main() {
            gl_Position = projection * vec4(aPos, 0.0, 1.0);
        }
    )";
    
    const char* bgFragSrc = R"(
        #version 330 core
        out vec4 FragColor;
        
        uniform vec4 color;
        
        void main() {
            FragColor = color;
        }
    )";
    
    // Compile shaders
    GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertShader, 1, &bgVertSrc, nullptr);
    glCompileShader(vertShader);
    
    GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragShader, 1, &bgFragSrc, nullptr);
    glCompileShader(fragShader);
    
    m_bgShader = glCreateProgram();
    glAttachShader(m_bgShader, vertShader);
    glAttachShader(m_bgShader, fragShader);
    glLinkProgram(m_bgShader);
    
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);
    
    // Create background quad
    glGenVertexArrays(1, &m_bgVAO);
    glGenBuffers(1, &m_bgVBO);
    
    glBindVertexArray(m_bgVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_bgVBO);
    glBufferData(GL_ARRAY_BUFFER, 6 * 2 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
}

void StatsOverlay::resize(int width, int height) {
    m_screenWidth = width;
    m_screenHeight = height;
    
    // Reposition to top-right
    m_boxX = width - m_boxWidth - 20;
    m_boxY = 20;
    
    m_textRenderer->resize(width, height);
}

void StatsOverlay::render(const BladeStats& stats, float windSpeed, float windAngle) {
    if (!m_visible) return;
    
    drawBackground();
    drawStats(stats, windSpeed, windAngle);
}

void StatsOverlay::drawBackground() {
    // Update quad vertices
    float x1 = static_cast<float>(m_boxX);
    float y1 = static_cast<float>(m_boxY);
    float x2 = static_cast<float>(m_boxX + m_boxWidth);
    float y2 = static_cast<float>(m_boxY + m_boxHeight);
    
    float vertices[] = {
        x1, y1,
        x2, y1,
        x2, y2,
        x1, y1,
        x2, y2,
        x1, y2
    };
    
    glBindBuffer(GL_ARRAY_BUFFER, m_bgVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
    
    // Set up orthographic projection
    float proj[16] = {
        2.0f / m_screenWidth, 0.0f, 0.0f, 0.0f,
        0.0f, -2.0f / m_screenHeight, 0.0f, 0.0f,
        0.0f, 0.0f, -1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f, 1.0f
    };
    
    glUseProgram(m_bgShader);
    glUniformMatrix4fv(glGetUniformLocation(m_bgShader, "projection"), 1, GL_FALSE, proj);
    glUniform4f(glGetUniformLocation(m_bgShader, "color"), 0.1f, 0.1f, 0.15f, 0.85f);
    
    glDisable(GL_DEPTH_TEST);
    glBindVertexArray(m_bgVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    glEnable(GL_DEPTH_TEST);
}

void StatsOverlay::drawStats(const BladeStats& stats, float windSpeed, float windAngle) {
    float x = static_cast<float>(m_boxX + m_padding);
    float y = static_cast<float>(m_boxY + m_padding);
    float scale = 0.5f;
    
    glm::vec3 titleColor(0.3f, 0.8f, 1.0f);
    glm::vec3 labelColor(0.8f, 0.8f, 0.8f);
    glm::vec3 valueColor(1.0f, 1.0f, 1.0f);
    glm::vec3 goodColor(0.3f, 1.0f, 0.5f);
    glm::vec3 warnColor(1.0f, 0.8f, 0.3f);
    
    // Title
    m_textRenderer->renderText("BLADE PERFORMANCE", x, y, 0.6f, titleColor);
    y += m_lineHeight * 1.5f;
    
    // Wind conditions
    m_textRenderer->renderText("=== Wind Conditions ===", x, y, scale, titleColor);
    y += m_lineHeight;
    m_textRenderer->renderText("Wind Speed: " + formatValue(windSpeed, "m/s", 1), x, y, scale, valueColor);
    y += m_lineHeight;
    m_textRenderer->renderText("Wind Angle: " + formatValue(windAngle, "deg", 1), x, y, scale, valueColor);
    y += m_lineHeight * 1.2f;
    
    // Geometry
    m_textRenderer->renderText("=== Geometry ===", x, y, scale, titleColor);
    y += m_lineHeight;
    m_textRenderer->renderText("Surface Area: " + formatValue(stats.surfaceArea, "m2"), x, y, scale, valueColor);
    y += m_lineHeight;
    m_textRenderer->renderText("Span Length: " + formatValue(stats.spanLength, "m"), x, y, scale, valueColor);
    y += m_lineHeight;
    m_textRenderer->renderText("Chord Length: " + formatValue(stats.chordLength, "m"), x, y, scale, valueColor);
    y += m_lineHeight;
    m_textRenderer->renderText("Aspect Ratio: " + formatValue(stats.aspectRatio, ""), x, y, scale, valueColor);
    y += m_lineHeight * 1.2f;
    
    // Aerodynamics
    m_textRenderer->renderText("=== Aerodynamics ===", x, y, scale, titleColor);
    y += m_lineHeight;
    m_textRenderer->renderText("Cl (Lift Coef): " + formatValue(stats.liftCoefficient, ""), x, y, scale, valueColor);
    y += m_lineHeight;
    m_textRenderer->renderText("Cd (Drag Coef): " + formatValue(stats.dragCoefficient, "", 4), x, y, scale, valueColor);
    y += m_lineHeight;
    m_textRenderer->renderText("L/D Ratio: " + formatValue(stats.liftToDragRatio, ""), x, y, scale, 
                               stats.liftToDragRatio > 10 ? goodColor : valueColor);
    y += m_lineHeight;
    m_textRenderer->renderText("Angle of Attack: " + formatValue(stats.angleOfAttack, "deg"), x, y, scale, valueColor);
    y += m_lineHeight * 1.2f;
    
    // Forces
    m_textRenderer->renderText("=== Forces ===", x, y, scale, titleColor);
    y += m_lineHeight;
    m_textRenderer->renderText("Lift Force: " + formatScientific(stats.liftMagnitude, "N"), x, y, scale, valueColor);
    y += m_lineHeight;
    m_textRenderer->renderText("Drag Force: " + formatScientific(stats.dragMagnitude, "N"), x, y, scale, valueColor);
    y += m_lineHeight * 1.2f;
    
    // Performance
    m_textRenderer->renderText("=== Performance ===", x, y, scale, titleColor);
    y += m_lineHeight;
    m_textRenderer->renderText("Power: " + formatScientific(stats.power, "W"), x, y, scale, valueColor);
    y += m_lineHeight;
    m_textRenderer->renderText("Torque: " + formatScientific(stats.torque, "Nm"), x, y, scale, valueColor);
    y += m_lineHeight;
    m_textRenderer->renderText("Cp (Power Coef): " + formatValue(stats.powerCoefficient, "", 3), x, y, scale, 
                               stats.powerCoefficient > 0.3f ? goodColor : valueColor);
    y += m_lineHeight;
    m_textRenderer->renderText("Efficiency: " + formatValue(stats.efficiency, "%", 1), x, y, scale,
                               stats.efficiency > 50.0f ? goodColor : warnColor);
    y += m_lineHeight * 1.2f;
    
    // Flow
    m_textRenderer->renderText("=== Flow Characteristics ===", x, y, scale, titleColor);
    y += m_lineHeight;
    m_textRenderer->renderText("Reynolds #: " + formatScientific(stats.reynoldsNumber, ""), x, y, scale, valueColor);
    y += m_lineHeight;
    m_textRenderer->renderText("Tip Speed Ratio: " + formatValue(stats.tipSpeedRatio, ""), x, y, scale, valueColor);
}

std::string StatsOverlay::formatValue(float value, const std::string& unit, int precision) const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(precision) << value;
    if (!unit.empty()) {
        ss << " " << unit;
    }
    return ss.str();
}

std::string StatsOverlay::formatScientific(float value, const std::string& unit) const {
    std::ostringstream ss;
    
    if (std::abs(value) >= 1e6 || (std::abs(value) < 0.01 && value != 0)) {
        ss << std::scientific << std::setprecision(2) << value;
    } else {
        ss << std::fixed << std::setprecision(2) << value;
    }
    
    if (!unit.empty()) {
        ss << " " << unit;
    }
    return ss.str();
}

void StatsOverlay::setPosition(int x, int y) {
    m_boxX = x;
    m_boxY = y;
}

void StatsOverlay::setSize(int width, int height) {
    m_boxWidth = width;
    m_boxHeight = height;
}
