#pragma once

#include <string>
#include <vector>
#include <memory>

class TextRenderer;
struct BladeStats;

class StatsOverlay {
public:
    StatsOverlay(int screenWidth, int screenHeight);
    ~StatsOverlay();
    
    void init();
    void resize(int width, int height);
    
    void render(const BladeStats& stats, float windSpeed, float windAngle);
    
    void setVisible(bool visible) { m_visible = visible; }
    bool isVisible() const { return m_visible; }
    
    void setPosition(int x, int y);
    void setSize(int width, int height);

private:
    void drawBackground();
    void drawStats(const BladeStats& stats, float windSpeed, float windAngle);
    std::string formatValue(float value, const std::string& unit, int precision = 2) const;
    std::string formatScientific(float value, const std::string& unit) const;
    
    std::unique_ptr<TextRenderer> m_textRenderer;
    
    int m_screenWidth;
    int m_screenHeight;
    int m_boxX;
    int m_boxY;
    int m_boxWidth;
    int m_boxHeight;
    int m_padding = 15;
    int m_lineHeight = 22;
    
    bool m_visible = true;
    
    // OpenGL objects for background
    unsigned int m_bgVAO = 0;
    unsigned int m_bgVBO = 0;
    unsigned int m_bgShader = 0;
};
