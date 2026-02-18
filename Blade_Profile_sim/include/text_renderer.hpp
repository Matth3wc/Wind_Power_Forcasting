#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <map>
#include <string>

struct Character {
    unsigned int textureID;
    glm::ivec2 size;
    glm::ivec2 bearing;
    unsigned int advance;
};

class TextRenderer {
public:
    TextRenderer(int screenWidth, int screenHeight);
    ~TextRenderer();
    
    bool init();
    void resize(int width, int height);
    
    void renderText(const std::string& text, float x, float y, float scale, const glm::vec3& color);
    
    float getTextWidth(const std::string& text, float scale) const;
    float getTextHeight(float scale) const;

private:
    void loadFont();
    void createFontTexture();
    
    std::map<char, Character> m_characters;
    unsigned int m_VAO = 0;
    unsigned int m_VBO = 0;
    unsigned int m_shaderProgram = 0;
    
    int m_screenWidth;
    int m_screenHeight;
    
    glm::mat4 m_projection;
    
    // Basic bitmap font data (embedded)
    static const int FONT_WIDTH = 8;
    static const int FONT_HEIGHT = 16;
};
