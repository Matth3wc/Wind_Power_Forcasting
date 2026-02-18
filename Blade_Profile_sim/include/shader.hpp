#pragma once
#include <glad/glad.h>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>

class Shader {
public:
    Shader();
    ~Shader();
    
    bool loadFromFile(const std::string& vertexPath, const std::string& fragmentPath);
    bool loadFromSource(const std::string& vertexSource, const std::string& fragmentSource);
    
    void use() const;
    
    void setInt(const std::string& name, int value) const;
    void setFloat(const std::string& name, float value) const;
    void setVec2(const std::string& name, const glm::vec2& value) const;
    void setVec3(const std::string& name, const glm::vec3& value) const;
    void setVec4(const std::string& name, const glm::vec4& value) const;
    void setMat3(const std::string& name, const glm::mat3& value) const;
    void setMat4(const std::string& name, const glm::mat4& value) const;
    
    GLuint getProgram() const { return m_program; }

private:
    GLuint compileShader(GLenum type, const std::string& source);
    bool linkProgram(GLuint vertex, GLuint fragment);
    
    GLuint m_program = 0;
};
