#pragma once
#include <glad/glad.h>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

class Shader;
class Mesh;
class Camera;

struct Particle {
    glm::vec3 position;
    glm::vec3 velocity;
    float life;
    float size;
    glm::vec3 color;
};

struct Streamline {
    std::vector<glm::vec3> points;
    glm::vec3 color;
};

class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();
    
    void init();
    void resize(int width, int height);
    void clear();
    
    void renderMesh(const Mesh& mesh, const Camera& camera, const glm::mat4& model);
    void renderParticles(const std::vector<Particle>& particles, const Camera& camera);
    void renderStreamlines(const std::vector<Streamline>& streamlines, const Camera& camera);
    void renderGrid(const Camera& camera);
    void renderAxes(const Camera& camera);
    void renderWindArrows(const glm::vec3& windDirection, float windSpeed, const Camera& camera);
    
    void setAmbientLight(const glm::vec3& color);
    void setDirectionalLight(const glm::vec3& direction, const glm::vec3& color);
    
private:
    void setupParticleBuffers();
    void setupStreamlineBuffers();
    void setupGridBuffers();
    void setupArrowBuffers();
    
    int m_width;
    int m_height;
    
    std::unique_ptr<Shader> m_meshShader;
    std::unique_ptr<Shader> m_particleShader;
    std::unique_ptr<Shader> m_lineShader;
    std::unique_ptr<Shader> m_gridShader;
    
    // Particle rendering
    GLuint m_particleVAO;
    GLuint m_particleVBO;
    GLuint m_particleInstanceVBO;
    
    // Streamline rendering
    GLuint m_streamlineVAO;
    GLuint m_streamlineVBO;
    
    // Grid rendering
    GLuint m_gridVAO;
    GLuint m_gridVBO;
    
    // Arrow rendering
    GLuint m_arrowVAO;
    GLuint m_arrowVBO;
    
    // Lighting
    glm::vec3 m_ambientLight = glm::vec3(0.2f);
    glm::vec3 m_lightDirection = glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f));
    glm::vec3 m_lightColor = glm::vec3(1.0f);
};
