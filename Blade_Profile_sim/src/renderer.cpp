#include "renderer.hpp"
#include "shader.hpp"
#include "mesh.hpp"
#include "camera.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <cmath>

Renderer::Renderer(int width, int height)
    : m_width(width)
    , m_height(height)
{
}

Renderer::~Renderer() {
    glDeleteVertexArrays(1, &m_particleVAO);
    glDeleteBuffers(1, &m_particleVBO);
    glDeleteBuffers(1, &m_particleInstanceVBO);
    glDeleteVertexArrays(1, &m_streamlineVAO);
    glDeleteBuffers(1, &m_streamlineVBO);
    glDeleteVertexArrays(1, &m_gridVAO);
    glDeleteBuffers(1, &m_gridVBO);
    glDeleteVertexArrays(1, &m_arrowVAO);
    glDeleteBuffers(1, &m_arrowVBO);
}

void Renderer::init() {
    // Create mesh shader
    m_meshShader = std::make_unique<Shader>();
    const char* meshVertSrc = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aNormal;
        layout (location = 2) in vec2 aTexCoord;
        
        out vec3 FragPos;
        out vec3 Normal;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform mat3 normalMatrix;
        
        void main() {
            FragPos = vec3(model * vec4(aPos, 1.0));
            Normal = normalMatrix * aNormal;
            gl_Position = projection * view * model * vec4(aPos, 1.0);
        }
    )";
    
    const char* meshFragSrc = R"(
        #version 330 core
        out vec4 FragColor;
        
        in vec3 FragPos;
        in vec3 Normal;
        
        uniform vec3 lightDir;
        uniform vec3 lightColor;
        uniform vec3 ambientColor;
        uniform vec3 objectColor;
        uniform vec3 viewPos;
        
        void main() {
            // Ambient
            vec3 ambient = ambientColor * objectColor;
            
            // Diffuse
            vec3 norm = normalize(Normal);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor * objectColor;
            
            // Specular
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
            vec3 specular = spec * lightColor * 0.5;
            
            vec3 result = ambient + diffuse + specular;
            FragColor = vec4(result, 1.0);
        }
    )";
    m_meshShader->loadFromSource(meshVertSrc, meshFragSrc);
    
    // Create particle shader
    m_particleShader = std::make_unique<Shader>();
    const char* particleVertSrc = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aInstancePos;
        layout (location = 2) in vec3 aInstanceColor;
        layout (location = 3) in float aInstanceSize;
        
        out vec3 Color;
        
        uniform mat4 view;
        uniform mat4 projection;
        
        void main() {
            Color = aInstanceColor;
            vec3 pos = aPos * aInstanceSize + aInstancePos;
            gl_Position = projection * view * vec4(pos, 1.0);
            gl_PointSize = aInstanceSize * 10.0;
        }
    )";
    
    const char* particleFragSrc = R"(
        #version 330 core
        out vec4 FragColor;
        
        in vec3 Color;
        
        void main() {
            vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
            float dist = length(circCoord);
            if (dist > 1.0) discard;
            
            float alpha = 1.0 - smoothstep(0.5, 1.0, dist);
            FragColor = vec4(Color, alpha * 0.8);
        }
    )";
    m_particleShader->loadFromSource(particleVertSrc, particleFragSrc);
    
    // Create line shader
    m_lineShader = std::make_unique<Shader>();
    const char* lineVertSrc = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aColor;
        
        out vec3 Color;
        
        uniform mat4 view;
        uniform mat4 projection;
        
        void main() {
            Color = aColor;
            gl_Position = projection * view * vec4(aPos, 1.0);
        }
    )";
    
    const char* lineFragSrc = R"(
        #version 330 core
        out vec4 FragColor;
        
        in vec3 Color;
        
        void main() {
            FragColor = vec4(Color, 0.8);
        }
    )";
    m_lineShader->loadFromSource(lineVertSrc, lineFragSrc);
    
    // Create grid shader
    m_gridShader = std::make_unique<Shader>();
    const char* gridVertSrc = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        
        uniform mat4 view;
        uniform mat4 projection;
        
        void main() {
            gl_Position = projection * view * vec4(aPos, 1.0);
        }
    )";
    
    const char* gridFragSrc = R"(
        #version 330 core
        out vec4 FragColor;
        
        uniform vec3 color;
        uniform float alpha;
        
        void main() {
            FragColor = vec4(color, alpha);
        }
    )";
    m_gridShader->loadFromSource(gridVertSrc, gridFragSrc);
    
    setupParticleBuffers();
    setupStreamlineBuffers();
    setupGridBuffers();
    setupArrowBuffers();
}

void Renderer::setupParticleBuffers() {
    // Particle quad vertices
    float quadVertices[] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.5f,  0.5f, 0.0f,
        -0.5f,  0.5f, 0.0f,
    };
    
    glGenVertexArrays(1, &m_particleVAO);
    glGenBuffers(1, &m_particleVBO);
    glGenBuffers(1, &m_particleInstanceVBO);
    
    glBindVertexArray(m_particleVAO);
    
    // Quad vertices
    glBindBuffer(GL_ARRAY_BUFFER, m_particleVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Instance data buffer (will be filled during render)
    glBindBuffer(GL_ARRAY_BUFFER, m_particleInstanceVBO);
    glBufferData(GL_ARRAY_BUFFER, 10000 * 7 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    
    // Instance position
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1);
    
    // Instance color
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);
    
    // Instance size
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(3);
    glVertexAttribDivisor(3, 1);
    
    glBindVertexArray(0);
}

void Renderer::setupStreamlineBuffers() {
    glGenVertexArrays(1, &m_streamlineVAO);
    glGenBuffers(1, &m_streamlineVBO);
    
    glBindVertexArray(m_streamlineVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_streamlineVBO);
    glBufferData(GL_ARRAY_BUFFER, 100000 * 6 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    
    // Position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
}

void Renderer::setupGridBuffers() {
    std::vector<float> gridVertices;
    float gridSize = 10.0f;
    int gridLines = 21;
    float step = gridSize * 2.0f / (gridLines - 1);
    
    for (int i = 0; i < gridLines; ++i) {
        float pos = -gridSize + i * step;
        // X-aligned lines
        gridVertices.push_back(-gridSize); gridVertices.push_back(0.0f); gridVertices.push_back(pos);
        gridVertices.push_back(gridSize); gridVertices.push_back(0.0f); gridVertices.push_back(pos);
        // Z-aligned lines
        gridVertices.push_back(pos); gridVertices.push_back(0.0f); gridVertices.push_back(-gridSize);
        gridVertices.push_back(pos); gridVertices.push_back(0.0f); gridVertices.push_back(gridSize);
    }
    
    glGenVertexArrays(1, &m_gridVAO);
    glGenBuffers(1, &m_gridVBO);
    
    glBindVertexArray(m_gridVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_gridVBO);
    glBufferData(GL_ARRAY_BUFFER, gridVertices.size() * sizeof(float), gridVertices.data(), GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
}

void Renderer::setupArrowBuffers() {
    // Arrow shape
    float arrowVertices[] = {
        // Shaft
        0.0f, 0.0f, 0.0f,
        0.8f, 0.0f, 0.0f,
        // Head
        0.8f, 0.0f, 0.0f,
        0.6f, 0.1f, 0.0f,
        0.8f, 0.0f, 0.0f,
        0.6f, -0.1f, 0.0f,
    };
    
    glGenVertexArrays(1, &m_arrowVAO);
    glGenBuffers(1, &m_arrowVBO);
    
    glBindVertexArray(m_arrowVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_arrowVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(arrowVertices), arrowVertices, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
}

void Renderer::resize(int width, int height) {
    m_width = width;
    m_height = height;
}

void Renderer::clear() {
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Renderer::renderMesh(const Mesh& mesh, const Camera& camera, const glm::mat4& model) {
    m_meshShader->use();
    
    glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(model)));
    
    m_meshShader->setMat4("model", model);
    m_meshShader->setMat4("view", camera.getViewMatrix());
    m_meshShader->setMat4("projection", camera.getProjectionMatrix());
    m_meshShader->setMat3("normalMatrix", normalMatrix);
    m_meshShader->setVec3("lightDir", m_lightDirection);
    m_meshShader->setVec3("lightColor", m_lightColor);
    m_meshShader->setVec3("ambientColor", m_ambientLight);
    m_meshShader->setVec3("objectColor", mesh.getColor());
    m_meshShader->setVec3("viewPos", camera.getPosition());
    
    mesh.draw();
}

void Renderer::renderParticles(const std::vector<Particle>& particles, const Camera& camera) {
    if (particles.empty()) return;
    
    // Prepare instance data
    std::vector<float> instanceData;
    instanceData.reserve(particles.size() * 7);
    
    for (const auto& p : particles) {
        if (p.life > 0.0f) {
            instanceData.push_back(p.position.x);
            instanceData.push_back(p.position.y);
            instanceData.push_back(p.position.z);
            instanceData.push_back(p.color.r);
            instanceData.push_back(p.color.g);
            instanceData.push_back(p.color.b);
            instanceData.push_back(p.size);
        }
    }
    
    if (instanceData.empty()) return;
    
    glBindBuffer(GL_ARRAY_BUFFER, m_particleInstanceVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, instanceData.size() * sizeof(float), instanceData.data());
    
    m_particleShader->use();
    m_particleShader->setMat4("view", camera.getViewMatrix());
    m_particleShader->setMat4("projection", camera.getProjectionMatrix());
    
    glEnable(GL_PROGRAM_POINT_SIZE);
    glBindVertexArray(m_particleVAO);
    glDrawArraysInstanced(GL_POINTS, 0, 1, static_cast<GLsizei>(instanceData.size() / 7));
    glBindVertexArray(0);
    glDisable(GL_PROGRAM_POINT_SIZE);
}

void Renderer::renderStreamlines(const std::vector<Streamline>& streamlines, const Camera& camera) {
    if (streamlines.empty()) return;
    
    std::vector<float> lineData;
    std::vector<int> lineSizes;
    
    for (const auto& sl : streamlines) {
        if (sl.points.size() < 2) continue;
        
        int startIdx = static_cast<int>(lineData.size() / 6);
        for (const auto& p : sl.points) {
            lineData.push_back(p.x);
            lineData.push_back(p.y);
            lineData.push_back(p.z);
            lineData.push_back(sl.color.r);
            lineData.push_back(sl.color.g);
            lineData.push_back(sl.color.b);
        }
        lineSizes.push_back(static_cast<int>(sl.points.size()));
    }
    
    if (lineData.empty()) return;
    
    glBindBuffer(GL_ARRAY_BUFFER, m_streamlineVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, lineData.size() * sizeof(float), lineData.data());
    
    m_lineShader->use();
    m_lineShader->setMat4("view", camera.getViewMatrix());
    m_lineShader->setMat4("projection", camera.getProjectionMatrix());
    
    glBindVertexArray(m_streamlineVAO);
    
    int offset = 0;
    for (int size : lineSizes) {
        glDrawArrays(GL_LINE_STRIP, offset, size);
        offset += size;
    }
    
    glBindVertexArray(0);
}

void Renderer::renderGrid(const Camera& camera) {
    m_gridShader->use();
    m_gridShader->setMat4("view", camera.getViewMatrix());
    m_gridShader->setMat4("projection", camera.getProjectionMatrix());
    m_gridShader->setVec3("color", glm::vec3(0.3f, 0.3f, 0.35f));
    m_gridShader->setFloat("alpha", 0.5f);
    
    glBindVertexArray(m_gridVAO);
    glDrawArrays(GL_LINES, 0, 21 * 4);
    glBindVertexArray(0);
}

void Renderer::renderAxes(const Camera& camera) {
    // Simple axis rendering using grid shader
    float axisLength = 2.0f;
    float axisVertices[] = {
        // X axis (red)
        0.0f, 0.0f, 0.0f, axisLength, 0.0f, 0.0f,
        // Y axis (green)  
        0.0f, 0.0f, 0.0f, 0.0f, axisLength, 0.0f,
        // Z axis (blue)
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, axisLength,
    };
    
    GLuint axisVAO, axisVBO;
    glGenVertexArrays(1, &axisVAO);
    glGenBuffers(1, &axisVBO);
    
    glBindVertexArray(axisVAO);
    glBindBuffer(GL_ARRAY_BUFFER, axisVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(axisVertices), axisVertices, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    m_gridShader->use();
    m_gridShader->setMat4("view", camera.getViewMatrix());
    m_gridShader->setMat4("projection", camera.getProjectionMatrix());
    m_gridShader->setFloat("alpha", 1.0f);
    
    glLineWidth(2.0f);
    
    m_gridShader->setVec3("color", glm::vec3(1.0f, 0.2f, 0.2f));
    glDrawArrays(GL_LINES, 0, 2);
    
    m_gridShader->setVec3("color", glm::vec3(0.2f, 1.0f, 0.2f));
    glDrawArrays(GL_LINES, 2, 2);
    
    m_gridShader->setVec3("color", glm::vec3(0.2f, 0.2f, 1.0f));
    glDrawArrays(GL_LINES, 4, 2);
    
    glLineWidth(1.0f);
    
    glDeleteVertexArrays(1, &axisVAO);
    glDeleteBuffers(1, &axisVBO);
}

void Renderer::renderWindArrows(const glm::vec3& windDirection, float windSpeed, const Camera& camera) {
    // Render wind direction indicators around the scene
    // Not fully implemented for brevity - shown as particles and streamlines
}

void Renderer::setAmbientLight(const glm::vec3& color) {
    m_ambientLight = color;
}

void Renderer::setDirectionalLight(const glm::vec3& direction, const glm::vec3& color) {
    m_lightDirection = glm::normalize(direction);
    m_lightColor = color;
}
