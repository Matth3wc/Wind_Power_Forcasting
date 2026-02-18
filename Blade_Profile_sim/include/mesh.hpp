#pragma once
#include <glad/glad.h>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoords;
};

struct BoundingBox {
    glm::vec3 min;
    glm::vec3 max;
    
    glm::vec3 center() const { return (min + max) * 0.5f; }
    glm::vec3 size() const { return max - min; }
    float maxExtent() const {
        glm::vec3 s = size();
        return std::max({s.x, s.y, s.z});
    }
};

struct Triangle {
    glm::vec3 v0, v1, v2;
    glm::vec3 normal;
    float area;
};

class Mesh {
public:
    Mesh();
    ~Mesh();
    
    void setVertices(const std::vector<Vertex>& vertices);
    void setIndices(const std::vector<unsigned int>& indices);
    void build();
    
    void draw() const;
    
    const BoundingBox& getBoundingBox() const { return m_boundingBox; }
    const std::vector<Triangle>& getTriangles() const { return m_triangles; }
    float getSurfaceArea() const { return m_surfaceArea; }
    float getVolume() const { return m_volume; }
    size_t getVertexCount() const { return m_vertices.size(); }
    size_t getTriangleCount() const { return m_indices.size() / 3; }
    
    glm::vec3 getColor() const { return m_color; }
    void setColor(const glm::vec3& color) { m_color = color; }
    
private:
    void calculateBoundingBox();
    void calculateTriangles();
    void calculateSurfaceArea();
    void calculateVolume();
    
    std::vector<Vertex> m_vertices;
    std::vector<unsigned int> m_indices;
    std::vector<Triangle> m_triangles;
    BoundingBox m_boundingBox;
    float m_surfaceArea = 0.0f;
    float m_volume = 0.0f;
    glm::vec3 m_color = glm::vec3(0.7f, 0.7f, 0.8f);
    
    GLuint m_VAO = 0;
    GLuint m_VBO = 0;
    GLuint m_EBO = 0;
    bool m_built = false;
};
