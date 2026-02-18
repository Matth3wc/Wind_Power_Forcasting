#include "mesh.hpp"

#include <cmath>
#include <algorithm>

Mesh::Mesh() {
}

Mesh::~Mesh() {
    if (m_VAO) {
        glDeleteVertexArrays(1, &m_VAO);
        glDeleteBuffers(1, &m_VBO);
        glDeleteBuffers(1, &m_EBO);
    }
}

void Mesh::setVertices(const std::vector<Vertex>& vertices) {
    m_vertices = vertices;
}

void Mesh::setIndices(const std::vector<unsigned int>& indices) {
    m_indices = indices;
}

void Mesh::build() {
    if (m_vertices.empty()) return;
    
    calculateBoundingBox();
    calculateTriangles();
    calculateSurfaceArea();
    calculateVolume();
    
    // Create OpenGL buffers
    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);
    glGenBuffers(1, &m_EBO);
    
    glBindVertexArray(m_VAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, m_vertices.size() * sizeof(Vertex), m_vertices.data(), GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_indices.size() * sizeof(unsigned int), m_indices.data(), GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    glEnableVertexAttribArray(1);
    
    // Texture coordinate attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));
    glEnableVertexAttribArray(2);
    
    glBindVertexArray(0);
    
    m_built = true;
}

void Mesh::draw() const {
    if (!m_built || !m_VAO) return;
    
    glBindVertexArray(m_VAO);
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(m_indices.size()), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Mesh::calculateBoundingBox() {
    if (m_vertices.empty()) return;
    
    m_boundingBox.min = m_vertices[0].position;
    m_boundingBox.max = m_vertices[0].position;
    
    for (const auto& v : m_vertices) {
        m_boundingBox.min = glm::min(m_boundingBox.min, v.position);
        m_boundingBox.max = glm::max(m_boundingBox.max, v.position);
    }
}

void Mesh::calculateTriangles() {
    m_triangles.clear();
    m_triangles.reserve(m_indices.size() / 3);
    
    for (size_t i = 0; i < m_indices.size(); i += 3) {
        Triangle tri;
        tri.v0 = m_vertices[m_indices[i]].position;
        tri.v1 = m_vertices[m_indices[i + 1]].position;
        tri.v2 = m_vertices[m_indices[i + 2]].position;
        
        glm::vec3 edge1 = tri.v1 - tri.v0;
        glm::vec3 edge2 = tri.v2 - tri.v0;
        glm::vec3 cross = glm::cross(edge1, edge2);
        
        tri.area = glm::length(cross) * 0.5f;
        tri.normal = glm::normalize(cross);
        
        m_triangles.push_back(tri);
    }
}

void Mesh::calculateSurfaceArea() {
    m_surfaceArea = 0.0f;
    for (const auto& tri : m_triangles) {
        m_surfaceArea += tri.area;
    }
}

void Mesh::calculateVolume() {
    // Calculate signed volume using divergence theorem
    m_volume = 0.0f;
    
    for (const auto& tri : m_triangles) {
        // Volume contribution from triangle
        float signedVolume = glm::dot(tri.v0, glm::cross(tri.v1, tri.v2)) / 6.0f;
        m_volume += signedVolume;
    }
    
    m_volume = std::abs(m_volume);
}
