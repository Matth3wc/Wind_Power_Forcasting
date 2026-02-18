#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
public:
    Camera(float fov = 45.0f, float aspectRatio = 16.0f/9.0f, float nearPlane = 0.1f, float farPlane = 1000.0f);
    
    void setPosition(const glm::vec3& position);
    void setTarget(const glm::vec3& target);
    void setAspectRatio(float aspectRatio);
    
    void orbit(float deltaYaw, float deltaPitch);
    void pan(float deltaX, float deltaY);
    void zoom(float delta);
    void reset();
    
    // Fit camera to view bounding box
    void fitToBoundingBox(const glm::vec3& min, const glm::vec3& max);
    
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;
    glm::mat4 getViewProjectionMatrix() const;
    
    glm::vec3 getPosition() const { return m_position; }
    glm::vec3 getTarget() const { return m_target; }
    glm::vec3 getForward() const;
    glm::vec3 getRight() const;
    glm::vec3 getUp() const;
    
    float getDistance() const { return m_distance; }
    float getYaw() const { return m_yaw; }
    float getPitch() const { return m_pitch; }

private:
    void updatePosition();
    
    glm::vec3 m_position = glm::vec3(0.0f, 0.0f, 10.0f);
    glm::vec3 m_target = glm::vec3(0.0f);
    glm::vec3 m_up = glm::vec3(0.0f, 1.0f, 0.0f);
    
    float m_fov;
    float m_aspectRatio;
    float m_nearPlane;
    float m_farPlane;
    
    // Orbital camera parameters
    float m_distance = 10.0f;
    float m_yaw = 0.0f;
    float m_pitch = 30.0f;
    
    // Limits
    float m_minPitch = -89.0f;
    float m_maxPitch = 89.0f;
    float m_minDistance = 0.1f;
    float m_maxDistance = 1000.0f;
};
