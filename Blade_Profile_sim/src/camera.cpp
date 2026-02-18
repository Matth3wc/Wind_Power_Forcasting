#include "camera.hpp"

#include <cmath>
#include <algorithm>

Camera::Camera(float fov, float aspectRatio, float nearPlane, float farPlane)
    : m_fov(fov)
    , m_aspectRatio(aspectRatio)
    , m_nearPlane(nearPlane)
    , m_farPlane(farPlane)
{
    updatePosition();
}

void Camera::setPosition(const glm::vec3& position) {
    m_position = position;
    m_distance = glm::length(position - m_target);
    
    // Calculate yaw and pitch from position
    glm::vec3 dir = glm::normalize(position - m_target);
    m_pitch = glm::degrees(std::asin(dir.y));
    m_yaw = glm::degrees(std::atan2(dir.x, dir.z));
}

void Camera::setTarget(const glm::vec3& target) {
    m_target = target;
    updatePosition();
}

void Camera::setAspectRatio(float aspectRatio) {
    m_aspectRatio = aspectRatio;
}

void Camera::orbit(float deltaYaw, float deltaPitch) {
    m_yaw += deltaYaw;
    m_pitch += deltaPitch;
    
    // Clamp pitch
    m_pitch = std::clamp(m_pitch, m_minPitch, m_maxPitch);
    
    updatePosition();
}

void Camera::pan(float deltaX, float deltaY) {
    glm::vec3 right = getRight();
    glm::vec3 up = getUp();
    
    glm::vec3 panOffset = right * deltaX + up * deltaY;
    m_target += panOffset;
    
    updatePosition();
}

void Camera::zoom(float delta) {
    m_distance *= (1.0f + delta * 0.1f);
    m_distance = std::clamp(m_distance, m_minDistance, m_maxDistance);
    
    updatePosition();
}

void Camera::reset() {
    m_target = glm::vec3(0.0f);
    m_distance = 10.0f;
    m_yaw = 0.0f;
    m_pitch = 30.0f;
    
    updatePosition();
}

void Camera::fitToBoundingBox(const glm::vec3& min, const glm::vec3& max) {
    glm::vec3 center = (min + max) * 0.5f;
    glm::vec3 size = max - min;
    float maxDim = std::max({size.x, size.y, size.z});
    
    m_target = center;
    
    // Calculate distance to fit object in view
    float halfFov = glm::radians(m_fov * 0.5f);
    m_distance = maxDim / (2.0f * std::tan(halfFov)) * 1.5f;
    m_distance = std::clamp(m_distance, m_minDistance, m_maxDistance);
    
    updatePosition();
}

void Camera::updatePosition() {
    float pitchRad = glm::radians(m_pitch);
    float yawRad = glm::radians(m_yaw);
    
    m_position.x = m_target.x + m_distance * std::cos(pitchRad) * std::sin(yawRad);
    m_position.y = m_target.y + m_distance * std::sin(pitchRad);
    m_position.z = m_target.z + m_distance * std::cos(pitchRad) * std::cos(yawRad);
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(m_position, m_target, m_up);
}

glm::mat4 Camera::getProjectionMatrix() const {
    return glm::perspective(glm::radians(m_fov), m_aspectRatio, m_nearPlane, m_farPlane);
}

glm::mat4 Camera::getViewProjectionMatrix() const {
    return getProjectionMatrix() * getViewMatrix();
}

glm::vec3 Camera::getForward() const {
    return glm::normalize(m_target - m_position);
}

glm::vec3 Camera::getRight() const {
    return glm::normalize(glm::cross(getForward(), m_up));
}

glm::vec3 Camera::getUp() const {
    return glm::normalize(glm::cross(getRight(), getForward()));
}
