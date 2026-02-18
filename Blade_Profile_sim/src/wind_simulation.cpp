#include "wind_simulation.hpp"
#include "mesh.hpp"

#include <cmath>
#include <algorithm>

WindSimulation::WindSimulation()
    : m_rng(std::random_device{}())
    , m_uniformDist(0.0f, 1.0f)
{
    m_domainMin = -m_domainSize * 0.5f;
    m_domainMax = m_domainSize * 0.5f;
    
    initParticles();
    initStreamlines();
}

WindSimulation::~WindSimulation() {
}

void WindSimulation::setWindSpeed(float speed) {
    m_windSpeed = std::max(0.1f, speed);
}

void WindSimulation::setWindDirection(const glm::vec3& direction) {
    m_windDirection = glm::normalize(direction);
}

void WindSimulation::setMesh(const Mesh* mesh) {
    m_mesh = mesh;
    if (mesh) {
        auto bbox = mesh->getBoundingBox();
        m_domainSize = bbox.size() * 3.0f;
        m_domainMin = bbox.center() - m_domainSize * 0.5f;
        m_domainMax = bbox.center() + m_domainSize * 0.5f;
    }
}

void WindSimulation::setDomainSize(const glm::vec3& size) {
    m_domainSize = size;
    m_domainMin = -size * 0.5f;
    m_domainMax = size * 0.5f;
}

void WindSimulation::update(float deltaTime) {
    updateParticles(deltaTime);
    updateStreamlines();
}

void WindSimulation::reset() {
    initParticles();
    initStreamlines();
}

void WindSimulation::initParticles() {
    m_particles.resize(m_particleCount);
    
    for (auto& p : m_particles) {
        spawnParticle(p);
    }
}

void WindSimulation::initStreamlines() {
    m_streamlines.clear();
    m_streamlines.resize(m_streamlineCount);
    
    // Create streamlines starting from upstream
    for (int i = 0; i < m_streamlineCount; ++i) {
        Streamline& sl = m_streamlines[i];
        
        // Random starting position on the inlet plane
        float y = m_domainMin.y + m_uniformDist(m_rng) * m_domainSize.y;
        float z = m_domainMin.z + m_uniformDist(m_rng) * m_domainSize.z;
        
        glm::vec3 startPos;
        if (m_windDirection.x > 0) {
            startPos = glm::vec3(m_domainMin.x, y, z);
        } else {
            startPos = glm::vec3(m_domainMax.x, y, z);
        }
        
        // Color based on height (blue = low, red = high)
        float t = (y - m_domainMin.y) / m_domainSize.y;
        sl.color = glm::vec3(t, 0.5f, 1.0f - t);
        
        // Trace streamline
        sl.points.clear();
        sl.points.push_back(startPos);
        
        glm::vec3 pos = startPos;
        float stepSize = m_domainSize.x / 100.0f;
        
        for (int step = 0; step < 200; ++step) {
            glm::vec3 vel = calculateFlowVelocity(pos);
            if (glm::length(vel) < 0.001f) break;
            
            pos += glm::normalize(vel) * stepSize;
            
            // Check bounds
            if (pos.x < m_domainMin.x || pos.x > m_domainMax.x ||
                pos.y < m_domainMin.y || pos.y > m_domainMax.y ||
                pos.z < m_domainMin.z || pos.z > m_domainMax.z) {
                break;
            }
            
            sl.points.push_back(pos);
        }
    }
}

void WindSimulation::spawnParticle(Particle& p) {
    // Spawn at inlet
    float y = m_domainMin.y + m_uniformDist(m_rng) * m_domainSize.y;
    float z = m_domainMin.z + m_uniformDist(m_rng) * m_domainSize.z;
    
    if (m_windDirection.x > 0) {
        p.position = glm::vec3(m_domainMin.x, y, z);
    } else {
        p.position = glm::vec3(m_domainMax.x, y, z);
    }
    
    p.velocity = m_windDirection * m_windSpeed;
    p.life = 5.0f + m_uniformDist(m_rng) * 3.0f;
    p.size = 0.02f + m_uniformDist(m_rng) * 0.03f;
    
    // Color based on speed (will be updated)
    float speedRatio = m_windSpeed / 20.0f;
    p.color = glm::vec3(0.3f + speedRatio * 0.5f, 0.6f, 1.0f - speedRatio * 0.3f);
}

void WindSimulation::updateParticles(float deltaTime) {
    for (auto& p : m_particles) {
        if (p.life <= 0.0f) {
            spawnParticle(p);
            continue;
        }
        
        // Get local flow velocity
        glm::vec3 flowVel = calculateFlowVelocity(p.position);
        
        // Update velocity with some smoothing
        p.velocity = glm::mix(p.velocity, flowVel, deltaTime * 5.0f);
        
        // Add turbulence
        if (m_turbulenceEnabled) {
            glm::vec3 turbulence(
                (m_uniformDist(m_rng) - 0.5f) * 2.0f,
                (m_uniformDist(m_rng) - 0.5f) * 2.0f,
                (m_uniformDist(m_rng) - 0.5f) * 2.0f
            );
            p.velocity += turbulence * m_turbulenceIntensity * m_windSpeed * deltaTime;
        }
        
        // Update position
        p.position += p.velocity * deltaTime;
        
        // Update life
        p.life -= deltaTime;
        
        // Update color based on velocity magnitude
        float speed = glm::length(p.velocity);
        float speedRatio = std::min(speed / (m_windSpeed * 2.0f), 1.0f);
        
        // Blue (slow) -> Cyan -> Green -> Yellow -> Red (fast)
        if (speedRatio < 0.25f) {
            p.color = glm::mix(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 1.0f), speedRatio * 4.0f);
        } else if (speedRatio < 0.5f) {
            p.color = glm::mix(glm::vec3(0.0f, 1.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f), (speedRatio - 0.25f) * 4.0f);
        } else if (speedRatio < 0.75f) {
            p.color = glm::mix(glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 0.0f), (speedRatio - 0.5f) * 4.0f);
        } else {
            p.color = glm::mix(glm::vec3(1.0f, 1.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), (speedRatio - 0.75f) * 4.0f);
        }
        
        // Respawn if out of bounds
        if (p.position.x < m_domainMin.x || p.position.x > m_domainMax.x ||
            p.position.y < m_domainMin.y || p.position.y > m_domainMax.y ||
            p.position.z < m_domainMin.z || p.position.z > m_domainMax.z) {
            spawnParticle(p);
        }
    }
}

void WindSimulation::updateStreamlines() {
    // Streamlines are static for now, but could be animated
}

glm::vec3 WindSimulation::calculateFlowVelocity(const glm::vec3& point) const {
    // Base flow velocity
    glm::vec3 baseVel = m_windDirection * m_windSpeed;
    
    if (!m_mesh) return baseVel;
    
    // Calculate distance to mesh surface
    float distToSurface = calculateDistanceToSurface(point);
    
    // If very close to or inside mesh, reduce velocity significantly
    if (distToSurface < 0.01f) {
        return glm::vec3(0.0f);
    }
    
    // Calculate potential flow around the object
    glm::vec3 potentialVel = calculatePotentialFlow(point);
    
    // Blend based on distance
    float blendFactor = std::min(distToSurface * 2.0f, 1.0f);
    
    return glm::mix(potentialVel, baseVel, blendFactor);
}

glm::vec3 WindSimulation::calculatePotentialFlow(const glm::vec3& point) const {
    if (!m_mesh) return m_windDirection * m_windSpeed;
    
    auto bbox = m_mesh->getBoundingBox();
    glm::vec3 center = bbox.center();
    float radius = bbox.maxExtent() * 0.5f;
    
    // Simplified potential flow around sphere (as approximation)
    glm::vec3 r = point - center;
    float rMag = glm::length(r);
    
    if (rMag < radius * 0.5f) {
        return glm::vec3(0.0f);
    }
    
    // Uniform flow + doublet (flow around sphere)
    float U = m_windSpeed;
    glm::vec3 dir = m_windDirection;
    
    // Potential flow around sphere formula
    float r3 = rMag * rMag * rMag;
    float R3 = radius * radius * radius;
    
    glm::vec3 rNorm = r / rMag;
    float dotProd = glm::dot(dir, rNorm);
    
    glm::vec3 velocity = U * dir + (U * R3 / (2.0f * r3)) * (3.0f * dotProd * rNorm - dir);
    
    // Add acceleration around the object (Bernoulli effect)
    float surfaceDist = rMag - radius;
    if (surfaceDist > 0 && surfaceDist < radius) {
        // Speed up flow near surface (simplified Bernoulli)
        float speedup = 1.0f + (1.0f - surfaceDist / radius) * 0.5f;
        velocity *= speedup;
    }
    
    return velocity;
}

float WindSimulation::calculateDistanceToSurface(const glm::vec3& point) const {
    if (!m_mesh) return 1000.0f;
    
    auto bbox = m_mesh->getBoundingBox();
    
    // Simple bounding box distance (approximate)
    glm::vec3 closest = glm::clamp(point, bbox.min, bbox.max);
    return glm::length(point - closest);
}

bool WindSimulation::isInsideMesh(const glm::vec3& point) const {
    if (!m_mesh) return false;
    
    auto bbox = m_mesh->getBoundingBox();
    return (point.x >= bbox.min.x && point.x <= bbox.max.x &&
            point.y >= bbox.min.y && point.y <= bbox.max.y &&
            point.z >= bbox.min.z && point.z <= bbox.max.z);
}

glm::vec3 WindSimulation::getWindVelocityAt(const glm::vec3& point) const {
    return calculateFlowVelocity(point);
}

float WindSimulation::getPressureAt(const glm::vec3& point) const {
    // Bernoulli equation: P + 0.5 * rho * v^2 = const
    glm::vec3 vel = calculateFlowVelocity(point);
    float v2 = glm::dot(vel, vel);
    
    // Reference: freestream
    float v_inf2 = m_windSpeed * m_windSpeed;
    
    // Dynamic pressure difference
    float p_dynamic = 0.5f * m_density * (v_inf2 - v2);
    
    // Pressure coefficient
    float Cp = (v_inf2 - v2) / v_inf2;
    
    // Return pressure relative to atmospheric (101325 Pa)
    return 101325.0f + p_dynamic;
}

void WindSimulation::setParticleCount(int count) {
    m_particleCount = std::max(100, std::min(count, 10000));
    initParticles();
}

void WindSimulation::setStreamlineCount(int count) {
    m_streamlineCount = std::max(10, std::min(count, 200));
    initStreamlines();
}

void WindSimulation::enableTurbulence(bool enable) {
    m_turbulenceEnabled = enable;
}

void WindSimulation::setTurbulenceIntensity(float intensity) {
    m_turbulenceIntensity = std::clamp(intensity, 0.0f, 1.0f);
}

void WindSimulation::setViscosity(float viscosity) {
    m_viscosity = viscosity;
}

void WindSimulation::setDensity(float density) {
    m_density = density;
}
