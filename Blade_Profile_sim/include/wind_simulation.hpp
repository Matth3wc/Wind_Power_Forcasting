#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <random>

class Mesh;
struct Particle;
struct Streamline;

// Wind field simulation using simplified CFD principles
class WindSimulation {
public:
    WindSimulation();
    ~WindSimulation();
    
    void setWindSpeed(float speed);
    void setWindDirection(const glm::vec3& direction);
    void setMesh(const Mesh* mesh);
    void setDomainSize(const glm::vec3& size);
    
    void update(float deltaTime);
    void reset();
    
    // Get visualization data
    const std::vector<Particle>& getParticles() const { return m_particles; }
    const std::vector<Streamline>& getStreamlines() const { return m_streamlines; }
    
    // Get wind properties at a point
    glm::vec3 getWindVelocityAt(const glm::vec3& point) const;
    float getPressureAt(const glm::vec3& point) const;
    
    // Parameters
    void setParticleCount(int count);
    void setStreamlineCount(int count);
    void enableTurbulence(bool enable);
    void setTurbulenceIntensity(float intensity);
    void setViscosity(float viscosity);
    void setDensity(float density);
    
    // Getters
    float getWindSpeed() const { return m_windSpeed; }
    glm::vec3 getWindDirection() const { return m_windDirection; }
    float getDensity() const { return m_density; }
    float getViscosity() const { return m_viscosity; }

private:
    void initParticles();
    void initStreamlines();
    void updateParticles(float deltaTime);
    void updateStreamlines();
    void spawnParticle(Particle& p);
    
    // Flow field calculations
    glm::vec3 calculateFlowVelocity(const glm::vec3& point) const;
    glm::vec3 calculatePotentialFlow(const glm::vec3& point) const;
    float calculateDistanceToSurface(const glm::vec3& point) const;
    bool isInsideMesh(const glm::vec3& point) const;
    
    // Wind parameters
    float m_windSpeed = 10.0f;
    glm::vec3 m_windDirection = glm::vec3(1.0f, 0.0f, 0.0f);
    float m_density = 1.225f;  // kg/m³ (air at sea level)
    float m_viscosity = 1.81e-5f;  // Pa·s (dynamic viscosity of air)
    float m_turbulenceIntensity = 0.1f;
    bool m_turbulenceEnabled = true;
    
    // Domain
    glm::vec3 m_domainSize = glm::vec3(20.0f, 10.0f, 10.0f);
    glm::vec3 m_domainMin;
    glm::vec3 m_domainMax;
    
    // Visualization
    std::vector<Particle> m_particles;
    std::vector<Streamline> m_streamlines;
    int m_particleCount = 2000;
    int m_streamlineCount = 50;
    
    // Mesh reference
    const Mesh* m_mesh = nullptr;
    
    // Random number generation
    std::mt19937 m_rng;
    std::uniform_real_distribution<float> m_uniformDist;
};
