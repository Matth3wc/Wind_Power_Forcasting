#pragma once

#include <glm/glm.hpp>
#include <vector>

class Mesh;
class WindSimulation;

// Blade performance statistics
struct BladeStats {
    // Geometric properties
    float surfaceArea = 0.0f;      // m²
    float volume = 0.0f;           // m³
    float chordLength = 0.0f;      // m (average)
    float spanLength = 0.0f;       // m
    float aspectRatio = 0.0f;
    
    // Aerodynamic properties
    float liftCoefficient = 0.0f;  // Cl
    float dragCoefficient = 0.0f;  // Cd
    float liftToDragRatio = 0.0f;  // L/D
    float momentCoefficient = 0.0f; // Cm
    
    // Forces (in Newtons)
    glm::vec3 liftForce = glm::vec3(0.0f);
    glm::vec3 dragForce = glm::vec3(0.0f);
    glm::vec3 totalForce = glm::vec3(0.0f);
    float liftMagnitude = 0.0f;
    float dragMagnitude = 0.0f;
    
    // Pressure
    float maxPressure = 0.0f;      // Pa
    float minPressure = 0.0f;      // Pa
    float avgPressure = 0.0f;      // Pa
    float pressureDifferential = 0.0f; // Pa
    
    // Performance
    float power = 0.0f;            // Watts
    float torque = 0.0f;           // N·m
    float efficiency = 0.0f;       // %
    float thrustCoefficient = 0.0f; // Ct
    float powerCoefficient = 0.0f;  // Cp
    
    // Flow characteristics
    float reynoldsNumber = 0.0f;
    float machNumber = 0.0f;
    float angleOfAttack = 0.0f;    // degrees
    float tipSpeedRatio = 0.0f;
    
    // Operational
    float rotationalSpeed = 0.0f;  // RPM
    float tipSpeed = 0.0f;         // m/s
};

class BladePhysics {
public:
    BladePhysics();
    ~BladePhysics();
    
    void setMesh(const Mesh* mesh);
    void setWindSimulation(const WindSimulation* windSim);
    
    void update(float deltaTime);
    void calculateStaticProperties();
    
    const BladeStats& getStats() const { return m_stats; }
    
    // Simulation parameters
    void setAngleOfAttack(float angle);
    void setRotationalSpeed(float rpm);
    void setBladeRadius(float radius);
    void setPitchAngle(float angle);
    
    // Get calculated values
    float getAngleOfAttack() const { return m_angleOfAttack; }
    float getRotationalSpeed() const { return m_rotationalSpeed; }
    
    // Pressure distribution on surface
    const std::vector<float>& getPressureField() const { return m_pressureField; }

private:
    void calculateGeometricProperties();
    void calculateAerodynamicCoefficients();
    void calculateForces();
    void calculatePressureDistribution();
    void calculatePerformance();
    void calculateReynoldsNumber();
    
    // Helper functions
    float calculateLiftCoefficient(float alpha) const;
    float calculateDragCoefficient(float alpha, float Cl) const;
    glm::vec3 calculateSurfaceNormal(int triangleIndex) const;
    
    const Mesh* m_mesh = nullptr;
    const WindSimulation* m_windSim = nullptr;
    
    BladeStats m_stats;
    std::vector<float> m_pressureField;
    
    // Blade parameters
    float m_angleOfAttack = 8.0f;     // degrees
    float m_rotationalSpeed = 15.0f;  // RPM
    float m_bladeRadius = 50.0f;      // meters
    float m_pitchAngle = 3.0f;        // degrees
    
    // Physical constants
    static constexpr float SPEED_OF_SOUND = 343.0f;  // m/s at 20°C
    static constexpr float BETZ_LIMIT = 0.593f;       // Maximum theoretical efficiency
};
