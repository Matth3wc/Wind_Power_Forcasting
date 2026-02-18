#include "blade_physics.hpp"
#include "mesh.hpp"
#include "wind_simulation.hpp"

#include <cmath>
#include <algorithm>

BladePhysics::BladePhysics() {
}

BladePhysics::~BladePhysics() {
}

void BladePhysics::setMesh(const Mesh* mesh) {
    m_mesh = mesh;
    if (mesh) {
        calculateStaticProperties();
    }
}

void BladePhysics::setWindSimulation(const WindSimulation* windSim) {
    m_windSim = windSim;
}

void BladePhysics::update(float deltaTime) {
    if (!m_mesh || !m_windSim) return;
    
    calculateAerodynamicCoefficients();
    calculateForces();
    calculatePressureDistribution();
    calculatePerformance();
    calculateReynoldsNumber();
}

void BladePhysics::calculateStaticProperties() {
    if (!m_mesh) return;
    
    calculateGeometricProperties();
}

void BladePhysics::calculateGeometricProperties() {
    auto bbox = m_mesh->getBoundingBox();
    glm::vec3 size = bbox.size();
    
    m_stats.surfaceArea = m_mesh->getSurfaceArea();
    m_stats.volume = m_mesh->getVolume();
    
    // Assume longest dimension is span, second is chord
    std::vector<float> dims = {size.x, size.y, size.z};
    std::sort(dims.begin(), dims.end(), std::greater<float>());
    
    m_stats.spanLength = dims[0];
    m_stats.chordLength = dims[1];
    m_stats.aspectRatio = (m_stats.chordLength > 0.001f) ? 
                          m_stats.spanLength / m_stats.chordLength : 0.0f;
}

void BladePhysics::calculateAerodynamicCoefficients() {
    float alpha = m_angleOfAttack;
    
    // Calculate lift coefficient using thin airfoil theory + stall model
    m_stats.liftCoefficient = calculateLiftCoefficient(alpha);
    m_stats.dragCoefficient = calculateDragCoefficient(alpha, m_stats.liftCoefficient);
    
    // Lift to drag ratio
    m_stats.liftToDragRatio = (m_stats.dragCoefficient > 0.0001f) ?
                              m_stats.liftCoefficient / m_stats.dragCoefficient : 0.0f;
    
    // Moment coefficient (simplified)
    float alphaRad = glm::radians(alpha);
    m_stats.momentCoefficient = -0.1f * std::sin(2.0f * alphaRad);
    
    m_stats.angleOfAttack = alpha;
}

float BladePhysics::calculateLiftCoefficient(float alpha) const {
    // Convert to radians
    float alphaRad = glm::radians(alpha);
    
    // Thin airfoil theory: Cl = 2*pi*alpha (for small angles)
    // With stall correction using Viterna model
    
    float Cl_linear = 2.0f * M_PI * alphaRad;
    
    // Stall angle (typical for NACA airfoils)
    float alphaStall = 12.0f;  // degrees
    float ClMax = 1.5f;
    
    if (std::abs(alpha) < alphaStall) {
        // Linear region
        return std::clamp(Cl_linear, -ClMax, ClMax);
    } else {
        // Post-stall (Viterna extrapolation)
        float sign = (alpha > 0) ? 1.0f : -1.0f;
        float alphaAbs = std::abs(alpha);
        
        // Post-stall Cl decay
        float Cl_stall = ClMax * std::cos(glm::radians(alphaAbs - alphaStall) * 2.0f);
        Cl_stall = std::max(Cl_stall, 0.0f);
        
        // Flat plate at high angles
        if (alphaAbs > 45.0f) {
            Cl_stall = 2.0f * std::sin(alphaRad) * std::cos(alphaRad);
        }
        
        return sign * Cl_stall;
    }
}

float BladePhysics::calculateDragCoefficient(float alpha, float Cl) const {
    // Parasitic drag (profile drag)
    float Cd0 = 0.008f;  // Zero-lift drag coefficient
    
    // Induced drag: Cdi = Cl^2 / (pi * e * AR)
    float e = 0.85f;  // Oswald efficiency factor
    float AR = std::max(m_stats.aspectRatio, 1.0f);
    float Cdi = (Cl * Cl) / (M_PI * e * AR);
    
    // Pressure drag increase at high angles
    float alphaRad = glm::radians(std::abs(alpha));
    float Cdp = 0.0f;
    
    if (std::abs(alpha) > 12.0f) {
        // Post-stall drag increase
        Cdp = 0.5f * std::pow(std::sin(alphaRad), 2.0f);
    }
    
    return Cd0 + Cdi + Cdp;
}

void BladePhysics::calculateForces() {
    if (!m_windSim) return;
    
    float V = m_windSim->getWindSpeed();
    float rho = m_windSim->getDensity();
    float S = m_stats.surfaceArea;
    
    // Dynamic pressure
    float q = 0.5f * rho * V * V;
    
    // Lift and drag magnitudes
    m_stats.liftMagnitude = q * S * m_stats.liftCoefficient;
    m_stats.dragMagnitude = q * S * m_stats.dragCoefficient;
    
    // Force directions
    glm::vec3 windDir = m_windSim->getWindDirection();
    glm::vec3 liftDir = glm::vec3(0.0f, 1.0f, 0.0f);  // Perpendicular to flow
    
    // Adjust lift direction based on angle of attack
    float alphaRad = glm::radians(m_angleOfAttack);
    liftDir = glm::vec3(-windDir.z, std::cos(alphaRad), windDir.x);
    liftDir = glm::normalize(liftDir);
    
    m_stats.liftForce = liftDir * m_stats.liftMagnitude;
    m_stats.dragForce = windDir * m_stats.dragMagnitude;
    m_stats.totalForce = m_stats.liftForce + m_stats.dragForce;
}

void BladePhysics::calculatePressureDistribution() {
    if (!m_mesh || !m_windSim) return;
    
    const auto& triangles = m_mesh->getTriangles();
    m_pressureField.resize(triangles.size());
    
    float V = m_windSim->getWindSpeed();
    float rho = m_windSim->getDensity();
    float q = 0.5f * rho * V * V;
    
    m_stats.maxPressure = -1e10f;
    m_stats.minPressure = 1e10f;
    float totalPressure = 0.0f;
    
    glm::vec3 windDir = m_windSim->getWindDirection();
    
    for (size_t i = 0; i < triangles.size(); ++i) {
        const auto& tri = triangles[i];
        
        // Pressure coefficient based on surface orientation
        float cosAngle = glm::dot(tri.normal, -windDir);
        
        // Stagnation pressure at front-facing surfaces
        float Cp;
        if (cosAngle > 0) {
            // Windward side - positive pressure
            Cp = cosAngle * cosAngle;
        } else {
            // Leeward side - suction (negative pressure)
            Cp = -0.5f * (1.0f - cosAngle * cosAngle);
        }
        
        float pressure = q * Cp;
        m_pressureField[i] = pressure;
        
        m_stats.maxPressure = std::max(m_stats.maxPressure, pressure);
        m_stats.minPressure = std::min(m_stats.minPressure, pressure);
        totalPressure += pressure * tri.area;
    }
    
    m_stats.avgPressure = totalPressure / m_stats.surfaceArea;
    m_stats.pressureDifferential = m_stats.maxPressure - m_stats.minPressure;
}

void BladePhysics::calculatePerformance() {
    if (!m_windSim) return;
    
    float V = m_windSim->getWindSpeed();
    float rho = m_windSim->getDensity();
    
    // Blade radius (use span length as approximation)
    float R = m_bladeRadius;
    
    // Rotational speed
    float omega = m_rotationalSpeed * 2.0f * M_PI / 60.0f;  // rad/s
    m_stats.rotationalSpeed = m_rotationalSpeed;
    
    // Tip speed
    m_stats.tipSpeed = omega * R;
    
    // Tip speed ratio
    m_stats.tipSpeedRatio = (V > 0.1f) ? m_stats.tipSpeed / V : 0.0f;
    
    // Torque (simplified: from tangential force component)
    float tangentialForce = m_stats.liftMagnitude * std::cos(glm::radians(m_pitchAngle)) -
                           m_stats.dragMagnitude * std::sin(glm::radians(m_pitchAngle));
    m_stats.torque = tangentialForce * R * 0.7f;  // 0.7R is effective radius
    
    // Power
    m_stats.power = m_stats.torque * omega;
    
    // Available wind power
    float A = M_PI * R * R;
    float P_wind = 0.5f * rho * A * V * V * V;
    
    // Power coefficient
    m_stats.powerCoefficient = (P_wind > 0.1f) ? m_stats.power / P_wind : 0.0f;
    m_stats.powerCoefficient = std::clamp(m_stats.powerCoefficient, 0.0f, BETZ_LIMIT);
    
    // Thrust coefficient
    float thrust = glm::length(m_stats.dragForce);
    m_stats.thrustCoefficient = thrust / (0.5f * rho * A * V * V);
    
    // Efficiency (relative to Betz limit)
    m_stats.efficiency = m_stats.powerCoefficient / BETZ_LIMIT * 100.0f;
}

void BladePhysics::calculateReynoldsNumber() {
    if (!m_windSim) return;
    
    float V = m_windSim->getWindSpeed();
    float rho = m_windSim->getDensity();
    float mu = m_windSim->getViscosity();
    float c = m_stats.chordLength;
    
    // Reynolds number: Re = rho * V * c / mu
    m_stats.reynoldsNumber = rho * V * c / mu;
    
    // Mach number: M = V / a (speed of sound)
    m_stats.machNumber = V / SPEED_OF_SOUND;
}

void BladePhysics::setAngleOfAttack(float angle) {
    m_angleOfAttack = std::clamp(angle, -90.0f, 90.0f);
}

void BladePhysics::setRotationalSpeed(float rpm) {
    m_rotationalSpeed = std::max(0.0f, rpm);
}

void BladePhysics::setBladeRadius(float radius) {
    m_bladeRadius = std::max(1.0f, radius);
}

void BladePhysics::setPitchAngle(float angle) {
    m_pitchAngle = std::clamp(angle, -30.0f, 30.0f);
}
