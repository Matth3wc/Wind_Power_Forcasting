#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <memory>

class Mesh;

// NACA 4-digit airfoil parameters
struct NACAParameters {
    float M;  // Maximum camber (0-9, represents 0-0.09 of chord)
    float P;  // Position of maximum camber (0-9, represents 0-0.9 of chord)
    float T;  // Maximum thickness (00-99, represents 0-0.99 of chord)
    
    // Returns NACA designation string (e.g., "2412")
    std::string getDesignation() const;
    
    // Create from NACA designation string
    static NACAParameters fromDesignation(const std::string& designation);
    
    // Create from actual values
    static NACAParameters fromValues(float maxCamber, float camberPos, float thickness);
};

// Blade geometry parameters
struct BladeParameters {
    NACAParameters rootAirfoil;   // Airfoil at blade root
    NACAParameters tipAirfoil;    // Airfoil at blade tip
    float spanLength;             // Total blade length (meters)
    float rootChord;              // Chord length at root (meters)
    float tipChord;               // Chord length at tip (meters)
    float twistAngle;             // Total twist from root to tip (degrees)
    float pitchAngle;             // Blade pitch angle (degrees)
    int spanSegments;             // Number of segments along span
    int chordPoints;              // Number of points around airfoil
    bool closedTrailingEdge;      // Use closed trailing edge
    
    BladeParameters();
};

// 2D airfoil point
struct AirfoilPoint {
    float x;  // Chordwise position (0-1)
    float y;  // Thickness position
};

// NACA 4-digit airfoil generator based on the equations:
// 
// Camber line (front, 0 <= x < p):
//   y_c = M/p^2 * (2*P*x - x^2)
//   dy_c/dx = 2*M/p^2 * (P - x)
//
// Camber line (back, p <= x <= 1):
//   y_c = M/(1-p)^2 * (1 - 2*P + 2*P*x - x^2)
//   dy_c/dx = 2*M/(1-p)^2 * (P - x)
//
// Thickness distribution:
//   y_t = T/0.2 * (a0*x^0.5 + a1*x + a2*x^2 + a3*x^3 + a4*x^4)
//   where a0=0.2969, a1=-0.126, a2=-0.3516, a3=0.2843, a4=-0.1015 (or -0.1036 for closed TE)
//
// Surface coordinates:
//   theta = atan(dy_c/dx)
//   Upper: x_u = x - y_t*sin(theta), y_u = y_c + y_t*cos(theta)
//   Lower: x_l = x + y_t*sin(theta), y_l = y_c - y_t*cos(theta)

class NACAGenerator {
public:
    NACAGenerator();
    ~NACAGenerator();
    
    // Generate 2D airfoil profile
    std::vector<AirfoilPoint> generateAirfoil(const NACAParameters& params, int numPoints = 100, bool closedTE = false) const;
    
    // Generate 3D blade mesh
    std::shared_ptr<Mesh> generateBlade(const BladeParameters& params) const;
    
    // Generate simple extruded airfoil (for testing)
    std::shared_ptr<Mesh> generateExtrudedAirfoil(const NACAParameters& params, float chord, float span, int numPoints = 50) const;
    
    // Get camber line
    std::vector<AirfoilPoint> getCamberLine(const NACAParameters& params, int numPoints = 100) const;
    
    // Get thickness distribution
    std::vector<float> getThicknessDistribution(const NACAParameters& params, int numPoints = 100, bool closedTE = false) const;
    
    // Calculate aerodynamic properties (thin airfoil theory)
    float calculateTheoreticalCl0(const NACAParameters& params) const;  // Lift at zero angle
    float calculateCmac(const NACAParameters& params) const;            // Moment about AC

private:
    // Camber line calculation
    float calculateCamber(float x, float M, float P) const;
    float calculateCamberGradient(float x, float M, float P) const;
    
    // Thickness distribution
    float calculateThickness(float x, float T, bool closedTE) const;
    
    // Surface coordinates
    void calculateSurfacePoint(float x, float M, float P, float T, bool closedTE,
                               float& xu, float& yu, float& xl, float& yl) const;
    
    // Thickness coefficients
    static constexpr float a0 = 0.2969f;
    static constexpr float a1 = -0.126f;
    static constexpr float a2 = -0.3516f;
    static constexpr float a3 = 0.2843f;
    static constexpr float a4_open = -0.1015f;
    static constexpr float a4_closed = -0.1036f;
};
