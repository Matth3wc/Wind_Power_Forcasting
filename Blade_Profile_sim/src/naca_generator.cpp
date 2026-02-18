#include "naca_generator.hpp"
#include "mesh.hpp"

#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

// NACAParameters implementation
std::string NACAParameters::getDesignation() const {
    std::ostringstream ss;
    ss << static_cast<int>(M * 100 + 0.5f)
       << static_cast<int>(P * 10 + 0.5f)
       << std::setfill('0') << std::setw(2) << static_cast<int>(T * 100 + 0.5f);
    return ss.str();
}

NACAParameters NACAParameters::fromDesignation(const std::string& designation) {
    NACAParameters params;
    
    if (designation.length() >= 4) {
        params.M = (designation[0] - '0') / 100.0f;
        params.P = (designation[1] - '0') / 10.0f;
        params.T = std::stoi(designation.substr(2, 2)) / 100.0f;
    }
    
    return params;
}

NACAParameters NACAParameters::fromValues(float maxCamber, float camberPos, float thickness) {
    NACAParameters params;
    params.M = maxCamber;
    params.P = camberPos;
    params.T = thickness;
    return params;
}

// BladeParameters implementation
BladeParameters::BladeParameters()
    : spanLength(50.0f)
    , rootChord(4.0f)
    , tipChord(1.0f)
    , twistAngle(15.0f)
    , pitchAngle(3.0f)
    , spanSegments(30)
    , chordPoints(50)
    , closedTrailingEdge(true)
{
    // Default to NACA 4412 at root, NACA 2412 at tip
    rootAirfoil = NACAParameters::fromDesignation("4412");
    tipAirfoil = NACAParameters::fromDesignation("2412");
}

// NACAGenerator implementation
NACAGenerator::NACAGenerator() {
}

NACAGenerator::~NACAGenerator() {
}

float NACAGenerator::calculateCamber(float x, float M, float P) const {
    if (M <= 0.0001f || P <= 0.0001f) {
        return 0.0f;  // Symmetric airfoil
    }
    
    float p = P;  // Position of max camber
    float m = M;  // Max camber value
    
    if (x < p) {
        // Front section: y_c = M/p^2 * (2*P*x - x^2)
        return (m / (p * p)) * (2.0f * p * x - x * x);
    } else {
        // Back section: y_c = M/(1-p)^2 * (1 - 2*P + 2*P*x - x^2)
        float denom = (1.0f - p) * (1.0f - p);
        return (m / denom) * (1.0f - 2.0f * p + 2.0f * p * x - x * x);
    }
}

float NACAGenerator::calculateCamberGradient(float x, float M, float P) const {
    if (M <= 0.0001f || P <= 0.0001f) {
        return 0.0f;  // Symmetric airfoil
    }
    
    float p = P;
    float m = M;
    
    if (x < p) {
        // Front: dy_c/dx = 2*M/p^2 * (P - x)
        return (2.0f * m / (p * p)) * (p - x);
    } else {
        // Back: dy_c/dx = 2*M/(1-p)^2 * (P - x)
        float denom = (1.0f - p) * (1.0f - p);
        return (2.0f * m / denom) * (p - x);
    }
}

float NACAGenerator::calculateThickness(float x, float T, bool closedTE) const {
    // y_t = T/0.2 * (a0*x^0.5 + a1*x + a2*x^2 + a3*x^3 + a4*x^4)
    float a4 = closedTE ? a4_closed : a4_open;
    
    float sqrtX = std::sqrt(std::max(x, 0.0f));
    float x2 = x * x;
    float x3 = x2 * x;
    float x4 = x3 * x;
    
    float yt = (T / 0.2f) * (a0 * sqrtX + a1 * x + a2 * x2 + a3 * x3 + a4 * x4);
    
    return std::max(yt, 0.0f);
}

void NACAGenerator::calculateSurfacePoint(float x, float M, float P, float T, bool closedTE,
                                          float& xu, float& yu, float& xl, float& yl) const {
    float yc = calculateCamber(x, M, P);
    float dydx = calculateCamberGradient(x, M, P);
    float yt = calculateThickness(x, T, closedTE);
    
    // theta = atan(dy_c/dx)
    float theta = std::atan(dydx);
    float sinTheta = std::sin(theta);
    float cosTheta = std::cos(theta);
    
    // Upper surface: x_u = x - y_t*sin(theta), y_u = y_c + y_t*cos(theta)
    xu = x - yt * sinTheta;
    yu = yc + yt * cosTheta;
    
    // Lower surface: x_l = x + y_t*sin(theta), y_l = y_c - y_t*cos(theta)
    xl = x + yt * sinTheta;
    yl = yc - yt * cosTheta;
}

std::vector<AirfoilPoint> NACAGenerator::generateAirfoil(const NACAParameters& params, int numPoints, bool closedTE) const {
    std::vector<AirfoilPoint> points;
    points.reserve(numPoints * 2 + 1);
    
    float M = params.M;
    float P = params.P;
    float T = params.T;
    
    // Use cosine spacing for better resolution at leading and trailing edges
    std::vector<float> xCoords;
    xCoords.reserve(numPoints);
    
    for (int i = 0; i <= numPoints; ++i) {
        float beta = static_cast<float>(i) / numPoints * M_PI;
        float x = 0.5f * (1.0f - std::cos(beta));
        xCoords.push_back(x);
    }
    
    // Generate upper surface (from trailing edge to leading edge)
    for (int i = numPoints; i >= 0; --i) {
        float x = xCoords[i];
        float xu, yu, xl, yl;
        calculateSurfacePoint(x, M, P, T, closedTE, xu, yu, xl, yl);
        points.push_back({xu, yu});
    }
    
    // Generate lower surface (from leading edge to trailing edge, skipping first point)
    for (int i = 1; i <= numPoints; ++i) {
        float x = xCoords[i];
        float xu, yu, xl, yl;
        calculateSurfacePoint(x, M, P, T, closedTE, xu, yu, xl, yl);
        points.push_back({xl, yl});
    }
    
    return points;
}

std::vector<AirfoilPoint> NACAGenerator::getCamberLine(const NACAParameters& params, int numPoints) const {
    std::vector<AirfoilPoint> points;
    points.reserve(numPoints + 1);
    
    for (int i = 0; i <= numPoints; ++i) {
        float x = static_cast<float>(i) / numPoints;
        float y = calculateCamber(x, params.M, params.P);
        points.push_back({x, y});
    }
    
    return points;
}

std::vector<float> NACAGenerator::getThicknessDistribution(const NACAParameters& params, int numPoints, bool closedTE) const {
    std::vector<float> thickness;
    thickness.reserve(numPoints + 1);
    
    for (int i = 0; i <= numPoints; ++i) {
        float x = static_cast<float>(i) / numPoints;
        float t = calculateThickness(x, params.T, closedTE);
        thickness.push_back(t);
    }
    
    return thickness;
}

std::shared_ptr<Mesh> NACAGenerator::generateExtrudedAirfoil(const NACAParameters& params, float chord, float span, int numPoints) const {
    auto airfoil = generateAirfoil(params, numPoints, true);
    
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    
    int numAirfoilPoints = static_cast<int>(airfoil.size());
    
    // Generate vertices for both ends
    for (int side = 0; side < 2; ++side) {
        float z = (side == 0) ? -span * 0.5f : span * 0.5f;
        
        for (const auto& pt : airfoil) {
            Vertex v;
            v.position = glm::vec3(pt.x * chord - chord * 0.25f, pt.y * chord, z);
            v.normal = glm::vec3(0.0f, 0.0f, (side == 0) ? -1.0f : 1.0f);
            v.texCoords = glm::vec2(pt.x, (side == 0) ? 0.0f : 1.0f);
            vertices.push_back(v);
        }
    }
    
    // Generate indices for side surfaces
    for (int i = 0; i < numAirfoilPoints; ++i) {
        int next = (i + 1) % numAirfoilPoints;
        
        int i0 = i;
        int i1 = next;
        int i2 = numAirfoilPoints + next;
        int i3 = numAirfoilPoints + i;
        
        // Calculate face normal
        glm::vec3 v0 = vertices[i0].position;
        glm::vec3 v1 = vertices[i1].position;
        glm::vec3 v2 = vertices[i2].position;
        
        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));
        
        // Update normals for surface vertices
        vertices[i0].normal = normal;
        vertices[i1].normal = normal;
        vertices[i2].normal = normal;
        vertices[i3].normal = normal;
        
        // Two triangles per quad
        indices.push_back(i0);
        indices.push_back(i1);
        indices.push_back(i2);
        
        indices.push_back(i0);
        indices.push_back(i2);
        indices.push_back(i3);
    }
    
    auto mesh = std::make_shared<Mesh>();
    mesh->setVertices(vertices);
    mesh->setIndices(indices);
    mesh->build();
    
    return mesh;
}

std::shared_ptr<Mesh> NACAGenerator::generateBlade(const BladeParameters& params) const {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    
    // Generate airfoil sections along the span
    for (int spanIdx = 0; spanIdx <= params.spanSegments; ++spanIdx) {
        float t = static_cast<float>(spanIdx) / params.spanSegments;
        float spanPos = t * params.spanLength;
        
        // Interpolate airfoil parameters
        NACAParameters localAirfoil;
        localAirfoil.M = glm::mix(params.rootAirfoil.M, params.tipAirfoil.M, t);
        localAirfoil.P = glm::mix(params.rootAirfoil.P, params.tipAirfoil.P, t);
        localAirfoil.T = glm::mix(params.rootAirfoil.T, params.tipAirfoil.T, t);
        
        // Interpolate chord
        float chord = glm::mix(params.rootChord, params.tipChord, t);
        
        // Calculate local twist
        float twist = t * params.twistAngle + params.pitchAngle;
        float twistRad = glm::radians(twist);
        float cosTwist = std::cos(twistRad);
        float sinTwist = std::sin(twistRad);
        
        // Generate airfoil profile
        auto airfoil = generateAirfoil(localAirfoil, params.chordPoints, params.closedTrailingEdge);
        
        // Add vertices for this section
        for (const auto& pt : airfoil) {
            // Scale by chord and apply twist
            float x = (pt.x - 0.25f) * chord;  // Rotate around quarter chord
            float y = pt.y * chord;
            
            // Apply twist rotation
            float xRot = x * cosTwist - y * sinTwist;
            float yRot = x * sinTwist + y * cosTwist;
            
            Vertex v;
            v.position = glm::vec3(xRot, yRot, spanPos);
            v.normal = glm::vec3(0.0f, 1.0f, 0.0f);  // Will be recalculated
            v.texCoords = glm::vec2(pt.x, t);
            vertices.push_back(v);
        }
    }
    
    int pointsPerSection = params.chordPoints * 2 + 1;
    
    // Generate indices
    for (int spanIdx = 0; spanIdx < params.spanSegments; ++spanIdx) {
        for (int ptIdx = 0; ptIdx < pointsPerSection; ++ptIdx) {
            int nextPt = (ptIdx + 1) % pointsPerSection;
            
            int i0 = spanIdx * pointsPerSection + ptIdx;
            int i1 = spanIdx * pointsPerSection + nextPt;
            int i2 = (spanIdx + 1) * pointsPerSection + nextPt;
            int i3 = (spanIdx + 1) * pointsPerSection + ptIdx;
            
            // Two triangles per quad
            indices.push_back(i0);
            indices.push_back(i1);
            indices.push_back(i2);
            
            indices.push_back(i0);
            indices.push_back(i2);
            indices.push_back(i3);
        }
    }
    
    // Recalculate normals
    std::vector<glm::vec3> normals(vertices.size(), glm::vec3(0.0f));
    
    for (size_t i = 0; i < indices.size(); i += 3) {
        glm::vec3 v0 = vertices[indices[i]].position;
        glm::vec3 v1 = vertices[indices[i + 1]].position;
        glm::vec3 v2 = vertices[indices[i + 2]].position;
        
        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        glm::vec3 faceNormal = glm::cross(edge1, edge2);
        
        normals[indices[i]] += faceNormal;
        normals[indices[i + 1]] += faceNormal;
        normals[indices[i + 2]] += faceNormal;
    }
    
    for (size_t i = 0; i < vertices.size(); ++i) {
        vertices[i].normal = glm::normalize(normals[i]);
    }
    
    auto mesh = std::make_shared<Mesh>();
    mesh->setVertices(vertices);
    mesh->setIndices(indices);
    mesh->build();
    
    return mesh;
}

float NACAGenerator::calculateTheoreticalCl0(const NACAParameters& params) const {
    // For thin airfoil theory, Cl at alpha=0 depends on camber
    // Cl_0 = 2*pi*(alpha_0), where alpha_0 is the zero-lift angle
    // For cambered airfoils: alpha_0 ≈ -2*M (in radians) for small camber
    
    if (params.M < 0.0001f) {
        return 0.0f;  // Symmetric airfoil
    }
    
    // Approximate Cl at zero angle of attack
    // This is a simplification; actual value depends on camber distribution
    float Cl0 = 4.0f * M_PI * params.M * (params.P - 0.5f + params.P * (1.0f - params.P));
    
    return Cl0;
}

float NACAGenerator::calculateCmac(const NACAParameters& params) const {
    // Moment coefficient about aerodynamic center
    // For thin airfoil theory: Cm_ac ≈ -pi/2 * (A1 - A2)
    // For NACA 4-digit: Cm_ac ≈ -pi * M / 4
    
    if (params.M < 0.0001f) {
        return 0.0f;  // Symmetric airfoil
    }
    
    return -M_PI * params.M / 4.0f;
}
