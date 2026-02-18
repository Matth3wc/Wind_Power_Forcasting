#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <random>
#include <functional>
#include <memory>

class Mesh;
class WindSimulation;
class BladePhysics;
struct NACAParameters;
struct BladeParameters;

// Constraint for optimization
struct OptimizationConstraint {
    enum class Type {
        MIN_LIFT_COEFFICIENT,
        MAX_DRAG_COEFFICIENT,
        MIN_LIFT_TO_DRAG,
        MAX_THICKNESS,
        MIN_THICKNESS,
        MAX_CAMBER,
        MAX_CHORD,
        MIN_CHORD,
        MAX_SPAN,
        MIN_SPAN,
        MAX_POWER,
        MIN_EFFICIENCY
    };
    
    Type type;
    float value;
    float weight;  // Penalty weight for constraint violation
    
    OptimizationConstraint(Type t, float v, float w = 1.0f)
        : type(t), value(v), weight(w) {}
};

// Result of a single evaluation
struct EvaluationResult {
    BladeParameters params;
    float fitness;
    float liftCoefficient;
    float dragCoefficient;
    float liftToDrag;
    float power;
    float efficiency;
    std::vector<float> constraintViolations;
    bool feasible;
};

// Optimization objectives
enum class OptimizationObjective {
    MAXIMIZE_LIFT,
    MINIMIZE_DRAG,
    MAXIMIZE_LIFT_TO_DRAG,
    MAXIMIZE_POWER,
    MAXIMIZE_EFFICIENCY,
    CUSTOM
};

// Callback for progress updates
using ProgressCallback = std::function<void(int iteration, const EvaluationResult& best)>;

// Blade optimizer using genetic algorithm and parameter sweeps
class BladeOptimizer {
public:
    BladeOptimizer();
    ~BladeOptimizer();
    
    // Set optimization parameters
    void setWindSpeed(float speed);
    void setAirDensity(float density);
    void setBladeRadius(float radius);
    void setRotationalSpeed(float rpm);
    
    // Set search ranges
    void setCamberRange(float min, float max);
    void setCamberPositionRange(float min, float max);
    void setThicknessRange(float min, float max);
    void setChordRange(float min, float max);
    void setTwistRange(float min, float max);
    
    // Add constraints
    void addConstraint(const OptimizationConstraint& constraint);
    void clearConstraints();
    
    // Set objective
    void setObjective(OptimizationObjective objective);
    void setCustomObjective(std::function<float(const EvaluationResult&)> func);
    
    // Optimization methods
    EvaluationResult randomSearch(int iterations, ProgressCallback callback = nullptr);
    EvaluationResult geneticAlgorithm(int populationSize, int generations, ProgressCallback callback = nullptr);
    EvaluationResult gridSearch(int stepsPerParam, ProgressCallback callback = nullptr);
    
    // Evaluate a specific configuration
    EvaluationResult evaluate(const BladeParameters& params);
    
    // Generate random blade parameters within constraints
    BladeParameters randomParameters();
    
    // Get best result from last optimization
    const EvaluationResult& getBestResult() const { return m_bestResult; }
    
    // Get all results from last optimization
    const std::vector<EvaluationResult>& getAllResults() const { return m_allResults; }

private:
    float calculateFitness(const EvaluationResult& result) const;
    float calculateConstraintPenalty(const EvaluationResult& result) const;
    BladeParameters crossover(const BladeParameters& p1, const BladeParameters& p2);
    BladeParameters mutate(const BladeParameters& params, float mutationRate);
    
    // Simulation components
    std::unique_ptr<WindSimulation> m_windSim;
    std::unique_ptr<BladePhysics> m_physics;
    
    // Optimization settings
    float m_windSpeed = 10.0f;
    float m_airDensity = 1.225f;
    float m_bladeRadius = 50.0f;
    float m_rotationalSpeed = 15.0f;
    
    // Search ranges
    float m_camberMin = 0.0f, m_camberMax = 0.09f;
    float m_camberPosMin = 0.2f, m_camberPosMax = 0.6f;
    float m_thicknessMin = 0.08f, m_thicknessMax = 0.24f;
    float m_chordRootMin = 2.0f, m_chordRootMax = 6.0f;
    float m_chordTipMin = 0.5f, m_chordTipMax = 2.0f;
    float m_twistMin = 0.0f, m_twistMax = 20.0f;
    
    // Constraints and objective
    std::vector<OptimizationConstraint> m_constraints;
    OptimizationObjective m_objective = OptimizationObjective::MAXIMIZE_LIFT_TO_DRAG;
    std::function<float(const EvaluationResult&)> m_customObjective;
    
    // Results
    EvaluationResult m_bestResult;
    std::vector<EvaluationResult> m_allResults;
    
    // Random number generation
    std::mt19937 m_rng;
    std::uniform_real_distribution<float> m_uniformDist;
};

// Helper function to print optimization results
void printOptimizationResults(const EvaluationResult& result);
