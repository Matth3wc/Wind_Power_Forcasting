#include "blade_optimizer.hpp"
#include "naca_generator.hpp"
#include "wind_simulation.hpp"
#include "blade_physics.hpp"
#include "mesh.hpp"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>

BladeOptimizer::BladeOptimizer()
    : m_rng(std::random_device{}())
    , m_uniformDist(0.0f, 1.0f)
{
    m_windSim = std::make_unique<WindSimulation>();
    m_physics = std::make_unique<BladePhysics>();
}

BladeOptimizer::~BladeOptimizer() {
}

void BladeOptimizer::setWindSpeed(float speed) {
    m_windSpeed = std::max(1.0f, speed);
}

void BladeOptimizer::setAirDensity(float density) {
    m_airDensity = density;
}

void BladeOptimizer::setBladeRadius(float radius) {
    m_bladeRadius = radius;
}

void BladeOptimizer::setRotationalSpeed(float rpm) {
    m_rotationalSpeed = rpm;
}

void BladeOptimizer::setCamberRange(float min, float max) {
    m_camberMin = min;
    m_camberMax = max;
}

void BladeOptimizer::setCamberPositionRange(float min, float max) {
    m_camberPosMin = min;
    m_camberPosMax = max;
}

void BladeOptimizer::setThicknessRange(float min, float max) {
    m_thicknessMin = min;
    m_thicknessMax = max;
}

void BladeOptimizer::setChordRange(float min, float max) {
    m_chordRootMin = min;
    m_chordRootMax = max;
    m_chordTipMin = min * 0.25f;
    m_chordTipMax = max * 0.5f;
}

void BladeOptimizer::setTwistRange(float min, float max) {
    m_twistMin = min;
    m_twistMax = max;
}

void BladeOptimizer::addConstraint(const OptimizationConstraint& constraint) {
    m_constraints.push_back(constraint);
}

void BladeOptimizer::clearConstraints() {
    m_constraints.clear();
}

void BladeOptimizer::setObjective(OptimizationObjective objective) {
    m_objective = objective;
}

void BladeOptimizer::setCustomObjective(std::function<float(const EvaluationResult&)> func) {
    m_objective = OptimizationObjective::CUSTOM;
    m_customObjective = func;
}

BladeParameters BladeOptimizer::randomParameters() {
    BladeParameters params;
    
    // Root airfoil
    params.rootAirfoil.M = m_camberMin + m_uniformDist(m_rng) * (m_camberMax - m_camberMin);
    params.rootAirfoil.P = m_camberPosMin + m_uniformDist(m_rng) * (m_camberPosMax - m_camberPosMin);
    params.rootAirfoil.T = m_thicknessMin + m_uniformDist(m_rng) * (m_thicknessMax - m_thicknessMin);
    
    // Tip airfoil (generally less camber and thinner)
    params.tipAirfoil.M = params.rootAirfoil.M * (0.3f + m_uniformDist(m_rng) * 0.5f);
    params.tipAirfoil.P = params.rootAirfoil.P * (0.8f + m_uniformDist(m_rng) * 0.4f);
    params.tipAirfoil.T = params.rootAirfoil.T * (0.5f + m_uniformDist(m_rng) * 0.3f);
    
    // Geometry
    params.rootChord = m_chordRootMin + m_uniformDist(m_rng) * (m_chordRootMax - m_chordRootMin);
    params.tipChord = m_chordTipMin + m_uniformDist(m_rng) * (m_chordTipMax - m_chordTipMin);
    params.tipChord = std::min(params.tipChord, params.rootChord * 0.5f);
    
    params.twistAngle = m_twistMin + m_uniformDist(m_rng) * (m_twistMax - m_twistMin);
    params.pitchAngle = -5.0f + m_uniformDist(m_rng) * 15.0f;
    
    params.spanLength = m_bladeRadius;
    params.spanSegments = 20;
    params.chordPoints = 30;
    params.closedTrailingEdge = true;
    
    return params;
}

EvaluationResult BladeOptimizer::evaluate(const BladeParameters& params) {
    EvaluationResult result;
    result.params = params;
    
    // Generate blade mesh
    NACAGenerator generator;
    auto mesh = generator.generateBlade(params);
    
    if (!mesh) {
        result.fitness = -1e10f;
        result.feasible = false;
        return result;
    }
    
    // Set up simulation
    m_windSim->setWindSpeed(m_windSpeed);
    m_windSim->setDensity(m_airDensity);
    m_windSim->setMesh(mesh.get());
    
    m_physics->setMesh(mesh.get());
    m_physics->setWindSimulation(m_windSim.get());
    m_physics->setBladeRadius(m_bladeRadius);
    m_physics->setRotationalSpeed(m_rotationalSpeed);
    m_physics->setAngleOfAttack(8.0f);  // Typical operating angle
    
    // Run physics calculations
    m_physics->calculateStaticProperties();
    m_physics->update(0.1f);
    
    // Extract results
    const auto& stats = m_physics->getStats();
    result.liftCoefficient = stats.liftCoefficient;
    result.dragCoefficient = stats.dragCoefficient;
    result.liftToDrag = stats.liftToDragRatio;
    result.power = stats.power;
    result.efficiency = stats.efficiency;
    
    // Check constraints
    result.constraintViolations.resize(m_constraints.size(), 0.0f);
    result.feasible = true;
    
    for (size_t i = 0; i < m_constraints.size(); ++i) {
        const auto& c = m_constraints[i];
        float violation = 0.0f;
        
        switch (c.type) {
            case OptimizationConstraint::Type::MIN_LIFT_COEFFICIENT:
                if (result.liftCoefficient < c.value)
                    violation = c.value - result.liftCoefficient;
                break;
            case OptimizationConstraint::Type::MAX_DRAG_COEFFICIENT:
                if (result.dragCoefficient > c.value)
                    violation = result.dragCoefficient - c.value;
                break;
            case OptimizationConstraint::Type::MIN_LIFT_TO_DRAG:
                if (result.liftToDrag < c.value)
                    violation = c.value - result.liftToDrag;
                break;
            case OptimizationConstraint::Type::MAX_THICKNESS:
                if (params.rootAirfoil.T > c.value)
                    violation = params.rootAirfoil.T - c.value;
                break;
            case OptimizationConstraint::Type::MIN_THICKNESS:
                if (params.rootAirfoil.T < c.value)
                    violation = c.value - params.rootAirfoil.T;
                break;
            case OptimizationConstraint::Type::MAX_CAMBER:
                if (params.rootAirfoil.M > c.value)
                    violation = params.rootAirfoil.M - c.value;
                break;
            case OptimizationConstraint::Type::MIN_EFFICIENCY:
                if (result.efficiency < c.value)
                    violation = c.value - result.efficiency;
                break;
            default:
                break;
        }
        
        result.constraintViolations[i] = violation;
        if (violation > 0.001f) {
            result.feasible = false;
        }
    }
    
    // Calculate fitness
    result.fitness = calculateFitness(result);
    
    return result;
}

float BladeOptimizer::calculateFitness(const EvaluationResult& result) const {
    float fitness = 0.0f;
    
    switch (m_objective) {
        case OptimizationObjective::MAXIMIZE_LIFT:
            fitness = result.liftCoefficient;
            break;
        case OptimizationObjective::MINIMIZE_DRAG:
            fitness = -result.dragCoefficient;
            break;
        case OptimizationObjective::MAXIMIZE_LIFT_TO_DRAG:
            fitness = result.liftToDrag;
            break;
        case OptimizationObjective::MAXIMIZE_POWER:
            fitness = result.power / 1e6f;  // Normalize
            break;
        case OptimizationObjective::MAXIMIZE_EFFICIENCY:
            fitness = result.efficiency;
            break;
        case OptimizationObjective::CUSTOM:
            if (m_customObjective) {
                fitness = m_customObjective(result);
            }
            break;
    }
    
    // Apply constraint penalty
    fitness -= calculateConstraintPenalty(result);
    
    return fitness;
}

float BladeOptimizer::calculateConstraintPenalty(const EvaluationResult& result) const {
    float penalty = 0.0f;
    
    for (size_t i = 0; i < m_constraints.size(); ++i) {
        penalty += m_constraints[i].weight * result.constraintViolations[i] * result.constraintViolations[i];
    }
    
    return penalty * 100.0f;  // Large penalty for constraint violations
}

EvaluationResult BladeOptimizer::randomSearch(int iterations, ProgressCallback callback) {
    m_allResults.clear();
    m_bestResult.fitness = -1e10f;
    
    for (int i = 0; i < iterations; ++i) {
        BladeParameters params = randomParameters();
        EvaluationResult result = evaluate(params);
        
        m_allResults.push_back(result);
        
        if (result.fitness > m_bestResult.fitness) {
            m_bestResult = result;
        }
        
        if (callback && (i % 10 == 0 || i == iterations - 1)) {
            callback(i, m_bestResult);
        }
    }
    
    return m_bestResult;
}

EvaluationResult BladeOptimizer::geneticAlgorithm(int populationSize, int generations, ProgressCallback callback) {
    m_allResults.clear();
    m_bestResult.fitness = -1e10f;
    
    // Initialize population
    std::vector<EvaluationResult> population;
    population.reserve(populationSize);
    
    for (int i = 0; i < populationSize; ++i) {
        BladeParameters params = randomParameters();
        EvaluationResult result = evaluate(params);
        population.push_back(result);
        
        if (result.fitness > m_bestResult.fitness) {
            m_bestResult = result;
        }
    }
    
    // Evolution loop
    for (int gen = 0; gen < generations; ++gen) {
        // Sort by fitness
        std::sort(population.begin(), population.end(),
                  [](const auto& a, const auto& b) { return a.fitness > b.fitness; });
        
        // Keep best
        if (population[0].fitness > m_bestResult.fitness) {
            m_bestResult = population[0];
        }
        
        // Create next generation
        std::vector<EvaluationResult> nextGen;
        nextGen.reserve(populationSize);
        
        // Elitism: keep top 10%
        int eliteCount = populationSize / 10;
        for (int i = 0; i < eliteCount; ++i) {
            nextGen.push_back(population[i]);
        }
        
        // Fill rest with crossover and mutation
        while (nextGen.size() < static_cast<size_t>(populationSize)) {
            // Tournament selection
            int idx1 = static_cast<int>(m_uniformDist(m_rng) * populationSize * 0.5f);
            int idx2 = static_cast<int>(m_uniformDist(m_rng) * populationSize * 0.5f);
            
            const auto& parent1 = population[idx1];
            const auto& parent2 = population[idx2];
            
            // Crossover
            BladeParameters childParams = crossover(parent1.params, parent2.params);
            
            // Mutation
            float mutationRate = 0.1f + 0.2f * (1.0f - static_cast<float>(gen) / generations);
            childParams = mutate(childParams, mutationRate);
            
            // Evaluate
            EvaluationResult child = evaluate(childParams);
            nextGen.push_back(child);
        }
        
        population = nextGen;
        m_allResults.insert(m_allResults.end(), population.begin(), population.end());
        
        if (callback) {
            callback(gen, m_bestResult);
        }
    }
    
    return m_bestResult;
}

EvaluationResult BladeOptimizer::gridSearch(int stepsPerParam, ProgressCallback callback) {
    m_allResults.clear();
    m_bestResult.fitness = -1e10f;
    
    int iteration = 0;
    int totalIterations = stepsPerParam * stepsPerParam * stepsPerParam;
    
    // Grid search over main parameters (camber, thickness, chord)
    for (int ci = 0; ci < stepsPerParam; ++ci) {
        float camber = m_camberMin + (m_camberMax - m_camberMin) * ci / (stepsPerParam - 1);
        
        for (int ti = 0; ti < stepsPerParam; ++ti) {
            float thickness = m_thicknessMin + (m_thicknessMax - m_thicknessMin) * ti / (stepsPerParam - 1);
            
            for (int chi = 0; chi < stepsPerParam; ++chi) {
                float chord = m_chordRootMin + (m_chordRootMax - m_chordRootMin) * chi / (stepsPerParam - 1);
                
                BladeParameters params;
                params.rootAirfoil.M = camber;
                params.rootAirfoil.P = 0.4f;  // Fixed at 40%
                params.rootAirfoil.T = thickness;
                params.tipAirfoil.M = camber * 0.5f;
                params.tipAirfoil.P = 0.4f;
                params.tipAirfoil.T = thickness * 0.7f;
                params.rootChord = chord;
                params.tipChord = chord * 0.3f;
                params.twistAngle = 12.0f;
                params.pitchAngle = 3.0f;
                params.spanLength = m_bladeRadius;
                
                EvaluationResult result = evaluate(params);
                m_allResults.push_back(result);
                
                if (result.fitness > m_bestResult.fitness) {
                    m_bestResult = result;
                }
                
                ++iteration;
                if (callback && (iteration % 10 == 0 || iteration == totalIterations)) {
                    callback(iteration, m_bestResult);
                }
            }
        }
    }
    
    return m_bestResult;
}

BladeParameters BladeOptimizer::crossover(const BladeParameters& p1, const BladeParameters& p2) {
    BladeParameters child;
    float t = m_uniformDist(m_rng);
    
    // Root airfoil
    child.rootAirfoil.M = glm::mix(p1.rootAirfoil.M, p2.rootAirfoil.M, t);
    child.rootAirfoil.P = glm::mix(p1.rootAirfoil.P, p2.rootAirfoil.P, t);
    child.rootAirfoil.T = glm::mix(p1.rootAirfoil.T, p2.rootAirfoil.T, t);
    
    // Tip airfoil
    child.tipAirfoil.M = glm::mix(p1.tipAirfoil.M, p2.tipAirfoil.M, t);
    child.tipAirfoil.P = glm::mix(p1.tipAirfoil.P, p2.tipAirfoil.P, t);
    child.tipAirfoil.T = glm::mix(p1.tipAirfoil.T, p2.tipAirfoil.T, t);
    
    // Geometry
    child.rootChord = glm::mix(p1.rootChord, p2.rootChord, t);
    child.tipChord = glm::mix(p1.tipChord, p2.tipChord, t);
    child.twistAngle = glm::mix(p1.twistAngle, p2.twistAngle, t);
    child.pitchAngle = glm::mix(p1.pitchAngle, p2.pitchAngle, t);
    child.spanLength = p1.spanLength;
    child.spanSegments = 20;
    child.chordPoints = 30;
    child.closedTrailingEdge = true;
    
    return child;
}

BladeParameters BladeOptimizer::mutate(const BladeParameters& params, float mutationRate) {
    BladeParameters mutated = params;
    
    auto mutateValue = [&](float& value, float min, float max) {
        if (m_uniformDist(m_rng) < mutationRate) {
            float delta = (m_uniformDist(m_rng) - 0.5f) * 2.0f * (max - min) * 0.2f;
            value = std::clamp(value + delta, min, max);
        }
    };
    
    mutateValue(mutated.rootAirfoil.M, m_camberMin, m_camberMax);
    mutateValue(mutated.rootAirfoil.P, m_camberPosMin, m_camberPosMax);
    mutateValue(mutated.rootAirfoil.T, m_thicknessMin, m_thicknessMax);
    mutateValue(mutated.tipAirfoil.M, 0.0f, mutated.rootAirfoil.M);
    mutateValue(mutated.tipAirfoil.T, m_thicknessMin * 0.5f, mutated.rootAirfoil.T);
    mutateValue(mutated.rootChord, m_chordRootMin, m_chordRootMax);
    mutateValue(mutated.tipChord, m_chordTipMin, mutated.rootChord * 0.5f);
    mutateValue(mutated.twistAngle, m_twistMin, m_twistMax);
    mutateValue(mutated.pitchAngle, -10.0f, 10.0f);
    
    return mutated;
}

void printOptimizationResults(const EvaluationResult& result) {
    std::cout << "\n=== Optimization Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    std::cout << "\nBlade Parameters:" << std::endl;
    std::cout << "  Root Airfoil: NACA " << result.params.rootAirfoil.getDesignation() << std::endl;
    std::cout << "    Camber: " << result.params.rootAirfoil.M * 100 << "%" << std::endl;
    std::cout << "    Camber Position: " << result.params.rootAirfoil.P * 100 << "% chord" << std::endl;
    std::cout << "    Thickness: " << result.params.rootAirfoil.T * 100 << "%" << std::endl;
    
    std::cout << "  Tip Airfoil: NACA " << result.params.tipAirfoil.getDesignation() << std::endl;
    std::cout << "  Root Chord: " << result.params.rootChord << " m" << std::endl;
    std::cout << "  Tip Chord: " << result.params.tipChord << " m" << std::endl;
    std::cout << "  Twist: " << result.params.twistAngle << " deg" << std::endl;
    std::cout << "  Pitch: " << result.params.pitchAngle << " deg" << std::endl;
    
    std::cout << "\nPerformance:" << std::endl;
    std::cout << "  Lift Coefficient (Cl): " << result.liftCoefficient << std::endl;
    std::cout << "  Drag Coefficient (Cd): " << result.dragCoefficient << std::endl;
    std::cout << "  Lift/Drag Ratio: " << result.liftToDrag << std::endl;
    std::cout << "  Power: " << result.power / 1000.0f << " kW" << std::endl;
    std::cout << "  Efficiency: " << result.efficiency << "%" << std::endl;
    
    std::cout << "\nFitness Score: " << result.fitness << std::endl;
    std::cout << "Feasible: " << (result.feasible ? "Yes" : "No") << std::endl;
}
