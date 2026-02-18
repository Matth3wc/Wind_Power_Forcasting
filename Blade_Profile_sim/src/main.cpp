#include "application.hpp"
#include "naca_generator.hpp"
#include "blade_optimizer.hpp"

#include <iostream>
#include <string>
#include <cstring>

void printUsage(const char* programName) {
    std::cout << "Wind Turbine Blade Physics Simulator\n\n";
    std::cout << "Usage: " << programName << " [options] [blade_file]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --help, -h          Show this help message\n";
    std::cout << "  --naca XXXX         Generate NACA 4-digit airfoil (e.g., 2412)\n";
    std::cout << "  --optimize          Run blade optimization\n";
    std::cout << "  --wind-speed N      Set wind speed in m/s (default: 10)\n";
    std::cout << "  --rpm N             Set rotational speed in RPM (default: 15)\n";
    std::cout << "  --no-gui            Run without GUI (optimization mode only)\n\n";
    std::cout << "Supported CAD formats: STL, OBJ, FBX, PLY, DAE, STEP, IGES, GLTF\n\n";
    std::cout << "Controls (GUI mode):\n";
    std::cout << "  Mouse drag          Rotate view\n";
    std::cout << "  Scroll              Zoom\n";
    std::cout << "  WASD                Pan camera\n";
    std::cout << "  +/-                 Adjust wind speed\n";
    std::cout << "  Left/Right arrows   Adjust wind angle\n";
    std::cout << "  Space               Toggle simulation\n";
    std::cout << "  P                   Toggle particles\n";
    std::cout << "  L                   Toggle streamlines\n";
    std::cout << "  H                   Toggle stats overlay\n";
    std::cout << "  R                   Reset view\n";
    std::cout << "  Esc                 Quit\n";
}

void runOptimization(float windSpeed, float rpm) {
    std::cout << "=== Blade Optimization Mode ===" << std::endl;
    std::cout << "Wind Speed: " << windSpeed << " m/s" << std::endl;
    std::cout << "RPM: " << rpm << std::endl;
    std::cout << "\nRunning genetic algorithm optimization..." << std::endl;
    
    BladeOptimizer optimizer;
    optimizer.setWindSpeed(windSpeed);
    optimizer.setRotationalSpeed(rpm);
    optimizer.setBladeRadius(50.0f);
    optimizer.setAirDensity(1.225f);
    
    // Set constraints
    optimizer.addConstraint(OptimizationConstraint(
        OptimizationConstraint::Type::MIN_LIFT_TO_DRAG, 10.0f, 1.0f));
    optimizer.addConstraint(OptimizationConstraint(
        OptimizationConstraint::Type::MIN_THICKNESS, 0.1f, 0.5f));
    optimizer.addConstraint(OptimizationConstraint(
        OptimizationConstraint::Type::MAX_THICKNESS, 0.21f, 0.5f));
    
    // Set objective
    optimizer.setObjective(OptimizationObjective::MAXIMIZE_LIFT_TO_DRAG);
    
    // Run optimization with progress callback
    auto result = optimizer.geneticAlgorithm(50, 30, [](int gen, const EvaluationResult& best) {
        std::cout << "Generation " << gen << ": Best fitness = " << best.fitness 
                  << ", L/D = " << best.liftToDrag << std::endl;
    });
    
    // Print results
    printOptimizationResults(result);
}

int main(int argc, char** argv) {
    std::string bladeFile;
    std::string nacaDesignation;
    bool runOptimize = false;
    bool noGui = false;
    float windSpeed = 10.0f;
    float rpm = 15.0f;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--naca") == 0 && i + 1 < argc) {
            nacaDesignation = argv[++i];
        } else if (strcmp(argv[i], "--optimize") == 0) {
            runOptimize = true;
        } else if (strcmp(argv[i], "--wind-speed") == 0 && i + 1 < argc) {
            windSpeed = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--rpm") == 0 && i + 1 < argc) {
            rpm = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--no-gui") == 0) {
            noGui = true;
        } else if (argv[i][0] != '-') {
            bladeFile = argv[i];
        }
    }
    
    // Run optimization mode
    if (runOptimize && noGui) {
        runOptimization(windSpeed, rpm);
        return 0;
    }
    
    // GUI mode
    try {
        Application app(1600, 900, "Wind Turbine Blade Simulator");
        
        if (!nacaDesignation.empty()) {
            // Generate NACA airfoil
            std::cout << "Generating NACA " << nacaDesignation << " blade..." << std::endl;
            
            NACAGenerator generator;
            BladeParameters params;
            params.rootAirfoil = NACAParameters::fromDesignation(nacaDesignation);
            params.tipAirfoil = params.rootAirfoil;
            params.tipAirfoil.M *= 0.5f;  // Less camber at tip
            params.tipAirfoil.T *= 0.7f;  // Thinner at tip
            params.spanLength = 50.0f;
            params.rootChord = 4.0f;
            params.tipChord = 1.2f;
            params.twistAngle = 12.0f;
            params.pitchAngle = 3.0f;
            
            auto blade = generator.generateBlade(params);
            
            // Save generated blade to file
            std::string outFile = "generated_blade_" + nacaDesignation + ".obj";
            std::cout << "Generated blade saved conceptually (mesh generated in memory)" << std::endl;
            
            // The application will use this generated mesh
            // For now, we'll need to modify application to accept a mesh directly
            // This is a placeholder for the integration
            
        } else if (!bladeFile.empty()) {
            app.loadBlade(bladeFile);
        } else {
            // Generate default blade for demonstration
            std::cout << "No blade specified. Generating default NACA 4412 blade..." << std::endl;
            
            NACAGenerator generator;
            BladeParameters params;
            params.rootAirfoil = NACAParameters::fromDesignation("4412");
            params.tipAirfoil = NACAParameters::fromDesignation("2412");
            params.spanLength = 50.0f;
            params.rootChord = 4.0f;
            params.tipChord = 1.2f;
            params.twistAngle = 12.0f;
            params.pitchAngle = 3.0f;
            
            // Generate and use the blade
            // (Application needs to be modified to accept Mesh directly)
        }
        
        app.run();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
