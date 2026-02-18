# Wind Turbine Blade Physics Simulator

A comprehensive C++ physics simulation for wind turbine blade analysis and optimization, featuring real-time 3D visualization of airflow and performance statistics.

## Features

- **CAD File Import**: Load blade geometry from STL, OBJ, FBX, PLY, STEP, IGES, GLTF, and other formats via Assimp
- **NACA Airfoil Generation**: Generate blades using NACA 4-digit airfoil equations
- **Real-time Wind Simulation**: Visualize particle flow and streamlines around the blade
- **Physics Calculations**: 
  - Lift and drag coefficients (Cl, Cd)
  - Lift-to-drag ratio (L/D)
  - Power and torque calculations
  - Reynolds number
  - Pressure distribution
- **Blade Optimization**: Genetic algorithm and random search optimization to find optimal blade designs
- **Interactive 3D Visualization**: OpenGL-based renderer with orbital camera controls

## NACA 4-Digit Airfoil Implementation

The simulator implements the standard NACA 4-digit airfoil equations:

### Camber Line

**Front section (0 ≤ x < p):**
```
y_c = M/p² × (2Px - x²)
dy_c/dx = 2M/p² × (P - x)
```

**Back section (p ≤ x ≤ 1):**
```
y_c = M/(1-p)² × (1 - 2P + 2Px - x²)
dy_c/dx = 2M/(1-p)² × (P - x)
```

### Thickness Distribution
```
y_t = T/0.2 × (a₀x^0.5 + a₁x + a₂x² + a₃x³ + a₄x⁴)
```

Where:
- a₀ = 0.2969
- a₁ = -0.126
- a₂ = -0.3516
- a₃ = 0.2843
- a₄ = -0.1015 (or -0.1036 for closed trailing edge)

### Surface Coordinates
```
θ = atan(dy_c/dx)

Upper surface:
  x_u = x - y_t × sin(θ)
  y_u = y_c + y_t × cos(θ)

Lower surface:
  x_l = x + y_t × sin(θ)
  y_l = y_c - y_t × cos(θ)
```

## Building

### Prerequisites

- CMake 3.16+
- C++17 compatible compiler
- OpenGL 3.3+
- GLFW 3.3+
- GLM
- Assimp

### macOS (Homebrew)

```bash
# Install dependencies
brew install cmake glfw glm assimp

# Build
cd Blade_Profile_sim
mkdir build && cd build
cmake ..
make

# Run
./WindTurbineBladeSim
```

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get install cmake libglfw3-dev libglm-dev libassimp-dev

# Build
cd Blade_Profile_sim
mkdir build && cd build
cmake ..
make
```

### Windows (vcpkg)

```powershell
# Install dependencies
vcpkg install glfw3 glm assimp

# Build
cd Blade_Profile_sim
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg root]/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

## Usage

### Basic Usage

```bash
# Run with default NACA 4412 blade
./WindTurbineBladeSim

# Load a CAD file
./WindTurbineBladeSim path/to/blade.stl

# Generate specific NACA airfoil
./WindTurbineBladeSim --naca 2412

# Run optimization
./WindTurbineBladeSim --optimize --no-gui --wind-speed 12 --rpm 18
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--help, -h` | Show help message |
| `--naca XXXX` | Generate NACA 4-digit airfoil |
| `--optimize` | Run blade optimization |
| `--wind-speed N` | Set wind speed (m/s, default: 10) |
| `--rpm N` | Set rotational speed (RPM, default: 15) |
| `--no-gui` | Run without GUI |

### Interactive Controls

| Key | Action |
|-----|--------|
| Mouse drag | Rotate view |
| Scroll | Zoom |
| WASD | Pan camera |
| +/- | Adjust wind speed |
| ←/→ | Adjust wind angle |
| Space | Toggle simulation |
| P | Toggle particles |
| L | Toggle streamlines |
| H | Toggle stats overlay |
| R | Reset view |
| Esc | Quit |

## Optimization

The optimizer supports multiple objectives:
- **MAXIMIZE_LIFT**: Maximize lift coefficient
- **MINIMIZE_DRAG**: Minimize drag coefficient
- **MAXIMIZE_LIFT_TO_DRAG**: Maximize L/D ratio
- **MAXIMIZE_POWER**: Maximize power output
- **MAXIMIZE_EFFICIENCY**: Maximize efficiency relative to Betz limit

### Constraints

You can add constraints such as:
- Minimum/maximum lift coefficient
- Maximum drag coefficient
- Minimum L/D ratio
- Thickness limits
- Chord length limits

### Example Optimization Output

```
=== Optimization Results ===

Blade Parameters:
  Root Airfoil: NACA 4415
    Camber: 4.0%
    Camber Position: 40% chord
    Thickness: 15%
  Tip Airfoil: NACA 2410
  Root Chord: 4.2 m
  Tip Chord: 1.3 m
  Twist: 14.5 deg
  Pitch: 2.8 deg

Performance:
  Lift Coefficient (Cl): 1.24
  Drag Coefficient (Cd): 0.0089
  Lift/Drag Ratio: 139.3
  Power: 2450 kW
  Efficiency: 78.5%

Fitness Score: 139.3
Feasible: Yes
```

## Stats Display

The stats overlay in the top-right corner shows:

**Wind Conditions**
- Wind Speed (m/s)
- Wind Angle (degrees)

**Geometry**
- Surface Area (m²)
- Span Length (m)
- Chord Length (m)
- Aspect Ratio

**Aerodynamics**
- Cl (Lift Coefficient)
- Cd (Drag Coefficient)
- L/D Ratio
- Angle of Attack

**Forces**
- Lift Force (N)
- Drag Force (N)

**Performance**
- Power (W)
- Torque (N·m)
- Cp (Power Coefficient)
- Efficiency (%)

**Flow Characteristics**
- Reynolds Number
- Tip Speed Ratio

## Project Structure

```
Blade_Profile_sim/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── application.hpp
│   ├── blade_optimizer.hpp
│   ├── blade_physics.hpp
│   ├── cad_loader.hpp
│   ├── camera.hpp
│   ├── mesh.hpp
│   ├── naca_generator.hpp
│   ├── renderer.hpp
│   ├── shader.hpp
│   ├── stats_overlay.hpp
│   ├── text_renderer.hpp
│   └── wind_simulation.hpp
├── src/
│   ├── main.cpp
│   ├── application.cpp
│   ├── blade_optimizer.cpp
│   ├── blade_physics.cpp
│   ├── cad_loader.cpp
│   ├── camera.cpp
│   ├── mesh.cpp
│   ├── naca_generator.cpp
│   ├── renderer.cpp
│   ├── shader.cpp
│   ├── stats_overlay.cpp
│   ├── text_renderer.cpp
│   ├── wind_simulation.cpp
│   └── glad.c
├── include/glad/
│   └── glad.h
├── include/KHR/
│   └── khrplatform.h
├── resources/
│   └── sample_blade.stl
└── shaders/
    └── (embedded in code)
```

## License

MIT License

## References

- NACA Report 460: "The Characteristics of 78 Related Airfoil Sections", 1933
- Thin Airfoil Theory
- Betz Limit for Wind Turbine Efficiency
- Blade Element Momentum Theory
