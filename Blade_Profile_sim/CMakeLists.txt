cmake_minimum_required(VERSION 3.16)
project(WindTurbineBladeSim VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(glm REQUIRED)
find_package(assimp REQUIRED)

# Add executable
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/application.cpp
    src/renderer.cpp
    src/shader.cpp
    src/mesh.cpp
    src/cad_loader.cpp
    src/camera.cpp
    src/wind_simulation.cpp
    src/blade_physics.cpp
    src/stats_overlay.cpp
    src/text_renderer.cpp
    src/glad.c
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OPENGL_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenGL::GL
    glfw
    glm::glm
    assimp::assimp
)

# Copy shaders and resources to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shaders DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/resources DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# macOS specific settings
if(APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreVideo"
    )
endif()

# Installation
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(DIRECTORY shaders/ DESTINATION share/${PROJECT_NAME}/shaders)
install(DIRECTORY resources/ DESTINATION share/${PROJECT_NAME}/resources)
