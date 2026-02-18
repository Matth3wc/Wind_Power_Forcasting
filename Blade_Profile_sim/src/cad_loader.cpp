#include "cad_loader.hpp"
#include "mesh.hpp"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <iostream>
#include <algorithm>

CADLoader::CADLoader() {
}

CADLoader::~CADLoader() {
}

std::shared_ptr<Mesh> CADLoader::loadFile(const std::string& filepath) {
    Assimp::Importer importer;
    
    // Import with processing flags
    const aiScene* scene = importer.ReadFile(filepath,
        aiProcess_Triangulate |
        aiProcess_GenNormals |
        aiProcess_GenUVCoords |
        aiProcess_OptimizeMeshes |
        aiProcess_JoinIdenticalVertices |
        aiProcess_ImproveCacheLocality |
        aiProcess_FixInfacingNormals
    );
    
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        m_lastError = importer.GetErrorString();
        return nullptr;
    }
    
    // Collect all vertices and indices from all meshes
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    
    // Process all meshes in the scene
    for (unsigned int meshIdx = 0; meshIdx < scene->mNumMeshes; ++meshIdx) {
        aiMesh* aiMeshPtr = scene->mMeshes[meshIdx];
        
        unsigned int indexOffset = static_cast<unsigned int>(vertices.size());
        
        // Process vertices
        for (unsigned int i = 0; i < aiMeshPtr->mNumVertices; ++i) {
            Vertex vertex;
            
            vertex.position = glm::vec3(
                aiMeshPtr->mVertices[i].x,
                aiMeshPtr->mVertices[i].y,
                aiMeshPtr->mVertices[i].z
            );
            
            if (aiMeshPtr->HasNormals()) {
                vertex.normal = glm::vec3(
                    aiMeshPtr->mNormals[i].x,
                    aiMeshPtr->mNormals[i].y,
                    aiMeshPtr->mNormals[i].z
                );
            } else {
                vertex.normal = glm::vec3(0.0f, 1.0f, 0.0f);
            }
            
            if (aiMeshPtr->mTextureCoords[0]) {
                vertex.texCoords = glm::vec2(
                    aiMeshPtr->mTextureCoords[0][i].x,
                    aiMeshPtr->mTextureCoords[0][i].y
                );
            } else {
                vertex.texCoords = glm::vec2(0.0f);
            }
            
            vertices.push_back(vertex);
        }
        
        // Process indices
        for (unsigned int i = 0; i < aiMeshPtr->mNumFaces; ++i) {
            aiFace face = aiMeshPtr->mFaces[i];
            for (unsigned int j = 0; j < face.mNumIndices; ++j) {
                indices.push_back(face.mIndices[j] + indexOffset);
            }
        }
    }
    
    if (vertices.empty()) {
        m_lastError = "No vertices found in file";
        return nullptr;
    }
    
    // Create mesh
    auto mesh = std::make_shared<Mesh>();
    mesh->setVertices(vertices);
    mesh->setIndices(indices);
    mesh->build();
    
    std::cout << "Loaded mesh with " << vertices.size() << " vertices and " 
              << indices.size() / 3 << " triangles" << std::endl;
    
    return mesh;
}

std::vector<std::string> CADLoader::getSupportedFormats() {
    return {
        ".stl",    // STL (Stereolithography)
        ".obj",    // Wavefront OBJ
        ".fbx",    // Autodesk FBX
        ".ply",    // Stanford PLY
        ".dae",    // COLLADA
        ".3ds",    // 3DS Max
        ".blend",  // Blender
        ".gltf",   // glTF
        ".glb",    // glTF Binary
        ".step",   // STEP CAD
        ".stp",    // STEP CAD
        ".iges",   // IGES CAD
        ".igs",    // IGES CAD
    };
}

bool CADLoader::isSupported(const std::string& extension) {
    auto formats = getSupportedFormats();
    std::string ext = extension;
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    for (const auto& fmt : formats) {
        if (fmt == ext) return true;
    }
    return false;
}
