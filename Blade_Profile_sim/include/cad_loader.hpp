#pragma once

#include <memory>
#include <string>
#include <vector>

class Mesh;

class CADLoader {
public:
    CADLoader();
    ~CADLoader();
    
    std::shared_ptr<Mesh> loadFile(const std::string& filepath);
    
    // Supported formats
    static std::vector<std::string> getSupportedFormats();
    static bool isSupported(const std::string& extension);
    
    // Get error message if loading failed
    const std::string& getLastError() const { return m_lastError; }
    
private:
    std::string m_lastError;
};
