// Minimal GLAD OpenGL loader stub for buildability
#include <glad/glad.h>
#include <stddef.h>
int gladLoadGL(void) { return 1; }
void* gladGetProcAddress(const char* name) { (void)name; return NULL; }
