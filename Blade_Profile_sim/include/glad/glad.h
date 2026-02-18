// GLAD OpenGL loader header (truncated for brevity)
// In a real project, generate this file using the GLAD web service for OpenGL 3.3 Core
// Here, we provide a minimal stub for buildability and integration
#ifndef GLAD_GLAD_H_
#define GLAD_GLAD_H_
#ifdef __cplusplus
extern "C" {
#endif
int gladLoadGL(void);
void* gladGetProcAddress(const char* name);
#ifdef __cplusplus
}
#endif
#endif // GLAD_GLAD_H_
