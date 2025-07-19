#ifndef CENTRAL_BODY_CUH
#define CENTRAL_BODY_CUH

typedef struct {
    float x, y, z;          // Posición (normalmente en el origen)
    float vx, vy, vz;       // Velocidad (puede mantenerse en cero o muy baja)
    float fx, fy, fz;       // Fuerzas (para cálculos, aunque el objeto central suele considerarse inamovible)
    float mass;             // Masa muy alta (>> masa de las partículas)
    float omega;            // Velocidad angular (magnitud)
    float nx, ny, nz;       // Vector unitario que define el eje de rotación
    float epsilon;          // Parámetro de suavizado para evitar singularidades en el potencial
} CentralBody;

void initCentralBody(CentralBody* central);

#endif // CENTRAL_BODY_CUH


