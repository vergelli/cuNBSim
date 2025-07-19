#include "centralBody.cuh"

void initCentralBody(CentralBody* central) {
    central->x = 0.0f;
    central->y = 0.0f;
    central->z = 0.0f;
    central->vx = 0.0f;
    central->vy = 0.0f;
    central->vz = 0.0f;
    central->fx = 0.0f;
    central->fy = 0.0f;
    central->fz = 0.0f;
    central->mass = 1e9f;
    central->omega = 1.0f;
    central->nx = 0.0f;
    central->ny = 0.0f;
    central->nz = 1.0f;
    central->epsilon = 0.1f;
}
