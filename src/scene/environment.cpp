/**
 * @file environment.cpp
 * @brief Implementation of the Environment class.
 */
#include "environment.h"

namespace rigid {

Environment::Environment() {
    // Default bounds: 20x20x20 box centered at origin
    setBounds(Vec3(-10.0f, -10.0f, -10.0f), Vec3(10.0f, 10.0f, 10.0f));
}

void Environment::setBounds(const Vec3& minCorner, const Vec3& maxCorner) {
    minCorner_ = minCorner;
    maxCorner_ = maxCorner;

    // Negative X boundary: normal points +X, pushes objects right
    planes_[kNegX].normal = Vec3(1.0f, 0.0f, 0.0f);
    planes_[kNegX].offset = minCorner.x();

    // Positive X boundary: normal points -X, pushes objects left
    planes_[kPosX].normal = Vec3(-1.0f, 0.0f, 0.0f);
    planes_[kPosX].offset = -maxCorner.x();

    // Negative Y boundary
    planes_[kNegY].normal = Vec3(0.0f, 1.0f, 0.0f);
    planes_[kNegY].offset = minCorner.y();

    // Positive Y boundary
    planes_[kPosY].normal = Vec3(0.0f, -1.0f, 0.0f);
    planes_[kPosY].offset = -maxCorner.y();

    // Negative Z boundary (floor)
    planes_[kNegZ].normal = Vec3(0.0f, 0.0f, 1.0f);
    planes_[kNegZ].offset = minCorner.z();

    // Positive Z boundary (ceiling)
    planes_[kPosZ].normal = Vec3(0.0f, 0.0f, -1.0f);
    planes_[kPosZ].offset = -maxCorner.z();
}

}  // namespace rigid
