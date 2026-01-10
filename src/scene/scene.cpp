/**
 * @file scene.cpp
 * @brief Implementation of the Scene class.
 */
#include "scene.h"

namespace rigid {

RigidBody& Scene::createBody(const String& name) {
    bodies_.emplace_back(std::make_unique<RigidBody>(name));
    return *bodies_.back();
}

RigidBody* Scene::body(Int index) {
    if (index < 0 || index >= static_cast<Int>(bodies_.size())) {
        return nullptr;
    }
    return bodies_[index].get();
}

const RigidBody* Scene::body(Int index) const {
    if (index < 0 || index >= static_cast<Int>(bodies_.size())) {
        return nullptr;
    }
    return bodies_[index].get();
}

}  // namespace rigid
