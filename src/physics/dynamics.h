/*
 * Force and Jacobian Assembly System
 */
#ifndef PHYS3D_FORCE_ASSEMBLER_HPP
#define PHYS3D_FORCE_ASSEMBLER_HPP

#include "intersection.h"
#include "core/types.h"
#include "scene/world.h"

namespace phys3d {

/*
 * ForceAssembler - Computes external forces and stiffness matrices
 */
class ForceAssembler 
{
public:
    ForceAssembler();

    void configure(IntType dofTotal);

    void buildForceVector(World& world, const DynArray<ContactPoint>& contacts, VectorN& forceOut);
    void buildStiffnessMatrix(World& world, const DynArray<ContactPoint>& contacts, SparseGrid& stiffnessOut);

    void setGravityVector(const Point3& g) { m_gravityAccel = g; }
    void setContactStiffness(RealType k) { m_contactStiffness = k; }
    void setFrictionStiffness(RealType k) { m_frictionStiffness = k; }

    [[nodiscard]] const Point3& gravityVector() const { return m_gravityAccel; }
    [[nodiscard]] RealType contactStiffness() const { return m_contactStiffness; }

private:
    void accumulateGravity(World& world, VectorN& forceOut);
    void accumulateContactForces(World& world, const DynArray<ContactPoint>& contacts, VectorN& forceOut);
    void accumulateContactStiffness(const DynArray<ContactPoint>& contacts, SparseGrid& stiffnessOut);

    Point3   m_gravityAccel;
    RealType m_contactStiffness;
    RealType m_frictionStiffness;
    IntType  m_dofTotal;
};

}  // namespace phys3d

namespace rigid {
    using ForceBuilder = phys3d::ForceAssembler;
}

#endif // PHYS3D_FORCE_ASSEMBLER_HPP
