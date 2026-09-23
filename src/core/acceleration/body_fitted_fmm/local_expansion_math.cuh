#pragma once

#include "../fmm/fmm_types.h"

namespace body_fitted_fmm {

// Gradient of the harmonic quartic part in cell-normalized coordinates.
// L2L needs this at the child center just as L2P does at a particle position.
__device__ __forceinline__
void add_local_quartic_gradient(const far_real_t* coeffs,
    far_real_t dx, far_real_t dy, far_real_t dz,
    far_real_t& fx, far_real_t& fy, far_real_t& fz)
{
    const far_real_t dx2=dx*dx,dy2=dy*dy,dz2=dz*dz;
    const far_real_t dx3=dx2*dx,dy3=dy2*dy,dz3=dz2*dz;
    far_real_t H_num = FAR_REAL(1.0) / FAR_REAL(420.0);
    fx += H_num * (coeffs[16] * (FAR_REAL(2.0)*dx3 - FAR_REAL(6.0)*dx*dz2) +
        coeffs[18] * (FAR_REAL(2.0)*dy3 - FAR_REAL(6.0)*dy*dz2) +
        coeffs[19] * (FAR_REAL(6.0)*dx2*dy - FAR_REAL(6.0)*dy*dz2) +
        coeffs[20] * (FAR_REAL(6.0)*dx2*dz - FAR_REAL(2.0)*dz3) +
        coeffs[22] * (FAR_REAL(6.0)*dx*dy2 - FAR_REAL(6.0)*dx*dz2) +
        coeffs[23] * (FAR_REAL(12.0)*dx*dy*dz) +
        coeffs[24] * (FAR_REAL(6.0)*dy2*dz - FAR_REAL(2.0)*dz3));
    fy += H_num * (coeffs[17] * (FAR_REAL(2.0)*dy3 - FAR_REAL(6.0)*dy*dz2) +
        coeffs[18] * (FAR_REAL(6.0)*dx*dy2 - FAR_REAL(6.0)*dx*dz2) +
        coeffs[19] * (FAR_REAL(2.0)*dx3 - FAR_REAL(6.0)*dx*dz2) +
        coeffs[21] * (FAR_REAL(6.0)*dy2*dz - FAR_REAL(2.0)*dz3) +
        coeffs[22] * (FAR_REAL(6.0)*dx2*dy - FAR_REAL(6.0)*dy*dz2) +
        coeffs[23] * (FAR_REAL(6.0)*dx2*dz - FAR_REAL(2.0)*dz3) +
        coeffs[24] * (FAR_REAL(12.0)*dx*dy*dz));
    fz += H_num * (coeffs[16] * (-FAR_REAL(6.0)*dx2*dz + FAR_REAL(2.0)*dz3) +
        coeffs[17] * (-FAR_REAL(6.0)*dy2*dz + FAR_REAL(2.0)*dz3) +
        coeffs[18] * (-FAR_REAL(12.0)*dx*dy*dz) +
        coeffs[19] * (-FAR_REAL(12.0)*dx*dy*dz) +
        coeffs[20] * (FAR_REAL(2.0)*dx3 - FAR_REAL(6.0)*dx*dz2) +
        coeffs[21] * (FAR_REAL(2.0)*dy3 - FAR_REAL(6.0)*dy*dz2) +
        coeffs[22] * (-FAR_REAL(6.0)*dx2*dz - FAR_REAL(6.0)*dy2*dz + FAR_REAL(4.0)*dz3) +
        coeffs[23] * (FAR_REAL(6.0)*dx2*dy - FAR_REAL(6.0)*dy*dz2) +
        coeffs[24] * (FAR_REAL(6.0)*dx*dy2 - FAR_REAL(6.0)*dx*dz2));

}

__device__ __forceinline__
void l2l_translate_coefficients_unified(
    const far_real_t* parent_coeffs, far_real_t* child_coeffs,
    far_real_t dx, far_real_t dy, far_real_t dz,
    far_real_t h_parent, far_real_t h_child)
{
    // Normalize offsets: ddx = dx/h_parent etc. are CONSTANT per octant (±0.25 or 0)
    // so all h-power factors cancel algebraically in the Taylor shift.
    far_real_t inv_hp = FAR_REAL(1.0) / h_parent;
    dx *= inv_hp; dy *= inv_hp; dz *= inv_hp;

    far_real_t dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    far_real_t dx3 = dx2 * dx, dy3 = dy2 * dy, dz3 = dz2 * dz;
    far_real_t dx4 = dx2 * dx2, dy4 = dy2 * dy2, dz4 = dz2 * dz2;
    far_real_t dx2dy = dx2 * dy, dx2dz = dx2 * dz;
    far_real_t dxdy2 = dx * dy2, dxdz2 = dx * dz2;
    far_real_t dy2dz = dy2 * dz, dydz2 = dy * dz2;
    far_real_t dxdy = dx * dy, dxdz = dx * dz, dydz = dy * dz;
    far_real_t dxdydz = dx * dy * dz;

    far_real_t L[fmm::FMM_ACTIVE_COEFFS];
    for (int i = 0; i < fmm::FMM_ACTIVE_COEFFS; i++) L[i] = FAR_REAL(0.0);

    far_real_t Q_xx = FAR_FMA(FAR_REAL(2.0), parent_coeffs[4], parent_coeffs[5]) / FAR_REAL(9.0);
    far_real_t Q_yy = FAR_FMA(FAR_REAL(-1.0), parent_coeffs[4], parent_coeffs[5]) / FAR_REAL(9.0);
    far_real_t Q_zz = -(Q_xx + Q_yy);
    far_real_t Q_xy = parent_coeffs[6] / FAR_REAL(3.0);
    far_real_t Q_xz = parent_coeffs[7] / FAR_REAL(3.0);
    far_real_t Q_yz = parent_coeffs[8] / FAR_REAL(3.0);

    L[0] += parent_coeffs[0] + parent_coeffs[1] * dx + parent_coeffs[2] * dy + parent_coeffs[3] * dz
          + FAR_REAL(0.5) * (Q_xx * dx2 + Q_yy * dy2 + Q_zz * dz2
                  + FAR_REAL(2.0) * Q_xy * dxdy + FAR_REAL(2.0) * Q_xz * dxdz + FAR_REAL(2.0) * Q_yz * dydz);

    L[0] += parent_coeffs[9]  * dx3    * (FAR_REAL(1.0) / FAR_REAL(30.0))
          + parent_coeffs[12] * dx2dy  * (FAR_REAL(1.0) / FAR_REAL(10.0))
          + parent_coeffs[13] * dx2dz  * (FAR_REAL(1.0) / FAR_REAL(10.0))
          + parent_coeffs[25] * dxdy2  * (FAR_REAL(1.0) / FAR_REAL(10.0))
          + parent_coeffs[15] * dxdydz * (FAR_REAL(1.0) / FAR_REAL(5.0))
          + parent_coeffs[26] * dxdz2  * (FAR_REAL(1.0) / FAR_REAL(10.0))
          + parent_coeffs[10] * dy3    * (FAR_REAL(1.0) / FAR_REAL(30.0))
          + parent_coeffs[14] * dy2dz  * (FAR_REAL(1.0) / FAR_REAL(10.0))
          + parent_coeffs[27] * dydz2  * (FAR_REAL(1.0) / FAR_REAL(10.0))
          + parent_coeffs[11] * dz3    * (FAR_REAL(1.0) / FAR_REAL(30.0));

    {
        const far_real_t s840 = FAR_REAL(1.0) / FAR_REAL(840.0);
        L[0] += s840 * (
            parent_coeffs[16] * dx4
          + parent_coeffs[19] * FAR_REAL(4.0) * dx3 * dy
          + parent_coeffs[20] * FAR_REAL(4.0) * dx3 * dz
          + parent_coeffs[22] * FAR_REAL(6.0) * dx2 * dy2
          + parent_coeffs[23] * FAR_REAL(12.0) * dx2 * dy * dz
          - parent_coeffs[16] * FAR_REAL(6.0) * dx2 * dz2
          - parent_coeffs[22] * FAR_REAL(6.0) * dx2 * dz2
          + parent_coeffs[18] * FAR_REAL(4.0) * dx * dy3
          + parent_coeffs[24] * FAR_REAL(12.0) * dx * dy2 * dz
          - parent_coeffs[18] * FAR_REAL(12.0) * dx * dy * dz2
          - parent_coeffs[19] * FAR_REAL(12.0) * dx * dy * dz2
          - parent_coeffs[20] * FAR_REAL(4.0) * dx * dz3
          - parent_coeffs[24] * FAR_REAL(4.0) * dx * dz3
          + parent_coeffs[17] * dy4
          + parent_coeffs[21] * FAR_REAL(4.0) * dy3 * dz
          - parent_coeffs[17] * FAR_REAL(6.0) * dy2 * dz2
          - parent_coeffs[22] * FAR_REAL(6.0) * dy2 * dz2
          - parent_coeffs[21] * FAR_REAL(4.0) * dy * dz3
          - parent_coeffs[23] * FAR_REAL(4.0) * dy * dz3
          + parent_coeffs[16] * dz4
          + parent_coeffs[17] * dz4
          + parent_coeffs[22] * FAR_REAL(2.0) * dz4
        );
    }

    L[1] += parent_coeffs[1] + Q_xx * dx + Q_xy * dy + Q_xz * dz
          + FAR_REAL(0.1) * (parent_coeffs[9] * dx2 + parent_coeffs[25] * dy2 + parent_coeffs[26] * dz2
                  + parent_coeffs[12] * FAR_REAL(2.0) * dxdy + parent_coeffs[13] * FAR_REAL(2.0) * dxdz
                  + parent_coeffs[15] * FAR_REAL(2.0) * dydz);
    L[2] += parent_coeffs[2] + Q_xy * dx + Q_yy * dy + Q_yz * dz
          + FAR_REAL(0.1) * (parent_coeffs[12] * dx2 + parent_coeffs[10] * dy2 + parent_coeffs[27] * dz2
                  + parent_coeffs[25] * FAR_REAL(2.0) * dxdy + parent_coeffs[14] * FAR_REAL(2.0) * dydz
                  + parent_coeffs[15] * FAR_REAL(2.0) * dxdz);
    L[3] += parent_coeffs[3] + Q_xz * dx + Q_yz * dy + Q_zz * dz
          + FAR_REAL(0.1) * (parent_coeffs[13] * dx2 + parent_coeffs[14] * dy2 + parent_coeffs[11] * dz2
                  + parent_coeffs[26] * FAR_REAL(2.0) * dxdz + parent_coeffs[27] * FAR_REAL(2.0) * dydz
                  + parent_coeffs[15] * FAR_REAL(2.0) * dxdy);

    // The child gradient is the complete parent gradient at its center.
    // Omitting this p=4 contribution makes force depend on translation depth.
    add_local_quartic_gradient(parent_coeffs,dx,dy,dz,L[1],L[2],L[3]);

    // Quadrupole L2L (indices 4-8)
    L[4] += parent_coeffs[4]
          + (FAR_REAL(3.0)/FAR_REAL(5.0)) * (parent_coeffs[9] * dx - parent_coeffs[25] * dx + parent_coeffs[12] * dy - parent_coeffs[10] * dy + parent_coeffs[13] * dz - parent_coeffs[14] * dz)
          + (FAR_REAL(3.0)/FAR_REAL(70.0)) * (parent_coeffs[16] * dx2 - parent_coeffs[22] * dx2 - parent_coeffs[17] * dy2 + parent_coeffs[22] * dy2 - parent_coeffs[16] * dz2 + parent_coeffs[17] * dz2)
          + (FAR_REAL(3.0)/FAR_REAL(35.0)) * (-parent_coeffs[18] * dxdy + parent_coeffs[19] * dxdy + parent_coeffs[20] * dxdz - parent_coeffs[21] * dydz + parent_coeffs[23] * dydz - parent_coeffs[24] * dxdz);

    L[5] += parent_coeffs[5]
          + (FAR_REAL(3.0)/FAR_REAL(5.0)) * (parent_coeffs[10] * dy - parent_coeffs[11] * dz + parent_coeffs[14] * dz + parent_coeffs[25] * dx - parent_coeffs[26] * dx - parent_coeffs[27] * dy)
          + (FAR_REAL(3.0)/FAR_REAL(70.0)) * (parent_coeffs[16] * dx2 + FAR_REAL(2.0) * parent_coeffs[22] * dx2 + FAR_REAL(2.0) * parent_coeffs[17] * dy2 + parent_coeffs[22] * dy2 - parent_coeffs[16] * dz2 - FAR_REAL(2.0) * parent_coeffs[17] * dz2 - FAR_REAL(3.0) * parent_coeffs[22] * dz2)
          + (FAR_REAL(3.0)/FAR_REAL(35.0)) * (FAR_REAL(2.0) * parent_coeffs[18] * dxdy + parent_coeffs[19] * dxdy + parent_coeffs[20] * dxdz + FAR_REAL(2.0) * parent_coeffs[24] * dxdz + FAR_REAL(2.0) * parent_coeffs[21] * dydz + parent_coeffs[23] * dydz);

    L[6] += parent_coeffs[6]
          + (FAR_REAL(3.0)/FAR_REAL(5.0)) * (parent_coeffs[12] * dx + parent_coeffs[25] * dy + parent_coeffs[15] * dz)
          + (FAR_REAL(3.0)/FAR_REAL(70.0)) * (parent_coeffs[19] * dx2 + parent_coeffs[18] * dy2 - parent_coeffs[18] * dz2 - parent_coeffs[19] * dz2)
          + (FAR_REAL(3.0)/FAR_REAL(35.0)) * (parent_coeffs[22] * dxdy + parent_coeffs[23] * dxdz + parent_coeffs[24] * dydz);

    L[7] += parent_coeffs[7]
          + (FAR_REAL(3.0)/FAR_REAL(5.0)) * (parent_coeffs[13] * dx + parent_coeffs[15] * dy + parent_coeffs[26] * dz)
          - (FAR_REAL(3.0)/FAR_REAL(35.0)) * (parent_coeffs[16] * dxdz + parent_coeffs[18] * dydz + parent_coeffs[19] * dydz + parent_coeffs[22] * dxdz)
          + (FAR_REAL(3.0)/FAR_REAL(70.0)) * (parent_coeffs[20] * dx2 + parent_coeffs[24] * dy2 - parent_coeffs[20] * dz2 - parent_coeffs[24] * dz2)
          + (FAR_REAL(3.0)/FAR_REAL(35.0)) * (parent_coeffs[23] * dxdy);

    L[8] += parent_coeffs[8]
          + (FAR_REAL(3.0)/FAR_REAL(5.0)) * (parent_coeffs[15] * dx + parent_coeffs[14] * dy + parent_coeffs[27] * dz)
          - (FAR_REAL(3.0)/FAR_REAL(35.0)) * (parent_coeffs[17] * dydz + parent_coeffs[18] * dxdz + parent_coeffs[19] * dxdz + parent_coeffs[22] * dydz)
          + (FAR_REAL(3.0)/FAR_REAL(70.0)) * (parent_coeffs[23] * dx2 + parent_coeffs[21] * dy2 - parent_coeffs[21] * dz2 - parent_coeffs[23] * dz2)
          + (FAR_REAL(3.0)/FAR_REAL(35.0)) * (parent_coeffs[24] * dxdy);

    // Octupole + Hexadecapole: identity copy (order-preserving)
    const far_real_t inv5 = FAR_REAL(1.0)/FAR_REAL(5.0);
    const far_real_t inv7 = FAR_REAL(1.0)/FAR_REAL(7.0);

    L[9]  += (FAR_REAL(2.0)/FAR_REAL(5.0)) * parent_coeffs[9] - (FAR_REAL(3.0)/FAR_REAL(5.0)) * (parent_coeffs[25] + parent_coeffs[26])
            + inv7 * (parent_coeffs[16] * dx + parent_coeffs[19] * dy + parent_coeffs[20] * dz);
    L[10] += (FAR_REAL(2.0)/FAR_REAL(5.0)) * parent_coeffs[10] - (FAR_REAL(3.0)/FAR_REAL(5.0)) * (parent_coeffs[12] + parent_coeffs[27])
            + inv7 * (parent_coeffs[18] * dx + parent_coeffs[17] * dy + parent_coeffs[21] * dz);
    L[11] += (FAR_REAL(2.0)/FAR_REAL(5.0)) * parent_coeffs[11] - (FAR_REAL(3.0)/FAR_REAL(5.0)) * (parent_coeffs[13] + parent_coeffs[14])
            + inv7 * ((-parent_coeffs[20] - parent_coeffs[24]) * dx + (-parent_coeffs[21] - parent_coeffs[23]) * dy
                    + (parent_coeffs[16] + parent_coeffs[17] + FAR_REAL(2.0) * parent_coeffs[22]) * dz);
    L[12] += -inv5 * parent_coeffs[10] + (FAR_REAL(4.0)/FAR_REAL(5.0)) * parent_coeffs[12] - inv5 * parent_coeffs[27]
            + inv7 * (parent_coeffs[19] * dx + parent_coeffs[22] * dy + parent_coeffs[23] * dz);
    L[13] += -inv5 * parent_coeffs[11] + (FAR_REAL(4.0)/FAR_REAL(5.0)) * parent_coeffs[13] - inv5 * parent_coeffs[14]
            + inv7 * (parent_coeffs[20] * dx + parent_coeffs[23] * dy + (-parent_coeffs[16] - parent_coeffs[22]) * dz);
    L[14] += -inv5 * parent_coeffs[11] - inv5 * parent_coeffs[13] + (FAR_REAL(4.0)/FAR_REAL(5.0)) * parent_coeffs[14]
            + inv7 * (parent_coeffs[24] * dx + parent_coeffs[21] * dy + (-parent_coeffs[17] - parent_coeffs[22]) * dz);
    L[15] += parent_coeffs[15] + inv7 * (parent_coeffs[23] * dx + parent_coeffs[24] * dy - (parent_coeffs[18] + parent_coeffs[19]) * dz);
    L[25] += -inv5 * parent_coeffs[9] + (FAR_REAL(4.0)/FAR_REAL(5.0)) * parent_coeffs[25] - inv5 * parent_coeffs[26]
            + inv7 * (parent_coeffs[22] * dx + parent_coeffs[18] * dy + parent_coeffs[24] * dz);
    L[26] += -inv5 * parent_coeffs[9] - inv5 * parent_coeffs[25] + (FAR_REAL(4.0)/FAR_REAL(5.0)) * parent_coeffs[26]
            + inv7 * ((-parent_coeffs[16] - parent_coeffs[22]) * dx + (-parent_coeffs[18] - parent_coeffs[19]) * dy
                    + (-parent_coeffs[20] - parent_coeffs[24]) * dz);
    L[27] += -inv5 * parent_coeffs[10] - inv5 * parent_coeffs[12] + (FAR_REAL(4.0)/FAR_REAL(5.0)) * parent_coeffs[27]
            + inv7 * ((-parent_coeffs[18] - parent_coeffs[19]) * dx + (-parent_coeffs[17] - parent_coeffs[22]) * dy
                    + (-parent_coeffs[21] - parent_coeffs[23]) * dz);

    L[16] += parent_coeffs[16];
    L[17] += parent_coeffs[17];
    L[18] += parent_coeffs[18];
    L[19] += parent_coeffs[19];
    L[20] += parent_coeffs[20];
    L[21] += parent_coeffs[21];
    L[22] += parent_coeffs[22];
    L[23] += parent_coeffs[23];
    L[24] += parent_coeffs[24];

    // Normalize: multiply by (h_child/h_parent)^order_i = 2^(-order_i)
    far_real_t p2[5]; p2[0]=FAR_REAL(1.0); p2[1]=FAR_REAL(0.5); p2[2]=FAR_REAL(0.25); p2[3]=FAR_REAL(0.125); p2[4]=FAR_REAL(0.0625);
    for (int i = 0; i < fmm::FMM_ACTIVE_COEFFS; i++) {
        child_coeffs[i] += L[i] * p2[fmm::get_coeff_order(i)];
    }
}

__device__ __forceinline__
void evaluate_local_expansion_force_unified(
    const far_real_t* coeffs,
    far_real_t dx, far_real_t dy, far_real_t dz,
    far_real_t& fx, far_real_t& fy, far_real_t& fz,
    far_real_t h_leaf)
{
    // Force evaluation with normalized coefficients.
    // All terms have common 1/h_leaf factor when using normalized coords.
    far_real_t inv_h = FAR_REAL(1.0) / h_leaf;
    dx *= inv_h; dy *= inv_h; dz *= inv_h;

    far_real_t dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    far_real_t dxdy = dx * dy, dxdz = dx * dz, dydz = dy * dz;
    far_real_t dx3 = dx2 * dx, dy3 = dy2 * dy, dz3 = dz2 * dz;

    fx = FAR_REAL(0.0); fy = FAR_REAL(0.0); fz = FAR_REAL(0.0);

    fx += coeffs[1]; fy += coeffs[2]; fz += coeffs[3];

    far_real_t H_xy = coeffs[6] / FAR_REAL(3.0);
    far_real_t H_xz = coeffs[7] / FAR_REAL(3.0);
    far_real_t H_yz = coeffs[8] / FAR_REAL(3.0);
    far_real_t H_xx = (FAR_REAL(2.0) * coeffs[4] + coeffs[5]) / FAR_REAL(9.0);
    far_real_t H_yy = (-coeffs[4] + coeffs[5]) / FAR_REAL(9.0);
    far_real_t H_zz = -(H_xx + H_yy);

    fx += (H_xx * dx + H_xy * dy + H_xz * dz);
    fy += (H_xy * dx + H_yy * dy + H_yz * dz);
    fz += (H_xz * dx + H_yz * dy + H_zz * dz);

    fx += FAR_REAL(0.1) * (coeffs[9] * dx2 + coeffs[25] * dy2 + coeffs[26] * dz2 +
        coeffs[12] * FAR_REAL(2.0) * dxdy + coeffs[13] * FAR_REAL(2.0) * dxdz + coeffs[15] * FAR_REAL(2.0) * dydz);
    fy += FAR_REAL(0.1) * (coeffs[12] * dx2 + coeffs[10] * dy2 + coeffs[27] * dz2 +
        coeffs[25] * FAR_REAL(2.0) * dxdy + coeffs[14] * FAR_REAL(2.0) * dydz + coeffs[15] * FAR_REAL(2.0) * dxdz);
    fz += FAR_REAL(0.1) * (coeffs[13] * dx2 + coeffs[14] * dy2 + coeffs[11] * dz2 +
        coeffs[26] * FAR_REAL(2.0) * dxdz + coeffs[27] * FAR_REAL(2.0) * dydz + coeffs[15] * FAR_REAL(2.0) * dxdy);

    add_local_quartic_gradient(coeffs,dx,dy,dz,fx,fy,fz);

    // All terms have common 1/h_leaf factor
    fx *= inv_h;
    fy *= inv_h;
    fz *= inv_h;
}

} // namespace body_fitted_fmm
