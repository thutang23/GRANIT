// =============================================================================
// P2M + M2M Unified Kernels (parent-side M2M, uses UnifiedNode)
// =============================================================================
// Included only by body_fitted_fmm_manager.cu.
// Separate from multipole_compute.cu to avoid breaking lbvh_fmm_manager.cu.
// =============================================================================

namespace body_fitted_fmm {

constexpr int MPU_BLOCK_SIZE = 256;

// =============================================================================
// Device Function: Accumulate Multipole Coefficients (p=4, trace-free)
// (Identical to the one in multipole_compute.cu)
// =============================================================================

__device__ __forceinline__
void accumulate_multipole_p4_unified(
    double coeffs[fmm::FMM_ACTIVE_COEFFS],
    far_real_t particle_pos_x, far_real_t particle_pos_y, far_real_t particle_pos_z,
    far_real_t particle_mass,
    far_real_t node_com_x, far_real_t node_com_y, far_real_t node_com_z)
{
    const far_real_t dx = particle_pos_x - node_com_x;
    const far_real_t dy = particle_pos_y - node_com_y;
    const far_real_t dz = particle_pos_z - node_com_z;
    
    const far_real_t r_sq = dx * dx + dy * dy + dz * dz;
    const far_real_t r = FAR_SQRT(r_sq);
    
    const far_real_t dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    const far_real_t dxdy = dx * dy, dxdz = dx * dz, dydz = dy * dz;
    const far_real_t dx3 = dx2 * dx, dy3 = dy2 * dy, dz3 = dz2 * dz;
    const far_real_t dx4 = dx2 * dx2, dy4 = dy2 * dy2, dz4 = dz2 * dz2;
    
    coeffs[0] += particle_mass;
    coeffs[1] += particle_mass * dx;
    coeffs[2] += particle_mass * dy;
    coeffs[3] += particle_mass * dz;
    coeffs[4] += particle_mass * 3.0 * (dx2 - dy2);
    coeffs[5] += particle_mass * 3.0 * (dy2 - dz2);
    coeffs[6] += particle_mass * 3.0 * dxdy;
    coeffs[7] += particle_mass * 3.0 * dxdz;
    coeffs[8] += particle_mass * 3.0 * dydz;
    coeffs[9] += particle_mass * (5.0 * dx3 - 3.0 * dx * r_sq);
    coeffs[10] += particle_mass * (5.0 * dy3 - 3.0 * dy * r_sq);
    coeffs[11] += particle_mass * (5.0 * dz3 - 3.0 * dz * r_sq);
    coeffs[12] += particle_mass * (5.0 * dx2 * dy - dy * r_sq);
    coeffs[13] += particle_mass * (5.0 * dx2 * dz - dz * r_sq);
    coeffs[14] += particle_mass * (5.0 * dy2 * dz - dz * r_sq);
    coeffs[15] += particle_mass * 5.0 * dx * dy * dz;
    coeffs[25] += particle_mass * (5.0 * dx * dy2 - dx * r_sq);
    coeffs[26] += particle_mass * (5.0 * dx * dz2 - dx * r_sq);
    coeffs[27] += particle_mass * (5.0 * dy * dz2 - dy * r_sq);
    coeffs[16] += particle_mass * (35.0 * dx4 - 30.0 * dx2 * r_sq + 3.0 * r_sq * r_sq);
    coeffs[17] += particle_mass * (35.0 * dy4 - 30.0 * dy2 * r_sq + 3.0 * r_sq * r_sq);
    coeffs[18] += particle_mass * (35.0 * dx * dy3 - 15.0 * dx * dy * r_sq);
    coeffs[19] += particle_mass * (35.0 * dx3 * dy - 15.0 * dx * dy * r_sq);
    coeffs[20] += particle_mass * (35.0 * dx3 * dz - 15.0 * dx * dz * r_sq);
    coeffs[21] += particle_mass * (35.0 * dy3 * dz - 15.0 * dy * dz * r_sq);
    coeffs[22] += particle_mass * (35.0 * dx2 * dy2 - 5.0 * (dx2 + dy2) * r_sq + r_sq * r_sq);
    coeffs[23] += particle_mass * (35.0 * dx2 * dy * dz - 5.0 * dy * dz * r_sq);
    coeffs[24] += particle_mass * (35.0 * dx * dy2 * dz - 5.0 * dx * dz * r_sq);
}

// =============================================================================
// Kernel: P2M Unified (Particle-to-Multipole) for Leaf Nodes
// =============================================================================

__global__ void fmm_p2m_leaf_multipoles_unified(
    const far_real_t* __restrict__ pos_x,
    const far_real_t* __restrict__ pos_y,
    const far_real_t* __restrict__ pos_z,
    const float* __restrict__ mass,
    const int* __restrict__ sorted_indices,
    const int2* __restrict__ node_range,
    double* __restrict__ multipoles_dbl,
    UnifiedNode* __restrict__ nodes,
    int num_nodes)
{
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx >= num_nodes) return;

    const uint8_t flags = nodes[node_idx].flags;
    if (!(flags & UNIFIED_NODE_FLAG_VALID) ||
        !(flags & UNIFIED_NODE_FLAG_IS_LEAF)) return;

    int p_begin = nodes[node_idx].begin;
    int p_count = nodes[node_idx].count;
    // A valid empty leaf must also overwrite its complete scratch result.

    // Compute mass-weighted COM (informational only, expansion center stays geometric)
    far_real_t sum_mass = 0.0;
    far_real_t sum_mx = 0.0, sum_my = 0.0, sum_mz = 0.0;
    for (int i = 0; i < p_count; i++) {
        int sorted_idx = p_begin + i;
        int orig_idx = sorted_indices[sorted_idx];
        far_real_t m = static_cast<far_real_t>(mass[orig_idx]);
        sum_mass += m;
        sum_mx += m * pos_x[orig_idx];
        sum_my += m * pos_y[orig_idx];
        sum_mz += m * pos_z[orig_idx];
    }

    // Use geometric center as expansion center (UnifiedNode.center already holds it)
    far_real_t expansion_center_x = static_cast<far_real_t>(nodes[node_idx].center.x);
    far_real_t expansion_center_y = static_cast<far_real_t>(nodes[node_idx].center.y);
    far_real_t expansion_center_z = static_cast<far_real_t>(nodes[node_idx].center.z);

    // Accumulate multipole coefficients around expansion center
    double coeffs[fmm::FMM_ACTIVE_COEFFS];
    for (int i = 0; i < fmm::FMM_ACTIVE_COEFFS; i++) coeffs[i] = 0.0;
    float leaf_R_max = 0.0f;

    for (int i = 0; i < p_count; i++) {
        int sorted_idx = p_begin + i;
        int orig_idx = sorted_indices[sorted_idx];
        far_real_t m = static_cast<far_real_t>(mass[orig_idx]);
        accumulate_multipole_p4_unified(coeffs,
            pos_x[orig_idx], pos_y[orig_idx], pos_z[orig_idx], m,
            expansion_center_x, expansion_center_y, expansion_center_z);
        far_real_t dx = pos_x[orig_idx] - expansion_center_x;
        far_real_t dy = pos_y[orig_idx] - expansion_center_y;
        far_real_t dz = pos_z[orig_idx] - expansion_center_z;
        far_real_t dist = FAR_SQRT(dx*dx + dy*dy + dz*dz);
        if (dist > leaf_R_max) leaf_R_max = static_cast<float>(dist);
    }

    for (int c = 0; c < fmm::FMM_ACTIVE_COEFFS; c++) {
        multipoles_dbl[node_idx * fmm::FMM_ACTIVE_COEFFS + c] = coeffs[c];
    }
    nodes[node_idx].radius = leaf_R_max;
}

// =============================================================================
// Kernel: Mass-Weighted COM Unified (bottom-up, for internal nodes)
// =============================================================================

__global__ void fmm_mass_weighted_com_unified(
    double* __restrict__ multipoles_dbl,
    const UnifiedNode* __restrict__ nodes,
    const int* __restrict__ children_compact,
    const int* __restrict__ level_list,
    const int* __restrict__ level_start,
    const int* __restrict__ level_count,
    int current_level,
    int max_nodes)
{
    int node_offset = blockIdx.x * blockDim.x + threadIdx.x;
    int level_start_idx = level_start[current_level];
    int level_cnt = level_count[current_level];
    if (node_offset >= level_cnt) return;

    int node_idx = level_list[level_start_idx + node_offset];
    if (node_idx < 0 || node_idx >= max_nodes) return;
    if (nodes[node_idx].childBase == UNIFIED_NODE_LEAF_CHILDBASE) return;

    far_real_t sum_mass = 0.0;
    for (int o = 0; o < 8; o++) {
        int child_idx = get_child_idx<3>(nodes, children_compact, node_idx, o);
        if (child_idx < 0 || child_idx >= max_nodes) continue;
        far_real_t child_mass = static_cast<far_real_t>(multipoles_dbl[child_idx * fmm::FMM_ACTIVE_COEFFS + 0]);
        if (child_mass <= 0.0) continue;
        sum_mass += child_mass;
    }
    // Coefficient zero is produced by this pass, including zero-mass parents.
    multipoles_dbl[node_idx * fmm::FMM_ACTIVE_COEFFS + 0] = static_cast<double>(sum_mass);
}

// =============================================================================
// Kernel: M2M Translation Parent-Side (no atomics, accumulates locally)
// =============================================================================

// Shared p=4 translation used by the ordinary upward pass and image reuse.
// The caller owns coefficient zero and radius bookkeeping.
__device__ __forceinline__ void translate_multipole_p4_unified(
    double* parent_coeffs, const double* child_coeffs,
    far_real_t dx, far_real_t dy, far_real_t dz)
{
    far_real_t dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    far_real_t dxdy = dx * dy, dxdz = dx * dz, dydz = dy * dz;
    far_real_t dx3 = dx2 * dx, dy3 = dy2 * dy, dz3 = dz2 * dz;
    far_real_t dx4 = dx2 * dx2, dy4 = dy2 * dy2, dz4 = dz2 * dz2;
    far_real_t r_sq = dx2 + dy2 + dz2;

    // === M2M Translation p=4 (SymPy-verified, same math as child-side) ===
    parent_coeffs[1] += child_coeffs[1] + child_coeffs[0] * dx;
    parent_coeffs[2] += child_coeffs[2] + child_coeffs[0] * dy;
    parent_coeffs[3] += child_coeffs[3] + child_coeffs[0] * dz;
    parent_coeffs[4] += child_coeffs[4] + 6.0*(child_coeffs[1]*dx - child_coeffs[2]*dy) + child_coeffs[0]*3.0*(dx2-dy2);
    parent_coeffs[5] += child_coeffs[5] + 6.0*(child_coeffs[2]*dy - child_coeffs[3]*dz) + child_coeffs[0]*3.0*(dy2-dz2);
    parent_coeffs[6] += child_coeffs[6] + 3.0*(child_coeffs[1]*dy + child_coeffs[2]*dx) + child_coeffs[0]*3.0*dxdy;
    parent_coeffs[7] += child_coeffs[7] + 3.0*(child_coeffs[1]*dz + child_coeffs[3]*dx) + child_coeffs[0]*3.0*dxdz;
    parent_coeffs[8] += child_coeffs[8] + 3.0*(child_coeffs[2]*dz + child_coeffs[3]*dy) + child_coeffs[0]*3.0*dydz;
    parent_coeffs[9] += child_coeffs[9] + child_coeffs[4]*(2.0*dx) + child_coeffs[5]*dx + child_coeffs[6]*(-2.0*dy) + child_coeffs[7]*(-2.0*dz) + child_coeffs[1]*(6.0*dx2-3.0*dy2-3.0*dz2) + child_coeffs[2]*(-6.0*dxdy) + child_coeffs[3]*(-6.0*dxdz) + child_coeffs[0]*dx*(2.0*dx2-3.0*dy2-3.0*dz2);
    parent_coeffs[10] += child_coeffs[10] + child_coeffs[4]*(-dy) + child_coeffs[5]*dy + child_coeffs[6]*(-2.0*dx) + child_coeffs[8]*(-2.0*dz) + child_coeffs[1]*(-6.0*dxdy) + child_coeffs[2]*(-3.0*dx2+6.0*dy2-3.0*dz2) + child_coeffs[3]*(-6.0*dydz) + child_coeffs[0]*dy*(-3.0*dx2+2.0*dy2-3.0*dz2);
    parent_coeffs[11] += child_coeffs[11] + child_coeffs[4]*(-dz) + child_coeffs[5]*(-2.0*dz) + child_coeffs[7]*(-2.0*dx) + child_coeffs[8]*(-2.0*dy) + child_coeffs[1]*(-6.0*dxdz) + child_coeffs[2]*(-6.0*dydz) + child_coeffs[3]*(-3.0*dx2-3.0*dy2+6.0*dz2) + child_coeffs[0]*dz*(-3.0*dx2-3.0*dy2+2.0*dz2);
    parent_coeffs[12] += child_coeffs[12] + child_coeffs[4]*(4.0*dy/3.0) + child_coeffs[5]*(dy/3.0) + child_coeffs[6]*(8.0*dx/3.0) + child_coeffs[8]*(-2.0*dz/3.0) + child_coeffs[1]*(8.0*dxdy) + child_coeffs[2]*(4.0*dx2-3.0*dy2-dz2) + child_coeffs[3]*(-2.0*dydz) + child_coeffs[0]*dy*(4.0*dx2-dy2-dz2);
    parent_coeffs[13] += child_coeffs[13] + child_coeffs[4]*(4.0*dz/3.0) + child_coeffs[5]*dz + child_coeffs[7]*(8.0*dx/3.0) + child_coeffs[8]*(-2.0*dy/3.0) + child_coeffs[1]*(8.0*dxdz) + child_coeffs[2]*(-2.0*dydz) + child_coeffs[3]*(4.0*dx2-dy2-3.0*dz2) + child_coeffs[0]*dz*(4.0*dx2-dy2-dz2);
    parent_coeffs[14] += child_coeffs[14] + child_coeffs[4]*(-dz/3.0) + child_coeffs[5]*dz + child_coeffs[7]*(-2.0*dx/3.0) + child_coeffs[8]*(8.0*dy/3.0) + child_coeffs[1]*(-2.0*dxdz) + child_coeffs[2]*(8.0*dydz) + child_coeffs[3]*(-dx2+4.0*dy2-3.0*dz2) + child_coeffs[0]*dz*(-dx2+4.0*dy2-dz2);
    parent_coeffs[15] += child_coeffs[15] + child_coeffs[6]*(5.0*dz/3.0) + child_coeffs[7]*(5.0*dy/3.0) + child_coeffs[8]*(5.0*dx/3.0) + child_coeffs[1]*(5.0*dydz) + child_coeffs[2]*(5.0*dxdz) + child_coeffs[3]*(5.0*dxdy) + child_coeffs[0]*5.0*dx*dy*dz;
    parent_coeffs[25] += child_coeffs[25] + child_coeffs[4]*(-dx) + child_coeffs[5]*(dx/3.0) + child_coeffs[6]*(8.0*dy/3.0) + child_coeffs[7]*(-2.0*dz/3.0) + child_coeffs[1]*(-3.0*dx2+4.0*dy2-dz2) + child_coeffs[2]*(8.0*dxdy) + child_coeffs[3]*(-2.0*dxdz) + child_coeffs[0]*dx*(-dx2+4.0*dy2-dz2);
    parent_coeffs[26] += child_coeffs[26] + child_coeffs[4]*(-dx) + child_coeffs[5]*(-4.0*dx/3.0) + child_coeffs[6]*(-2.0*dy/3.0) + child_coeffs[7]*(8.0*dz/3.0) + child_coeffs[1]*(-3.0*dx2-dy2+4.0*dz2) + child_coeffs[2]*(-2.0*dxdy) + child_coeffs[3]*(8.0*dxdz) + child_coeffs[0]*dx*(-dx2-dy2+4.0*dz2);
    parent_coeffs[27] += child_coeffs[27] + child_coeffs[4]*(-dy/3.0) + child_coeffs[5]*(-4.0*dy/3.0) + child_coeffs[6]*(-2.0*dx/3.0) + child_coeffs[8]*(8.0*dz/3.0) + child_coeffs[1]*(-2.0*dxdy) + child_coeffs[2]*(-dx2-3.0*dy2+4.0*dz2) + child_coeffs[3]*(8.0*dydz) + child_coeffs[0]*dy*(-dx2-dy2+4.0*dz2);

    far_real_t dxdydz = dx*dy*dz;
    parent_coeffs[16] += child_coeffs[16] + 16.0*child_coeffs[9]*dx - 12.0*child_coeffs[12]*dy - 12.0*child_coeffs[13]*dz + child_coeffs[4]*(16.0*dx2-8.0*dy2-8.0*dz2) + child_coeffs[5]*(8.0*dx2-2.0*dy2-6.0*dz2) - 32.0*child_coeffs[6]*dxdy - 32.0*child_coeffs[7]*dxdz + 8.0*child_coeffs[8]*dydz + child_coeffs[1]*(32.0*dx3-48.0*dx*dy2-48.0*dx*dz2) + child_coeffs[2]*(-48.0*dx2*dy+12.0*dy3+12.0*dy*dz2) + child_coeffs[3]*(-48.0*dx2*dz+12.0*dy2*dz+12.0*dz3) + child_coeffs[0]*(35.0*dx4-30.0*dx2*r_sq+3.0*r_sq*r_sq);
    parent_coeffs[17] += child_coeffs[17] + 16.0*child_coeffs[10]*dy + 12.0*child_coeffs[11]*dz + 12.0*child_coeffs[13]*dz - 12.0*child_coeffs[25]*dx + child_coeffs[4]*(6.0*dx2-8.0*dy2+2.0*dz2) + child_coeffs[5]*(-2.0*dx2+8.0*dy2-6.0*dz2) - 32.0*child_coeffs[6]*dxdy + 8.0*child_coeffs[7]*dxdz - 32.0*child_coeffs[8]*dydz + child_coeffs[1]*(12.0*dx3-48.0*dx*dy2+12.0*dx*dz2) + child_coeffs[2]*(-48.0*dx2*dy+32.0*dy3-48.0*dy*dz2) + child_coeffs[3]*(12.0*dx2*dz-48.0*dy2*dz+12.0*dz3) + child_coeffs[0]*(35.0*dy4-30.0*dy2*r_sq+3.0*r_sq*r_sq);
    parent_coeffs[18] += child_coeffs[18] + 7.0*child_coeffs[10]*dx - 6.0*child_coeffs[12]*dx - 6.0*child_coeffs[15]*dz + 15.0*child_coeffs[25]*dy - 15.0*child_coeffs[4]*dxdy + 5.0*child_coeffs[5]*dxdy + child_coeffs[6]*(-15.0*dx2+20.0*dy2-5.0*dz2) - 10.0*child_coeffs[7]*dydz - 10.0*child_coeffs[8]*dxdz + child_coeffs[1]*(-45.0*dx2*dy+20.0*dy3-15.0*dy*dz2) + child_coeffs[2]*(-15.0*dx3+60.0*dx*dy2-15.0*dx*dz2) - 30.0*child_coeffs[3]*dxdydz + child_coeffs[0]*(35.0*dx*dy3-15.0*dx*dy*r_sq);
    parent_coeffs[19] += child_coeffs[19] + 7.0*child_coeffs[9]*dy + 15.0*child_coeffs[12]*dx - 6.0*child_coeffs[15]*dz - 6.0*child_coeffs[25]*dy + 20.0*child_coeffs[4]*dxdy + 5.0*child_coeffs[5]*dxdy + child_coeffs[6]*(20.0*dx2-15.0*dy2-5.0*dz2) - 10.0*child_coeffs[7]*dydz - 10.0*child_coeffs[8]*dxdz + child_coeffs[1]*(60.0*dx2*dy-15.0*dy3-15.0*dy*dz2) + child_coeffs[2]*(20.0*dx3-45.0*dx*dy2-15.0*dx*dz2) - 30.0*child_coeffs[3]*dxdydz + child_coeffs[0]*(35.0*dx3*dy-15.0*dx*dy*r_sq);
    parent_coeffs[20] += child_coeffs[20] + 13.0*child_coeffs[9]*dz + 15.0*child_coeffs[13]*dx - 6.0*child_coeffs[15]*dy + 6.0*child_coeffs[25]*dz + 20.0*child_coeffs[4]*dxdz + 15.0*child_coeffs[5]*dxdz - 10.0*child_coeffs[6]*dydz + child_coeffs[7]*(20.0*dx2-5.0*dy2-15.0*dz2) - 10.0*child_coeffs[8]*dxdy + child_coeffs[1]*(60.0*dx2*dz-15.0*dy2*dz-15.0*dz3) - 30.0*child_coeffs[2]*dxdydz + child_coeffs[3]*(20.0*dx3-15.0*dx*dy2-45.0*dx*dz2) + child_coeffs[0]*(35.0*dx3*dz-15.0*dx*dz*r_sq);
    parent_coeffs[21] += child_coeffs[21] - 15.0*child_coeffs[11]*dy + 13.0*child_coeffs[10]*dz + 6.0*child_coeffs[12]*dz - 15.0*child_coeffs[13]*dy - 6.0*child_coeffs[15]*dx - 5.0*child_coeffs[4]*dydz + 15.0*child_coeffs[5]*dydz - 10.0*child_coeffs[6]*dxdz - 10.0*child_coeffs[7]*dxdy + child_coeffs[8]*(-5.0*dx2+20.0*dy2-15.0*dz2) - 30.0*child_coeffs[1]*dxdydz + child_coeffs[2]*(-15.0*dx2*dz+60.0*dy2*dz-15.0*dz3) + child_coeffs[3]*(-15.0*dx2*dy+20.0*dy3-45.0*dy*dz2) + child_coeffs[0]*(35.0*dy3*dz-15.0*dy*dz*r_sq);
    parent_coeffs[22] += child_coeffs[22] - 2.0*child_coeffs[9]*dx - 2.0*child_coeffs[10]*dy + 2.0*child_coeffs[11]*dz + 12.0*child_coeffs[12]*dy + 12.0*child_coeffs[25]*dx + child_coeffs[4]*(-8.0*dx2+9.0*dy2-dz2) + child_coeffs[5]*(dx2+dy2-2.0*dz2) + 36.0*child_coeffs[6]*dxdy - 4.0*child_coeffs[7]*dxdz - 4.0*child_coeffs[8]*dydz + child_coeffs[1]*(-16.0*dx3+54.0*dx*dy2-6.0*dx*dz2) + child_coeffs[2]*(54.0*dx2*dy-16.0*dy3-6.0*dy*dz2) + child_coeffs[3]*(-6.0*dx2*dz-6.0*dy2*dz+4.0*dz3) + child_coeffs[0]*(35.0*dx2*dy2-5.0*(dx2+dy2)*r_sq+r_sq*r_sq);
    parent_coeffs[23] += child_coeffs[23] + 12.0*child_coeffs[15]*dx + 2.0*child_coeffs[10]*dz + 2.0*child_coeffs[11]*dy + 9.0*child_coeffs[12]*dz + 9.0*child_coeffs[13]*dy + 10.0*child_coeffs[4]*dydz + 5.0*child_coeffs[5]*dydz + 20.0*child_coeffs[6]*dxdz + 20.0*child_coeffs[7]*dxdy + child_coeffs[8]*(10.0*dx2-5.0*dy2-5.0*dz2) + 60.0*child_coeffs[1]*dxdydz + child_coeffs[2]*(30.0*dx2*dz-15.0*dy2*dz-5.0*dz3) + child_coeffs[3]*(30.0*dx2*dy-5.0*dy3-15.0*dy*dz2) + child_coeffs[0]*(35.0*dx2*dy*dz-5.0*dy*dz*r_sq);
    parent_coeffs[24] += child_coeffs[24] + 2.0*child_coeffs[9]*dz - 7.0*child_coeffs[11]*dx + 12.0*child_coeffs[15]*dy - 9.0*child_coeffs[13]*dx + 9.0*child_coeffs[25]*dz - 5.0*child_coeffs[4]*dxdz + 5.0*child_coeffs[5]*dxdz + 20.0*child_coeffs[6]*dydz + child_coeffs[7]*(-5.0*dx2+10.0*dy2-5.0*dz2) + 20.0*child_coeffs[8]*dxdy + child_coeffs[1]*(-15.0*dx2*dz+30.0*dy2*dz-5.0*dz3) + 60.0*child_coeffs[2]*dxdydz + child_coeffs[3]*(-5.0*dx3+30.0*dx*dy2-15.0*dx*dz2) + child_coeffs[0]*(35.0*dx*dy2*dz-5.0*dx*dz*r_sq);

}

__global__ void fmm_m2m_translate_parent_side(
    double* __restrict__ multipoles_dbl,
    const UnifiedNode* __restrict__ nodes,
    const int* __restrict__ children_compact,
    const int* __restrict__ level_list,
    const int* __restrict__ level_start,
    const int* __restrict__ level_count,
    int parent_level,
    int max_nodes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lvl_count = level_count[parent_level];
    if (tid >= lvl_count) return;

    int lvl_start = level_start[parent_level];
    int parent_idx = level_list[lvl_start + tid];
    if (parent_idx < 0 || parent_idx >= max_nodes) return;

    const UnifiedNode& parent = nodes[parent_idx];
    if (parent.childBase == UNIFIED_NODE_LEAF_CHILDBASE) return;

    double parent_coeffs[fmm::FMM_ACTIVE_COEFFS] = {0};
    far_real_t parent_cx = static_cast<far_real_t>(parent.center.x);
    far_real_t parent_cy = static_cast<far_real_t>(parent.center.y);
    far_real_t parent_cz = static_cast<far_real_t>(parent.center.z);
    float parent_R_max = 0.0f;

    for (int c = 0; c < 8; c++) {
        int child_idx = get_child_idx<3>(nodes, children_compact, parent_idx, c);
        if (child_idx < 0) continue;

        const UnifiedNode& child = nodes[child_idx];
        far_real_t dx = static_cast<far_real_t>(child.center.x) - parent_cx;
        far_real_t dy = static_cast<far_real_t>(child.center.y) - parent_cy;
        far_real_t dz = static_cast<far_real_t>(child.center.z) - parent_cz;
        far_real_t dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;

        double child_coeffs[fmm::FMM_ACTIVE_COEFFS];
        for (int i = 0; i < fmm::FMM_ACTIVE_COEFFS; i++)
            child_coeffs[i] = multipoles_dbl[child_idx * fmm::FMM_ACTIVE_COEFFS + i];

        translate_multipole_p4_unified(parent_coeffs, child_coeffs, dx, dy, dz);

        float child_dist = FAR_SQRT(dx2 + dy2 + dz2);
        float candidate_R_max = child_dist + nodes[child_idx].radius;
        if (candidate_R_max > parent_R_max) parent_R_max = candidate_R_max;
    }

    for (int i = 1; i < fmm::FMM_ACTIVE_COEFFS; i++) {
        // This parent owns its complete new result; never read old scratch.
        multipoles_dbl[parent_idx * fmm::FMM_ACTIVE_COEFFS + i] = parent_coeffs[i];
    }
    if (parent_R_max > 0.0f) {
        const_cast<UnifiedNode&>(parent).radius = parent_R_max;
    }
}

// =============================================================================
// Kernel: Clear R_max for all nodes
// =============================================================================

__global__ void fmm_clear_R_max_unified(
    UnifiedNode* __restrict__ nodes,
    int num_nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes &&
        (nodes[idx].flags & UNIFIED_NODE_FLAG_VALID))
        nodes[idx].radius = 0.0f;
}

// =============================================================================
// Kernel: Pack Double-Precision Multipoles to Float Format
// =============================================================================

__global__ void fmm_pack_multipoles_unified(
    const double* __restrict__ multipoles_dbl,
    fmm::MultipoleExpansion* __restrict__ multipoles,
    const UnifiedNode* __restrict__ nodes,
    int num_nodes)
{
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx >= num_nodes ||
        !(nodes[node_idx].flags & UNIFIED_NODE_FLAG_VALID)) return;
    fmm::MultipoleExpansion& m = multipoles[node_idx];
    for (int i = 0; i < fmm::FMM_ACTIVE_COEFFS; i++)
        m.coeffs[i] = static_cast<far_real_t>(multipoles_dbl[node_idx * fmm::FMM_ACTIVE_COEFFS + i]);
    for (int i = fmm::FMM_ACTIVE_COEFFS; i < fmm::FMM_MAX_COEFFS; i++)
        m.coeffs[i] = FAR_REAL(0.0);
}

// =============================================================================
// Host Function: Compute All Multipoles Unified (P2M + M2M parent-side)
// =============================================================================

void fmm_compute_all_multipoles_unified(
    SimulationState& state,
    const SimConfig& config,
    fmm::FMMHostData& host_data,
    fmm::FMMDeviceData& device_data,
    cudaStream_t stream)
{
    const int BLOCK = MPU_BLOCK_SIZE;
    int N = state.particles.num_particles;
    int num_nodes = device_data.num_nodes;
    if (N <= 0 || num_nodes <= 0) return;

    const far_real_t* d_pos_x = state.particles.pos_x;
    const far_real_t* d_pos_y = state.particles.pos_y;
    const far_real_t* d_pos_z = state.particles.pos_z;
    const float* d_mass = state.particles.mass;
    const int* d_sorted_indices = host_data.sorted_indices;

    double* d_multipoles_dbl = device_data.multipoles_dbl;
    if (!d_multipoles_dbl)
        throw std::logic_error("persistent P2M/M2M scratch is not allocated");
    // P2M assigns all 28 leaf coefficients. The mass pass assigns internal
    // coefficient zero; M2M assigns coefficients 1..27 in bottom-up order.
    // Invalid slots are never consumed, so no capacity-wide clear is needed.

    // Step 0: Clear R_max
    {
        int grid = (num_nodes + BLOCK - 1) / BLOCK;
        fmm_clear_R_max_unified<<<grid, BLOCK, 0, stream>>>(device_data.unified_nodes, num_nodes);
        CUDA_CHECK(cudaGetLastError());
        if (getenv("BF_FMM_VALIDATE_PHASE_SYNC")) {
            fprintf(stderr, "[BF-FMM validate] P2M clear radius\n");
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    // Step 1: P2M
    {
        int grid = (num_nodes + BLOCK - 1) / BLOCK;
        fmm_p2m_leaf_multipoles_unified<<<grid, BLOCK, 0, stream>>>(
            d_pos_x, d_pos_y, d_pos_z, d_mass, d_sorted_indices,
            host_data.node_range, d_multipoles_dbl,
            device_data.unified_nodes, num_nodes);
        CUDA_CHECK(cudaGetLastError());
        if (getenv("BF_FMM_VALIDATE_PHASE_SYNC")) {
            fprintf(stderr, "[BF-FMM validate] P2M leaves\n");
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    // Step 2: Mass-Weighted COM (bottom-up)
    {
        for (int level = device_data.max_level - 1; level >= 0; level--) {
            const int grid = (num_nodes + BLOCK - 1) / BLOCK;
            fmm_mass_weighted_com_unified<<<grid, BLOCK, 0, stream>>>(
                d_multipoles_dbl, device_data.unified_nodes,
                device_data.children_compact,
                device_data.level_list, device_data.level_start,
                device_data.level_count, level, num_nodes);
            CUDA_CHECK(cudaGetLastError());
            if (getenv("BF_FMM_VALIDATE_PHASE_SYNC")) {
                fprintf(stderr, "[BF-FMM validate] COM level %d\n", level);
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
        }
    }

    // Step 3: M2M parent-side (bottom-up)
    {
        for (int level = device_data.max_level - 1; level >= 0; level--) {
            const int grid = (num_nodes + BLOCK - 1) / BLOCK;
            fmm_m2m_translate_parent_side<<<grid, BLOCK, 0, stream>>>(
                d_multipoles_dbl, device_data.unified_nodes,
                device_data.children_compact,
                device_data.level_list, device_data.level_start,
                device_data.level_count, level, num_nodes);
            CUDA_CHECK(cudaGetLastError());
            if (getenv("BF_FMM_VALIDATE_PHASE_SYNC")) {
                fprintf(stderr, "[BF-FMM validate] M2M level %d\n", level);
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
        }
    }

    // Step 4: Pack
    {
        const int PACK_BLOCK = 128;
        int grid = (num_nodes + PACK_BLOCK - 1) / PACK_BLOCK;
        fmm_pack_multipoles_unified<<<grid, PACK_BLOCK, 0, stream>>>(
            d_multipoles_dbl, device_data.multipoles,
            device_data.unified_nodes, num_nodes);
        CUDA_CHECK(cudaGetLastError());
        if (getenv("BF_FMM_VALIDATE_PHASE_SYNC")) {
            fprintf(stderr, "[BF-FMM validate] multipole pack\n");
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
}

} // namespace body_fitted_fmm
