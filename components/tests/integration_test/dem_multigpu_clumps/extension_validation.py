#!/usr/bin/env python3
"""Separate DEM clump restart, shear-image and constrained-depth regressions.

The original validation fixtures and their successful evidence remain unchanged.
All preparation/readback/oracle commands are CPU-only. Solver invocation requires
the explicit run command and existing binaries.
"""
from __future__ import annotations
import argparse
import copy
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import re
import shutil
import struct

import validate as base

KIN_ATOL, KIN_RTOL = 2e-8, 2e-6
WRENCH_ATOL, WRENCH_RTOL = 1e-12, 2e-5
CONTACT = struct.Struct('<iiBB2x9f')  # padding and local-index cache excluded semantically
FIXTURES = ('restart-contact', 'restart-gravity', 'shear-contact', 'shear-gravity',
            'adaptive-occupancy', 'adaptive-too-coarse', 'adaptive-clump-conflict',
            'adaptive-growth', 'shear-adaptive-contact', 'adaptive-restart', 'shear-near-gravity')


def read_record(path):
    """Strict unified-v3 DEM input/output/checkpoint reader, including histories."""
    blob = path.read_bytes()
    header = base.HEADER.unpack_from(blob)
    magic, version, hlen, n, step, time, dt, nf, physics, kind, stride, contacts, material_hash, clump_hash = header
    assert (magic, version, hlen, physics) == (b'GRANIT\0\0', 3, 72, 1), path
    assert n > 0 and kind in (1, 2, 3) and contacts <= n*stride
    assert all(math.isfinite(x) for x in (time, dt)) and step >= 0
    tier1 = 72+16*nf
    tier2 = tier1+52*n
    coords = struct.unpack_from(f'<{6*n}d', blob, tier1)
    gids = struct.unpack_from(f'<{n}i', blob, tier1+48*n)
    assert set(gids) == set(range(n)), 'missing, duplicate or noncanonical IDs'
    assert all(math.isfinite(x) for x in coords)
    fields, spans = {}, []
    for index in range(nf):
        fid, dtype, count, reserved, offset = base.DESC.unpack_from(blob, 72+16*index)
        assert fid not in fields and dtype in base.TYPES and count > 0 and reserved == 0
        schema = {6: (2, 1), 0x10: (1, 3), 0x11: (0, 1), 0x12: (0, 1),
            0x13: (0, 1), 0x14: (0, 3), 0x15: (0, 3), 0x16: (1, 4),
            0x17: (2, 1), 0x18: (2, 1), 0x19: (0, 3),
            0x1a: (6, 1), 0x1b: (6, 1), 0x1c: (6, 1), 0x1d: (4, 1)}
        if fid in schema:
            assert (dtype, count) == schema[fid], (fid, dtype, count)
        fmt = base.TYPES[dtype]
        values = struct.unpack_from(f'<{n*count}{fmt}', blob, tier2+offset)
        assert all(math.isfinite(x) for x in values), (path, fid)
        fields[fid] = {gids[i]: tuple(values[a*n+i] for a in range(count)) for i in range(n)}
        spans.append((offset, offset+struct.calcsize(fmt)*n*count))
    end = 0
    for low, high in sorted(spans):
        assert low == end, 'overlapping or gapped field payload'
        end = high
    end += tier2
    required = {6, 0x11, 0x18}
    if kind in (2, 3):
        required |= {0x10, 0x12, 0x13, 0x14, 0x15, 0x1d}
    if kind == 3:
        required |= {0x16, 0x17, 0x19}
    assert required <= fields.keys(), (path, required-fields.keys())
    assert all(row[0] > 0 for row in fields[0x11].values())
    if kind in (2, 3):
        assert all(row[0] > 0 for row in fields[0x12].values())
        assert all(row == (0,) for row in fields[0x1d].values()), 'unexpected nonlive fixture particle'
    history = {}
    if kind == 3 and (contacts or len(blob) > end):
        assert stride > 0 and len(blob) == end+n*stride*CONTACT.size
        for i, gid in enumerate(gids):
            count = fields[0x17][gid][0]
            assert 0 <= count <= stride
            for slot in range(stride):
                partner, cached_local, active, seen, *spring = CONTACT.unpack_from(blob, end+(i*stride+slot)*CONTACT.size)
                assert active in (0, 1) and seen in (0, 1)
                if not active:
                    continue
                assert slot < count and partner in set(gids) and partner != gid
                assert all(math.isfinite(x) for x in spring)
                assert not (fields[0x18][gid][0] > 0 and fields[0x18][gid] == fields[0x18][partner]), 'internal clump contact history'
                key = (gid, partner)
                assert key not in history, 'duplicate active directed history'
                history[key] = (bool(active), bool(seen), *spring)
        assert len(history) <= contacts, 'active history exceeds declared contact count'
    else:
        assert len(blob) == end, 'trailing bytes or unexpected contact blob'
    return {'path': str(path), 'n': n, 'step': step, 'time': time, 'dt': dt,
        'kind': kind, 'stride': stride, 'total_contacts': contacts,
        'material_fingerprint': material_hash, 'clump_fingerprint': clump_hash,
        'rows': {gids[i]: tuple(coords[a*n+i] for a in range(6)) for i in range(n)},
        'fields': fields, 'histories': history}


def semantic_history_compare(reference, candidate):
    a, b = reference['histories'], candidate['histories']
    keys_equal = a.keys() == b.keys()
    differences = [list(key) for key in a.keys() & b.keys() if a[key] != b[key]]
    return {'pass': keys_equal and not differences, 'exact': True,
        'reference_active_directed_contacts': len(a), 'candidate_active_directed_contacts': len(b),
        'keys_equal': keys_equal, 'different_values': differences,
        'nonzero_spring_contacts': sum(any(x != 0 for x in row[2:]) for row in a.values()),
        'ignored': ['partner_local_id cache', 'two ABI padding bytes', 'inactive unused slots']}


def gate(a, b, atol=KIN_ATOL, rtol=KIN_RTOL):
    assert a.keys() == b.keys()
    maximum = max((abs(value-b[key][i])/(atol+rtol*abs(value))
                   for key, row in a.items() for i, value in enumerate(row)), default=0.)
    return {'pass': maximum <= 1., 'max_gate_fraction': maximum, 'atol': atol, 'rtol': rtol}


def compare_records(reference, candidate, exact_history=True):
    assert reference['step'] == candidate['step']
    assert abs(reference['time']-candidate['time']) <= 1e-12
    assert abs(reference['dt']-candidate['dt']) <= 1e-15
    checks = {'member_kinematics': gate(reference['rows'], candidate['rows'])}
    for fid in (6, 0x11, 0x12, 0x13, 0x18, 0x1a, 0x1b, 0x1c, 0x1d):
        if fid in reference['fields'] or fid in candidate['fields']:
            checks[f'exact_{fid:04x}'] = {'pass': reference['fields'].get(fid) == candidate['fields'].get(fid)}
    for fid in (0x10, 0x16, 0x19):
        if fid in reference['fields'] or fid in candidate['fields']:
            assert fid in reference['fields'] and fid in candidate['fields']
            checks[f'field_{fid:04x}'] = gate(reference['fields'][fid], candidate['fields'][fid])
    for fid in (0x14, 0x15):
        if fid in reference['fields'] or fid in candidate['fields']:
            assert fid in reference['fields'] and fid in candidate['fields']
            checks[f'field_{fid:04x}'] = gate(reference['fields'][fid], candidate['fields'][fid], 2e-7, 2e-5)
    if exact_history:
        checks['semantic_contact_history'] = semantic_history_compare(reference, candidate)
    return {'pass': all(row['pass'] for row in checks.values()), 'checks': checks}


def shear_phase(config, time):
    lo, hi = config['domain']['box_min'], config['domain']['box_max']
    length = [hi[a]-lo[a] for a in range(3)]
    shear = config['shear_box']
    speed = shear.get('shear_parameter', 1.5)*shear['omega']*length[0]
    full = speed*(time-shear.get('time_origin', 0.))
    reduced = full-length[1]*math.floor(full/length[1]+.5)
    return length, reduced, speed


def image_position(point, nx, ny, config, time):
    length, shift, _ = shear_phase(config, time)
    y = point[1]
    if nx:
        low = config['domain']['box_min'][1]
        y = y-nx*shift
        y -= length[1]*math.floor((y-low)/length[1])
    return (point[0]+nx*length[0], y+ny*length[1], point[2])


def nearest_image(target, source, config, time, reach=2):
    candidates = []
    for nx, ny in itertools.product(range(-reach, reach+1), repeat=2):
        point = image_position(source, nx, ny, config, time)
        candidates.append((math.dist(target, point), nx, ny, point))
    return min(candidates, key=lambda row: row[0])


def finite_image_direct(record, config):
    """Independent finite-image Newton forces with nearest-copy self exclusion.

    Search for the internal rigid copy extends beyond the configured physical
    image window. Nonlocal replicas with the same positive ID remain sources.
    This deliberately does not reproduce the production diameter near-filter.
    """
    fields, rows, time = record['fields'], record['rows'], record['time']
    options = config['shear_box']['gravity_images']
    lx, ly = options.get('layers_x', 1), options.get('layers_y', 1)
    cutoff = options.get('cutoff_radius', 0.) or 0.
    G = config['accelerations']['self_gravity']['unified_fmm']['gravitational_constant']
    result, internal_copies = {}, {}
    for gid, state in rows.items():
        target, force = state[:3], [0., 0., 0.]
        cid = fields[0x18][gid][0]
        for source, source_state in rows.items():
            same_body = cid > 0 and cid == fields[0x18][source][0]
            internal = nearest_image(target, source_state[:3], config, time, max(lx, ly)+2)[1:3] if same_body else None
            if internal is not None:
                internal_copies[(gid, source)] = internal
            for nx, ny in itertools.product(range(-lx, lx+1), range(-ly, ly+1)):
                if (same_body and (nx, ny) == internal) or (not same_body and source == gid and nx == ny == 0):
                    continue
                point = image_position(source_state[:3], nx, ny, config, time)
                delta = [point[a]-target[a] for a in range(3)]
                radius = base.norm(delta)
                if cutoff and radius > cutoff:
                    continue
                assert radius > 0., ('unexcluded coincident gravity source', gid, source, nx, ny)
                factor = G*fields[0x12][gid][0]*fields[0x12][source][0]/radius**3
                for a in range(3):
                    force[a] += factor*delta[a]
        result[gid] = tuple(force)
    return result, internal_copies


def global_leaf_counts(points, root_min, root_size, depth):
    """Count canonical physical rows globally; do not combine local maxima."""
    cells = 1 << depth
    counts = Counter()
    for point in points:
        cell = tuple(math.floor((point[a]-root_min[a])*cells/root_size) for a in range(3))
        assert all(0 <= value < cells for value in cell), ('particle outside recorded common root', point, cell)
        counts[cell] += 1
    return counts


def admissible_depths(points, root_min, root_size, diameter, min_depth, max_depth):
    rows = []
    for depth in range(min_depth, max_depth+1):
        maximum = max(global_leaf_counts(points, root_min, root_size, depth).values(), default=0)
        width = root_size/(1 << depth)
        rows.append({'depth': depth, 'global_max_leaf_count': maximum, 'leaf_width': width,
                     'diameter': diameter, 'pass': maximum <= 128 and diameter < width*(1-1e-6)})
    return rows


def rotate(q, vector):
    xyz = q[1:]
    twice = tuple(2*x for x in base.cross(xyz, vector))
    second = base.cross(xyz, twice)
    return tuple(vector[a]+q[0]*twice[a]+second[a] for a in range(3))


def validate_rigid_frame(record, bodies, config, initial_distances=None):
    shear = config.get('shear_box', {}).get('enabled', False)
    groups = {}
    for gid, (cid,) in record['fields'][0x18].items():
        if cid:
            groups.setdefault(cid, []).append(gid)
    assert set(groups) == set(bodies['rows'])
    worst_distance = worst_velocity = worst_offset = 0.
    for cid, group in groups.items():
        body = bodies['rows'][cid]
        assert abs(base.norm(body[6:10])-1.) <= 1e-10
        world_omega = rotate(body[6:10], body[10:13])
        for gid in group:
            state = record['rows'][gid]
            point = nearest_image(body[:3], state[:3], config, record['time'])[3] if shear else state[:3]
            arm = tuple(point[a]-body[a] for a in range(3))
            spin = base.cross(world_omega, arm)
            vy_correction = 0.
            if shear:
                image = nearest_image(body[:3], state[:3], config, record['time'])
                vy_correction = -image[1]*shear_phase(config, record['time'])[2]
            differences = [state[a+3]+(vy_correction if a == 1 else 0.)-body[a+3]-spin[a] for a in range(3)]
            worst_velocity = max(worst_velocity, *(abs(x) for x in differences))
            if 0x19 in record['fields']:
                expected = rotate(body[6:10], record['fields'][0x19][gid])
                worst_offset = max(worst_offset, *(abs(arm[a]-expected[a]) for a in range(3)))
        for a, b in itertools.combinations(group, 2):
            point = nearest_image(record['rows'][a][:3], record['rows'][b][:3], config, record['time'])[3] if shear else record['rows'][b][:3]
            distance = math.dist(record['rows'][a][:3], point)
            if initial_distances is not None:
                worst_distance = max(worst_distance, abs(distance-initial_distances[(a, b)]))
    return {'pass': worst_distance <= 2e-8 and worst_velocity <= 2e-8 and worst_offset <= 2e-8,
            'max_rigid_distance_error': worst_distance, 'max_rigid_velocity_error': worst_velocity,
            'max_body_offset_error': worst_offset}


def hill_body_wrench(body, config):
    """Analytic continuum Hill wrench, including each sphere's intrinsic inertia."""
    shear = config['shear_box']
    omega, q = shear['omega'], shear.get('shear_parameter', 1.5)
    nu = shear.get('vertical_frequency', omega)
    mass, quaternion = body[13], body[6:10]
    force = (mass*(2*omega*body[4]+2*q*omega*omega*body[0]),
             -2*mass*omega*body[3], -mass*nu*nu*body[2])
    a, b, c, xy, xz, yz = body[14:20]
    inertia = ((a, xy, xz), (xy, b, yz), (xz, yz, c))
    basis = [rotate(quaternion, tuple(1. if i == j else 0. for i in range(3))) for j in range(3)]
    world = [[sum(basis[i][r]*inertia[i][j]*basis[j][s] for i in range(3) for j in range(3))
              for s in range(3)] for r in range(3)]
    half_trace = .5*sum(world[i][i] for i in range(3))
    covariance = [[(half_trace if i == j else 0.)-world[i][j] for j in range(3)] for i in range(3)]
    wx, wy, wz = rotate(quaternion, body[10:13])
    gradient = ((2*q*omega*omega+2*omega*wz, 0., -2*omega*wx),
                (0., 2*omega*wz, -2*omega*wy), (0., 0., -nu*nu))
    torque = tuple(sum(gradient[k][j]*covariance[l][j]-gradient[l][j]*covariance[k][j]
                       for j in range(3)) for l, k in ((1, 2), (2, 0), (0, 1)))
    return force, torque


def pure_gravity_sidecar(record):
    path = Path(record['path']+'.gravity_force')
    blob = path.read_bytes()
    magic, version, n, step, time, dt = struct.unpack_from('<8sIiqdd', blob)
    assert magic == b'DEMOUT\0\0' and version == 1 and n == record['n']
    assert step == record['step'] and abs(time-record['time']) <= 1e-12 and len(blob) == 40+24*n
    values = struct.unpack_from(f'<{3*n}d', blob, 40)
    assert all(math.isfinite(x) for x in values)
    return {gid: tuple(values[a*n+gid] for a in range(3)) for gid in range(n)}


def vector_accuracy(reference, candidate):
    assert reference.keys() == candidate.keys()
    reference_norm = math.sqrt(sum(base.norm(v)**2 for v in reference.values()))
    errors = [base.norm([candidate[key][a]-row[a] for a in range(3)]) for key, row in reference.items()]
    absolute_l2 = base.norm(errors)
    rms = reference_norm/math.sqrt(len(reference))
    relative = absolute_l2/max(reference_norm, 1e-300)
    maximum = max(errors, default=0.)/max(rms, 1e-300)
    return {'pass': relative <= 1e-4 and maximum <= 5e-4,
        'relative_l2_error': relative, 'max_error_over_reference_rms': maximum,
        'absolute_l2_error': absolute_l2, 'absolute_max_error': max(errors, default=0.),
        'reference_l2_norm': reference_norm, 'relative_l2_gate': 1e-4, 'maximum_rms_gate': 5e-4}


def shear_gravity_accuracy(record, bodies, config):
    reference, excluded = finite_image_direct(record, config)
    result = {'member_force': vector_accuracy(reference, pure_gravity_sidecar(record)),
        'internal_copy_noncentral_pairs': sum(nx != 0 or ny != 0 for nx, ny in excluded.values())}
    expected_force, expected_torque = {}, {}
    conditioning = {}
    for cid, body in bodies['rows'].items():
        force, torque, sum_torque = [0.]*3, [0.]*3, 0.
        for gid, (member_cid,) in record['fields'][0x18].items():
            if cid != member_cid:
                continue
            point = nearest_image(body[:3], record['rows'][gid][:3], config, record['time'])[3]
            arm = tuple(point[a]-body[a] for a in range(3))
            moment = base.cross(arm, reference[gid])
            sum_torque += base.norm(moment)
            for a in range(3):
                force[a] += reference[gid][a]; torque[a] += moment[a]
        expected_force[cid], expected_torque[cid] = tuple(force), tuple(torque)
        conditioning[cid] = {'sum_member_torque_norms': sum_torque,
            'net_torque_norm': base.norm(torque),
            'cancellation_condition': sum_torque/base.norm(torque) if base.norm(torque) else None}
    actual_force, actual_torque = {}, {}
    for cid, body in bodies['rows'].items():
        hill_force, hill_torque = hill_body_wrench(body, config)
        actual_force[cid] = tuple(body[39+a]-hill_force[a] for a in range(3))
        actual_torque[cid] = tuple(body[42+a]-hill_torque[a] for a in range(3))
    result['body_force'] = vector_accuracy(expected_force, actual_force)
    result['body_torque'] = vector_accuracy(expected_torque, actual_torque)
    result['torque_conditioning'] = conditioning
    result['pass'] = all(result[key]['pass'] for key in ('member_force', 'body_force', 'body_torque'))
    return result


def adaptive_diagnostics(path):
    result = []
    for line in path.read_text().splitlines():
        if '[clumps-adaptive]' not in line and '[clumps-depth]' not in line:
            continue
        row = {}
        for key, value in re.findall(r'(\w+)=([^\s]+)', line):
            try:
                row[key] = int(value) if re.fullmatch(r'[+-]?\d+', value) else float(value)
            except ValueError:
                row[key] = value
        aliases = {'selected_base': 'depth', 'actual_depth': 'effective_depth',
                   'global_max_leaf': 'global_max_leaf_count', 'max_diameter': 'max_clump_diameter'}
        for source, target in aliases.items():
            if source in row:
                row[target] = row[source]
        result.append(row)
    return result


def census_points(record, config):
    if config.get('shear_box', {}).get('enabled', False):
        images = config['shear_box']['gravity_images']
        return [image_position(state[:3], nx, ny, config, record['time'])
                for state in record['rows'].values()
                for nx, ny in itertools.product(range(-images['layers_x'], images['layers_x']+1),
                                                range(-images['layers_y'], images['layers_y']+1))]
    return [state[:3] for state in record['rows'].values()]


def audit_adaptive_row(row, record, config, independent_diameter):
    root_min = [row[f'root_{axis}'] for axis in 'xyz']
    depth, size = row.get('effective_depth', row['depth']), row['root_size']
    points = census_points(record, config)
    counts = global_leaf_counts(points, root_min, size, depth)
    maximum = max(counts.values(), default=0)
    width = size/(1 << depth)
    checks = {'global_occupancy': maximum <= 128,
        'reported_occupancy_matches': maximum == row['global_max_leaf_count'],
        'full_clump_diameter': independent_diameter < width*(1-1e-6),
        'reported_diameter_not_understated': row['max_clump_diameter']+1e-8 >= independent_diameter,
        'reported_leaf_width_matches': abs(width-row['leaf_width']) <= 1e-10*max(1., width)}
    if 'source_particles' in row:
        checks['logical_source_count'] = row['source_particles'] == len(points)
    if 'physical_particles' in row:
        checks['physical_particle_count'] = row['physical_particles'] == record['n']
    return {'pass': all(checks.values()), 'checks': checks, 'independent_global_max_leaf_count': maximum,
        'effective_depth': depth, 'independent_diameter': independent_diameter, 'leaf_width': width}


def initial_geometry(source, config):
    groups = {}
    for gid, (cid,) in source['fields'][0x18].items():
        if cid:
            groups.setdefault(cid, []).append(gid)
    distances, diameter = {}, 0.
    shear = config.get('shear_box', {}).get('enabled', False)
    for group in groups.values():
        reference = source['rows'][group[0]][:3]
        points = {gid: (nearest_image(reference, source['rows'][gid][:3], config, source['time'])[3]
                        if shear else source['rows'][gid][:3]) for gid in group}
        # Uniform material fixture: mass is proportional to radius cubed.
        weights = {gid: source['fields'][0x11][gid][0]**3 for gid in group}
        total = sum(weights.values())
        com = tuple(sum(weights[gid]*points[gid][a] for gid in group)/total for a in range(3))
        diameter = max(diameter, *(2*(math.dist(points[gid], com)+source['fields'][0x11][gid][0]) for gid in group))
        for a, b in itertools.combinations(group, 2):
            distances[(a, b)] = math.dist(points[a], points[b])
    return distances, diameter


def validate_run(work, require_seam=None, gravity_oracle=False):
    config = json.loads((work/'config.json').read_text())
    manifest = json.loads((work/'manifest.json').read_text())
    assert base.digest(work/'input.bin') == manifest['input_sha256']
    assert base.digest(work/'config.json') == manifest['config_sha256']
    source = read_record(work/'input.bin')
    distances, diameter = initial_geometry(source, config)
    files = sorted((work/'output').glob('output_step_*.bin'))
    assert files, f'no output in {work}'
    frames, checks = {}, []
    winding_changed, nonzero_torque = False, False
    first_winding = None
    for path in files:
        record = read_record(path)
        assert record['kind'] == 2 and record['n'] == source['n']
        assert record['fields'][0x18] == source['fields'][0x18]
        assert abs(record['dt']-manifest['dt']) <= 1e-15
        assert record['step'] not in frames
        frames[record['step']] = record
        body_path = path.parent/f"clump_states_step_{record['step']:010d}.bin"
        if record['step'] == 0 and not body_path.exists():
            continue
        bodies = base.read_clumps(body_path)
        assert bodies['step'] == record['step'] and abs(bodies['time']-record['time']) <= 1e-12
        checks.append(validate_rigid_frame(record, bodies, config, distances))
        nonzero_torque |= any(base.norm(row[42:45]) > 1e-8 for row in bodies['rows'].values())
        if config.get('shear_box', {}).get('enabled', False):
            assert {0x1a, 0x1b, 0x1c} <= record['fields'].keys(), 'missing winding fields'
            windings = tuple((gid, record['fields'][0x1a][gid], record['fields'][0x1b][gid]) for gid in sorted(record['rows']))
            first_winding = first_winding or windings
            winding_changed |= windings != first_winding
    assert len(frames) >= 2 and max(frames) == manifest['steps']
    assert all(step+1 in frames for step in range(min(frames), max(frames)))
    result = {'pass': all(row['pass'] for row in checks), 'frames': len(frames),
        'first_step': min(frames), 'final_step': max(frames),
        'max_rigid_distance_error': max((r['max_rigid_distance_error'] for r in checks), default=0.),
        'max_rigid_velocity_error': max((r['max_rigid_velocity_error'] for r in checks), default=0.),
        'max_body_offset_error': max((r['max_body_offset_error'] for r in checks), default=0.),
        'independent_max_clump_diameter': diameter, 'winding_changed': winding_changed,
        'nonzero_body_torque': nonzero_torque}
    if require_seam is None:
        require_seam = manifest['fixture'] == 'shear-contact' and min(frames) == 0
    if require_seam:
        result['pass'] &= winding_changed and nonzero_torque
    diagnostic_rows = adaptive_diagnostics(work/'run.log') if (work/'run.log').exists() else []
    if config['accelerations']['self_gravity']['unified_fmm']['adaptive_depth']['enabled']:
        audits, unavailable = [], []
        for row in diagnostic_rows:
            # The diagnostic runs before the integrator increments its step;
            # match the physical force endpoint, including resumed endpoints.
            matches = [record for record in frames.values()
                       if 'time' in row and abs(record['time']-row['time']) <= 1e-12]
            if len(matches) != 1:
                unavailable.append(row)
                continue
            audits.append(audit_adaptive_row(row, matches[0], config, diameter))
        result['adaptive'] = {'pass': bool(audits) and not unavailable and all(x['pass'] for x in audits),
            'diagnostics': diagnostic_rows, 'independent_audits': audits, 'unmatched_evaluations': unavailable}
        result['pass'] &= result['adaptive']['pass']
    if gravity_oracle:
        record = frames[min(frames)]
        bodies = base.read_clumps(work/'output'/f"clump_states_step_{record['step']:010d}.bin")
        result['finite_image_gravity'] = shear_gravity_accuracy(record, bodies, config)
        result['pass'] &= result['finite_image_gravity']['pass']
    return result


def fixture_rows(name):
    if name.startswith('restart-'):
        yield from base.fixture_rows('integrated-contact' if name.endswith('gravity') else 'contact-migration', 0)
        return
    if name == 'shear-near-gravity':
        # An anisotropic, contact-free near pair gives a resolved gravity
        # torque while one actual body spans the radial seam at bootstrap.
        for cid, center in enumerate(((.9995, .0003, -.0003), (-.9885, .0009, .0008)), 1):
            spin = (0., 0., 3. if cid == 1 else -2.)
            for original in base.OFFSETS:
                off = tuple(original[a]*(.15, .11, .07)[a] for a in range(3))
                velocity = base.cross(spin, off)
                yield tuple(center[a]+off[a] for a in range(3)), velocity, spin, .001, cid
        return
    if name.startswith('shear-'):
        contact = name in ('shear-contact', 'shear-adaptive-contact')
        centers = ((.987, .011, -.003), (-.987 if contact else -.80, .011 if contact else .13, -.003))
        for cid, center in enumerate(centers, 1):
            spin = (0., 0., 3. if cid == 1 else -2.)
            velocity = (40., .2 if cid == 1 else -.2, 0.) if contact else (0., 0., 0.)
            for off in base.OFFSETS:
                rotational = base.cross(spin, off)
                yield tuple(center[a]+off[a] for a in range(3)), tuple(velocity[a]+rotational[a] for a in range(3)), spin, .01, cid
        return
    # Dense, nonoverlapping, tiny normal particles; occupancy is physical count,
    # not mass. The cluster exceeds128 at coarse depths even at negligible mass.
    for index in range(256):
        xyz = (index % 4, (index//4) % 4, index//16)
        yield tuple((.013, .017, .019)[a]+5e-5*(xyz[a]-(1.5 if a < 2 else 7.5)) for a in range(3)), (0.,)*3, (0.,)*3, 1e-6, 0
    for p in ((-1., -1., -1.), (1., 1., 1.)):
        yield p, ((100., 0., 0.) if name == 'adaptive-growth' and p[0] > 0 else (0.,)*3), (0.,)*3, 1e-5, 0
    scale = 20. if name == 'adaptive-clump-conflict' else .001
    for off in base.OFFSETS:
        yield tuple(-.3+scale*x for x in off), (0.,)*3, (0.,)*3, (.02 if scale > 1 else 1e-5), 1


def prepare(work, fixture, ranks=2, steps=40, adaptive=None):
    work.mkdir(parents=True, exist_ok=False)
    n, k, normal = base.write_input(work/'input.bin', fixture_rows(fixture))
    young = 1. if fixture.startswith('adaptive-') else 1e6
    base.dump(work/'materials.json', {'version': 1, 'materials': [{'id': 0, 'name': 'extension',
        'dem': {'density': 1000., 'youngs_modulus': young, 'poisson_ratio': .25}}],
        'dem_interactions': [{'materials': [0, 0], 'restitution': .8,
            'friction': {'static': .4, 'rolling': .03, 'twisting': .02},
            'shape_parameter': 1., 'cohesion': {'model': 'none'}}]})
    config = base.make_config(work, fixture, k, ranks, gravity=fixture != 'restart-contact', steps=steps)
    config['io']['checkpoint'] = {'enabled': True, 'interval': max(1, steps//2),
        'path': str(work/'checkpoints'), 'force_fresh': True}
    config['io']['output']['fields'] = ['gravity_force']
    if fixture.startswith('shear-'):
        config['domain'] = {'box_min': [-1.]*3, 'box_max': [1.]*3}
        config['shear_box'] = {'enabled': True, 'omega': .001, 'shear_parameter': 1.5,
            'vertical_frequency': .001, 'time_origin': 0., 'input_velocity': 'total',
            'gravity_images': {'layers_x': 1, 'layers_y': 1, 'cutoff_radius': 0.},
            'force_balance_warn_threshold': 1e-3}
        if fixture == 'shear-near-gravity':
            # The near pair remains in the P2P stencil without an occupancy
            # fallback; distant finite image interactions still use M2L.
            config['accelerations']['self_gravity']['unified_fmm']['tree_depth'] = 2
            config['multigpu']['partition_depth'] = 2
    if fixture.startswith('adaptive-') or fixture == 'shear-adaptive-contact' or adaptive:
        options = {'enabled': True, 'min_depth': 2, 'max_depth': 16}
        if fixture in ('shear-adaptive-contact', 'adaptive-restart', 'adaptive-growth'):
            options.update(trial_interval_evaluations=2, trial_warmup_evaluations=0,
                trial_samples=1, confirmation_trials=1)
        if fixture == 'shear-adaptive-contact':
            options.update(max_depth=6)
        if fixture == 'adaptive-too-coarse':
            options.update(min_depth=4, max_depth=4)
        if adaptive:
            options.update(adaptive)
        fmm = config['accelerations']['self_gravity']['unified_fmm']
        fmm['tree_depth'] = options['min_depth']
        fmm['adaptive_depth'] = options
        config['multigpu']['partition_depth'] = options['min_depth']
    if fixture == 'adaptive-growth':
        # The positive anchor moves fromx=1 to1.04, beyond this initial root.
        # Actual root changes remain an observed gate, not an input assumption.
        config['domain'] = {'box_min': [-1.001]*3, 'box_max': [1.001]*3}
    base.dump(work/'config.json', config)
    manifest = {'schema': 'granit-clump-extensions/v1', 'fixture': fixture,
        'ranks': ranks, 'particles': n, 'clumps': k, 'normals': normal,
        'steps': steps, 'dt': config['simulation']['time']['dt'],
        'input_sha256': base.digest(work/'input.bin'), 'config_sha256': base.digest(work/'config.json'),
        'material_sha256': base.digest(work/'materials.json')}
    base.dump(work/'manifest.json', manifest)
    return manifest


def latest_checkpoint(work):
    files = sorted((work/'checkpoints').glob('checkpoint_step_*.bin'))
    assert files, f'no checkpoint in {work}'
    return files[-1]


def verify_restart(reference, resumed, reference_step=None):
    checkpoint = (reference/'checkpoints'/f'checkpoint_step_{reference_step:010d}.bin'
                  if reference_step is not None else latest_checkpoint(reference))
    a, b = read_record(checkpoint), read_record(latest_checkpoint(resumed))
    result = compare_records(a, b)
    assert a['kind'] == b['kind'] == 3
    result['checkpoint_steps'] = [a['step'], b['step']]
    result['reference_sha256'] = base.digest(Path(a['path']))
    result['resumed_sha256'] = base.digest(Path(b['path']))
    ra = base.read_clumps(reference/'output'/f"clump_states_step_{a['step']:010d}.bin")
    rb = base.read_clumps(resumed/'output'/f"clump_states_step_{b['step']:010d}.bin")
    result['body_state'] = gate({k: tuple(v[:13]) for k, v in ra['rows'].items()}, {k: tuple(v[:13]) for k, v in rb['rows'].items()})
    result['body_wrench'] = gate({k: tuple(v[39:45]) for k, v in ra['rows'].items()}, {k: tuple(v[39:45]) for k, v in rb['rows'].items()}, WRENCH_ATOL, WRENCH_RTOL)
    result['pass'] &= result['body_state']['pass'] and result['body_wrench']['pass']
    return result


def update_config(work, mutate):
    config = json.loads((work/'config.json').read_text())
    mutate(config)
    base.dump(work/'config.json', config)
    manifest = json.loads((work/'manifest.json').read_text())
    manifest['config_sha256'] = base.digest(work/'config.json')
    base.dump(work/'manifest.json', manifest)
    return config


def compare_saved_runs(reference, candidate):
    checks = []
    for path in sorted((candidate/'output').glob('output_step_*.bin')):
        other = reference/'output'/path.name
        assert other.exists(), f'missing matching reference frame {other}'
        a, b = read_record(other), read_record(path)
        row = compare_records(a, b, exact_history=False)
        ac = reference/'output'/f"clump_states_step_{a['step']:010d}.bin"
        bc = candidate/'output'/ac.name
        if bc.exists():
            assert ac.exists()
            ra, rb = base.read_clumps(ac), base.read_clumps(bc)
            row['body_state'] = gate({k: tuple(v[:13]) for k, v in ra['rows'].items()}, {k: tuple(v[:13]) for k, v in rb['rows'].items()})
            row['body_wrench'] = gate({k: tuple(v[39:45]) for k, v in ra['rows'].items()}, {k: tuple(v[39:45]) for k, v in rb['rows'].items()}, WRENCH_ATOL, WRENCH_RTOL)
            row['pass'] &= row['body_state']['pass'] and row['body_wrench']['pass']
        checks.append(row)
    return {'pass': bool(checks) and all(x['pass'] for x in checks), 'frames': len(checks),
        'max_member_gate_fraction': max(x['checks']['member_kinematics']['max_gate_fraction'] for x in checks),
        'max_body_wrench_gate_fraction': max((x['body_wrench']['max_gate_fraction'] for x in checks if 'body_wrench' in x), default=0.)}


def prepare_sphere_causality_control(clump_run, work):
    """Clone accepted frame0 bytes as normal spheres, preserving source geometry."""
    work.mkdir(parents=True, exist_ok=False)
    frame = read_record(clump_run/'output'/'output_step_0000000000.bin')
    assert frame['step'] == 0, 'causality control requires the actual bootstrap frame'
    assert all(value == (0,) for value in frame['fields'][6].values()), 'this fixture control expects material0'
    original = json.loads((clump_run/'config.json').read_text())
    rows = ((frame['rows'][gid][:3], frame['rows'][gid][3:6],
             frame['fields'][0x10][gid], frame['fields'][0x11][gid][0], 0)
            for gid in sorted(frame['rows']))
    base.write_input(work/'input.bin', rows)
    shutil.copy2(original['material_database'], work/'materials.json')
    config = copy.deepcopy(original)
    config['material_database'] = str(work/'materials.json')
    config['clumps'] = {'instances': []}
    config['multigpu'].update(enabled=True, max_gpus=1)
    dt = config['simulation']['time']['dt']
    config['simulation']['time'].update(start=0., stop=dt, output_step=dt)
    config['simulation']['name'] = 'source-identical-unclumped-gravity-control'
    config['io'] = {'input': {'particle_file': str(work/'input.bin')},
        'output': {'directory': str(work/'output'), 'initial_state': True, 'fields': ['gravity_force']},
        'checkpoint': {'enabled': False}}
    base.dump(work/'config.json', config)
    manifest = {'schema': 'granit-clump-source-causality-control/v1',
        'original_clump_run': str(clump_run), 'source_frame_sha256': base.digest(Path(frame['path'])),
        'input_sha256': base.digest(work/'input.bin'), 'config_sha256': base.digest(work/'config.json'),
        'material_sha256': base.digest(work/'materials.json'), 'particles': frame['n'],
        'preserved': ['frame0 F64 positions', 'F64 velocities', 'F64 omega', 'F32 radii',
                      'canonical IDs', 'material bytes', 'finite-image model', 'FMM depth/root controls'],
        'changed': ['positive clump IDs become0', 'clump instances removed', 'one distributed GPU']}
    base.dump(work/'manifest.json', manifest)
    return manifest


def analyze_sphere_causality_control(clump_run, sphere_run):
    a = read_record(clump_run/'output'/'output_step_0000000000.bin')
    b = read_record(sphere_run/'output'/'output_step_0000000000.bin')
    ca, cb = [json.loads((p/'config.json').read_text()) for p in (clump_run, sphere_run)]
    assert ca['accelerations']['self_gravity'] == cb['accelerations']['self_gravity']
    assert ca['domain'] == cb['domain'] and ca['shear_box'] == cb['shear_box']
    assert ca['accelerations']['self_gravity']['unified_fmm']['softening_factor'] == 0.
    assert a['rows'] == b['rows'], 'control geometry or prescribed kinematics changed at bootstrap'
    for fid in (6, 0x10, 0x11, 0x12):
        assert a['fields'][fid] == b['fields'][fid], ('control field differs', fid)
    assert all(cid == (0,) for cid in b['fields'][0x18].values())
    assert ca['clumps']['instances'] and not cb['clumps']['instances']
    clump_force, sphere_force = pure_gravity_sidecar(a), pure_gravity_sidecar(b)
    exact, excluded = finite_image_direct(a, ca)
    G = ca['accelerations']['self_gravity']['unified_fmm']['gravitational_constant']
    corrected, internal = {}, {}
    for gid, state in a['rows'].items():
        own = a['fields'][0x18][gid][0]
        force = [0.]*3
        for other, source in a['rows'].items():
            if other == gid or own <= 0 or a['fields'][0x18][other][0] != own:
                continue
            nx, ny = excluded[(gid, other)]
            point = image_position(source[:3], nx, ny, ca, a['time'])
            delta = [point[i]-state[i] for i in range(3)]
            factor = G*a['fields'][0x12][gid][0]*a['fields'][0x12][other][0]/base.norm(delta)**3
            for i in range(3):force[i] += factor*delta[i]
        internal[gid] = tuple(force)
        corrected[gid] = tuple(sphere_force[gid][i]-force[i] for i in range(3))
    clump_residual = {gid: tuple(clump_force[gid][i]-exact[gid][i] for i in range(3)) for gid in exact}
    sphere_residual = {gid: tuple(corrected[gid][i]-exact[gid][i] for i in range(3)) for gid in exact}
    equivalence = vector_accuracy(clump_force, corrected)
    residual_match = vector_accuracy(clump_residual, sphere_residual)
    return {'schema': 'granit-clump-source-causality-analysis/v1',
        'pass': equivalence['pass'] and residual_match['pass'],
        'source_identical': True, 'same_finite_image_model_and_root_controls': True,
        'clump_vs_spheres_minus_exact_internal': equivalence,
        'far_error_pattern_equivalence': residual_match,
        'original_clump_vs_direct': vector_accuracy(exact, clump_force),
        'unclumped_corrected_vs_direct': vector_accuracy(exact, corrected),
        'internal_force_l2_norm': math.sqrt(sum(base.norm(row)**2 for row in internal.values())),
        'per_particle_corrected_forces': corrected,
        'interpretation': 'Matching corrected forces and residuals isolates the measured far error to the common unclumped FMM path; the original accuracy failure remains a failure.'}


class Campaign:
    def __init__(self, binary, work, ranks, reference_campaign=None):
        self.binary, self.work, self.ranks = binary.resolve(), work.resolve(), ranks
        self.reference_campaign = reference_campaign.resolve() if reference_campaign else None
        self.work.mkdir(parents=True, exist_ok=False)
        self.report = {'schema': 'granit-clump-extension-campaign/v1', 'pass': False,
            'binary': str(self.binary), 'binary_sha256': base.digest(self.binary),
            'checks': {}, 'commands': [], 'limitations': []}

    def save(self):
        base.dump(self.work/'report.json', self.report)

    def run(self, label, fixture, ranks, steps=40, resume=None, mutate=None):
        directory = self.work/label
        prepare(directory, fixture, ranks, steps)
        def apply(config):
            config['io']['checkpoint']['interval'] = 20
            if resume:
                config['io']['checkpoint'].update(resume_from=str(resume), force_fresh=False,
                                                   rebootstrap_forces=False)
            if mutate:
                mutate(config)
        update_config(directory, apply)
        launch = base.launch(self.binary, directory, 90)
        self.report['commands'].append(launch)
        self.save()
        assert launch['returncode'] == 0 and not launch['timeout'], (label, launch)
        return directory

    def check(self, label, result):
        self.report['checks'][label] = result
        self.save()
        print(json.dumps({'check': label, 'pass': result['pass']}), flush=True)
        assert result['pass'], (label, result)

    def restart(self, fixture, ranks):
        stem = f'{fixture}-gpu{ranks}'
        full = self.run(stem+'-full', fixture, ranks)
        self.check(stem+'-full', validate_run(full))
        # Resume the checkpoint written at an ordinary full-step boundary of
        # the uninterrupted run. A shortened stop time could clip that step
        # or introduce an extra residual step through binary floating arithmetic.
        checkpoint = full/'checkpoints'/'checkpoint_step_0000000020.bin'
        assert checkpoint.exists(), 'missing scheduled midpoint checkpoint'
        assert Path(str(checkpoint)+'.distributed_clumps').exists(), 'missing mandatory exact body/F64 sidecar'
        resumed = self.run(stem+'-resumed', fixture, ranks, resume=checkpoint)
        self.check(stem+'-resumed', validate_run(resumed, require_seam=False))
        self.check(stem+'-trajectory-restart', compare_saved_runs(full, resumed))
        restart = verify_restart(full, resumed)
        restart['active_contact_history_exercised'] = restart['checks']['semantic_contact_history']['nonzero_spring_contacts'] > 0
        if fixture in ('restart-contact', 'restart-gravity', 'shear-contact'):
            restart['pass'] &= restart['active_contact_history_exercised']
        self.check(stem+'-checkpoint-restart', restart)
        if ranks == self.ranks[0]:
            saved = read_record(checkpoint)
            roundtrip = self.run(stem+'-roundtrip', fixture, ranks, steps=20, resume=checkpoint,
                mutate=lambda config: config['simulation']['time'].update(stop=saved['time']))
            self.check(stem+'-zero-step-roundtrip', verify_restart(full, roundtrip, reference_step=20))
        return full

    def reject(self, label, fixture, ranks, expected, corrupt=None, resume=None, mutate=None):
        directory = self.work/label
        prepare(directory, fixture, ranks, steps=2)
        if resume or mutate:
            def apply(config):
                if resume:
                    config['io']['checkpoint'].update(resume_from=str(resume),
                        force_fresh=False, rebootstrap_forces=False)
                if mutate:
                    mutate(config)
            update_config(directory, apply)
        if corrupt:
            checkpoint, mode = corrupt
            copied = directory/'checkpoint-copy.bin'
            shutil.copy2(checkpoint, copied)
            for sidecar in checkpoint.parent.glob(checkpoint.name+'.*'):
                shutil.copy2(sidecar, Path(str(copied)+str(sidecar)[len(str(checkpoint)):]))
            exact = Path(str(copied)+'.distributed_clumps')
            if mode == 'missing':
                exact.unlink()
            else:
                blob = bytearray(exact.read_bytes()); blob[-1] ^= 1; exact.write_bytes(blob)
            update_config(directory, lambda c: c['io']['checkpoint'].update(
                resume_from=str(copied), force_fresh=False, rebootstrap_forces=False))
        result = base.launch(self.binary, directory, 30)
        self.report['commands'].append(result)
        log = (directory/'run.log').read_text()
        result.update(pass_=result['returncode'] == 1 and not result['timeout'] and expected in log,
                      expected_message=expected)
        result['pass'] = result.pop('pass_')
        self.check(label, result)
        return result

    def combined_extensions(self):
        """New combined coverage; old successful phase semantics are unchanged."""
        isolated_adaptive = {}
        for ranks in self.ranks:
            shear = self.restart('shear-adaptive-contact', ranks)
            self.check(f'shear-adaptive-seam-coverage-gpu{ranks}', validate_run(shear, require_seam=True))
            contact = read_record(latest_checkpoint(shear))
            self.check(f'shear-adaptive-nonzero-history-gpu{ranks}', {
                'pass': any(any(x != 0 for x in row[2:]) for row in contact['histories'].values()),
                'active_directed_contacts': len(contact['histories'])})
            isolated_adaptive[ranks] = self.restart('adaptive-restart', ranks)
        growth_rank = self.ranks[-1]
        growth = self.restart('adaptive-growth', growth_rank)
        rows = adaptive_diagnostics(growth/'run.log')
        roots = {(row.get('root_x'), row.get('root_y'), row.get('root_z'), row.get('root_size')) for row in rows}
        self.check(f'adaptive-root-growth-observed-gpu{growth_rank}', {
            'pass': len(roots) > 1, 'distinct_roots': [list(root) for root in sorted(roots)],
            'note': 'Each recorded force endpoint independently passed occupancy/diameter audit.'})
        if self.reference_campaign:
            fixed_reference = self.reference_campaign/'restart-gravity-gpu2-full'
            assert fixed_reference.exists(), 'reference campaign lacks accepted fixed two-GPU full run'
            launch = json.loads((fixed_reference/'launch.json').read_text())
            assert launch['binary_sha256'] == base.digest(self.binary), 'fixed reference binary differs'
            self.report['reused_fixed_reference'] = str(fixed_reference)
        else:
            fixed_reference = self.restart('restart-gravity', 2)
        fixed_checkpoint = fixed_reference/'checkpoints'/'checkpoint_step_0000000020.bin'
        transported = self.run('fixed-restart-gpu2-to4', 'restart-gravity', 4, resume=fixed_checkpoint)
        self.check('fixed-restart-gpu2-to4-trajectory', compare_saved_runs(fixed_reference, transported))
        self.check('fixed-restart-gpu2-to4-checkpoint', verify_restart(fixed_reference, transported))
        if 2 in isolated_adaptive:
            adaptive_checkpoint = isolated_adaptive[2]/'checkpoints'/'checkpoint_step_0000000020.bin'
            self.reject('adaptive-changed-rank-rejection', 'adaptive-restart', 4,
                'Invalid/mismatched common FMM checkpoint', resume=adaptive_checkpoint)
        else:
            self.report['limitations'].append('Adaptive2-to4 rejection requires --ranks including2; not exercised.')

    def shear_near_control(self):
        baseline = None
        for ranks in self.ranks:
            directory = self.run(f'shear-near-gravity-gpu{ranks}', 'shear-near-gravity', ranks, steps=4)
            result = validate_run(directory, gravity_oracle=True)
            first = read_record(directory/'output'/'output_step_0000000000.bin')
            split = []
            for cid in range(1, 3):
                images = {first['fields'][0x1a][gid][0] for gid, value in first['fields'][0x18].items() if value == (cid,)}
                if len(images) > 1:
                    split.append(cid)
            result['actual_seam_split_clumps'] = split
            result['pass'] &= bool(split) and result['finite_image_gravity']['internal_copy_noncentral_pairs'] > 0
            self.check(f'shear-near-gravity-oracle-gpu{ranks}', result)
            if baseline:
                self.check(f'shear-near-gravity-rank-parity-gpu{ranks}', compare_saved_runs(baseline, directory))
            else:
                baseline = directory

    def execute(self, phase):
        try:
            if phase in ('restart', 'all'):
                for fixture in ('restart-contact', 'restart-gravity'):
                    for ranks in self.ranks:
                        full = self.restart(fixture, ranks)
                        if fixture == 'restart-contact' and ranks == self.ranks[0]:
                            checkpoint = latest_checkpoint(full)
                            self.reject('restart-sidecar-missing', fixture, ranks,
                                'Missing, corrupt or mismatched distributed-clump checkpoint sidecar', (checkpoint, 'missing'))
                            self.reject('restart-sidecar-corrupt', fixture, ranks,
                                'Corrupt distributed-clump checkpoint payload', (checkpoint, 'corrupt'))
            if phase in ('shear', 'all'):
                baseline = {}
                for ranks in self.ranks:
                    full = self.restart('shear-contact', ranks)
                    if 'contact' in baseline:
                        self.check(f'shear-contact-rank-parity-gpu{ranks}', compare_saved_runs(baseline['contact'], full))
                    else:
                        baseline['contact'] = full
                    gravity = self.run(f'shear-gravity-gpu{ranks}', 'shear-gravity', ranks, steps=4)
                    self.check(f'shear-gravity-oracle-gpu{ranks}', validate_run(gravity, gravity_oracle=True))
                    if 'gravity' in baseline:
                        self.check(f'shear-gravity-rank-parity-gpu{ranks}', compare_saved_runs(baseline['gravity'], gravity))
                    else:
                        baseline['gravity'] = gravity
            if phase in ('adaptive', 'all'):
                for ranks in self.ranks:
                    adaptive = self.run(f'adaptive-occupancy-gpu{ranks}', 'adaptive-occupancy', ranks, steps=4)
                    result = validate_run(adaptive)
                    self.check(f'adaptive-occupancy-gpu{ranks}', result)
                    chosen = {row['depth'] for row in result.get('adaptive', {}).get('diagnostics', [])}
                    if len(chosen) == 1:
                        depth = next(iter(chosen))
                        def fixed(config):
                            config['accelerations']['self_gravity']['unified_fmm'].update(
                                tree_depth=depth, adaptive_depth={'enabled': False})
                        control = self.run(f'adaptive-fixed-control-gpu{ranks}', 'adaptive-occupancy', ranks, steps=4, mutate=fixed)
                        self.check(f'adaptive-fixed-parity-gpu{ranks}', compare_saved_runs(control, adaptive))
                    else:
                        self.report['limitations'].append(f'adaptive rank{ranks} changed depths; no single-depth control claimed')
                self.reject('adaptive-occupancy-no-valid-depth', 'adaptive-too-coarse', self.ranks[-1], '128')
                self.reject('adaptive-diameter-occupancy-conflict', 'adaptive-clump-conflict', self.ranks[-1], 'clump')
            if phase == 'combined':
                self.combined_extensions()
            if phase == 'shear-control':
                self.shear_near_control()
            self.report['pass'] = bool(self.report['checks']) and all(x['pass'] for x in self.report['checks'].values())
        except Exception as error:
            self.report['failure'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            self.save()
        return 0 if self.report['pass'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--fixture', choices=FIXTURES, required=True)
    p.add_argument('--ranks', type=int, default=2)
    p.add_argument('--steps', type=int, default=40)
    c = sub.add_parser('compare-restart')
    c.add_argument('--reference', type=Path, required=True)
    c.add_argument('--resumed', type=Path, required=True)
    r = sub.add_parser('run')
    r.add_argument('--binary', type=Path, required=True)
    r.add_argument('--work-dir', type=Path, required=True)
    r.add_argument('--phase', choices=['restart', 'shear', 'adaptive', 'combined', 'shear-control', 'all'], default='all')
    r.add_argument('--ranks', nargs='+', type=int, default=[1, 2, 4])
    r.add_argument('--reference-campaign', type=Path,
                   help='reuse accepted fixed-depth restart-gravity-gpu2-full for2-to4 test')
    v = sub.add_parser('verify-run')
    v.add_argument('--work-dir', type=Path, required=True)
    v.add_argument('--gravity-oracle', action='store_true')
    s = sub.add_parser('prepare-sphere-control')
    s.add_argument('--clump-run', type=Path, required=True)
    s.add_argument('--work-dir', type=Path, required=True)
    a = sub.add_parser('analyze-sphere-control')
    a.add_argument('--clump-run', type=Path, required=True)
    a.add_argument('--sphere-run', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.work_dir.resolve(), args.fixture, args.ranks, args.steps), indent=2))
    elif args.command == 'compare-restart':
        result = verify_restart(args.reference.resolve(), args.resumed.resolve())
        base.dump(args.resumed/'restart-verification.json', result)
        print(json.dumps(result, indent=2))
        return 0 if result['pass'] else 1
    elif args.command == 'verify-run':
        result = validate_run(args.work_dir.resolve(), gravity_oracle=args.gravity_oracle)
        base.dump(args.work_dir/'extension-verification.json', result)
        print(json.dumps(result, indent=2))
        return 0 if result['pass'] else 1
    elif args.command == 'prepare-sphere-control':
        print(json.dumps(prepare_sphere_causality_control(args.clump_run.resolve(), args.work_dir.resolve()), indent=2))
    elif args.command == 'analyze-sphere-control':
        result = analyze_sphere_causality_control(args.clump_run.resolve(), args.sphere_run.resolve())
        base.dump(args.sphere_run/'causality-analysis.json', result)
        print(json.dumps(result, indent=2))
        return 0 if result['pass'] else 1
    else:
        return Campaign(args.binary, args.work_dir, args.ranks, args.reference_campaign).execute(args.phase)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
