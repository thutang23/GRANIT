#!/usr/bin/env python3
"""Dependency-free small-clump DEM multi-GPU fixtures and physical validation.

JSON configuration is valid YAML. No GPU is used by generate/self-test/verify.
Run uses an existing executable and records every command, hash and failure.
"""
from __future__ import annotations
import argparse
from array import array
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
HEADER = struct.Struct('<8sHHQqddHBBIIQQ')
DESC = struct.Struct('<HBBIQ')
TYPES = {0: 'f', 1: 'd', 2: 'i', 3: 'h', 4: 'B', 5: 'B', 6: 'q'}
OFFSETS = [(-.008, -.008, -.008), (-.008, .008, .008),
           (.008, -.008, .008), (.008, .008, -.008)]
FIXTURES = ('all-clump', 'contact-migration', 'mixed', 'empty-rank',
            'near-gravity', 'populated-fmm', 'inferred-spin', 'integrated-contact', 'oversized', 'throughput')
CONTACT_FIXTURES = ('contact-migration', 'mixed', 'inferred-spin', 'integrated-contact')


def dump(path, data):
    with path.open('w') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def norm(a):
    return math.sqrt(sum(x*x for x in a))


def write_input(path, rows, include_omega=True):
    """Stream fixture rows into compact SoA arrays; 1M particles needs ~85 MiB."""
    pos = [array('d') for _ in range(3)]
    vel = [array('d') for _ in range(3)]
    omega = [array('d') for _ in range(3)]
    radius, clump, material, gids = array('f'), array('i'), array('i'), array('i')
    for i, (p, v, w, r, cid) in enumerate(rows):
        for a in range(3):
            pos[a].append(p[a]); vel[a].append(v[a]); omega[a].append(w[a])
        radius.append(r); clump.append(cid); material.append(0); gids.append(i)
    n = len(gids)
    assert n > 0 and sys.byteorder == 'little'
    fields = [(6, 2, [material]), (0x11, 0, [radius]), (0x18, 2, [clump])]
    if include_omega:
        fields.insert(1, (0x10, 1, omega))
    offset = 0
    with path.open('wb') as f:
        f.write(HEADER.pack(b'GRANIT\0\0', 3, 72, n, 0, 0., 0., len(fields), 1, 1, 0, 0, 0, 0))
        for fid, dtype, arrays in fields:
            f.write(DESC.pack(fid, dtype, len(arrays), 0, offset))
            offset += sum(len(a)*a.itemsize for a in arrays)
        for a in pos + vel + [gids]:
            a.tofile(f)
        for _, _, arrays in fields:
            for a in arrays:
                a.tofile(f)
    return n, max(clump), sum(cid == 0 for cid in clump)


def fixture_rows(name, count):
    def body(cid, center, velocity=(0., 0., 0.), spin=(0., 0., 0.), scale=1., radius=.01):
        for off in OFFSETS:
            off = tuple(scale*x for x in off)
            rotational = cross(spin, off)
            yield (tuple(center[a]+off[a] for a in range(3)),
                   tuple(velocity[a]+rotational[a] for a in range(3)), spin, radius, cid)
    if name == 'throughput':
        side = math.ceil(count**(1/3))
        for i in range(count):
            xyz = (i % side, (i // side) % side, i // (side*side))
            center = tuple(.1*(x-.5*(side-1)) for x in xyz)
            yield from body(i+1, center)
        return
    if name == 'empty-rank':
        # One whole Morton leaf, no normal spheres, >1 owners => empty ranks.
        yield from body(1, (.12, .13, .14), spin=(.3, -.7, 2.), scale=0.)
        return
    if name == 'near-gravity':
        # Tiny normal anchors define a wide root with negligible force. Both
        # clumps lie in neighboring finest leaves; self force would dominate.
        for p in [(-1., -1., -1.), (1., 1., 1.)]:
            yield p, (0.,)*3, (0.,)*3, 1e-5, 0
        yield from body(1, (-.024, .0003, -.0003), scale=.15, radius=.001)
        yield from body(2, (.024, .0007, .0004), scale=.15, radius=.001)
        return
    if name == 'populated-fmm':
        for i, xyz in enumerate(itertools.product(range(6), repeat=3), 1):
            yield from body(i, tuple(.3*(x-2.5) for x in xyz))
        return
    if name == 'oversized':
        yield from body(1, (0., 0., 0.), scale=70.)
        yield from body(2, (.7, .7, .7))
        return
    contact = name in CONTACT_FIXTURES
    yield from body(1, (-.003 if contact else -.10, .011, -.003 if contact else .013),
                    (20., .2, 40.) if contact else (0.,)*3, (0., 0., 3.))
    yield from body(2, (.023 if contact else .10, .011, -.003 if contact else .013),
                    (20., -.2, 40.) if contact else (0.,)*3, (0., 0., -2.))
    # Far clumps keep every body small relative to the dynamic root without
    # normal particles, explicitly exercising a zero-size normal segment.
    for cid, xyz in enumerate(itertools.product((-1., 1.), repeat=3), 3):
        yield from body(cid, xyz)
    if name == 'mixed':
        for p in [(-.30, .22, .31), (.25, -.22, -.31), (.7, .2, -.1)]:
            yield p, (.1, 0., 0.), (0.,)*3, .01, 0


def make_config(work, name, k, ranks, gravity=True, steps=None):
    contact = name in CONTACT_FIXTURES
    steps = (40 if contact else 4) if steps is None else steps
    dt = 1e-5
    return {'material_database': str(work/'materials.json'),
        'simulation': {'name': name, 'physics': 'dem', 'time': {
            'start': 0., 'stop': steps*dt, 'output_step': dt, 'dt': dt}},
        'dem': {'damping_model': 'restitution'},
        'domain': {'box_min': [-10.]*3, 'box_max': [10.]*3},
        'clumps': {'instances': [{'id': i+1} for i in range(k)]},
        'neighbor': {'method': 'lbvh', 'backend': 'cuda', 'update_frequency': 1,
                     'verlet_buffer_enabled': False},
        'accelerations': {'self_gravity': {'enabled': gravity, 'method': 'unified_fmm',
            'gravity_update_rate': 1, 'direct': {'gravitational_constant': .001},
            'unified_fmm': {'gravitational_constant': .001, 'theta_far': .8,
                'tree_dimension': 3, 'tree_depth': 4, 'adaptive_depth': {'enabled': False},
                'softening_factor': 0., 'enable_global_projection': False,
                'direct_threshold_leaf': 0, 'direct_threshold_internal': 0}}},
        'io': {'input': {'particle_file': str(work/'input.bin')},
            'output': {'directory': str(work/'output'), 'initial_state': True, 'fields': ['gravity_force']},
            'checkpoint': {'enabled': False}},
        'multigpu': {'enabled': ranks > 0, 'max_gpus': max(1, ranks),
            'morton_ownership': True, 'partition_depth': 4,
            'rebalance_min_interval': 1000, 'halo_width_factor': 1.5,
            'buffer_capacity_factor': 3.}, 'gpu': {'device_id': 0}}


def generate(work, name, count=250000, ranks=2, steps=None, spheres=False):
    work.mkdir(parents=True, exist_ok=False)
    rows = fixture_rows(name, count)
    if spheres:
        if name != 'throughput':
            raise ValueError('sphere baseline is defined only for nonoverlapping throughput geometry')
        rows = ((p, v, w, r, 0) for p, v, w, r, _ in rows)
    n, k, normals = write_input(work/'input.bin', rows, include_omega=name != 'inferred-spin')
    dump(work/'materials.json', {'version': 1, 'materials': [{'id': 0, 'name': 'test',
        'dem': {'density': 1000., 'youngs_modulus': 1. if name == 'near-gravity' else 1e6,
                'poisson_ratio': .25}}],
        'dem_interactions': [{'materials': [0, 0], 'restitution': .8,
            'friction': {'static': .4, 'rolling': .03, 'twisting': .02},
            'shape_parameter': 1., 'cohesion': {'model': 'none'}}]})
    cfg = make_config(work, name, k, ranks, gravity=name in (
        'near-gravity', 'populated-fmm', 'integrated-contact', 'oversized', 'throughput'), steps=steps)
    dump(work/'config.json', cfg)
    manifest = {'schema': 'dem-multigpu-clumps-fixture/v1', 'fixture': name,
        'particles': n, 'clumps': k, 'normals': normals, 'sphere_baseline': spheres,
        'input_sha256': digest(work/'input.bin'), 'config_sha256': digest(work/'config.json'),
        'steps': round(cfg['simulation']['time']['stop']/cfg['simulation']['time']['dt']),
        'dt': cfg['simulation']['time']['dt'], 'ranks': ranks,
        'limitations': ['Rank migration, empty ranks and remote contact require runtime ownership diagnostics.']}
    dump(work/'manifest.json', manifest)
    return manifest


def read_particles(path, output=True):
    blob = path.read_bytes()
    if len(blob) < HEADER.size:
        raise ValueError(f'truncated header: {path}')
    magic, version, hlen, n, step, t, dt, nf, physics, kind, stride, contacts, material_hash, clump_hash = HEADER.unpack_from(blob)
    assert (magic, version, hlen, physics) == (b'GRANIT\0\0', 3, 72, 1), path
    assert kind == (2 if output else 1) and n > 0 and stride == 0 and contacts == 0, path
    assert all(math.isfinite(x) for x in (t, dt)) and step >= 0
    base = 72+16*nf
    fbase = base+52*n
    coords = struct.unpack_from(f'<{6*n}d', blob, base)
    gids = struct.unpack_from(f'<{n}i', blob, base+48*n)
    assert set(gids) == set(range(n)), f'canonical IDs missing/duplicated in {path}'
    assert all(math.isfinite(x) for x in coords)
    fields, spans = {}, []
    for i in range(nf):
        fid, dtype, count, reserved, off = DESC.unpack_from(blob, 72+16*i)
        assert fid not in fields and dtype in TYPES and reserved == 0 and count > 0
        fmt = TYPES[dtype]
        values = struct.unpack_from(f'<{n*count}{fmt}', blob, fbase+off)
        assert all(math.isfinite(x) for x in values), (path, fid)
        schema = {6: (2, 1), 0x10: (1, 3), 0x11: (0, 1), 0x12: (0, 1),
                  0x13: (0, 1), 0x14: (0, 3), 0x15: (0, 3), 0x18: (2, 1), 0x1d: (4, 1)}
        if fid in schema:
            assert (dtype, count) == schema[fid], (path, fid, dtype, count)
        fields[fid] = {gids[j]: tuple(values[a*n+j] for a in range(count)) for j in range(n)}
        spans.append((off, off+struct.calcsize(fmt)*n*count))
    end = 0
    for lo, hi in sorted(spans):
        assert lo == end, 'overlapping or gapped fields'
        end = hi
    assert fbase+end == len(blob), f'truncated/trailing bytes in {path}'
    required = {6, 0x11, 0x18} | ({0x10, 0x12, 0x13, 0x14, 0x15, 0x1d} if output else set())
    assert required <= fields.keys(), (path, required-fields.keys())
    assert all(r[0] > 0 for r in fields[0x11].values())
    if output:
        assert all(x[0] > 0 for x in fields[0x12].values())
        assert all(x == (0,) for x in fields[0x1d].values())
    rows = {gids[j]: tuple(coords[a*n+j] for a in range(6)) for j in range(n)}
    return {'path': str(path), 'step': step, 'time': t, 'dt': dt, 'n': n, 'rows': rows, 'fields': fields,
            'file_order': gids}


def read_clumps(path):
    blob = path.read_bytes()
    magic, version, n, step, t = struct.unpack_from('<8sIiqd', blob)
    assert magic == b'CLUMP\0\0\0' and version == 2 and n > 0
    assert len(blob) == 32+n*364 and math.isfinite(t)
    rows = {}
    for i in range(n):
        cid, *v = struct.unpack_from('<i45d', blob, 32+364*i)
        assert cid == i+1 and all(math.isfinite(x) for x in v)
        assert abs(norm(v[6:10])-1.) <= 1e-10 and v[13] > 0
        rows[cid] = v
    return {'rows': rows, 'step': step, 'time': t}


def check_trajectory(work):
    manifest = json.loads((work/'manifest.json').read_text())
    assert digest(work/'input.bin') == manifest['input_sha256'], 'input bytes changed after generation'
    assert digest(work/'config.json') == manifest['config_sha256'], 'configuration changed after generation'
    source = read_particles(work/'input.bin', False)
    expected_membership = source['fields'][0x18]
    members = {}
    for gid, (cid,) in expected_membership.items():
        if cid:
            members.setdefault(cid, []).append(gid)
    distances = {(a, b): math.dist(source['rows'][a][:3], source['rows'][b][:3])
                 for group in members.values() for a, b in itertools.combinations(group, 2)}
    files = sorted((work/'output').glob('output_step_*.bin'))
    assert len(files) == manifest['steps']+1, (len(files), manifest['steps']+1)
    worst_distance, worst_velocity = 0., 0.
    first_bodies, last_bodies, maximum_torque = None, None, 0.
    frames = []
    for expected_step, path in enumerate(files):
        p = read_particles(path)
        assert p['step'] == expected_step and p['n'] == manifest['particles']
        assert abs(p['time']-expected_step*manifest['dt']) <= 1e-12
        assert abs(p['dt']-manifest['dt']) <= 1e-15, ('unexpected actual timestep', p['dt'], manifest['dt'])
        assert p['fields'][0x18] == expected_membership
        for pair, d in distances.items():
            err = abs(math.dist(p['rows'][pair[0]][:3], p['rows'][pair[1]][:3])-d)
            worst_distance = max(worst_distance, err)
            assert err <= 2e-8, ('rigid distance', path, pair, err)
        cp = path.parent/f"clump_states_step_{p['step']:010d}.bin"
        # The established SG runner writes initial members after bootstrap,
        # then emits body snapshots with each completed integration step.
        if members and (expected_step > 0 or cp.exists()):
            c = read_clumps(cp)
            assert c['step'] == p['step'] and abs(c['time']-p['time']) <= 1e-14
            assert set(c['rows']) == set(members)
            first_bodies = first_bodies or c
            last_bodies = c
            for cid, group in members.items():
                body = c['rows'][cid]
                maximum_torque = max(maximum_torque, norm(body[-3:]))
                for gid in group:
                    offset = tuple(p['rows'][gid][a]-body[a] for a in range(3))
                    world_omega = p['fields'][0x10][gid]
                    spin_velocity = cross(world_omega, offset)
                    error = max(abs(p['rows'][gid][a+3]-body[a+3]-spin_velocity[a]) for a in range(3))
                    worst_velocity = max(worst_velocity, error)
                    assert error <= 2e-8, ('rigid velocity', path, gid, error)
        frames.append(p)
    contact = manifest['fixture'] in CONTACT_FIXTURES
    activation = {'contact_torque': maximum_torque,
        'body1_crossed_x_zero': bool(first_bodies and first_bodies['rows'][1][0] < 0 < last_bodies['rows'][1][0])}
    if contact:
        assert maximum_torque > 1e-8, 'contact fixture did not generate a body torque'
        assert activation['body1_crossed_x_zero'], 'migration fixture did not cross x=0'
        assert norm([last_bodies['rows'][1][i]-first_bodies['rows'][1][i] for i in range(10, 13)]) > 1e-5
    diagnostics = ownership_diagnostics(work)
    coverage = {'pass': True, 'required': manifest['ranks'] > 1}
    required_checks = []
    if coverage['required']:
        coverage['diagnostics_present'] = diagnostics['available']
        required_checks.append('diagnostics_present')
        if manifest['fixture'] in CONTACT_FIXTURES:
            coverage['remote_contact_exercised'] = diagnostics['remote_interclump_overlap_pairs'] > 0
            coverage['member_migration_exercised'] = diagnostics['member_count_changes'] > 0
            config = json.loads((work/'config.json').read_text())
            coverage['cross_rank_contact_required'] = config['accelerations']['self_gravity']['enabled']
            if coverage['cross_rank_contact_required']:
                required_checks.extend(['remote_contact_exercised', 'member_migration_exercised'])
            else:
                coverage['scope'] = 'Mechanical SG parity; integrated-contact requires cross-rank contact/migration.'
        if manifest['fixture'] == 'near-gravity':
            coverage['remote_body_participants_exercised'] = diagnostics['remote_body_participants'] > 0
            required_checks.append('remote_body_participants_exercised')
        if manifest['fixture'] == 'empty-rank':
            coverage['empty_rank_exercised'] = bool(diagnostics['empty_particle_ranks'])
            required_checks.append('empty_rank_exercised')
        coverage['pass'] = all(coverage[key] for key in required_checks)
    return {'pass': coverage['pass'], 'frames': len(frames), 'max_rigid_distance_error': worst_distance,
            'max_rigid_velocity_error': worst_velocity, 'activation': activation,
            'ownership_diagnostics': diagnostics, 'coverage': coverage}, frames


def ownership_diagnostics(work):
    log_path = work/'run.log'
    if not log_path.exists():
        return {'available': False}
    categories = {'routes': [], 'migration': [], 'contact': []}
    for line in log_path.read_text().splitlines():
        for category in categories:
            if f'[clumps-{category}]' in line:
                categories[category].append({k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line)})
    return {'available': any(categories.values()),
        'empty_particle_ranks': sorted({x['rank'] for x in categories['routes'] if x.get('owned_members') == 0}),
        'remote_body_participants': max((x.get('remote_bodies', 0) for x in categories['routes']), default=0),
        'member_count_changes': sum(x.get('previous_members') != x.get('owned_members') for x in categories['migration']),
        'remote_interclump_overlap_pairs': max((x.get('remote_interclump_overlap_pairs', 0) for x in categories['contact']), default=0),
        'records': categories}


def compare(reference, candidate):
    ra, ca = check_trajectory(reference), check_trajectory(candidate)
    assert len(ra[1]) == len(ca[1])
    worst = {}
    # Double kinematics; FP32 contact wrench output has its own quantization.
    for r, c in zip(ra[1], ca[1]):
        for label, x, y, atol, rtol in [
            ('member_state', r['rows'], c['rows'], 2e-8, 2e-6),
            ('member_omega', r['fields'][0x10], c['fields'][0x10], 2e-7, 2e-5)]:
            assert x.keys() == y.keys()
            fraction = max((abs(v-y[k][i])/(atol+rtol*abs(v)) for k, row in x.items() for i, v in enumerate(row)), default=0.)
            worst[label] = max(worst.get(label, 0.), fraction)
        if (reference/'output'/f"clump_states_step_{r['step']:010d}.bin").exists():
            rc = read_clumps(reference/'output'/f"clump_states_step_{r['step']:010d}.bin")
            cc = read_clumps(candidate/'output'/f"clump_states_step_{c['step']:010d}.bin")
            for label, indices, atol, rtol in [('body_state', range(13), 2e-8, 2e-6),
                    ('body_wrench', range(39, 45), 1e-12, 2e-5)]:
                fraction = max(abs(v[i]-cc['rows'][k][i])/(atol+rtol*abs(v[i])) for k, v in rc['rows'].items() for i in indices)
                worst[label] = max(worst.get(label, 0.), fraction)
    return {'pass': ra[0]['pass'] and ca[0]['pass'] and all(v <= 1 for v in worst.values()), 'max_gate_fractions': worst,
            'reference': ra[0], 'candidate': ca[0]}


def direct_gravity(work):
    """Independent Newton pair sum against aggregate body force and torque.

    At update rate 1 the SG member force/cache does not include its body's
    gravity; clump v2 stores the physical body wrench and is the proper oracle.
    """
    body_path = sorted((work/'output').glob('clump_states_step_*.bin'))[0]
    snapshot = read_clumps(body_path)
    p = read_particles(work/'output'/f"output_step_{snapshot['step']:010d}.bin")
    gids, rows, fields = sorted(p['rows']), p['rows'], p['fields']
    bodies = snapshot['rows']
    reference = {cid: [0.]*6 for cid in bodies}
    torque_abs_sum = {cid: 0. for cid in bodies}
    member_reference = {}
    for i, gid in enumerate(gids):
        cid = fields[0x18][gid][0]
        if not cid:
            continue
        force = [0., 0., 0.]
        for other in gids:
            if other == gid or (fields[0x18][gid][0] > 0 and fields[0x18][gid] == fields[0x18][other]):
                continue
            d = [rows[other][a]-rows[gid][a] for a in range(3)]
            factor = .001*fields[0x12][gid][0]*fields[0x12][other][0]/norm(d)**3
            for a in range(3):
                force[a] += factor*d[a]
        arm = [rows[gid][a]-bodies[cid][a] for a in range(3)]
        torque = cross(arm, force)
        torque_abs_sum[cid] += norm(torque)
        for a in range(3):
            reference[cid][a] += force[a]
            reference[cid][a+3] += torque[a]
        member_reference[gid] = force
    result = {'pass': True, 'step': snapshot['step'], 'fields': {},
        'torque_conditioning': {cid: {'sum_member_torque_norms': torque_abs_sum[cid],
            'net_reference_torque_norm': norm(row[3:6]),
            'cancellation_condition': torque_abs_sum[cid]/norm(row[3:6]) if norm(row[3:6]) else None}
            for cid, row in reference.items()}}
    for label, offset in [('force', 0), ('torque', 3)]:
        squared = sum(norm(row[offset:offset+3])**2 for row in reference.values())
        errors = [norm([bodies[cid][39+offset+a]-row[offset+a] for a in range(3)])
                  for cid, row in reference.items()]
        rms = math.sqrt(squared/len(bodies))
        relative = math.sqrt(sum(x*x for x in errors)/max(squared, 1e-300))
        maximum = max(errors)/max(rms, 1e-300)
        result['fields'][label] = {'pass': relative <= 1e-4 and maximum <= 5e-4,
            'relative_l2_error': relative, 'max_error_over_reference_rms': maximum,
            'absolute_l2_error': norm(errors), 'absolute_max_error': max(errors),
            'reference_l2_norm': math.sqrt(squared), 'reference_rms_norm': rms,
            'relative_l2_gate': 1e-4, 'max_error_over_reference_rms_gate': 5e-4}
        result['pass'] &= result['fields'][label]['pass']
    config = json.loads((work/'config.json').read_text())
    if config['multigpu']['enabled']:
        # MG retains physical per-member force before body reduction. This
        # check catches same-clump leakage even when internal Newton forces
        # cancel in the body force and torque sums.
        squared = sum(norm(row)**2 for row in member_reference.values())
        errors = [norm([fields[0x14][gid][a]-row[a] for a in range(3)])
                  for gid, row in member_reference.items()]
        rms = math.sqrt(squared/len(member_reference))
        relative = math.sqrt(sum(x*x for x in errors)/max(squared, 1e-300))
        maximum = max(errors)/max(rms, 1e-300)
        result['fields']['member_force'] = {'pass': relative <= 1e-4 and maximum <= 5e-4,
            'relative_l2_error': relative, 'max_error_over_reference_rms': maximum,
            'relative_l2_gate': 1e-4, 'max_error_over_reference_rms_gate': 5e-4}
        result['pass'] &= result['fields']['member_force']['pass']
    return result


def launch(binary, work, timeout):
    command = [str(binary.resolve()), str(work/'config.json')]
    start = time.monotonic()
    result = {'argv': command, 'binary_sha256': digest(binary), 'config_sha256': digest(work/'config.json')}
    with (work/'run.log').open('w') as log:
        try:
            env = dict(os.environ, GRANIT_CLUMP_DIAGNOSTICS='1')
            result['environment'] = {key: env[key] for key in ('GRANIT_CLUMP_DIAGNOSTICS', 'CUDA_VISIBLE_DEVICES') if key in env}
            result['returncode'] = subprocess.run(command, cwd=ROOT, stdout=log, env=env,
                stderr=subprocess.STDOUT, timeout=timeout).returncode
            result['timeout'] = False
        except subprocess.TimeoutExpired:
            result.update(returncode=None, timeout=True)
    result['wall_seconds'] = time.monotonic()-start
    dump(work/'launch.json', result)
    return result


def reject(args):
    generate(args.work_dir, args.fixture, ranks=args.ranks)
    config = json.loads((args.work_dir/'config.json').read_text())
    config['accelerations']['self_gravity']['enabled'] = True
    def merge(target, overlay):
        for key, value in overlay.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value
    merge(config, json.loads(args.overlay))
    dump(args.work_dir/'config.json', config)
    manifest = json.loads((args.work_dir/'manifest.json').read_text())
    manifest['config_sha256'] = digest(args.work_dir/'config.json')
    manifest['rejection_overlay'] = json.loads(args.overlay)
    dump(args.work_dir/'manifest.json', manifest)
    result = launch(args.binary, args.work_dir, args.timeout)
    log = (args.work_dir/'run.log').read_text()
    result['expected_message'] = args.expect
    result['normal_error_exit'] = result['returncode'] is not None and result['returncode'] > 0
    result['pass'] = not result['timeout'] and result['normal_error_exit'] and args.expect in log
    dump(args.work_dir/'rejection.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['pass'] else 1


def run(args):
    args.work_dir.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'dem-multigpu-clumps-validation/v1', 'passed': False, 'checks': {},
              'limitations': ['Physical parity applies to the bounded fixtures and declared tolerances.',
                'x=0 crossing alone does not establish rank migration; inspect ownership diagnostics.',
                'Outputs do not serialize contact history; friction persistence is checked through trajectory parity.']}
    try:
        for fixture in args.fixtures:
            for ranks in [0]+args.ranks:
                work = args.work_dir/f'{fixture}-gpu{ranks}'
                generate(work, fixture, ranks=ranks, steps=args.steps)
                result = launch(args.single if ranks == 0 else args.multi, work, args.timeout)
                assert not result['timeout'] and result['returncode'] == 0, (work, result)
                validation, _ = check_trajectory(work)
                report['checks'][work.name] = validation
                assert validation['pass'], (work, validation)
                if fixture == 'near-gravity':
                    report['checks'][work.name+'-direct'] = direct_gravity(work)
                    assert report['checks'][work.name+'-direct']['pass']
                if ranks:
                    key = work.name+'-parity'
                    # Distributed 1/2/4 GPUs share a common Morton root.
                    # Standalone SG uses a dynamic root, so a populated FMM
                    # tree may accept a different approximation there.
                    baseline = 1 if fixture in ('populated-fmm', 'integrated-contact') else 0
                    if baseline == 1 and ranks == 1:
                        continue
                    reference = args.work_dir/f'{fixture}-gpu{baseline}'
                    assert reference.exists(), 'populated-fmm requires --ranks to start with 1'
                    report['checks'][key] = compare(reference, work)
                    assert report['checks'][key]['pass'], (key, report['checks'][key])
                dump(args.work_dir/'report.json', report)
        report['passed'] = True
    except Exception as error:
        report['failure'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        dump(args.work_dir/'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    g = sub.add_parser('generate')
    g.add_argument('--output-dir', type=Path, required=True)
    g.add_argument('--fixture', choices=FIXTURES, required=True)
    g.add_argument('--clumps', type=int, default=250000)
    g.add_argument('--ranks', type=int, default=2)
    g.add_argument('--steps', type=int)
    g.add_argument('--spheres', action='store_true')
    v = sub.add_parser('verify')
    v.add_argument('--work-dir', type=Path, required=True)
    v.add_argument('--reference', type=Path)
    v.add_argument('--direct-gravity', action='store_true')
    r = sub.add_parser('run')
    r.add_argument('--single', type=Path, required=True)
    r.add_argument('--multi', type=Path, required=True)
    r.add_argument('--work-dir', type=Path, required=True)
    r.add_argument('--ranks', nargs='+', type=int, default=[1, 2, 4])
    r.add_argument('--fixtures', nargs='+', choices=FIXTURES[:-2], default=list(FIXTURES[:-2]))
    r.add_argument('--timeout', type=float, default=180.)
    r.add_argument('--steps', type=int)
    n = sub.add_parser('reject')
    n.add_argument('--binary', type=Path, required=True)
    n.add_argument('--work-dir', type=Path, required=True)
    n.add_argument('--fixture', choices=['all-clump', 'oversized'], default='all-clump')
    n.add_argument('--ranks', type=int, default=2)
    n.add_argument('--overlay', default='{}', help='JSON object recursively overlaid on generated config')
    n.add_argument('--expect', required=True, help='required diagnostic substring')
    n.add_argument('--timeout', type=float, default=30.)
    args = parser.parse_args()
    for attribute in ('output_dir', 'work_dir', 'reference'):
        if getattr(args, attribute, None):
            setattr(args, attribute, getattr(args, attribute).resolve())
    if args.command == 'generate':
        print(json.dumps(generate(args.output_dir, args.fixture, args.clumps, args.ranks, args.steps, args.spheres), indent=2))
    elif args.command == 'verify':
        result = compare(args.reference, args.work_dir) if args.reference else check_trajectory(args.work_dir)[0]
        if args.direct_gravity:
            result['direct_gravity'] = direct_gravity(args.work_dir)
            result['pass'] &= result['direct_gravity']['pass']
        dump(args.work_dir/'verification.json', result)
        print(json.dumps(result, indent=2))
        return 0 if result['pass'] else 1
    elif args.command == 'reject':
        return reject(args)
    else:
        run(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
