#!/usr/bin/env python3
"""Production DEM checkpoint/RESPA regression for member-based clump gravity.

Dependency-free. Runs bounded fresh, interrupted, and resumed trajectories in a
new output directory; compares actual particle and rigid-body endpoint states.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess


def write_input(path, large):
    points = [(-16., -16., -16.), (16., 16., 16.),
              (.12, .15, .09), (-2.7, .9, 1.8), (4.1, -3.2, .7)]
    ids = [0] * len(points)
    offsets = [(-.78, -.19, .11), (.43, -.31, -.23),
               (.08, .57, .37), (.34, .12, -.46)]
    scale = .9 if large else .006
    for cid, center in enumerate(((.21, .23, .19), (-1.3, -.7, 1.1)), 1):
        points.extend(tuple(center[a] + scale * r[a] for a in range(3))
                      for r in offsets)
        ids.extend([cid] * len(offsets))
    n = len(points)
    fields = [(6, 2, struct.pack('<' + 'i' * n, *([0] * n))),
              (0x11, 0, struct.pack('<' + 'f' * n, *([.001] * n))),
              (0x18, 2, struct.pack('<' + 'i' * n, *ids))]
    descriptors, payload, offset = [], [], 0
    for field, dtype, blob in fields:
        descriptors.append(struct.pack('<HBBIQ', field, dtype, 1, 0, offset))
        payload.append(blob)
        offset += len(blob)
    header = struct.pack('<8sHHQqddHBBIIQQ', b'GRANIT\0\0', 3, 72,
                         n, 0, 0., 0., len(fields), 1, 1, 0, 0, 0, 0)
    positions = struct.pack('<' + 'd' * (3 * n),
                            *(p[a] for a in range(3) for p in points))
    path.write_bytes(header + b''.join(descriptors) + positions + bytes(24 * n)
                     + struct.pack('<' + 'i' * n, *range(n)) + b''.join(payload))


def read_particles(path):
    blob = path.read_bytes()
    magic, version, hlen, n, step, time, dt, nf, physics, kind, *_ = struct.unpack_from(
        '<8sHHQqddHBBIIQQ', blob)
    assert (magic, version, hlen, physics, kind) == (b'GRANIT\0\0', 3, 72, 1, 2)
    base = hlen + 16 * nf
    values = struct.unpack_from('<' + 'd' * (6 * n), blob, base)
    gids = struct.unpack_from('<' + 'i' * n, blob, base + 48 * n)
    assert len(set(gids)) == n and all(math.isfinite(x) for x in values)
    rows = {gids[i]: tuple(values[a * n + i] for a in range(6)) for i in range(n)}
    return {'step': step, 'time': time, 'dt': dt, 'rows': rows}


def read_clumps(path):
    blob = path.read_bytes()
    magic, version, n, step, time = struct.unpack_from('<8sIiqd', blob)
    assert magic == b'CLUMP\0\0\0' and version == 2
    rows, offset = {}, 32
    for _ in range(n):
        cid, *values = struct.unpack_from('<i45d', blob, offset)
        offset += 4 + 45 * 8
        assert all(math.isfinite(x) for x in values)
        rows[cid] = values
    assert offset == len(blob)
    return {'step': step, 'time': time, 'rows': rows}


def compare_endpoints(a, b):
    assert a['rows'].keys() == b['rows'].keys()
    assert a['step'] == b['step'] and abs(a['time'] - b['time']) <= 1e-14
    return [max(abs(a['rows'][key][i] - b['rows'][key][i])
                for key in a['rows']) for i in range(len(next(iter(a['rows'].values()))))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--large', action='store_true')
    args = parser.parse_args()
    exe, work = args.executable.resolve(), args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    particle_file, material_file = work / 'members.bin', work / 'materials.yaml'
    write_input(particle_file, args.large)
    material_file.write_text(json.dumps({'version': 1, 'materials': [
        {'id': 0, 'name': 'test', 'dem': {'density': 1e8,
         'youngs_modulus': 1e5, 'poisson_ratio': .25}}],
        'dem_interactions': [{'materials': [0, 0], 'restitution': 1.,
            'friction': {'static': 0., 'rolling': 0., 'twisting': 0.},
            'shape_parameter': 1., 'cohesion': {'model': 'none'}}]}, indent=2))
    results = []
    for method in ('direct', 'unified_fmm'):
        for rate in (1, 2):
            root = work / f'{method}_k{rate}'
            root.mkdir()
            def run(label, stop, split=False):
                output = root / label
                checkpoints = root / ('split_checkpoints' if split else 'full_checkpoints')
                config = {
                    'material_database': str(material_file),
                    'simulation': {'physics': 'dem', 'time': {
                        'start': 0., 'stop': stop, 'output_step': 2e-5, 'dt': 2e-5}},
                    'dem': {'damping_model': 'restitution'},
                    'domain': {'box_min': [-17.] * 3, 'box_max': [17.] * 3},
                    'clumps': {'instances': [{'id': 1}, {'id': 2}]},
                    'accelerations': {'self_gravity': {'enabled': True, 'method': method,
                        'gravity_update_rate': rate, 'direct': {'gravitational_constant': 1.},
                        'unified_fmm': {'tree_depth': 5, 'gravitational_constant': 1.,
                            'softening_factor': .001, 'enable_global_projection': False}}},
                    'io': {'input': {'particle_file': str(particle_file)},
                        'output': {'directory': str(output), 'initial_state': True},
                        'checkpoint': {'enabled': True, 'interval': 2, 'path': str(checkpoints),
                                       'force_fresh': label != 'resumed'}},
                    'gpu': {'device_id': 0}}
                cfg = root / f'{label}.yaml'
                cfg.write_text(json.dumps(config, indent=2))
                with (root / f'{label}.log').open('w') as log:
                    subprocess.run([str(exe), str(cfg)], stdout=log,
                                   stderr=subprocess.STDOUT, check=True, timeout=120)
                particles = read_particles(sorted(output.glob('output_step_*.bin'))[-1])
                clumps = read_clumps(sorted(output.glob('clump_states_step_*.bin'))[-1])
                expected_step = round(stop / 2e-5)
                for endpoint in (particles, clumps):
                    assert endpoint['step'] == expected_step
                    assert abs(endpoint['time'] - stop) <= 1e-14
                return particles, clumps
            full_p, full_c = run('full', 8e-5)
            run('interrupted', 4e-5, True)
            resumed_p, resumed_c = run('resumed', 8e-5, True)
            assert 'Resuming from latest checkpoint' in (root / 'resumed.log').read_text()
            dp, dc = compare_endpoints(full_p, resumed_p), compare_endpoints(full_c, resumed_c)
            # The serialized body offsets are FP32, so rebuilt inertias/moments
            # can differ at that precision. Compare dynamical state and wrench,
            # reporting all coefficient differences without demanding identity.
            state_pass = max(dp[:3]) <= 1e-9 and max(dp[3:]) <= 1e-9
            state_pass &= max(dc[:13]) <= 1e-8
            wrench_scale = max(1., *(abs(x) for row in full_c['rows'].values() for x in row[-6:]))
            state_pass &= max(dc[-6:]) <= 1e-7 * wrench_scale
            results.append({'method': method, 'gravity_update_rate': rate,
                'particle_component_max_abs_differences': dp,
                'clump_component_max_abs_differences': dc, 'pass': bool(state_pass)})
    report = {'executable': str(exe), 'executable_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
              'large': args.large, 'results': results}
    (work / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if all(row['pass'] for row in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
