#!/usr/bin/env python3
"""Bounded native DEM benchmark; defaults to PREPARE ONLY, never GPU execution.

Examples (all large artifacts must live under ~/data):
  python3 benchmark.py --root ~/data/clump-bench-NEW --counts 10000
  python3 benchmark.py run --root ~/data/clump-bench-NEW --binary /abs/granit_multigpu --devices 0 1 2 3
  python3 benchmark.py verify --root ~/data/clump-bench-NEW

Clumps versus spheres is a kernel/workload comparison: removing rigid membership
changes gravity exclusions and dynamics. Dense-contact runs are clump-only health
checks; neither fixture establishes physical convergence or force accuracy.
"""
from __future__ import annotations
import argparse
from array import array
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import mmap
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import threading
import time

import validate as fixtures

DEPTHS = {10000: 4, 100000: 5, 1000000: 6}
DENSE_DEPTHS = {10000: 2, 100000: 3, 1000000: 5}
KINDS = ('clumps', 'spheres', 'dense-contact')
BODY = struct.Struct('<i45d')
BODY_HEADER = struct.Struct('<8sIiqd')
SCHEMA = {6: (2, 1), 0x10: (1, 3), 0x11: (0, 1), 0x12: (0, 1),
          0x13: (0, 1), 0x14: (0, 3), 0x15: (0, 3), 0x16: (1, 4),
          0x18: (2, 1), 0x1d: (4, 1)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dump(path, obj):
    # Caller creates a new campaign or unique attempt; evidence is never replaced.
    with path.open('x') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')


def data_root(path):
    path = path.expanduser().resolve()
    require(path.is_relative_to((Path.home()/'data').resolve()),
            'campaign root must be inside ~/data (symlink aliases are resolved)')
    return path


def sampled_ids(count, limit):
    return sorted({i*(count-1)//max(1, min(count, limit)-1)
                   for i in range(min(count, limit))})


class ParticleMap:
    """mmap metadata and zero-copy scalar views; no per-particle dictionaries."""
    def __init__(self, path, kind=2):
        self.path = path
        self.file = path.open('rb')
        try:
            self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            require(len(self.mm) >= 72, f'truncated header: {path}')
            h = fixtures.HEADER.unpack_from(self.mm)
            magic, version, hlen, self.n, self.step, self.time, self.dt, nf, physics, record, stride, contacts, _, _ = h
            require((magic, version, hlen, physics, record, stride, contacts) ==
                    (b'GRANIT\0\0', 3, 72, 1, kind, 0, 0), f'wrong record kind/format: {path}')
            require(self.n > 0 and self.step >= 0 and math.isfinite(self.time) and math.isfinite(self.dt),
                    f'invalid count/time: {path}')
            self.base = 72+16*nf
            self.fields = {}
            end = self.base+52*self.n
            self.fbase = end
            spans = []
            require(end <= len(self.mm), f'truncated fixed arrays: {path}')
            for i in range(nf):
                fid, dtype, components, reserved, off = fixtures.DESC.unpack_from(self.mm, 72+16*i)
                require(fid not in self.fields and dtype in fixtures.TYPES and components > 0 and reserved == 0,
                        f'invalid descriptor: {path}, {fid}')
                fmt = fixtures.TYPES[dtype]
                size = struct.calcsize(fmt)
                if fid in SCHEMA:
                    require((dtype, components) == SCHEMA[fid], f'wrong field schema: {path}, {fid}')
                self.fields[fid] = (fmt, components, self.fbase+off)
                spans.append((off, off+size*components*self.n))
            cursor = 0
            for lo, hi in sorted(spans):
                require(lo == cursor, f'gapped/overlapping descriptors: {path}')
                cursor = hi
            require(self.fbase+cursor == len(self.mm), f'truncated/trailing bytes: {path}')
        except BaseException:
            self.close()
            raise

    def close(self):
        if hasattr(self, 'mm'):
            self.mm.close()
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def view(self, fid=None, component=0):
        if fid is None:
            fmt, offset = 'd', self.base+8*self.n*component
        elif fid == 'id':
            fmt, offset = 'i', self.base+48*self.n
        else:
            fmt, nc, offset = self.fields[fid]
            require(component < nc, 'component outside field')
            offset += component*self.n*struct.calcsize(fmt)
        return memoryview(self.mm)[offset:offset+self.n*struct.calcsize(fmt)].cast(fmt)

    def value(self, fid, row, component=0):
        with self.view(fid, component) as values:
            return values[row]


def geometry_digest(path):
    h = hashlib.sha256()
    with ParticleMap(path, 1) as source:
        h.update(struct.pack('<Q', source.n))
        # All stored geometry/kinematics, IDs and materials match; membership differs.
        for start in range(source.base, source.fbase, 1024*1024):
            h.update(source.mm[start:min(start+1024*1024, source.fbase)])
        for fid in sorted(source.fields):
            if fid == 0x18:
                continue
            fmt, count, offset = source.fields[fid]
            h.update(struct.pack('<HB', fid, count))
            length = count*source.n*struct.calcsize(fmt)
            for start in range(offset, offset+length, 1024*1024):
                h.update(source.mm[start:min(start+1024*1024, offset+length)])
    return h.hexdigest()


def rows(count, spacing, spheres=False):
    bodies = count//4
    side = math.ceil(bodies**(1/3))
    while side**3 < bodies:
        side += 1
    for body in range(bodies):
        xyz = (body % side, body//side % side, body//(side*side))
        center = tuple(spacing*(x-.5*(side-1)) for x in xyz)
        for offset in fixtures.OFFSETS:
            yield tuple(center[a]+offset[a] for a in range(3)), (0.,)*3, (0.,)*3, .01, 0 if spheres else body+1


def prepare(args):
    root = data_root(args.root)
    root.mkdir(parents=True, exist_ok=False)
    (root/'assets').mkdir()
    (root/'rows').mkdir()
    warmup, measure = args.warmup, args.measure
    require(warmup >= 0 and measure > 0, 'invalid timing window')
    steps = warmup+measure
    kinds = ['clumps', 'spheres']+(['dense-contact'] if args.dense_contact else [])
    assets, plan = {}, []
    for count in args.counts:
        for kind in kinds:
            asset = root/'assets'/f'{count}-{kind}'
            asset.mkdir()
            n, k, normals = fixtures.write_input(asset/'input.bin', rows(count, .026 if kind == 'dense-contact' else .1, kind == 'spheres'))
            require(n == count and k == (0 if kind == 'spheres' else count//4), 'generator returned incorrect population')
            material = {'version': 1, 'materials': [{'id': 0, 'name': 'benchmark', 'dem': {
                'density': 1000., 'youngs_modulus': 1e6, 'poisson_ratio': .25}}],
                'dem_interactions': [{'materials': [0, 0], 'restitution': .8,
                    'friction': {'static': .4, 'rolling': .03, 'twisting': .02},
                    'shape_parameter': 1., 'cohesion': {'model': 'none'}}]}
            dump(asset/'materials.json', material)
            identity = {'particles': n, 'clumps': k, 'normal_rows': normals,
                'input_sha256': fixtures.digest(asset/'input.bin'),
                'geometry_sha256_excluding_membership': geometry_digest(asset/'input.bin'),
                'materials_sha256': fixtures.digest(asset/'materials.json')}
            assets[f'{count}-{kind}'] = identity
            depth = (DENSE_DEPTHS if kind == 'dense-contact' else DEPTHS)[count]
            for ranks in args.ranks:
                row = root/'rows'/f'{count}-{kind}-gpu{ranks}'
                row.mkdir()
                cfg = fixtures.make_config(row, 'throughput', k, ranks, True, steps)
                dt = cfg['simulation']['time']['dt']
                # DEM clips its last timestep to the stop boundary. Preserve
                # the requested duration and dt up to floating-point rounding.
                # Follow runtime repeated addition, then move one ULP inward:
                # an N*dt product can leave a tiny (N+1)th residual step.
                stop = math.nextafter(sum(dt for _ in range(steps)), -math.inf)
                cfg['simulation']['time'].update(stop=stop, output_step=(steps+1)*dt)
                cfg['domain']['bounds'] = 'auto'
                cfg['material_database'] = str(asset/'materials.json')
                cfg['io']['input']['particle_file'] = str(asset/'input.bin')
                cfg['io']['output'].update(fields=[], scalar_interval=0,
                    timing={'warmup_steps': warmup, 'measure_steps': measure},
                    timing_receipt_path=str(row/'timing.json'))
                cfg['accelerations']['self_gravity']['unified_fmm']['tree_depth'] = depth
                cfg['multigpu'].update(partition_depth=min(4, depth), buffer_capacity_factor=1.5)
                dump(row/'config.json', cfg)
                plan.append({'name': row.name, 'count': count, 'kind': kind, 'ranks': ranks,
                    'depth': depth, 'asset': str(asset.relative_to(root)),
                    'config_sha256': fixtures.digest(row/'config.json')})
        require(assets[f'{count}-clumps']['geometry_sha256_excluding_membership'] ==
                assets[f'{count}-spheres']['geometry_sha256_excluding_membership'], 'baseline geometry differs')
    manifest = {'schema': 'granit-clump-benchmark/v1', 'prepared_only': True,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'root': str(root),
        'binary': str(args.binary.resolve()) if args.binary else None,
        'binary_sha256': fixtures.digest(args.binary) if args.binary else None,
        'warmup_steps': warmup, 'measure_steps': measure, 'completed_steps_expected': steps,
        'process_timeout_seconds': min(args.timeout, 120), 'assets': assets, 'rows': plan,
        'scope': ['native full DEM timing, sparse initial/final output',
            'clump/sphere comparison changes rigid constraints and same-clump gravity exclusion',
            'dense-contact is intentionally preloaded clump-only structural health',
            'no physical convergence or independent force-accuracy claim',
            'ordinary timing rows disable correctness diagnostics; dense health rows enable contact diagnostics',
            'normal output omits member quaternion; all clump body quaternions are checked']}
    dump(root/'manifest.json', manifest)
    print(json.dumps({'prepared': str(root), 'rows': len(plan), 'gpu_executed': False}, indent=2))


def validate_particles(path, count, kind, step, expected_time, body_samples):
    clumps = kind != 'spheres'
    wanted = {4*cid+a for cid in body_samples for a in range(4)} if clumps else set()
    indices = {}
    with ParticleMap(path) as p:
        require(p.n == count and p.step == step and abs(p.time-expected_time) <= 1e-12,
                f'wrong endpoint/count: {path}')
        require({6, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x18, 0x1d} <= p.fields.keys(), 'missing required DEM fields')
        seen = bytearray(count)
        with p.view('id') as ids, p.view(0x18) as membership:
            for row, (gid, cid) in enumerate(zip(ids, membership)):
                require(0 <= gid < count and not seen[gid], f'missing/duplicate/global ID: {path}, {gid}')
                seen[gid] = 1
                require(cid == (gid//4+1 if clumps else 0), f'changed membership: {path}, {gid}')
                if gid in wanted:
                    indices[gid] = row
        require(all(seen), 'incomplete global ID coverage')
        # Scan every stored numeric array, including forces and any extra fields.
        for fid in [None]+list(p.fields):
            components = 6 if fid is None else p.fields[fid][1]
            for component in range(components):
                with p.view(fid, component) as values:
                    require(all(math.isfinite(v) for v in values), f'nonfinite field: {path}, {fid}, {component}')
                    if fid in (0x11, 0x12, 0x13):
                        require(all(v > 0 for v in values), f'nonpositive radius/mass/inertia: {path}')
                    if fid == 0x1d:
                        require(all(v == 0 for v in values), f'inactive output row: {path}')
        if 0x16 in p.fields:
            for row in range(count):
                require(abs(sum(p.value(0x16, row, a)**2 for a in range(4))-1) <= 2e-10, 'member quaternion norm')
        samples = {gid: {'pos': tuple(p.value(None, row, a) for a in range(3)),
                          'vel': tuple(p.value(None, row, a+3) for a in range(3)),
                          'omega': tuple(p.value(0x10, row, a) for a in range(3)),
                          'mass': p.value(0x12, row)} for gid, row in indices.items()}
        return {'n': p.n, 'step': p.step, 'time': p.time, 'all_fields_finite': True,
                'all_ids_and_membership_checked': True, 'member_quaternion_stored': 0x16 in p.fields, 'sha256': fixtures.digest(path)}, samples


def rotate(q, vector):
    axis = q[1:]
    t = tuple(2*x for x in fixtures.cross(axis, vector))
    c = fixtures.cross(axis, t)
    return tuple(vector[a]+q[0]*t[a]+c[a] for a in range(3))


def validate_bodies(path, count, step, expected_time, samples):
    require(path.stat().st_size == 32+364*count, f'body count/length mismatch: {path}')
    worst_pos = worst_velocity = worst_distance = worst_spin = 0.
    with path.open('rb') as file:
        mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            magic, version, n, actual_step, actual_time = BODY_HEADER.unpack_from(mm)
            require((magic, version, n, actual_step) == (b'CLUMP\0\0\0', 2, count, step) and
                    abs(actual_time-expected_time) <= 1e-12, f'invalid body header: {path}')
            for index in range(count):
                cid, *v = BODY.unpack_from(mm, 32+364*index)
                require(cid == index+1 and all(math.isfinite(x) for x in v) and v[13] > 0, 'invalid/nonfinite body')
                q = v[6:10]
                require(abs(sum(x*x for x in q)-1) <= 2e-10, 'body quaternion norm')
                if 4*index not in samples:
                    continue
                world_spin = rotate(q, v[10:13])
                member_positions = []
                for a, offset in enumerate(fixtures.OFFSETS):
                    member = samples[4*index+a]
                    # Body offsets are stored FP32 by GRANIT, including initial sync.
                    offset = tuple(struct.unpack('<f', struct.pack('<f', x))[0] for x in offset)
                    expected_offset = rotate(q, offset)
                    lever = tuple(member['pos'][b]-v[b] for b in range(3))
                    spin_velocity = fixtures.cross(world_spin, lever)
                    worst_pos = max(worst_pos, max(abs(lever[b]-expected_offset[b]) for b in range(3)))
                    worst_velocity = max(worst_velocity, max(abs(member['vel'][b]-v[3+b]-spin_velocity[b]) for b in range(3)))
                    worst_spin = max(worst_spin, max(abs(member['omega'][b]-world_spin[b]) for b in range(3)))
                    require(abs(member['mass']*4-v[13]) <= 1e-6*v[13], 'member/body mass mismatch')
                    member_positions.append(member['pos'])
                for a, b in itertools.combinations(range(4), 2):
                    expected = math.dist(fixtures.OFFSETS[a], fixtures.OFFSETS[b])
                    worst_distance = max(worst_distance, abs(math.dist(member_positions[a], member_positions[b])-expected))
        finally:
            mm.close()
    require(max(worst_pos, worst_velocity, worst_spin, worst_distance) <= 2e-8,
            f'sampled rigid state mismatch: {path}: {worst_pos}, {worst_velocity}, {worst_spin}, {worst_distance}')
    return {'all_body_rows_finite': True, 'all_body_quaternions_checked': True,
            'body_count': count, 'sampled_bodies': len(samples)//4,
            'max_pose_error': worst_pos, 'max_velocity_error': worst_velocity,
            'max_spin_error': worst_spin, 'max_pair_distance_error': worst_distance,
            'sha256': fixtures.digest(path)}


def validate_attempt(root, manifest, row, attempt, sample_count):
    cfg = json.loads((attempt/'config.json').read_text())
    if (attempt/'launch.json').exists():
        launch = json.loads((attempt/'launch.json').read_text())
        require(fixtures.digest(attempt/'config.json') == launch['config_sha256'], 'attempt config hash changed')
        require(fixtures.digest(Path(cfg['io']['input']['particle_file'])) == launch['input_sha256'] and
                fixtures.digest(Path(cfg['material_database'])) == launch['materials_sha256'], 'attempt input/material hash changed')
    dt = cfg['simulation']['time']['dt']
    steps = manifest['completed_steps_expected']
    timing = json.loads((attempt/'timing.json').read_text())
    require(timing['compute_window_completed'] and timing['completed_steps'] == steps and
            timing['start_step'] == 0 and timing['end_step'] == steps and
            timing['compute_window_start_step'] == manifest['warmup_steps'] and
            timing['compute_window_end_step']-timing['compute_window_start_step'] == manifest['measure_steps'] and
            timing['configured_warmup_steps'] == manifest['warmup_steps'] and
            timing['configured_measure_steps'] == manifest['measure_steps'] and
            timing['num_gpus'] == row['ranks'] and timing['live_particles'] == row['count'], 'native timing receipt incomplete or population/window mismatch')
    seconds = timing['compute_window_wall_seconds']
    require(math.isfinite(seconds) and seconds > 0, 'invalid native compute time')
    out = attempt/'output'
    files = sorted(out.glob('output_step_*.bin'))
    require([p.name for p in files] == [f'output_step_{s:010d}.bin' for s in (0, steps)], 'expected exactly initial and final particle files')
    sample = sampled_ids(row['count']//4, sample_count) if row['kind'] != 'spheres' else []
    frames = []
    for path, step in zip(files, (0, steps)):
        report, members = validate_particles(path, row['count'], row['kind'], step, step*dt, sample)
        if row['kind'] != 'spheres':
            report['bodies'] = validate_bodies(out/f'clump_states_step_{step:010d}.bin', row['count']//4, step, step*dt, members)
        frames.append(report)
    dense_evidence = None
    if row['kind'] == 'dense-contact':
        with ParticleMap(Path(cfg['io']['input']['particle_file']), 1) as source:
            first = [tuple(source.value(None, i, a) for a in range(3)) for i in range(4)]
            second = [tuple(source.value(None, i+4, a) for a in range(3)) for i in range(4)]
            external_distance = min(math.dist(a, b) for a in first for b in second)
            require(external_distance < .02, 'dense fixture lacks intended external overlap')
            dense_evidence = {'initial_external_pair_min_distance': external_distance,
                'radius_sum': .02, 'initial_external_overlap': .02-external_distance,
                'remote_contact_diagnostics': fixtures.ownership_diagnostics(attempt),
                'scope': 'preloaded clump-only structural health; diagnostic overhead included'}
    body_files = list(out.glob('clump_states_step_*.bin'))
    require(len(body_files) == (2 if sample else 0), 'unexpected body output frequency')
    return {'pass': True, 'native_compute_seconds': seconds,
            'native_ms_per_step': 1000*seconds/manifest['measure_steps'],
            'constituents_per_second': row['count']*manifest['measure_steps']/seconds,
            'frames': frames, 'dense_contact_evidence': dense_evidence, 'force_accuracy_tested': False, 'physical_convergence_tested': False}


def gpu_snapshot():
    command = ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version',
               '--format=csv,noheader,nounits']
    result = subprocess.run(command, capture_output=True, text=True, timeout=5)
    require(result.returncode == 0, 'nvidia-smi unavailable: '+result.stderr.strip())
    return [dict(zip(('index', 'uuid', 'name', 'total_mib', 'used_mib', 'utilization_percent', 'driver_version'),
                     (part.strip() for part in line.split(','))))
            for line in result.stdout.splitlines() if line.strip()]


def run_rows(args):
    root = data_root(args.root)
    manifest = json.loads((root/'manifest.json').read_text())
    binary = (args.binary or Path(manifest['binary'] or '')).expanduser().resolve()
    require(binary.is_file(), '--binary must name an existing granit_multigpu executable')
    binary_hash = fixtures.digest(binary)
    if manifest['binary_sha256']:
        require(binary_hash == manifest['binary_sha256'], 'binary differs from prepared campaign')
    devices = args.devices or [x.strip() for x in os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',') if x.strip()]
    results = []
    for row in manifest['rows']:
        if row['count'] not in args.counts or row['ranks'] not in args.ranks or row['kind'] not in args.kinds:
            continue
        require(len(devices) >= row['ranks'], 'insufficient selected GPU devices')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        attempt = root/'rows'/row['name']/('attempt-'+stamp)
        attempt.mkdir(exist_ok=False)
        cfg = json.loads((attempt.parent/'config.json').read_text())
        require(fixtures.digest(attempt.parent/'config.json') == row['config_sha256'], 'prepared config changed')
        asset = root/row['asset']
        identity = manifest['assets'][asset.name]
        require(fixtures.digest(asset/'input.bin') == identity['input_sha256'] and
                fixtures.digest(asset/'materials.json') == identity['materials_sha256'], 'prepared input/material changed')
        cfg['io']['output'].update(directory=str(attempt/'output'), timing_receipt_path=str(attempt/'timing.json'))
        dump(attempt/'config.json', cfg)
        command = [str(binary), str(attempt/'config.json')]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(devices[:row['ranks']]), GRANIT_CLUMP_DIAGNOSTICS='1' if row['kind'] == 'dense-contact' else '0')
        receipt = {'command': command, 'binary_sha256': binary_hash,
            'input_sha256': identity['input_sha256'], 'materials_sha256': identity['materials_sha256'],
            'config_sha256': fixtures.digest(attempt/'config.json'), 'environment': {
                k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'GRANIT_CLUMP_DIAGNOSTICS')},
            'timeout_seconds': min(args.timeout, manifest['process_timeout_seconds'], 120),
            'memory_scope': 'sampled total device usage, not exact process allocator highwater',
            'memory_samples': []}
        started = time.monotonic()
        proc = None
        try:
            available = gpu_snapshot()
            selected = []
            for token in devices[:row['ranks']]:
                matches = [gpu for gpu in available if gpu['index'] == token or gpu['uuid'].startswith(token)]
                require(len(matches) == 1, 'selected CUDA device cannot be matched to GPU identity: '+token)
                selected.append(matches[0])
            require(len({gpu['uuid'] for gpu in selected}) == row['ranks'], 'duplicate selected GPU')
            selected_uuids = {gpu['uuid'] for gpu in selected}
            receipt['gpu_identity_before'] = selected
            with (attempt/'run.log').open('x') as log:
                proc = subprocess.Popen(command, cwd=fixtures.ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                stop_monitor = threading.Event()
                def monitor():
                    while not stop_monitor.is_set():
                        try:
                            snapshot = [gpu for gpu in gpu_snapshot() if gpu['uuid'] in selected_uuids]
                            receipt['memory_samples'].append({'seconds': time.monotonic()-started, 'gpus': snapshot})
                        except Exception as error:
                            receipt.setdefault('monitor_errors', []).append(str(error))
                        stop_monitor.wait(.5)
                monitor_thread = threading.Thread(target=monitor, daemon=True)
                monitor_thread.start()
                try:
                    # GPU telemetry cannot extend the solver process deadline.
                    proc.wait(timeout=receipt['timeout_seconds'])
                except subprocess.TimeoutExpired:
                    receipt['timed_out'] = True
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=3)
                finally:
                    stop_monitor.set()
                    monitor_thread.join(timeout=6)
                    require(not monitor_thread.is_alive(), 'GPU telemetry worker did not stop')
            receipt.update(returncode=proc.returncode, timed_out=receipt.get('timed_out', False))
            receipt['process_and_monitor_seconds'] = time.monotonic()-started
            require(proc.returncode == 0 and not receipt['timed_out'], 'solver failed or timed out')
            result = validate_attempt(root, manifest, row, attempt, args.samples)
        except Exception as error:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=3)
            receipt.update(returncode=proc.returncode if proc else None,
                           timed_out=receipt.get('timed_out', False), error=str(error))
            result = {'pass': False, 'error': str(error)}
        receipt['attempt_including_verification_seconds'] = time.monotonic()-started
        receipt.setdefault('process_and_monitor_seconds', receipt['attempt_including_verification_seconds'])
        peaks = {}
        for sample in receipt['memory_samples']:
            for gpu in sample['gpus']:
                peaks[gpu['uuid']] = max(peaks.get(gpu['uuid'], 0), int(gpu['used_mib']))
        receipt['sampled_total_device_peak_mib'] = peaks
        dump(attempt/'launch.json', receipt)
        dump(attempt/'verification.json', result)
        results.append({'row': row['name'], 'attempt': str(attempt), **result})
        print(json.dumps({'row': row['name'], 'pass': result['pass'], 'attempt': str(attempt)}), flush=True)
        if not result['pass'] and not args.keep_going:
            break
    require(results, 'no rows selected')
    paired = []
    for count in args.counts:
        for ranks in args.ranks:
            matched = {kind: next((r for r in results if r['row'] == f'{count}-{kind}-gpu{ranks}' and r['pass']), None)
                       for kind in ('clumps', 'spheres')}
            if all(matched.values()):
                paired.append({'count': count, 'ranks': ranks,
                    'clump_over_sphere_native_time_ratio': matched['clumps']['native_ms_per_step']/matched['spheres']['native_ms_per_step'],
                    'scope': 'kernel/workload comparison; changed physical constraints and self-gravity exclusions'})
    summary_path = root/('run-report-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    dump(summary_path, {'binary_sha256': binary_hash, 'results': results, 'matched_pairs': paired})
    print(json.dumps({'report': str(summary_path), 'passed': all(r['pass'] for r in results)}), flush=True)
    return 0 if all(r['pass'] for r in results) else 1


def verify(args):
    root = data_root(args.root)
    manifest = json.loads((root/'manifest.json').read_text())
    results = []
    for row in manifest['rows']:
        for attempt in sorted((root/'rows'/row['name']).glob('attempt-*')):
            try:
                launch = json.loads((attempt/'launch.json').read_text())
                require(launch['returncode'] == 0 and not launch.get('timed_out'), 'unsuccessful launch')
                result = validate_attempt(root, manifest, row, attempt, args.samples)
            except Exception as error:
                result = {'pass': False, 'error': str(error)}
            results.append({'row': row['name'], 'attempt': str(attempt), **result})
    print(json.dumps({'results': results, 'pass': bool(results) and all(r['pass'] for r in results)}, indent=2))
    return 0 if results and all(r['pass'] for r in results) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', nargs='?', choices=('prepare', 'run', 'verify'), default='prepare')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--counts', type=int, nargs='+', choices=tuple(DEPTHS), default=list(DEPTHS))
    parser.add_argument('--ranks', type=int, nargs='+', choices=(1, 2, 4), default=[1, 2, 4])
    parser.add_argument('--kinds', nargs='+', choices=KINDS, default=list(KINDS))
    parser.add_argument('--devices', nargs='+', help='physical CUDA indices or GPU UUIDs; first N used for N-rank rows')
    parser.add_argument('--dense-contact', action='store_true', help='prepare additional preloaded clump-only health rows')
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--measure', type=int, default=16)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--samples', type=int, default=1024, help='deterministically sampled bodies for rigid geometry checks; all rows scanned for finiteness')
    parser.add_argument('--keep-going', action='store_true')
    args = parser.parse_args()
    require(sys.byteorder == 'little' and 0 < args.samples <= 4096 and 0 < args.timeout <= 120,
            'requires little endian, 1..4096 sampled bodies and timeout in (0,120]')
    if args.action == 'prepare':
        prepare(args)
        return 0
    return run_rows(args) if args.action == 'run' else verify(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'benchmark: {error}', file=sys.stderr)
        raise SystemExit(1)
