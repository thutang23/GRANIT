#!/usr/bin/env python3
"""Actual-signal checkpoint-stop regression; all solver results require CUDA.

Reuses the DEM contact fixture and semantic contact-history oracle. CPU unit
tests exercise only the signal harness and its rejection paths, never claim
solver/GPU success. Put --work on the linked large-data storage directory.
"""
from __future__ import annotations
import argparse
import csv
import dataclasses
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
DEM_DT = 2.0**-24  # Exact binary endpoint controls avoid accidental residual steps.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'dem_multigpu_clumps'))
spec = importlib.util.spec_from_file_location('clump_validation', Path(__file__).resolve().parents[1]/'dem_multigpu_clumps/validate.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
import extension_validation as ext


def read_checkpoint(path):
    """Read every descriptor without assuming the old F32 force/torque ABI."""
    path = path.resolve(strict=True)
    blob = path.read_bytes()
    magic, version, hlen, n, step, t, dt, nf, physics, kind, stride, contacts, mh, ch = base.HEADER.unpack_from(blob)
    assert (magic, version, hlen, kind) == (b'GRANIT\0\0', 3, 72, 3)
    assert n > 0 and math.isfinite(t) and math.isfinite(dt)
    start = 72+16*nf; payload = start+52*n
    coords = struct.unpack_from(f'<{6*n}d', blob, start)
    gids = struct.unpack_from(f'<{n}i', blob, start+48*n)
    assert len(set(gids)) == n
    fields = {}; spans = []
    for i in range(nf):
        fid, dtype, count, reserved, offset = base.DESC.unpack_from(blob, 72+16*i)
        assert fid not in fields and reserved == 0 and count > 0
        fmt = base.TYPES[dtype]; size = struct.calcsize(fmt)*n*count
        vals = struct.unpack_from(f'<{n*count}{fmt}', blob, payload+offset)
        assert all(math.isfinite(v) for v in vals)
        fields[fid] = {gid: tuple(vals[j*n+i] for j in range(count)) for i, gid in enumerate(gids)}
        spans.append((offset, offset+size))
    end = 0
    for a, b in sorted(spans):
        assert a == end
        end = b
    end += payload
    histories = {}
    if physics == 1 and len(blob) > end:
        assert len(blob) == end+n*stride*ext.CONTACT.size
        for i, gid in enumerate(gids):
            for slot in range(stride):
                partner, cache, active, seen, *spring = ext.CONTACT.unpack_from(blob, end+(i*stride+slot)*ext.CONTACT.size)
                if active:
                    assert slot < fields[0x17][gid][0] and partner in fields[0x17]
                    assert (gid, partner) not in histories
                    histories[gid, partner] = (bool(active), bool(seen), *spring)
    else:
        assert len(blob) == end
    return {'path': str(path), 'step': step, 'time': t, 'dt': dt, 'physics': physics,
            'kind': kind, 'n': n, 'material_fingerprint': mh, 'clump_fingerprint': ch,
            'rows': {gid: tuple(coords[j*n+i] for j in range(6)) for i, gid in enumerate(gids)},
            'fields': fields, 'histories': histories}


def exact_state(a, b):
    keys = ('step', 'time', 'dt', 'physics', 'kind', 'n', 'material_fingerprint',
            'clump_fingerprint', 'rows', 'fields')
    different = [key for key in keys if a[key] != b[key]]
    history = ext.semantic_history_compare(a, b)
    return {'pass': not different and history['pass'], 'different': different,
            'semantic_contact_history': history}


def observe_step(work):
    steps = []
    for path in (work/'output').glob('output_step_*.bin'):
        match = re.fullmatch(r'output_step_(\d+)\.bin', path.name)
        if match and path.stat().st_size >= 72:
            steps.append(int(match[1]))
    return max(steps, default=-1)


def launch(binary, work, timeout, requested_signal=None, after_step=4, env=None):
    """Signal only after visible solver progress; kill a timed-out child group."""
    command = [str(binary), str(work/'config.json')]
    started = time.monotonic(); sent = None; observed = None
    with (work/'run.log').open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        try:
            while proc.poll() is None:
                now = time.monotonic()
                if now-started > timeout:
                    raise TimeoutError(f'{command} exceeded {timeout}s')
                if requested_signal is not None and sent is None:
                    observed = observe_step(work)
                    if observed >= after_step:
                        os.kill(proc.pid, requested_signal)
                        sent = time.monotonic()
                time.sleep(.002)
            result = {'command': command, 'pid': proc.pid, 'returncode': proc.returncode,
                      'config_sha256': base.digest(work/'config.json'),
                      'elapsed_seconds': time.monotonic()-started,
                      'signal': requested_signal, 'signal_after_observed_step': observed,
                      'signal_to_exit_seconds': None if sent is None else time.monotonic()-sent}
            if requested_signal is not None and sent is None:
                raise AssertionError(f'process exited before requested signal: {result}')
            return result
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)


def prepare_dem(work, ranks, enabled, resume=None, stop_time=None):
    work = work.resolve()
    ext.prepare(work, 'restart-contact', ranks, steps=200000)
    cfg = json.loads((work/'config.json').read_text())
    # Retain overlap and tangential memory over the signal-observation window.
    cfg['simulation']['time'].update(dt=DEM_DT, output_step=4*DEM_DT)
    cfg['simulation']['time']['stop'] = 200000*DEM_DT if stop_time is None else stop_time
    # Distributed rigid clumps currently support only k=1; unclumped DEM and
    # SPH variants below exercise the supported k=4 closed macro paths.
    cfg['accelerations']['self_gravity'].update(enabled=True, gravity_update_rate=1 if ranks else 4)
    cfg['io']['checkpoint'].update(enabled=enabled, interval=1000000)
    if not enabled: cfg['io']['checkpoint'].pop('path')
    if resume:
        cfg['io']['checkpoint'].update(force_fresh=False, resume_from=str(resume), rebootstrap_forces=False)
    base.dump(work/'config.json', cfg)
    return cfg


def prepare_variant(work, ranks, enabled, resume=None, stop_time=None, variant='yorp1'):
    cfg = prepare_dem(work, ranks, enabled, resume, stop_time)
    work = work.resolve()
    rows = []
    for x in (-.03, 0., .03):
        for y in (-.03, 0., .03):
            for z in (-.03, 0., .03):
                # Adaptive energy removes rigid rotation. A small symmetric
                # strain supplies a known nonrigid residual for pause checks.
                strain = (.002*x, -.001*y) if variant == 'yorp-adaptive' else (0., 0.)
                rows.append(((x,y,z), (-.01*y+strain[0],.01*x+strain[1],0.), (0.,0.,.01), .002, 0))
    base.write_input(work/'input.bin', rows)
    cfg['clumps'] = {'instances': []}
    cfg['accelerations']['self_gravity']['gravity_update_rate'] = 4
    cfg['domain'] = {'box_min': [-1.]*3, 'box_max': [1.]*3}
    if variant in ('nonmorton-adaptive', 'morton-adaptive'):
        assert ranks > 0
        cfg['multigpu']['morton_ownership'] = variant == 'morton-adaptive'
        cfg['multigpu']['partition_depth'] = 2
        cfg['accelerations']['self_gravity']['unified_fmm'].update(tree_depth=2,
            adaptive_depth={'enabled': True, 'min_depth': 2, 'max_depth': 4,
                            'trial_interval_evaluations': 2, 'trial_warmup_evaluations': 0,
                            'trial_samples': 1, 'confirmation_trials': 1})
    else:
        mode = {'yorp1': 1, 'yorp3': 3, 'yorp-adaptive': 2}[variant]
        yorp = {'enabled': True, 'mode': mode, 'omega_init': .01,
                'time': {'start': 0., 'stop': 200000*DEM_DT},
                'increment': .001, 'dt_check': 4*DEM_DT, 'epsilon': .01,
                'mode_3': {'T_start': 5., 'T_inter': 3., 'T_final': 2.26,
                           'fast_spinup_time': 16*DEM_DT, 'slow_spinup_time': 32*DEM_DT}}
        if variant == 'yorp-adaptive':
            cfg['simulation']['termination'] = {'condition': 'adaptive_yorp'}
            yorp['adaptive'] = {'enabled': True, 'check_interval_seconds': 4*DEM_DT,
                'pause_specific_energy_j_per_kg': 1e-12, 'resume_specific_energy_j_per_kg': 1e-14,
                'pause_confirmation_checks': 1, 'resume_confirmation_checks': 1,
                'axis_ratio_drop_fraction': .01, 'epsilon_reduction_fraction_per_cycle': .1,
                'minimum_epsilon_fraction': .1, 'max_omega_rad_s': 100.}
        cfg['accelerations']['yorp'] = yorp
    base.dump(work/'config.json', cfg)
    base.dump(work/'manifest.json', {'fixture': variant, 'particles': len(rows),
                                   'input_sha256': base.digest(work/'input.bin')})
    return cfg


def sph_tools():
    sys.path.insert(0, str(ROOT/'src/plugins/scripts'))
    spec = importlib.util.spec_from_file_location('sph_replay_stop', ROOT/'components/tests/integration_test/sph_neighbor_validation/replay_sph_backends.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def ordered_sph(replay, path):
    data = replay.read_particle_binary(path)
    order = replay.np.argsort(data.global_id)
    for field in dataclasses.fields(data):
        value = getattr(data, field.name)
        if isinstance(value, replay.np.ndarray) and len(value) == data.num_particles:
            setattr(data, field.name, value[order])
    return data


def sidecar_state(path, suffix):
    """Compare defined fields while excluding native header ABI padding."""
    blob = Path(str(path)+suffix).read_bytes()
    if suffix == '.respa_sync':
        assert len(blob) >= 32
        return blob[:12]+blob[16:]
    if suffix == '.wall_history':
        assert len(blob) >= 48
        return blob[:44]+blob[48:]
    if suffix == '.single_clumps':
        assert len(blob) >= 56
        return blob[:20]+blob[24:]
    if suffix == '.distributed_clumps':
        # MGCLMP1/v3, MAX_MULTI_GPUS=8: the checksum fields derive from
        # physical payload bytes (including discarded cache/padding bytes).
        # Each source bundle's integrity is checked independently by resume.
        assert len(blob) >= 272 and blob[:8] == b'MGCLMP1\0'
        assert struct.unpack_from('<I', blob, 8)[0] == 3
        return blob[:56]+blob[72:244]+blob[248:]
    if suffix == '.adaptive_fmm_depth' and struct.unpack_from('<I', blob, 8)[0] == 4:
        # Header(40) + NonMorton prefix(40) + DomainParams bool offset(8).
        # This marker tells initialization to retain the restored boxes;
        # it is set on restore and is not an integrator/controller quantity.
        assert len(blob) > 88 and blob[88] in (0, 1)
        return blob[:88]+blob[89:]
    if suffix == '.yorp_runtime':
        header = struct.unpack_from('<8s4IqdQ', blob)
        magic, version, ranks, size, external, step, t, fingerprint = header
        assert magic == b'YORPRUN1' and version == 1 and size == 296 and external in (0, 1)
        assert len(blob) == 48+ranks*(size+8*external)
        rows = []
        for rank in range(ranks):
            offset = 48+rank*(size+8*external)
            row = struct.unpack_from('<4I3dq2d', blob, offset)
            last = struct.unpack_from('<7i4x8d', blob, offset+64)
            device = struct.unpack_from('<7i4x8d', blob, offset+160)
            state = struct.unpack_from('<4di4x', blob, offset+256)
            schedule = struct.unpack_from('<d', blob, offset+size) if external else ()
            rows.append((row, last, device, state, schedule))
        return header, rows
    return blob


def prepare_sph(work, ranks, enabled, resume=None, stop_time=None):
    work = work.resolve()
    replay = sph_tools(); work.mkdir(parents=True)
    source = work/'input.bin'; replay.fixture(source, 4)
    cfg = replay.config(work, source, ROOT/'components/config/materials_example.yaml', 'cuda', 1, 200000, 4, enabled)
    cfg['accelerations']['self_gravity'].update(enabled=True, method='unified_fmm', gravity_update_rate=4)
    cfg['accelerations']['self_gravity']['unified_fmm'].update(
        tree_depth=4, gravitational_constant=1., adaptive_depth={'enabled': False})
    cfg['multigpu'] = {'enabled': ranks > 0, 'max_gpus': max(1, ranks),
                       'morton_ownership': True, 'halo_width_factor': 1.5, 'buffer_capacity_factor': 3.}
    cfg['io']['checkpoint']['interval'] = 1000000
    if not enabled: cfg['io']['checkpoint'].pop('path')
    if stop_time is not None: cfg['simulation']['time']['stop'] = stop_time
    if resume: cfg['io']['checkpoint'].update(force_fresh=False, resume_from=str(resume))
    base.dump(work/'config.json', cfg)
    return cfg


def checkpoint(work):
    paths = sorted([*(work/'checkpoints').glob('checkpoint_step_*.bin'),
                    *(work/'output/checkpoints').glob('checkpoint_step_*.bin')], key=lambda p: p.name)
    assert paths, f'no committed checkpoint: {work}'
    path = paths[-1]
    assert path.is_symlink(), f'checkpoint was not atomically published: {path}'
    resolved = path.resolve(strict=True)
    assert (resolved.parent/'COMPLETE').is_file(), 'missing bundle commit manifest'
    return path, resolved, read_checkpoint(path)


def verify_gpu_release(checks):
    """Read driver process ownership; other GPU users are left untouched."""
    pids = {row['interrupted']['pid'] for row in checks.values() if 'interrupted' in row}
    pids.update(run['pid'] for row in checks.values() for run in row['runs'])
    command = ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']
    remaining = set()
    for attempt in range(10):
        query = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
        active = {int(line.strip()) for line in query.stdout.splitlines() if line.strip().isdigit()}
        remaining = pids & active
        if not remaining: break
        time.sleep(.1)
    return {'pass': not remaining, 'command': command, 'solver_pids': sorted(pids),
            'solver_pids_still_on_gpu': sorted(remaining)}


def audit_rank_diagnostics(work):
    result = {}
    for path in sorted((work/'output').glob('adaptive_depth_rank_*/adaptive_fmm_depth.csv')):
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        sequences = [int(row['sequence']) for row in rows]
        origin = Path(str(path)+'.start')
        first = int(origin.read_text().strip()) if origin.exists() else 1
        assert sequences == list(range(first, first+len(sequences))), 'rank diagnostic sequence gap or duplicate'
        result[str(path)] = {'rows': len(rows), 'first_sequence': first,
                             'last_sequence': sequences[-1] if sequences else first-1}
    assert result and sum(row['rows'] for row in result.values()) >= 2, 'capacity-service diagnostic backlog was not persisted'
    return result


def run_case(binary, root, physics, ranks, enabled, sig, timeout, env, variant='contact'):
    name = f'{physics}-{variant}-r{ranks}-enabled{int(enabled)}-{signal.Signals(sig).name}'
    if variant == 'contact':
        prepare = prepare_dem if physics == 'dem' else prepare_sph
    else:
        prepare = lambda *a, **kw: prepare_variant(*a, **kw, variant=variant)
    work = root/name; prepare(work, ranks, enabled)
    if variant == 'nonmorton-adaptive' and ranks > 1:
        # This released backend is forbidden by an existing physics/config
        # guard. Preserve that boundary instead of bypassing it for a test.
        result = launch(binary, work, timeout, env=env)
        diagnostic = 'Multi-GPU BFFMM requires multigpu.morton_ownership=true'
        assert result['returncode'] == 1 and diagnostic in (work/'run.log').read_text()
        return {'pass': True, 'expected_rejection': True, 'diagnostic': diagnostic,
                'coverage': 'configuration guard only; no checkpoint/runtime claim', 'runs': [result]}
    interrupted = launch(binary, work, timeout, sig, after_step=8 if variant != 'contact' else 4, env=env)
    assert interrupted['returncode'] == 0, interrupted
    assert interrupted['signal_to_exit_seconds'] < timeout, interrupted
    log = (work/'run.log').read_text()
    assert '[cooperative-stop]' in log, 'missing successful cooperative-stop receipt'
    alias, payload, record = checkpoint(work)
    receipt = re.search(r'\[cooperative-stop\] signal=(\d+) checkpoint=(\S+) step=(\d+) time=(\S+)', log)
    assert receipt and int(receipt[1]) == sig and int(receipt[3]) == record['step'], 'stop receipt disagrees with committed state'
    assert 0 < record['step'] < 200000 and record['step'] % 1000000 != 0
    macro_k = 1 if physics == 'dem' and variant == 'contact' and ranks > 0 else 4
    required = ['.respa_sync'] if macro_k > 1 else []
    if physics == 'dem':
        required += ['.wall_history']
        if ranks > 0: required += ['.distributed_clumps']
        elif variant == 'contact': required += ['.single_clumps']
    if variant.startswith('yorp'): required += ['.yorp_runtime']
    if variant in ('nonmorton-adaptive', 'morton-adaptive'): required += ['.adaptive_fmm_depth']
    for suffix in required:
        assert Path(str(payload)+suffix).is_file(), f'missing authoritative state {suffix}'
    diagnostics = audit_rank_diagnostics(work) if variant == 'nonmorton-adaptive' else None
    nonzero = None
    if physics == 'dem' and variant == 'contact':
        nonzero = sum(any(x != 0 for x in h[2:]) for h in record['histories'].values())
        assert nonzero > 0, 'stop checkpoint did not exercise active contact memory'
    yorp_observed = None
    if variant.startswith('yorp'):
        _, rows = sidecar_state(payload, '.yorp_runtime')
        yorp_observed = []
        for host, last, device, state, schedule in rows:
            if variant != 'yorp-adaptive' or ranks == 0:
                assert host[7] > 0, 'YORP runtime was never active before stop'
            if variant == 'yorp1': assert host[6] != .01, 'mode1 did not apply an increment'
            if variant == 'yorp3': assert host[8] >= 0, 'mode3 clock did not start'
            if variant == 'yorp-adaptive':
                assert host[1] == 1 and state[4] == 1 and device[6] == 0
                assert device[1] in (0, 2), 'adaptive fixture did not reach pause hysteresis'
                assert device[7] >= 4*DEM_DT, 'adaptive diagnostics did not advance'
                if ranks > 0:
                    # Distributed adaptive torque uses its collective path;
                    # the legacy manager update counter correctly stays zero.
                    assert schedule and schedule[0] > device[7]
            yorp_observed.append({'step_counter': host[7], 'accumulated_omega': host[6],
                                  'mode3_start_time': host[8], 'adaptive_state': device[1],
                                  'baseline_recorded': state[4]})
    assert record['step'] % macro_k == 0, 'stop checkpoint is inside an open RESPA macro'
    # Zero-step roundtrip catches restored forces/rates/body state being rebuilt
    # from rounded output fields instead of restored from the checkpoint.
    zero = root/(name+'-roundtrip'); prepare(zero, ranks, True, alias, record['time'])
    result = launch(binary, zero, timeout, env=env); assert result['returncode'] == 0, result
    za, zp, zr = checkpoint(zero)
    exact = exact_state(record, zr); assert exact['pass'], exact
    sidecar_exact = {}
    for suffix in required:
        sidecar_exact[suffix] = sidecar_state(payload, suffix) == sidecar_state(zp, suffix)
    assert all(sidecar_exact.values()), sidecar_exact
    # Short continuation shares the same final endpoint with an uninterrupted
    # control. This also proves a stop checkpoint is actually resumable.
    dt = DEM_DT if physics == 'dem' else sph_tools().DT
    endpoint = record['time']+32*dt
    resumed = root/(name+'-resume'); prepare(resumed, ranks, True, alias, endpoint)
    control = root/(name+'-control'); prepare(control, ranks, True, stop_time=endpoint)
    runs = [launch(binary, d, timeout, env=env) for d in (resumed, control)]
    assert all(r['returncode'] == 0 for r in runs), runs
    ca, cp, cr = checkpoint(control); ra, rp, rr = checkpoint(resumed)
    if physics == 'dem':
        continuation = ext.compare_records(cr, rr)
    else:
        replay = sph_tools()
        continuation = replay.compare_pair(ordered_sph(replay, cp), ordered_sph(replay, rp))
    assert continuation['pass'], continuation
    return {'pass': True, 'interrupted': interrupted, 'stop_step': record['step'],
            'atomic_payload': str(payload), 'macro_inner_count': macro_k,
            'closed_macro': record['step'] % macro_k == 0,
            'nonzero_active_spring_contacts': nonzero,
            'yorp_observed': yorp_observed,
            'rank_diagnostics': diagnostics,
            'roundtrip_excludes': ['contact partner-local cache', 'inactive contact slots',
                                  'native sidecar header ABI padding', 'derived distributed integrity hashes',
                                  'NonMortonDepthCheckpoint.ownership.restore_morton_partition at .adaptive_fmm_depth byte 88'],
            'roundtrip': exact, 'sidecars_exact': sidecar_exact,
            'continuation': continuation, 'runs': [result]+runs}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dem-binary', type=Path)
    ap.add_argument('--sph-binary', type=Path)
    ap.add_argument('--multi-binary', type=Path)
    ap.add_argument('--ranks', nargs='+', type=int, default=[1, 2])
    ap.add_argument('--variants', nargs='+', choices=['contact', 'yorp1', 'yorp3', 'yorp-adaptive', 'nonmorton-adaptive', 'morton-adaptive'], default=['contact'])
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--timeout', type=float, default=120)
    args = ap.parse_args(); args.work = args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=False)
    assert args.dem_binary or args.sph_binary or args.multi_binary, 'provide at least one actual solver'
    # Keep every subprocess on the exact same executable if another terminal
    # relinks the build tree while a multi-case regression is running.
    binaries = args.work/'binaries'; binaries.mkdir()
    sources = {}; hashes = {}; resources = {}
    for attr in ('dem_binary', 'sph_binary', 'multi_binary'):
        source = getattr(args, attr)
        if source is None: continue
        source = source.resolve(strict=True); frozen = binaries/attr
        shutil.copy2(source, frozen)
        sources[attr] = str(source); hashes[str(frozen)] = base.digest(frozen)
        table = source.parent.parent/'share/granit/fmm/g_table_v1.bin'
        if table.is_file():
            target = args.work/'share/granit/fmm/g_table_v1.bin'
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert base.digest(target) == base.digest(table), 'selected solver builds use different G tables'
            else:
                shutil.copy2(table, target)
            resources[str(target)] = base.digest(target)
        setattr(args, attr, frozen)
    report = {'pass': False, 'checks': {},
              'binary_sources': sources, 'binary_sha256': hashes,
              'resource_sha256': resources,
              'limitation': 'Tiny correctness fixtures only; no production or performance claim.'}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
    try:
        cases = [(args.dem_binary, 'dem', 0), (args.sph_binary, 'sph', 0)]
        cases += [(args.multi_binary, p, r) for r in args.ranks for p in ('dem', 'sph')]
        for variant in args.variants:
            for binary, physics, ranks in cases:
                if binary is None or (variant != 'contact' and physics != 'dem'): continue
                if variant == 'nonmorton-adaptive' and ranks == 0: continue
                if variant == 'morton-adaptive' and ranks < 2: continue
                signals = ((True, signal.SIGUSR1), (False, signal.SIGTERM)) if variant == 'contact' else ((False, signal.SIGUSR1),)
                for enabled, sig in signals:
                    key = f'{physics}-{variant}-r{ranks}-enabled{int(enabled)}'
                    report['active_case'] = key
                    report['checks'][key] = run_case(binary.resolve(), args.work, physics, ranks, enabled, sig, args.timeout, env, variant)
                    base.dump(args.work/'report.json', report)
        assert report['checks'], 'selected binaries/variants produced no runnable GPU checks'
        report['gpu_release'] = verify_gpu_release(report['checks'])
        assert report['gpu_release']['pass'], report['gpu_release']
        report['pass'] = True
    except Exception as failure:
        report['failure'] = f'{type(failure).__name__}: {failure}'
        raise
    finally:
        base.dump(args.work/'report.json', report)
    print(json.dumps({'pass': report['pass'], 'report': str(args.work/'report.json')}))


if __name__ == '__main__': main()
