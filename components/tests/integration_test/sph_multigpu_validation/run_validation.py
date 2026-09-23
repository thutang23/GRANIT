#!/usr/bin/env python3
"""Retained, numerical four-GPU SPH gravity, halo and restart regressions.

Use the existing SPH backend replay's fixed field tolerances for nongravitational
partition invariance and restart. Gravity is checked separately against direct
pair-mean spline summation: relative RMS <= 1%, max absolute error <= 5% of the
reference RMS force. Those approximation gates are fixed before execution.
"""
from __future__ import annotations
import argparse
import copy
import dataclasses
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src/plugins/scripts'))
from cudadem_pipeline.common.io import ParticleData, read_particle_binary, write_particle_binary
spec = importlib.util.spec_from_file_location('replay', ROOT / 'components/tests/integration_test/sph_neighbor_validation/replay_sph_backends.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def ordered(data):
    order = np.argsort(data.global_id)
    for field in dataclasses.fields(data):
        value = getattr(data, field.name)
        if isinstance(value, np.ndarray) and len(value) == data.num_particles:
            setattr(data, field.name, value[order])
    return data


def total_force(data):
    # SPH checkpoints retain fast force and slow acceleration separately.
    return data.force + data.mass[:, None] * data.gravity_acceleration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--single', type=Path, required=True)
    parser.add_argument('--multi', type=Path, required=True)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    material = work / 'materials.yaml'
    material.write_bytes((ROOT / 'components/config/materials_example.yaml').read_bytes())
    input_path = work / 'input.bin'
    replay.fixture(input_path, 6)
    report = {'schema': 'granit-sph-multigpu-validation/v1',
              'commands': [], 'checks': {}, 'passed': False,
              'binaries': {str(p): replay.digest(p) for p in (args.single, args.multi)}}

    def save():
        (work / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    def config(name, ranks, *, mode=1, steps=8, gravity=None, resume=None, source=input_path, correction=True, rebootstrap=False, card=material, overrides=None):
        directory = work / name
        directory.mkdir()
        c = replay.config(directory, source, card, 'cuda', mode, steps, 1, True)
        c['neighbor'].update(verlet_buffer_enabled=False, update_frequency=1)
        c['sph_parameters']['use_kernel_correction'] = correction
        c['multigpu'] = {'enabled': ranks > 0, 'max_gpus': max(1, ranks),
                         'morton_ownership': True, 'halo_width_factor': 1.5,
                         'buffer_capacity_factor': 3.0, 'rebalance_min_interval': 1000}
        c['accelerations']['self_gravity'].update(enabled=gravity is not None,
                                                 method=gravity or 'direct')
        c['accelerations']['self_gravity']['direct'] = {'gravitational_constant': 1.0}
        c['accelerations']['self_gravity']['unified_fmm'].update(
            gravitational_constant=1.0, tree_depth=5, softening_factor=0.0,
            enable_global_projection=False, theta_far=0.8)
        if resume:
            c['io']['checkpoint'].update(force_fresh=False, resume_from=str(resume), rebootstrap_forces=rebootstrap)
        if overrides:
            overrides(c)
        return directory, c

    def run(name, ranks, **kwargs):
        directory, c = config(name, ranks, **kwargs)
        path = directory / 'config.yaml'
        path.write_text(yaml.safe_dump(c, sort_keys=False))
        binary = args.multi if ranks else args.single
        command = [str(binary), str(path)]
        start = time.monotonic()
        with (directory / 'run.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    cwd=ROOT, timeout=300)
        report['commands'].append({'argv': command, 'returncode': result.returncode,
                                   'wall_seconds': time.monotonic()-start,
                                   'config_sha256': replay.digest(path)})
        save()
        if result.returncode:
            raise RuntimeError(f'{name} exited {result.returncode}: {directory / "run.log"}')
        snapshots = sorted((directory / 'checkpoints').glob('checkpoint_step_*.bin'))
        if not snapshots:
            raise RuntimeError(f'{name} produced no checkpoint')
        return snapshots[-1], ordered(read_particle_binary(snapshots[-1]))

    def reject(name, ranks, expected, **kwargs):
        directory, c = config(name, ranks, **kwargs)
        path = directory/'config.yaml'; path.write_text(yaml.safe_dump(c,sort_keys=False))
        binary = args.multi if ranks else args.single
        start = time.monotonic()
        with (directory/'run.log').open('w') as log:
            result = subprocess.run([str(binary),str(path)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,timeout=60)
        output = (directory/'run.log').read_text()
        passed = result.returncode != 0 and expected in output
        report['checks'][name] = {'pass':passed,'returncode':result.returncode,'expected_error':expected}
        report['commands'].append({'argv':[str(binary),str(path)],'returncode':result.returncode,
            'wall_seconds':time.monotonic()-start,'expected_rejection':True,'config_sha256':replay.digest(path)})
        save()
        if not passed: raise AssertionError(f'{name}: expected rejection {expected}')
        print(f'PASS {name}',flush=True)

    def parity(name, reference, candidate):
        ref, got = ordered(copy.deepcopy(reference)), ordered(copy.deepcopy(candidate))
        check = replay.compare_pair(ref, got)
        for field, floor in {'energy_cold':1e-7, 'energy_thermal_peak':1e-7,
                             'pressure_peak':1e-2, 'fracture_area':1e-10,
                             'fragment_length':1e-12, 'damping_energy':1e-9}.items():
            a, b = getattr(ref, field), getattr(got, field)
            if a is None and b is None: continue
            if a is None or b is None: raise AssertionError(f'missing history {field}')
            atol = max(floor, replay.ROUNDOFF * float(np.max(np.abs(a))))
            difference = np.abs(b-a)
            ok = bool(np.isfinite(a).all() and np.isfinite(b).all() and
                      np.all(difference <= atol + replay.RTOL*np.abs(a)))
            check['fields'][field] = {'pass':ok,'atol':atol,
                'max_abs_difference':float(np.max(difference)),
                'max_gate_fraction':float(np.max(difference/(atol+replay.RTOL*np.abs(a))))}
            check['pass'] = check['pass'] and ok
        report['checks'][name] = check
        save()
        if not check['pass']:
            bad = {k:v for k,v in check['fields'].items() if not v['pass']}
            raise AssertionError(f'{name}: {bad}')
        print(f'PASS {name}', flush=True)

    try:
        # Both continuity and density summation exercise changed neighbor state.
        for mode in (0, 1):
            _, single = run(f'hydro-mode{mode}-single', 0, mode=mode)
            for ranks in (1, 2, 4):
                _, multi = run(f'hydro-mode{mode}-gpu{ranks}', ranks, mode=mode)
                parity(f'hydro-mode{mode}-gpu{ranks}', single, multi)

        _, standard_single = run('standard-sph-single', 0, correction=False)
        _, standard_multi = run('standard-sph-four-gpu', 4, correction=False)
        parity('standard-sph-single-vs-four-gpu', standard_single, standard_multi)

        fluid_card=work/'zero-tension-materials.yaml'
        table=yaml.safe_load(material.read_text());fluid=table['materials'][0]['sph']
        fluid['strength']={'model':'none'};fluid['damage']={'model':'none'}
        fluid['eos']['negative_pressure']='zero'
        fluid_card.write_text(yaml.safe_dump(table,sort_keys=False))
        fluid_input=work/'zero-tension-input.bin';fluid_data=read_particle_binary(input_path)
        fluid_data.lifecycle_state=None;fluid_data.velocities[:]=0;fluid_data.energy[:]=0
        fluid_data.density=np.where(fluid_data.positions[:,0]<0,.97*2700.,1.03*2700.).astype(np.float32)
        write_particle_binary(fluid_input,fluid_data,physics='sph',record_kind='input')
        _,fluid_single=run('zero-tension-single',0,card=fluid_card,source=fluid_input,correction=False)
        _,fluid_full=run('zero-tension-four-gpu',4,card=fluid_card,source=fluid_input,correction=False)
        parity('zero-tension-single-vs-four-gpu',fluid_single,fluid_full)
        if not (np.all(fluid_full.pressure>=0) and np.any(fluid_full.pressure==0) and
                np.all(fluid_full.c_sound[fluid_full.pressure==0]==1)):
            raise AssertionError('zero-tension pressure/sound branch was not preserved in distributed evolution')
        fluid_split,_=run('zero-tension-first',4,steps=4,card=fluid_card,source=fluid_input,correction=False)
        _,fluid_resume=run('zero-tension-resume',4,resume=fluid_split,card=fluid_card,source=fluid_input,correction=False)
        parity('zero-tension-four-gpu-continuation',fluid_full,fluid_resume)

        full_path, full = run('restart-full', 4)
        split_path, split = run('restart-first', 4, steps=4)
        _, resumed = run('restart-second', 4, resume=split_path)
        parity('continuous-vs-resumed-four-gpu', full, resumed)
        _, imported = run('restart-four-to-one', 1, resume=split_path)
        parity('four-to-one-continuation', full, imported)
        _, rebooted = run('restart-force-rebootstrap', 4, resume=split_path, rebootstrap=True)
        # Rebootstrap intentionally rebuilds velocity-dependent forces/rates at
        # stored full-step velocities. Its reference is the same operation in
        # the established single-GPU driver, not ordinary closed-state replay.
        _, reboot_reference = run('restart-force-rebootstrap-single', 0, resume=split_path, rebootstrap=True)
        parity('four-gpu-force-rebootstrap', reboot_reference, rebooted)
        # Zero-duration load/save tests preserve all persistent fields exactly,
        # including nonzero stress and the deterministic flaw realization.
        _, roundtrip = run('checkpoint-roundtrip', 4, steps=4, resume=split_path)
        parity('checkpoint-roundtrip', split, roundtrip)
        for key in replay.EXACT + tuple(replay.FLOORS):
            if not np.array_equal(getattr(split, key), getattr(roundtrip, key)):
                raise AssertionError(f'Checkpoint roundtrip changed {key}')

        _, uncorrected_full = run('uncorrected-full', 4, correction=False)
        uncorrected_path, _ = run('uncorrected-first', 4, correction=False, steps=4)
        _, uncorrected_resumed = run('uncorrected-second', 4, correction=False,
                                    resume=uncorrected_path)
        parity('heating-resume-without-kernel-correction', uncorrected_full, uncorrected_resumed)

        # Pressure-free starting state isolates the first gravitational kick.
        gravity_input = work / 'gravity-input.bin'
        data = read_particle_binary(input_path)
        data.lifecycle_state = None
        data.velocities[:] = 0.0
        data.energy[:] = 0.0
        write_particle_binary(gravity_input, data, physics='sph', record_kind='input')
        # One very short step, direct reference; evaluate total force because
        # ordinary SPH output deliberately omits FORCE.
        _, reference = run('gravity-direct', 0, source=gravity_input, gravity='direct', steps=1)
        norm = float(np.linalg.norm(total_force(reference)))
        rms = norm / np.sqrt(reference.num_particles)
        for ranks in (0, 1, 2, 4):
            _, candidate = run(f'gravity-fmm-gpu{ranks}', ranks, source=gravity_input,
                               gravity='unified_fmm', steps=1)
            delta = total_force(candidate)-total_force(reference)
            relative = float(np.linalg.norm(delta)/norm)
            maximum = float(np.max(np.linalg.norm(delta,axis=1))/rms)
            check = {'relative_rms': relative, 'max_error_over_reference_rms': maximum,
                     'relative_rms_limit': .01, 'max_error_limit': .05,
                     'pass': relative <= .01 and maximum <= .05}
            report['checks'][f'gravity-fmm-gpu{ranks}'] = check
            save()
            if not check['pass']:
                raise AssertionError(f'gravity-fmm-gpu{ranks}: {check}')
            print(f'PASS gravity-fmm-gpu{ranks}: {relative:.6g}', flush=True)

        # Spread particles across enough common-grid leaves to require far
        # routes; the compact fixture mainly exercises remote near-field pairs.
        far_input = work / 'far-field-input.bin'
        far_data = copy.deepcopy(data)
        far_data.positions *= 12.
        write_particle_binary(far_input, far_data, physics='sph', record_kind='input')
        _, far_reference = run('far-field-direct', 0, source=far_input, gravity='direct', steps=1)
        far_norm = float(np.linalg.norm(total_force(far_reference)))
        far_rms = far_norm / np.sqrt(far_reference.num_particles)
        for ranks in (0, 1, 2, 4):
            _, far_candidate = run(f'far-field-fmm-gpu{ranks}', ranks, source=far_input,
                                   gravity='unified_fmm', steps=1)
            residual = float(np.linalg.norm(total_force(far_candidate)-total_force(far_reference))/far_norm)
            maximum = float(np.max(np.linalg.norm(total_force(far_candidate)-total_force(far_reference),axis=1))/far_rms)
            report['checks'][f'far-field-fmm-gpu{ranks}'] = {
                'relative_rms':residual,'max_error_over_reference_rms':maximum,
                'relative_rms_limit':.01,'max_error_limit':.05,
                'pass':residual<=.01 and maximum<=.05}
            save()
            if not report['checks'][f'far-field-fmm-gpu{ranks}']['pass']:
                raise AssertionError(f'far-field gravity mismatch: {residual}, {maximum}')
            print(f'PASS far-field-fmm-gpu{ranks}: {residual:.6g}',flush=True)

        close_input = work / 'near-pair-input.bin'
        close_data = copy.deepcopy(data)
        close_data.positions[0] = 0.
        close_data.positions[1] = np.array([1e-9, 0., 0.])
        write_particle_binary(close_input, close_data, physics='sph', record_kind='input')
        _, close_reference = run('near-pair-direct', 0, source=close_input, gravity='direct', steps=1)
        close_norm = float(np.linalg.norm(close_total_force(reference)))
        for ranks in (0, 4):
            _, close_candidate = run(f'near-pair-fmm-gpu{ranks}', ranks, source=close_input,
                                     gravity='unified_fmm', steps=1)
            residual = float(np.linalg.norm(close_total_force(candidate)-close_total_force(reference))/close_norm)
            report['checks'][f'near-pair-fmm-gpu{ranks}'] = {'relative_rms':residual,
                                                            'limit':.01,'pass':residual<=.01}
            save()
            if not residual <= .01: raise AssertionError(f'near-pair gravity mismatch {residual}')

        _, gravity_full = run('gravity-restart-full', 4, source=far_input, gravity='unified_fmm', correction=False)
        gravity_split, _ = run('gravity-restart-first', 4, source=far_input, gravity='unified_fmm', steps=4, correction=False)
        _, gravity_resumed = run('gravity-restart-second', 4, source=far_input, gravity='unified_fmm', resume=gravity_split, correction=False)
        parity('self-gravitating-four-gpu-continuation', gravity_full, gravity_resumed)

        # Updated EOS and wide Weibull data, continuous activation and all histories.
        history_input = work / 'history-input.bin'
        history_data = read_particle_binary(input_path)
        history_data.lifecycle_state = None
        history_data.density[:] *= .99
        history_data.damage[:] = .01
        history_data.velocities *= -1
        write_particle_binary(history_input, history_data, physics='sph', record_kind='input')
        for eos in ('tillotson', 'simplified_tillotson', 'elastic'):
            table = yaml.safe_load(material.read_text()); sph = table['materials'][0]['sph']
            sph.update(bulk_modulus=2.7e10, elastic_modulus=5.4e10, continuous_activation=True)
            sph['damage'].update(k=1e44, m=9.5)
            if eos == 'simplified_tillotson': sph['eos'] = {'model':eos, 'A':2.67e10, 'c':2.0}
            if eos == 'elastic': sph['eos'] = {'model':eos, 'A':2.67e10}
            card = work / f'{eos}-history-materials.yaml'
            card.write_text(yaml.safe_dump(table, sort_keys=False))
            def history(c):
                c['sph_parameters']['track_history'] = True
                c['sph_parameters']['relaxation']['disable_damage'] = False
            _, ref = run(f'{eos}-history-single', 0, source=history_input, card=card, overrides=history)
            for ranks in (1, 2, 4):
                cp, got = run(f'{eos}-history-gpu{ranks}', ranks, source=history_input, card=card, overrides=history)
                parity(f'{eos}-history-gpu{ranks}', ref, got)
            # Compare cross-driver restart of the same closed state.
            split = cp.parent / 'checkpoint_step_0000000004.bin'
            _, restart_single = run(f'{eos}-history-resume-single', 0, source=history_input,
                                    card=card, overrides=history, resume=split)
            _, restart_four = run(f'{eos}-history-resume-four', 4, source=history_input,
                                  card=card, overrides=history, resume=split)
            parity(f'{eos}-history-restart-drivers', restart_single, restart_four)
            parity(f'{eos}-history-continuation', got, restart_four)

        # Morton ownership, distributed gravity and histories must work together.
        _, morton_ref = run('history-gravity-single',0,source=history_input,
                            gravity='unified_fmm',overrides=history,correction=False)
        cp, morton_got = run('history-gravity-four',4,source=history_input,
                            gravity='unified_fmm',overrides=history,correction=False)
        parity('history-gravity-single-vs-four',morton_ref,morton_got)
        split = cp.parent/'checkpoint_step_0000000004.bin'
        _, morton_resumed = run('history-gravity-resume-four',4,source=history_input,
            gravity='unified_fmm',overrides=history,correction=False,resume=split)
        parity('history-gravity-continuation',morton_got,morton_resumed)

        # Porosity closure and historical compaction travel through refreshed halos.
        table = yaml.safe_load(material.read_text())
        table['materials'][0]['sph']['porosity'] = {
            'model':'p_alpha','alpha_initial':1.3, 'P_elastic_limit':1e5,
            'P_solid_crush':1e8,'crushing_exponent':2.0}
        porous_card = work / 'porous-materials.yaml'
        porous_card.write_text(yaml.safe_dump(table,sort_keys=False))
        _, porous_ref = run('porous-single',0,card=porous_card,overrides=history)
        for ranks in (1,2,4):
            cp, porous_got = run(f'porous-gpu{ranks}',ranks,card=porous_card,overrides=history)
            parity(f'porous-gpu{ranks}',porous_ref,porous_got)
        if not (np.all(porous_ref.distension <= 1.3) and np.any(porous_ref.distension < 1.3)):
            raise AssertionError('porosity compaction not exercised')
        split = cp.parent/'checkpoint_step_0000000004.bin'
        _, porous_resumed = run('porous-resume-four',4,card=porous_card,overrides=history,resume=split)
        parity('porous-continuation',porous_got,porous_resumed)

        # Mixed materials and bulk translation exercise ownership migration.
        moving = read_particle_binary(input_path); moving.lifecycle_state = None
        moving.material_id = (moving.global_id % 2).astype(np.int32)
        moving.density = np.where(moving.material_id == 0,2700.,1185.).astype(np.float32)
        moving.mass = (moving.density * .01**3).astype(np.float32)
        moving.velocities += np.array([30000.,-15000.,10000.])
        moving_path = work/'moving-mixed-input.bin'
        write_particle_binary(moving_path,moving,physics='sph',record_kind='input')
        _, moving_ref = run('moving-mixed-single',0,source=moving_path,overrides=history)
        for ranks in (1,2,4):
            _, moving_got = run(f'moving-mixed-gpu{ranks}',ranks,source=moving_path,overrides=history)
            parity(f'moving-mixed-gpu{ranks}',moving_ref,moving_got)

        # Fractional stage boundaries, two damping half steps and stage sidecars.
        def staged(c):
            c['sph_parameters'].update(track_history=True, stages=[
                {'mode':'RELAX','end_time':2.5*replay.DT,'damping_rate':1e5},
                {'mode':'SHOCK','end_time':5.5*replay.DT},
                {'mode':'LATE','end_time':8*replay.DT}])
            c['sph_parameters']['relaxation']['disable_damage'] = False
        ref_cp, stage_ref = run('stages-single', 0, source=history_input, overrides=staged)
        for ranks in (1, 2, 4):
            cp, got = run(f'stages-gpu{ranks}', ranks, source=history_input, overrides=staged)
            parity(f'stages-gpu{ranks}', stage_ref, got)
            # Every closed checkpoint, including both fractional boundaries.
            for p in sorted(cp.parent.glob('checkpoint_step_*.bin')):
                ref_p = ref_cp.parent / p.name
                parity(f'stages-gpu{ranks}-{p.stem}', read_particle_binary(ref_p), read_particle_binary(p))
                if p.with_suffix('.bin.sph_stage').read_bytes() != ref_p.with_suffix('.bin.sph_stage').read_bytes():
                    raise AssertionError('different stage sidecars')
        if not np.any(stage_ref.damping_energy > 0): raise AssertionError('damping not exercised')
        for step in (3, 4, 6):
            split = cp.parent / f'checkpoint_step_{step:010d}.bin'
            _, one = run(f'stage-resume-{step}-single', 0, source=history_input, overrides=staged, resume=split)
            _, four = run(f'stage-resume-{step}-four', 4, source=history_input, overrides=staged, resume=split)
            parity(f'stage-resume-{step}-drivers', one, four)
            parity(f'stage-resume-{step}-continuation', got, four)
        final_stage = sorted((work/'stages-gpu4/checkpoints').glob('checkpoint_step_*.bin'))[-1]
        _, final_stage_ref = run('stage-final-roundtrip-single',0,source=history_input,overrides=staged,resume=final_stage)
        _, final_stage_got = run('stage-final-roundtrip-four',4,source=history_input,overrides=staged,resume=final_stage)
        stored_stage = read_particle_binary(final_stage)
        parity('stage-final-roundtrip-single',stored_stage,final_stage_ref)
        parity('stage-final-roundtrip-four',stored_stage,final_stage_got)
        for field in dataclasses.fields(stored_stage):
            a, b = getattr(ordered(copy.deepcopy(stored_stage)),field.name), getattr(final_stage_got,field.name)
            if isinstance(a,np.ndarray) and not np.array_equal(a,b):
                raise AssertionError(f'final staged roundtrip changed {field.name}')

        def fractional(c): c['simulation']['time']['stop'] = 8.5*replay.DT
        _, ref = run('fractional-stop-single', 0, overrides=fractional)
        for ranks in (1, 2, 4):
            _, got = run(f'fractional-stop-gpu{ranks}', ranks, overrides=fractional)
            parity(f'fractional-stop-gpu{ranks}', ref, got)
        if ref.current_time != 8.5*replay.DT: raise AssertionError('terminal time not exact')

        # Continuation contracts must fail closed if history or stage metadata is lost.
        original = work/'stages-gpu4/checkpoints/checkpoint_step_0000000004.bin'
        broken = work/'missing-stage.bin'; broken.write_bytes(original.read_bytes())
        reject('missing-stage-rejected-single',0,'stage',source=history_input,overrides=staged,resume=broken)
        reject('missing-stage-rejected-four',4,'stage',source=history_input,overrides=staged,resume=broken)
        missing = read_particle_binary(original); missing.energy_cold = None
        broken_history = work/'missing-history.bin'
        write_particle_binary(broken_history,missing,physics='sph',record_kind='checkpoint')
        broken_history.with_suffix('.bin.sph_stage').write_bytes(original.with_suffix('.bin.sph_stage').read_bytes())
        reject('missing-history-rejected-single',0,'history',source=history_input,overrides=staged,resume=broken_history)
        reject('missing-history-rejected-four',4,'history',source=history_input,overrides=staged,resume=broken_history)
        def short_halo(c): c['sph_parameters']['kernel_cutoff_factor'] = 1.0
        reject('short-gravity-support-rejected',4,'kernel_cutoff_factor >= 2',gravity='unified_fmm',overrides=short_halo)

        report['passed'] = True
    except Exception as exc:
        report['failure'] = str(exc)
        raise
    finally:
        save()
    print(f'PASS all validation: {work / "report.json"}', flush=True)

if __name__ == '__main__':
    main()
