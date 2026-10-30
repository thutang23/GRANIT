#!/usr/bin/env python3
"""Tiny CUDA DEM seconds/restart regression. No build or production launch.

Run with --single and --multi pointing at built binaries and --work pointing
inside linked ~/data evidence storage. Every solver invocation is bounded.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import subprocess

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'dem_multigpu_clumps'))
spec = importlib.util.spec_from_file_location('stop_validation', HERE.parent/'cooperative_stop/validate.py')
stop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stop)
base, ext = stop.base, stop.ext


def campaign(single, multi, work):
    work.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'granit-timestep-seconds/v1', 'pass': False, 'runs': [], 'checks': {},
              'binary_sha256': {str(p): base.digest(p) for p in (single, multi)},
              'script_sha256': {str(p): base.digest(p) for p in (Path(__file__), HERE.parent/'cooperative_stop/validate.py', HERE.parent/'dem_multigpu_clumps/validate.py', HERE.parent/'dem_multigpu_clumps/extension_validation.py')}}

    def save():
        base.dump(work/'report.json', report)

    def run(label, binary, ranks, dt=None, steps=8, resume=None, omit=False,
            factor=None, expect=None, probe=False, shear=False):
        directory = work/label
        ext.prepare(directory, 'restart-contact', ranks, steps=math.ceil(steps))
        cfg = json.loads((directory/'config.json').read_text())
        timing = cfg['simulation']['time']
        timing.update(start=0., stop=steps*(dt if dt and math.isfinite(dt) and dt > 0 else 1e-6),
                      output_step=(dt if dt and math.isfinite(dt) and dt > 0 else 1e-6))
        if omit or dt is None:
            timing.pop('dt', None)
        else:
            timing['dt'] = dt if math.isfinite(dt) else ('.nan' if math.isnan(dt) else '.inf')
        if probe:
            timing.update(stop=1e-6, output_step=1e-6)
        if factor is not None:
            timing['dt_factor'] = factor
        if shear:
            cfg['clumps'] = {'instances': []}
            base.write_input(directory/'input.bin', ((pos, vel, spin, radius, 0)
                for pos, vel, spin, radius, _ in ext.fixture_rows('restart-contact')))
            cfg['shear_box'] = {'enabled': True, 'omega': 1., 'shear_parameter': 1.5,
                                'vertical_frequency': 1., 'orbital_dt_factor': dt/2,
                                'input_velocity': 'total'}
        cfg['io']['checkpoint']['interval'] = 4
        if resume:
            cfg['io']['checkpoint'].update(resume_from=str(resume), force_fresh=False,
                                           rebootstrap_forces=False)
        base.dump(directory/'config.json', cfg)
        launched = stop.launch(binary, directory, 90)
        launched['label'] = label
        launched['process_reaped'] = True  # launch polls/reaps before returning
        report['runs'].append(launched)
        save()
        log = (directory/'run.log').read_text()
        if expect:
            assert launched['returncode'] != 0, (label, 'expected failure')
            assert re.search(expect, log, re.I), (label, expect, log[-3000:])
            assert not list((directory/'checkpoints').glob('checkpoint_step_*.bin')), (label, 'rejected run wrote checkpoint')
            report['checks'][label] = {'pass': True, 'expected_diagnostic': expect,
                                       'returncode': launched['returncode'], 'no_checkpoint': True}
            save()
        else:
            assert launched['returncode'] == 0, (label, log[-3000:])
        return directory

    try:
        for ranks, binary in ((0, single), (1, multi), (2, multi)):
            prefix = f'gpu{ranks}'
            probe = run(prefix+'-auto', binary, ranks, probe=True)
            match = re.search(r'Critical timestep:\s*([0-9.eE+-]+)', (probe/'run.log').read_text())
            assert match, 'missing physical critical bound in solver log'
            critical = float(match[1])
            dt = 2.**math.floor(math.log2(.65*critical))
            assert .2*critical < dt < critical
            report['checks'][prefix+'-choice'] = {'dt_seconds': dt, 'critical_log_seconds': critical,
                                                 'above_legacy_auto': True}
            full = run(prefix+'-full', binary, ranks, dt)
            checkpoint = full/'checkpoints/checkpoint_step_0000000004.bin'
            assert checkpoint.exists(), checkpoint
            saved = stop.read_checkpoint(checkpoint)
            final = stop.read_checkpoint(ext.latest_checkpoint(full))
            assert saved['dt'] == final['dt'] == dt
            assert final['step'] == 8 and final['time'] == 8*dt
            assert saved['histories'], 'continuation fixture must carry active history'
            clipped = run(prefix+'-clipped', binary, ranks, dt, steps=8.5)
            clipped_path = ext.latest_checkpoint(clipped)
            clipped_state = stop.read_checkpoint(clipped_path)
            assert clipped_state['dt'] == dt and clipped_state['time'] == 8.5*dt
            roundtrip = run(prefix+'-zero-step', binary, ranks, dt, steps=8.5, resume=clipped_path)
            roundtrip_state = stop.read_checkpoint(ext.latest_checkpoint(roundtrip))
            comparison = stop.exact_state(clipped_state, roundtrip_state)
            report['checks'][prefix+'-clipped-zero-step'] = comparison
            assert comparison['pass'], comparison
            for omit in (False, True):
                resumed = run(prefix+('-resume-implicit' if omit else '-resume-explicit'),
                              binary, ranks, dt, resume=checkpoint, omit=omit)
                actual = stop.read_checkpoint(ext.latest_checkpoint(resumed))
                comparison = stop.exact_state(final, actual)
                report['checks'][resumed.name] = comparison
                assert comparison['pass'], comparison
            rejection = r'simulation\.time\.dt must be finite'
            run(prefix+'-mismatch', binary, ranks, dt/2, steps=16, resume=checkpoint, expect='differs from the checkpoint timestep')
            run(prefix+'-oversize', binary, ranks, 2*critical, expect='exceeds the physical DEM critical timestep')
            run(prefix+'-both', binary, ranks, dt, factor=.2, expect='mutually exclusive')
            run(prefix+'-orbital', binary, ranks, dt, shear=True, expect='exceeds the shear orbital timestep limit')
            for label, invalid in (('zero', 0.), ('negative', -dt), ('nan', float('nan')), ('infinity', float('inf'))):
                run(prefix+'-'+label, binary, ranks, invalid, expect=rejection)
            save()
        release = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                 text=True, capture_output=True, timeout=10)
        report['gpu_release'] = {'returncode': release.returncode, 'stdout': release.stdout, 'stderr': release.stderr}
        assert release.returncode == 0, report['gpu_release']
        gpu_pids = {int(line.strip()) for line in release.stdout.splitlines() if line.strip().isdigit()}
        child_pids = {row['pid'] for row in report['runs']}
        assert not gpu_pids & child_pids, ('solver PIDs still on GPU', gpu_pids & child_pids)
        report['gpu_release']['solver_pids_absent'] = True
        report['pass'] = True
    finally:
        save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--single', type=Path, required=True)
    parser.add_argument('--multi', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    result = campaign(args.single.resolve(strict=True), args.multi.resolve(strict=True), args.work.resolve())
    print(json.dumps({'pass': result['pass'], 'report': str(args.work/'report.json')}))
