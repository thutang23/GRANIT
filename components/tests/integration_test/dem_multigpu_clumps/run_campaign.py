#!/usr/bin/env python3
"""Run the fixed small-clump acceptance matrix against already-built binaries."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import validate as v


def gravity_overlay(**changes):
    return {'accelerations': {'self_gravity': changes}}


def guards():
    return [
        ('oversized2', 2, 'oversized', {}, 'distributed small-clump diameter'),
        ('oversized4', 4, 'oversized', {}, 'distributed small-clump diameter'),
        ('projection', 4, 'all-clump', gravity_overlay(unified_fmm={'enable_global_projection': True}), 'enable_global_projection=false'),
        ('rate', 4, 'all-clump', gravity_overlay(gravity_update_rate=2), 'gravity_update_rate=1'),
        ('macro-dt', 4, 'all-clump', gravity_overlay(gravity_macro_dt=2e-5), 'gravity_macro_dt=0'),
        ('threshold-leaf', 4, 'all-clump', gravity_overlay(unified_fmm={'direct_threshold_leaf': 1}), 'direct thresholds 0/0'),
        ('threshold-internal', 4, 'all-clump', gravity_overlay(unified_fmm={'direct_threshold_internal': 1}), 'direct thresholds 0/0'),
        ('dimension', 4, 'all-clump', gravity_overlay(unified_fmm={'tree_dimension': 2}), 'explicit 3D FMM'),
        ('ownership', 4, 'all-clump', {'multigpu': {'morton_ownership': False}}, 'multigpu.morton_ownership=true'),
        ('direct-neighbor', 4, 'all-clump', {'neighbor': {'method': 'direct'}}, 'requires neighbor.method: LBVH'),
        ('cll-neighbor', 4, 'all-clump', {'neighbor': {'method': 'cll'}}, 'requires neighbor.method: LBVH'),
        ('yorp', 4, 'all-clump', {'accelerations': {'yorp': {'enabled': True}}}, 'YORP is unsupported'),
        ('damping', 4, 'all-clump', {'damping': {'enabled': True}}, 'global damping is unsupported'),
        ('rotating', 4, 'all-clump', {'simulation': {'coordinate_system': 'rotating'}}, 'rotating and inertial frames are unsupported'),
        ('inertial', 4, 'all-clump', {'accelerations': {'inertial': {'enabled': True}}}, 'rotating and inertial frames are unsupported'),
        ('acceleration-limit', 4, 'all-clump', {'accelerations': {'acceleration_limit': {'enabled': True}}}, 'acceleration limiting is unsupported'),
        ('periodic', 4, 'all-clump', {'walls': [{'id': 1, 'material_id': 0, 'type': 'periodic', 'axis': 'x', 'bounds': [-10., 10.]}]}, 'periodic boundaries are unsupported'),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fp64', type=Path, required=True)
    parser.add_argument('--default', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True,
                        help='directory containing the accepted sg-mechanical-* and sg-preflight-near-gravity-02 runs')
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=['mechanical', 'gravity', 'guards', 'all'], default='all')
    parser.add_argument('--reuse-runs', type=Path,
                        help='reverify completed runs with matching binary/config hashes, preserving their prior receipts')
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    source = sorted([*v.ROOT.glob('main*.cu'), *[p for p in (v.ROOT/'src').rglob('*')
                     if p.is_file() and p.suffix in ('.h', '.cuh', '.cpp', '.cu', '.hpp')]])
    report = {'schema': 'granit-clump-small-campaign/v1', 'pass': False, 'phase': args.phase,
        'binaries': {str(p.resolve()): v.digest(p) for p in (args.fp64, args.default)},
        'source_files_sha256': {str(p.relative_to(v.ROOT)): v.digest(p) for p in source},
        'checks': {}}

    def save():
        v.dump(work/'report.json', report)

    def execute(name, ranks, binary, reference=None, direct=False, tag=None):
        directory = work/f'{tag or name}-gpu{ranks}'
        retained = args.reuse_runs/f'{tag or name}-gpu{ranks}' if args.reuse_runs else None
        if retained and (retained/'launch.json').exists():
            launch = json.loads((retained/'launch.json').read_text())
            assert launch['binary_sha256'] == v.digest(binary), 'retained binary differs'
            assert launch['config_sha256'] == v.digest(retained/'config.json'), 'retained config differs'
            directory = retained
        else:
            v.generate(directory, name, ranks=ranks)
            launch = v.launch(binary, directory, 90)
        report['checks'][directory.name] = {'launch': launch}
        report['checks'][directory.name]['run_directory'] = str(directory)
        save()
        assert not launch['timeout'] and launch['returncode'] == 0, directory
        if reference:
            check = v.compare(reference, directory)
        else:
            check, _ = v.check_trajectory(directory)
        parity_pass = check['pass']
        if direct:
            check['direct_gravity'] = v.direct_gravity(directory)
            check['pass'] &= check['direct_gravity']['pass']
        receipt = work/f'{directory.name}.reverified.json' if directory.parent != work else directory/'verification.json'
        v.dump(receipt, check)
        report['checks'][directory.name]['verification'] = check
        save()
        print(json.dumps({'case': directory.name, 'pass': check['pass'],
            'max_gate_fractions': check.get('max_gate_fractions')}), flush=True)
        assert parity_pass, directory
        return directory

    try:
        if args.phase in ('mechanical', 'all'):
            for fixture in ('contact-migration', 'mixed', 'inferred-spin', 'all-clump', 'empty-rank'):
                reference = args.references/f'sg-mechanical-{fixture}-01'
                for ranks in (1, 2, 4):
                    execute(fixture, ranks, args.fp64, reference)
        if args.phase in ('gravity', 'all'):
            for fixture, binary in [('integrated-contact', args.fp64), ('populated-fmm', args.default)]:
                baseline = execute(fixture, 1, binary)
                for ranks in (2, 4):
                    execute(fixture, ranks, binary, baseline)
            for ranks in (2, 4):
                execute('near-gravity', ranks, args.default,
                        args.references/'sg-preflight-near-gravity-02', direct=True)
            for ranks in (2, 4):
                execute('near-gravity', ranks, args.fp64,
                        args.references/'sg-preflight-near-gravity-02', direct=True, tag='near-gravity-fp64')
        if args.phase in ('guards', 'all'):
            for name, ranks, fixture, overlay, expected in guards():
                directory = work/f'reject-{name}'
                code = v.reject(SimpleNamespace(binary=args.fp64, work_dir=directory,
                    fixture=fixture, ranks=ranks, overlay=json.dumps(overlay), expect=expected, timeout=30))
                report['checks']['reject-'+name] = json.loads((directory/'rejection.json').read_text())
                save()
                assert code == 0, directory
        report['pass'] = all(row.get('verification', row).get('pass', False) for row in report['checks'].values())
    except Exception as error:
        report['failure'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        save()
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
