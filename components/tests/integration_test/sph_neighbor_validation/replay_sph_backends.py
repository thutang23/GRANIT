#!/usr/bin/env python3
"""Replay deterministic unified-v3 SPH inputs through DIRECT, CUDA LBVH and RT.

Examples (all artifacts are retained):
  python3 replay_sph_backends.py --work-dir /tmp/sph-parity --prepare-only
  python3 replay_sph_backends.py --work-dir /tmp/sph-parity --run-prepared --binary PATH
  python3 replay_sph_backends.py --work-dir /tmp/sph-timing --benchmark --binary PATH

The two 16-step validation cases exercise summation and continuous density,
heterogeneous changing h, strength, energy, and kernel correction. Checkpoints
are essential: ordinary SPH outputs do not contain force, stress or rates.
Exact adjacency is tested by test_sph_neighbor, not inferred from these outputs.

The fixed numeric gate is |candidate-reference| <= 2e-5*|reference| +
32*eps_float32*max(abs(reference_field)), with a documented dimensional floor
for identically zero/near-zero fields. The absolute term allows cancellation
roundoff in single-precision SPH accumulations; candidate data never sets it.
No runtime tolerance adjustment is offered.

Optional timing uses the solver's native elapsed-ms progress timestamps after
warmup. It excludes initialization, includes SPH physics, timestep work and
periodic output/scalar I/O, and is not a kernel-only or I/O-free speed claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src/plugins/scripts"))
from cudadem_pipeline.common.io import ParticleData, read_particle_binary, write_particle_binary

DT = 2.0 ** -24  # Exactly representable in float32 and float64; fixed output steps.
RTOL = 2.0e-5
ROUNDOFF = float(32 * np.finfo(np.float32).eps)
FLOORS = {
    "positions": 1e-12, "velocities": 1e-9, "force": 1e-7,
    "density": 1e-6, "pressure": 1e-2, "c_sound": 1e-6,
    "smoothing_length": 1e-12, "energy": 1e-7, "damage": 1e-10,
    "crack_strain": 1e-10, "distension": 1e-9, "drhodt": 1e-5,
    "dedt": 1e-4, **{f"S_{s}": 1e-2 for s in ("xx", "yy", "zz", "xy", "xz", "yz")},
}
EXACT = ("global_id", "material_id", "clump_id", "mass", "flaws_ntot",
         "flaws_strain_min", "flaws_strain_max", "flaws_m0")
META = ("num_particles", "current_step", "current_time", "dt", "record_kind",
        "format_version", "material_database_fingerprint", "clump_metadata_fingerprint")
PROGRESS = re.compile(r"Step (\d+) \(t=.*?\) - (\d+) ms elapsed")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fixture(path: Path, side: int) -> dict:
    """Perturbed lattice, reproducible permutation and nonuniform h; no dataset."""
    n = side ** 3
    rng = np.random.default_rng(20260907)
    pos = np.indices((side, side, side)).reshape(3, -1).T.astype(np.float64)
    pos = (pos - 0.5 * (side - 1)) * 0.01
    pos += rng.uniform(-2e-4, 2e-4, size=pos.shape)
    vel = np.column_stack((-8 * pos[:, 0] + 3 * pos[:, 1],
                           -5 * pos[:, 1] + 2 * pos[:, 2], -6 * pos[:, 2]))
    h = (0.0115 * (1 + 0.12 * np.sin(np.arange(n) * 0.73))).astype(np.float32)
    order = rng.permutation(n)
    data = ParticleData(
        positions=pos[order], velocities=vel[order],
        global_id=(np.arange(n, dtype=np.int32) * 7 + 100)[order],
        material_id=np.zeros(n, dtype=np.int32), clump_id=np.zeros(n, dtype=np.int32),
        mass=np.full(n, 2700.0 * 0.01 ** 3, dtype=np.float32),
        density=np.full(n, 2700.0, dtype=np.float32), smoothing_length=h[order],
        energy=np.full(n, 1000.0, dtype=np.float64), damage=np.zeros(n, dtype=np.float32),
    )
    write_particle_binary(path, data, physics="sph", record_kind="input")
    restored = read_particle_binary(path)
    if not np.array_equal(restored.global_id, data.global_id):
        raise ValueError("Fixture roundtrip changed particle IDs")
    return {"path": str(path), "sha256": digest(path), "particles": n,
            "seed": 20260907, "geometry": "0.01 m perturbed cubic lattice",
            "h_min_m": float(h.min()), "h_max_m": float(h.max())}


def config(work: Path, input_path: Path, material_path: Path, backend: str,
           mode: int, steps: int, output_interval: int, checkpoint: bool) -> dict:
    return {
        "material_database": str(material_path),
        "simulation": {"name": "SPH backend replay", "physics": "sph",
                       "time": {"start": 0.0, "stop": DT * steps,
                                "output_step": DT * output_interval},
                       "termination": {"condition": "time"}},
        "domain": {"box_min": [-1., -1., -1.], "box_max": [1., 1., 1.]},
        "accelerations": {"inertial": {"enabled": False}, "tidal": {"enabled": False},
                          "self_gravity": {"enabled": False, "method": "direct",
                                           "unified_fmm": {"tree_dimension": 3, "tree_depth": 6}}},
        "sph_parameters": {"viscosity": 0.1, "smoothing_length": 0.0115,
                           "density_update_mode": mode, "enable_shepard_filter": False,
                           "use_kernel_correction": True, "use_artificial_stress": True,
                           "cfl_number": 0.3, "dt_min": DT, "dt_max": DT,
                           "kernel_cutoff_factor": 2., "hsml_min": 0.005,
                           "hsml_max": 0.025, "damage_seed": 12345678,
                           "relaxation": {"disable_strength": False, "disable_damage": True}},
        "neighbor": {"method": "direct" if backend == "direct" else "lbvh",
                     "backend": "cuda" if backend == "direct" else backend,
                     "update_frequency": 10, "verlet_buffer_enabled": True,
                     "verlet_skin": 0.002},
        "damping": {"enabled": False, "type": "local", "coefficient": 0.0},
        "io": {"input": {"particle_file": str(input_path)},
               "output": {"directory": str(work / "output"), "initial_state": True,
                          "scalar_interval": 0},
               "checkpoint": {"enabled": checkpoint, "force_fresh": True,
                              "interval": 1, "path": str(work / "checkpoints")}},
        "gpu": {"device_id": 0, "block_size": 128}, "multigpu": {"enabled": False},
    }


def prepare(args: argparse.Namespace) -> dict:
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    material = work / "materials.yaml"
    material.write_bytes((ROOT / "components/config/materials_example.yaml").read_bytes())
    small = work / "validation-input.bin"
    inputs = [fixture(small, 6)]
    backends = ["direct", "cuda"] + ([] if args.skip_rt else ["rt"])
    cases = []

    def add(name: str, backend: str, mode: int, steps: int, interval: int,
            checkpoint: bool, input_path: Path, warmup: int = 0) -> None:
        directory = work / name
        directory.mkdir()
        path = directory / "config.yaml"
        path.write_text(yaml.safe_dump(config(directory, input_path, material, backend,
                                               mode, steps, interval, checkpoint), sort_keys=False))
        cases.append({"name": name, "backend": backend, "density_mode": mode,
                      "steps": steps, "kind": "validation" if checkpoint else "benchmark",
                      "warmup_steps": warmup, "config": str(path),
                      "config_sha256": digest(path), "directory": str(directory)})

    for mode in (0, 1):
        for backend in backends:
            add(f"density{mode}-{backend}", backend, mode, args.steps, 1, True, small)
    if args.benchmark:
        large = work / "benchmark-input.bin"
        inputs.append(fixture(large, args.benchmark_side))
        for repeat in range(args.repeats):
            # Alternate backend order to reduce a fixed ordering bias.
            order = [b for b in backends if b != "direct"]
            if repeat % 2:
                order.reverse()
            for backend in order:
                add(f"benchmark-{repeat}-{backend}", backend, 1, args.benchmark_steps,
                    args.warmup_steps, False, large, args.warmup_steps)
    manifest = {"schema": "granit-sph-backend-replay/v1", "inputs": inputs,
                "materials": {"path": str(material), "sha256": digest(material)},
                "cases": cases, "dt": DT, "rt_requested": not args.skip_rt,
                "numeric_gate": {"rtol": RTOL, "roundoff_multiplier": ROUNDOFF,
                                 "absolute_floors_in_field_units": FLOORS},
                "generator": {"path": str(Path(__file__).resolve()),
                              "sha256": digest(Path(__file__))}}
    write_json(work / "manifest.json", manifest)
    return manifest


def compare_pair(reference, candidate) -> dict:
    for name in META:
        if getattr(reference, name) != getattr(candidate, name):
            raise ValueError(f"Metadata differs: {name}")
    for name in EXACT:
        left, right = getattr(reference, name), getattr(candidate, name)
        if left is None or right is None or not np.array_equal(left, right):
            raise ValueError(f"Missing/different exact field: {name}")
    fields = {}
    for name, floor in FLOORS.items():
        left, right = getattr(reference, name), getattr(candidate, name)
        if left is None or right is None or left.shape != right.shape:
            raise ValueError(f"Missing/different numeric field shape: {name}")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"Nonfinite numeric field: {name}")
        left, right = left.astype(np.float64), right.astype(np.float64)
        scale = float(np.max(np.abs(left)))
        atol = max(floor, ROUNDOFF * scale)
        delta = np.abs(right - left)
        fields[name] = {"max_abs_difference": float(delta.max()), "atol": atol,
                        "max_gate_fraction": float(np.max(delta / (atol + RTOL * np.abs(left)))),
                        "pass": bool(np.all(delta <= atol + RTOL * np.abs(left)))}
    return {"pass": all(v["pass"] for v in fields.values()), "fields": fields}


def compare(manifest: dict) -> dict:
    comparisons = {}
    coverage = {}
    cases = {c["name"]: c for c in manifest["cases"] if c["kind"] == "validation"}
    for mode in (0, 1):
        baseline = cases[f"density{mode}-direct"]
        initial = read_particle_binary(Path(baseline["directory"]) / "output/output_step_0000000000.bin")
        final = None
        for backend in ("cuda", "rt"):
            name = f"density{mode}-{backend}"
            if name not in cases:
                continue
            steps = {}
            for step in range(1, baseline["steps"] + 1):
                filename = f"checkpoint_step_{step:010d}.bin"
                ref_path = Path(baseline["directory"]) / "checkpoints" / filename
                got_path = Path(cases[name]["directory"]) / "checkpoints" / filename
                reference, candidate = read_particle_binary(ref_path), read_particle_binary(got_path)
                if reference.record_kind != "checkpoint" or reference.current_step != step:
                    raise ValueError(f"Invalid checkpoint: {ref_path}")
                steps[str(step)] = {**compare_pair(reference, candidate),
                                    "reference_sha256": digest(ref_path), "candidate_sha256": digest(got_path)}
                final = reference
            comparisons[name] = {"pass": all(s["pass"] for s in steps.values()), "steps": steps}
        if final is None:
            raise ValueError("No CUDA comparison produced")
        coverage[str(mode)] = {
            "heterogeneous_initial_h": bool(np.ptp(initial.smoothing_length) > 0),
            "h_changed": bool(np.any(initial.smoothing_length != final.smoothing_length)),
            "nonzero_force": bool(np.any(final.force != 0)),
            "nonzero_stress": bool(any(np.any(getattr(final, f"S_{s}") != 0)
                                       for s in ("xx", "yy", "zz", "xy", "xz", "yz"))),
            "energy_changed": bool(np.any(initial.energy != final.energy)),
        }
    return {"pass": all(c["pass"] for c in comparisons.values()) and
                     all(all(c.values()) for c in coverage.values()),
            "comparisons": comparisons, "coverage": coverage,
            "exact_adjacency": "Verified separately by test_sph_neighbor; not inferred from solver fields"}


def execute(args: argparse.Namespace, manifest: dict) -> dict:
    binary = args.binary.resolve()
    identity = {"path": str(binary), "sha256": digest(binary)}
    for item in manifest["inputs"] + [manifest["materials"]]:
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Prepared input changed: {item['path']}")
    results = {}
    for case in manifest["cases"]:
        path, directory = Path(case["config"]), Path(case["directory"])
        if digest(path) != case["config_sha256"]:
            raise ValueError(f"Prepared config changed: {path}")
        if (directory / "run.json").exists():
            raise ValueError(f"Run already exists; use a new work directory: {directory}")
        started = time.perf_counter()
        completed = subprocess.run([str(binary), str(path)], cwd=ROOT, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=args.timeout)
        wall_seconds = time.perf_counter() - started
        (directory / "solver.log").write_text(completed.stdout)
        row = {"returncode": completed.returncode, "process_wall_seconds": wall_seconds,
               "binary": identity, "config_sha256": digest(path)}
        if case["kind"] == "benchmark" and completed.returncode == 0:
            ticks = {int(step): int(ms) for step, ms in PROGRESS.findall(completed.stdout)}
            first, last = case["warmup_steps"], case["steps"]
            if first not in ticks or last not in ticks or ticks[last] <= ticks[first]:
                raise ValueError(f"Native timing interval missing or below ms resolution: {directory}")
            row["timing"] = {"warmup_steps": first, "measured_steps": last - first,
                             "native_elapsed_ms": ticks[last] - ticks[first],
                             "ms_per_step": (ticks[last] - ticks[first]) / (last - first),
                             "scope": "Solver loop after warmup, including output/scalar I/O; initialization excluded"}
        write_json(directory / "run.json", row)
        results[case["name"]] = row
        print(f"{case['name']}: exit={completed.returncode}, process={wall_seconds:.3f}s", flush=True)
        if completed.returncode:
            raise RuntimeError(f"Solver failed; inspect {directory / 'solver.log'}")
        final_path = directory / "output" / f"output_step_{case['steps']:010d}.bin"
        final = read_particle_binary(final_path)
        if final.current_step != case["steps"] or final.current_time != case["steps"] * DT:
            raise ValueError(f"Final output does not match requested endpoint: {final_path}")
        for name in ("positions", "velocities", "density", "pressure", "smoothing_length", "energy"):
            if not np.isfinite(getattr(final, name)).all():
                raise ValueError(f"Nonfinite final {name}: {final_path}")
        if digest(binary) != identity["sha256"]:
            raise ValueError("Solver executable changed during replay")
    validation = compare(manifest)
    timings = {}
    for backend in ("cuda", "rt"):
        samples = [results[c["name"]]["timing"]["ms_per_step"] for c in manifest["cases"]
                   if c["kind"] == "benchmark" and c["backend"] == backend]
        if samples:
            timings[backend] = {"ms_per_step_samples": samples, "median_ms_per_step": statistics.median(samples),
                                "minimum_ms_per_step": min(samples), "maximum_ms_per_step": max(samples)}
    return {"schema": "granit-sph-backend-replay-result/v1", "pass": validation["pass"],
            "binary": identity, "runs": results, "validation": validation, "timing": timings,
            "timing_limitations": "Synthetic finite-domain case; native 1 ms timestamps; output/scalar overhead included; no general speed claim"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "cmake/bin/cudadem_sph")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--skip-rt", action="store_true", help="Explicit CUDA-only validation; records RT as unrequested")
    parser.add_argument("--prepare-only", action="store_true", help="Write/read input fixtures and configs without executing GPU code")
    parser.add_argument("--run-prepared", action="store_true", help="Execute the manifest already in work-dir without regenerating it")
    parser.add_argument("--compare-only", action="store_true", help="Recompare existing checkpoints without executing GPU code")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-side", type=int, default=16, help="Particles = side cubed; default 4096")
    parser.add_argument("--benchmark-steps", type=int, default=240)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.)
    args = parser.parse_args()
    if not 10 <= args.steps <= 20:
        parser.error("--steps must be 10..20 for this bounded validation")
    if not 10 <= args.benchmark_side <= 21:
        parser.error("--benchmark-side must be 10..21 (1000..9261 particles)")
    if args.warmup_steps <= 0 or args.benchmark_steps <= args.warmup_steps or args.benchmark_steps % args.warmup_steps:
        parser.error("benchmark steps must exceed and be a multiple of positive warmup steps")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    try:
        manifest = (json.loads((args.work_dir / "manifest.json").read_text())
                    if args.run_prepared or args.compare_only else prepare(args))
        if args.prepare_only:
            print(f"Prepared {len(manifest['cases'])} cases at {args.work_dir.resolve()}; no GPU code executed")
            return 0
        result = compare(manifest) if args.compare_only else execute(args, manifest)
        write_json(args.work_dir / ("comparison.json" if args.compare_only else "result.json"), result)
        print("PASS" if result["pass"] else "FAIL")
        return 0 if result["pass"] else 1
    except Exception as error:
        if args.work_dir.is_dir():
            write_json(args.work_dir / "failure.json", {"pass": False, "error": str(error)})
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
