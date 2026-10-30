"""Strict reader and writer for GRANIT unified particle binary version 3.

Version 3 is the only runtime particle format.  Legacy formats deliberately
live in the migration utility; do not add compatibility decoding here.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

MAGIC = b"GRANIT\0\0"
HEADER_VERSION = 3
HEADER_SIZE = 72
DESCRIPTOR_SIZE = 16
CONTACT_HISTORY_SIZE = 48
_HEADER = struct.Struct("<8sHHQqddHBBIIQQ")
_DESCRIPTOR = struct.Struct("<HBBIQ")
assert _HEADER.size == HEADER_SIZE
assert _DESCRIPTOR.size == DESCRIPTOR_SIZE

PHYSICS_DEM = 1
PHYSICS_SPH = 2
RECORD_INPUT = 1
RECORD_OUTPUT = 2
RECORD_CHECKPOINT = 3
_PHYSICS_BY_NAME = {"dem": PHYSICS_DEM, "sph": PHYSICS_SPH}
_PHYSICS_BY_VALUE = {value: name for name, value in _PHYSICS_BY_NAME.items()}
_RECORD_BY_NAME = {"input": RECORD_INPUT, "output": RECORD_OUTPUT, "checkpoint": RECORD_CHECKPOINT}

# Field IDs are stable across C++ and Python.  Every v3 record carries the
# material ID in Tier 2; global IDs remain in the fixed Tier 1 section.
FIELD_MATERIAL_ID = 0x0006
FIELD_CLUMP_ID = 0x0018
FIELD_CLUMP_OFFSET = 0x0019
FIELD_WINDING_X = 0x001A
FIELD_WINDING_Y = 0x001B
FIELD_WINDING_Z = 0x001C
FIELD_LIFECYCLE_STATE = 0x001D
FIELD_FLAWS_NTOT = 0x0022
FIELD_FLAWS_STRAIN_MIN = 0x0023
FIELD_FLAWS_STRAIN_MAX = 0x0024
FIELD_FLAWS_M0 = 0x0025
FLAW_METADATA_FIELDS = (FIELD_FLAWS_NTOT, FIELD_FLAWS_STRAIN_MIN,
                        FIELD_FLAWS_STRAIN_MAX, FIELD_FLAWS_M0)
WINDING_FIELDS = (FIELD_WINDING_X, FIELD_WINDING_Y, FIELD_WINDING_Z)
# These SPH fields remain f32 in fresh inputs for v3 compatibility, but C++
# runtime outputs/checkpoints promote them to f64.
SPH_RUNTIME_F64_FIELDS = {0x0001, 0x0041}
FIELD_REGISTRY: Dict[int, Tuple[str, int, int]] = {
    0x0001: ("density", 0, 1),
    0x0002: ("pressure", 0, 1),
    0x0003: ("c_sound", 0, 1),
    0x0004: ("smoothing_length", 0, 1),
    0x0005: ("energy", 1, 1),
    FIELD_MATERIAL_ID: ("material_id", 2, 1),
    0x0010: ("omega", 1, 3),
    0x0011: ("radius", 0, 1),
    0x0012: ("mass", 0, 1),
    0x0013: ("moment_of_inertia", 0, 1),
    0x0014: ("force", 0, 3),
    0x001E: ("gravity_acceleration", 1, 3),
    0x0015: ("torque", 0, 3),
    0x0016: ("quaternion", 1, 4),
    0x0017: ("num_contacts", 2, 1),
    FIELD_CLUMP_ID: ("clump_id", 2, 1),
    FIELD_CLUMP_OFFSET: ("clump_offset", 0, 3),
    FIELD_WINDING_X: ("winding_x", 6, 1),
    FIELD_WINDING_Y: ("winding_y", 6, 1),
    FIELD_WINDING_Z: ("winding_z", 6, 1),
    FIELD_LIFECYCLE_STATE: ("lifecycle_state", 4, 1),
    0x0020: ("damage", 0, 1),
    0x0021: ("crack_strain", 0, 1),
    FIELD_FLAWS_NTOT: ("flaws_ntot", 2, 1),
    FIELD_FLAWS_STRAIN_MIN: ("flaws_strain_min", 0, 1),
    FIELD_FLAWS_STRAIN_MAX: ("flaws_strain_max", 0, 1),
    FIELD_FLAWS_M0: ("flaws_m0", 0, 1),
    0x0030: ("S_xx", 0, 1), 0x0031: ("S_yy", 0, 1),
    0x0032: ("S_zz", 0, 1), 0x0033: ("S_xy", 0, 1),
    0x0034: ("S_xz", 0, 1), 0x0035: ("S_yz", 0, 1),
    0x0040: ("distension", 0, 1), 0x0041: ("drhodt", 0, 1),
    0x0042: ("dedt", 1, 1),
    0x0050: ("energy_cold", 1, 1),
    0x0051: ("energy_thermal_peak", 1, 1),
    0x0052: ("pressure_peak", 1, 1),
    0x0053: ("fracture_area", 1, 1),
    0x0054: ("fragment_length", 1, 1),
    0x0055: ("damping_energy", 1, 1),

}
DTYPE_MAP = {0: np.dtype("<f4"), 1: np.dtype("<f8"), 2: np.dtype("<i4"),
             3: np.dtype("<i2"), 4: np.dtype("u1"), 5: np.dtype("?"),
             6: np.dtype("<i8")}
_DTYPE_REVERSE = {np.dtype("float32"): 0, np.dtype("float64"): 1,
                  np.dtype("int32"): 2, np.dtype("int16"): 3,
                  np.dtype("uint8"): 4, np.dtype("bool"): 5,
                  np.dtype("int64"): 6}
_NAME_REMAP = {"radius": "radii"}


@dataclass
class ParticleData:
    """In-memory v3 particle record.

    ``global_id`` and ``material_id`` are intentionally required by the writer:
    silently inventing either ID makes multiphysics/reordered results unsafe.
    """
    positions: np.ndarray
    velocities: Optional[np.ndarray] = None
    global_id: Optional[np.ndarray] = None
    material_id: Optional[np.ndarray] = None
    clump_id: Optional[np.ndarray] = None
    lifecycle_state: Optional[np.ndarray] = None
    clump_offset: Optional[np.ndarray] = None
    winding_x: Optional[np.ndarray] = None
    winding_y: Optional[np.ndarray] = None
    winding_z: Optional[np.ndarray] = None
    radii: Optional[np.ndarray] = None
    omega: Optional[np.ndarray] = None
    mass: Optional[np.ndarray] = None
    moment_of_inertia: Optional[np.ndarray] = None
    gravity_acceleration: Optional[np.ndarray] = None
    force: Optional[np.ndarray] = None
    torque: Optional[np.ndarray] = None
    density: Optional[np.ndarray] = None
    pressure: Optional[np.ndarray] = None
    c_sound: Optional[np.ndarray] = None
    smoothing_length: Optional[np.ndarray] = None
    energy: Optional[np.ndarray] = None
    damage: Optional[np.ndarray] = None
    crack_strain: Optional[np.ndarray] = None
    flaws_ntot: Optional[np.ndarray] = None
    flaws_strain_min: Optional[np.ndarray] = None
    flaws_strain_max: Optional[np.ndarray] = None
    flaws_m0: Optional[np.ndarray] = None
    # False denotes a legacy SPH checkpoint whose flaw realization must be
    # regenerated by the calling runtime from its configured seed.
    has_damage_flaw_metadata: bool = False
    quaternion: Optional[np.ndarray] = None
    num_contacts: Optional[np.ndarray] = None
    distension: Optional[np.ndarray] = None
    drhodt: Optional[np.ndarray] = None
    dedt: Optional[np.ndarray] = None
    energy_cold: Optional[np.ndarray] = None
    energy_thermal_peak: Optional[np.ndarray] = None
    pressure_peak: Optional[np.ndarray] = None
    fracture_area: Optional[np.ndarray] = None
    fragment_length: Optional[np.ndarray] = None
    damping_energy: Optional[np.ndarray] = None

    S_xx: Optional[np.ndarray] = None
    S_yy: Optional[np.ndarray] = None
    S_zz: Optional[np.ndarray] = None
    S_xy: Optional[np.ndarray] = None
    S_xz: Optional[np.ndarray] = None
    S_yz: Optional[np.ndarray] = None
    num_particles: int = 0
    current_step: int = 0
    current_time: float = 0.0
    dt: float = 0.0
    format_version: int = HEADER_VERSION
    material_database_fingerprint: int = 0
    clump_metadata_fingerprint: int = 0
    record_kind: str = "input"
    extra_fields: Optional[Dict[str, np.ndarray]] = None
    target_radii: Optional[np.ndarray] = None
    flag: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float64)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self.num_particles == 0:
            self.num_particles = len(self.positions)
        if self.num_particles != len(self.positions):
            raise ValueError("num_particles does not match positions")

    @property
    def physics(self) -> str:
        if self.density is not None or self.smoothing_length is not None:
            return "sph"
        return "dem"

    def center_on_mass_center(self, material_densities: Optional[Dict[int, float]] = None) -> "ParticleData":
        """Translate to COM using stored mass, or explicit material densities.

        Radius-cubed alone is not mass for a multi-material particle set.
        """
        if self.mass is not None:
            masses = np.asarray(self.mass, dtype=np.float64)
        else:
            if self.radii is None or material_densities is None or self.material_id is None:
                raise ValueError("COM requires stored mass or material_densities plus radii/material_id")
            density = np.asarray([material_densities[int(i)] for i in self.material_id], dtype=np.float64)
            masses = density * (4.0 / 3.0) * np.pi * np.asarray(self.radii, dtype=np.float64) ** 3
        total = float(np.sum(masses))
        if total <= 0.0:
            raise ValueError("total mass must be positive")
        self.positions = self.positions - np.sum(self.positions * masses[:, None], axis=0) / total
        return self


def _fail(message: str) -> None:
    raise ValueError(f"invalid GRANIT v3 particle binary: {message}")


def _validate_contiguous_clump_ids(clump_id: np.ndarray) -> None:
    positive = np.unique(clump_id[clump_id > 0])
    if len(positive) and not np.array_equal(
            positive, np.arange(1, int(positive[-1]) + 1, dtype=positive.dtype)):
        raise ValueError("positive clump_id values must be contiguous 1..K")


def _validate_sph_part_ids(part_id: np.ndarray) -> None:
    """Validate the unified-v3 ``clump_id`` field in its SPHSOL role.

    The on-disk field keeps its historical name, but SPH uses it as the exact
    Part index.  Live Parts must be a dense 0..K range; -1 is an inactive row.
    """
    active = np.unique(part_id[part_id >= 0])
    if len(active) and not np.array_equal(
            active, np.arange(int(active[-1]) + 1, dtype=active.dtype)):
        raise ValueError("SPH Part IDs must be dense 0..K")


def _validate_winding_members(values) -> bool:
    """Require the three signed winding counters to travel as one field set."""
    present = tuple(value is not None for value in values)
    if any(present) and not all(present):
        raise ValueError("winding_x, winding_y, and winding_z must be present together")
    return all(present)


def _checked_array(value: Optional[np.ndarray], name: str, n: int, dtype: np.dtype, count: int) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is required")
    arr = np.asarray(value)
    expected_shape = (n,) if count == 1 else (n, count)
    if arr.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {arr.shape}")
    if arr.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {arr.dtype}")
    return arr


def _write_soa(f, arr: np.ndarray, count: int) -> None:
    # Tier 2 vectors are component-major SoA, matching the device writer.
    if count == 1:
        f.write(np.ascontiguousarray(arr).tobytes())
    else:
        f.write(np.ascontiguousarray(arr.T).tobytes())


def read_particle_binary(filepath: str | Path) -> ParticleData:
    """Read and validate a v3 particle record.  Invalid data raises ValueError."""
    path = Path(filepath)
    raw = path.read_bytes()
    if len(raw) < HEADER_SIZE:
        _fail("truncated header")
    magic, version, header_size, n, step, time, dt, descriptor_count, physics_value, record_value, max_contacts, total_contacts, fingerprint, clump_fingerprint = _HEADER.unpack_from(raw)
    if magic != MAGIC or version != HEADER_VERSION or header_size != HEADER_SIZE:
        _fail("magic, version, or header size mismatch")
    if physics_value not in _PHYSICS_BY_VALUE or record_value not in _RECORD_BY_NAME.values():
        _fail("unknown physics mode or record kind")
    if step < 0 or not np.isfinite(time) or not np.isfinite(dt) or dt < 0:
        _fail("step/time/timestep metadata is invalid")
    if record_value == RECORD_INPUT and (step != 0 or time != 0.0 or dt != 0.0 or
                                         max_contacts != 0 or total_contacts != 0):
        _fail("input record metadata must be zero")
    if record_value == RECORD_OUTPUT and (max_contacts != 0 or total_contacts != 0):
        _fail("output record cannot carry contact metadata")
    if n > np.iinfo(np.int32).max:
        _fail("particle count exceeds runtime int32 limit")
    if n == 0:
        _fail("particle count must be positive")
    descriptor_end = HEADER_SIZE + descriptor_count * DESCRIPTOR_SIZE
    if descriptor_end > len(raw):
        _fail("truncated descriptor table")
    n = int(n)
    tier1_size = n * (6 * 8 + 4)
    tier1_start = descriptor_end
    tier2_start = tier1_start + tier1_size
    if tier2_start > len(raw):
        _fail("truncated Tier 1 data")

    descriptors = []
    seen = set()
    intervals = []
    for index in range(descriptor_count):
        field_id, dtype_value, count, reserved, offset = _DESCRIPTOR.unpack_from(raw, HEADER_SIZE + index * DESCRIPTOR_SIZE)
        if reserved != 0 or field_id in seen or field_id not in FIELD_REGISTRY:
            _fail("duplicate, unknown, or reserved descriptor")
        expected_name, expected_dtype, expected_count = FIELD_REGISTRY[field_id]
        flexible_runtime_dtype = (field_id in SPH_RUNTIME_F64_FIELDS or field_id in (0x0014, 0x0015)) and dtype_value in (0, 1)
        if (dtype_value != expected_dtype and not flexible_runtime_dtype) or count != expected_count:
            _fail(f"wrong dtype/count for {expected_name}")
        itemsize = DTYPE_MAP[dtype_value].itemsize
        size = n * count * itemsize
        end = offset + size
        if end < offset or tier2_start + end > len(raw):
            _fail("Tier 2 descriptor offset is out of bounds")
        seen.add(field_id)
        intervals.append((offset, end))
        descriptors.append((field_id, dtype_value, count, offset))
    for (_, end), (next_start, _) in zip(sorted(intervals), sorted(intervals)[1:]):
        if end > next_start:
            _fail("overlapping Tier 2 descriptors")
    if FIELD_MATERIAL_ID not in seen:
        _fail("MATERIAL_ID descriptor is required")
    winding_present = tuple(field_id in seen for field_id in WINDING_FIELDS)
    if any(winding_present) and not all(winding_present):
        _fail("WINDING_X, WINDING_Y, and WINDING_Z must be present together")
    physics = _PHYSICS_BY_VALUE[physics_value]
    required = {FIELD_MATERIAL_ID, 0x0011} if physics == "dem" else {FIELD_MATERIAL_ID, 0x0012, 0x0001}
    if not required.issubset(seen):
        _fail(f"required {physics.upper()} fields are missing")
    if record_value != RECORD_INPUT and FIELD_CLUMP_ID not in seen:
        _fail("CLUMP_ID is mandatory in output and checkpoint records")
    if FIELD_CLUMP_OFFSET in seen and (physics != "dem" or record_value != RECORD_CHECKPOINT):
        _fail("CLUMP_OFFSET is permitted only in DEM checkpoints")
    if physics == "dem" and record_value == RECORD_CHECKPOINT and FIELD_CLUMP_OFFSET not in seen:
        _fail("DEM checkpoints require CLUMP_OFFSET")
    flaw_presence = tuple(field_id in seen for field_id in FLAW_METADATA_FIELDS)
    if any(flaw_presence) and (not all(flaw_presence) or physics != "sph" or
                               record_value != RECORD_CHECKPOINT):
        _fail("Grady-Kipp flaw metadata must be a complete SPH checkpoint set")
    if record_value == RECORD_INPUT:
        allowed = ({FIELD_MATERIAL_ID, 0x0011, 0x0010, FIELD_CLUMP_ID} if physics == "dem" else
                   {FIELD_MATERIAL_ID, 0x0012, 0x0001, 0x0004, 0x0005, 0x0020,
                    FIELD_CLUMP_ID})
        allowed.update(WINDING_FIELDS)
        if not seen.issubset(allowed):
            _fail(f"unsupported field in {physics.upper()} input record")
    # A non-empty checkpoint blob is exactly the packed C++ ContactHistory array.
    tier2_end = max((end for _, end in intervals), default=0)
    payload_end = tier2_start + tier2_end
    if record_value != RECORD_CHECKPOINT and payload_end != len(raw):
        _fail("trailing bytes in non-checkpoint record")
    if record_value == RECORD_CHECKPOINT and total_contacts == 0 and payload_end != len(raw):
        _fail("unexpected checkpoint contact blob")
    if record_value == RECORD_CHECKPOINT and total_contacts > 0:
        expected_end = payload_end + n * max_contacts * CONTACT_HISTORY_SIZE
        if expected_end != len(raw):
            _fail("checkpoint contact blob has the wrong size")
    if total_contacts > n * max_contacts:
        _fail("total contacts exceeds declared capacity")

    def tier1(offset: int, dtype: np.dtype) -> np.ndarray:
        return np.frombuffer(raw, dtype=dtype, count=n, offset=tier1_start + offset).copy()
    positions = np.column_stack([tier1(0, np.dtype("<f8")), tier1(n * 8, np.dtype("<f8")), tier1(2 * n * 8, np.dtype("<f8"))])
    velocities = np.column_stack([tier1(3 * n * 8, np.dtype("<f8")), tier1(4 * n * 8, np.dtype("<f8")), tier1(5 * n * 8, np.dtype("<f8"))])
    global_id = tier1(6 * n * 8, np.dtype("<i4"))
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        _fail("positions and velocities must be finite")
    if np.any(global_id < 0) or len(np.unique(global_id)) != n:
        _fail("global_id values must be unique and nonnegative")
    data = ParticleData(positions=positions, velocities=velocities, global_id=global_id,
                        num_particles=n, current_step=step, current_time=time, dt=dt,
                        format_version=version, material_database_fingerprint=fingerprint,
                        clump_metadata_fingerprint=clump_fingerprint,
                        record_kind=next(name for name, value in _RECORD_BY_NAME.items() if value == record_value))
    for field_id, dtype_value, count, offset in descriptors:
        name = _NAME_REMAP.get(FIELD_REGISTRY[field_id][0], FIELD_REGISTRY[field_id][0])
        arr = np.frombuffer(raw, dtype=DTYPE_MAP[dtype_value], count=n * count, offset=tier2_start + offset).copy()
        if count > 1:
            arr = arr.reshape(count, n).T.copy()
        setattr(data, name, arr)
    data.has_damage_flaw_metadata = (
        physics == "sph" and record_value == RECORD_CHECKPOINT and all(flaw_presence)
    )
    if np.any(data.material_id < 0):
        _fail("material_id values must be nonnegative")
    if data.clump_id is None:
        if record_value != RECORD_INPUT:
            _fail("CLUMP_ID is mandatory in output and checkpoint records")
        data.clump_id = np.zeros(n, dtype=np.int32)
    if data.lifecycle_state is None:
        # Pre-external-body v3 records legitimately omit this field.  They
        # decode as entirely live; the C++ runner rejects that legacy form
        # when external bodies are enabled, where silent revival is unsafe.
        data.lifecycle_state = np.zeros(n, dtype=np.uint8)
    if np.any((data.lifecycle_state < 0) | (data.lifecycle_state > 1)):
        _fail("lifecycle_state values must be 0 or 1")
    if np.any(data.clump_id < -1):
        _fail("clump_id values must be at least -1")
    if physics == "sph":
        _validate_sph_part_ids(data.clump_id)
    else:
        _validate_contiguous_clump_ids(data.clump_id)
    if data.clump_offset is not None:
        if physics != "dem" or record_value != RECORD_CHECKPOINT:
            _fail("CLUMP_OFFSET is permitted only in DEM checkpoints")
        if not np.isfinite(data.clump_offset).all():
            _fail("clump_offset must be finite")
    _validate_winding_members((data.winding_x, data.winding_y, data.winding_z))
    if physics == "dem" and (not np.isfinite(data.radii).all() or np.any(data.radii <= 0)):
        _fail("DEM radii must be finite and positive")
    for name in ("omega", "force", "torque", "quaternion"):
        value = getattr(data, name)
        if value is not None and not np.isfinite(value).all():
            _fail(f"{name} must be finite")
    for name in ("mass", "moment_of_inertia"):
        value = getattr(data, name)
        if value is not None and (not np.isfinite(value).all() or np.any(value <= 0)):
            _fail(f"{name} must be finite and positive")
    if data.num_contacts is not None and np.any(data.num_contacts < 0):
        _fail("num_contacts must be nonnegative")
    if physics == "sph":
        if not np.isfinite(data.mass).all() or np.any(data.mass <= 0):
            _fail("SPH mass must be finite and positive")
        if not np.isfinite(data.density).all() or np.any(data.density <= 0):
            _fail("SPH density must be finite and positive")
        if data.energy is not None and not np.isfinite(data.energy).all():
            _fail("SPH energy must be finite")
        if data.damage is not None and (not np.isfinite(data.damage).all() or
                                        np.any((data.damage < 0) | (data.damage > 1))):
            _fail("SPH damage must be finite and in [0, 1]")
        for name in ("pressure", "crack_strain", "flaws_strain_min",
                     "flaws_strain_max", "flaws_m0", "drhodt", "dedt",
                     "S_xx", "S_yy", "S_zz", "S_xy", "S_xz", "S_yz"):
            value = getattr(data, name)
            if value is not None and not np.isfinite(value).all():
                _fail(f"SPH {name} must be finite")
        if data.c_sound is not None and (not np.isfinite(data.c_sound).all() or
                                         np.any(data.c_sound < 0)):
            _fail("SPH c_sound must be finite and nonnegative")
        if data.smoothing_length is not None and (
                not np.isfinite(data.smoothing_length).all() or
                np.any(data.smoothing_length <= 0)):
            _fail("SPH smoothing_length must be finite and positive")
        if data.flaws_ntot is not None and np.any(data.flaws_ntot < 0):
            _fail("SPH flaw count must be nonnegative")
        if data.distension is not None and (not np.isfinite(data.distension).all() or
                                            np.any(data.distension < 1)):
            _fail("SPH distension must be finite and at least one")
    return data


def write_particle_binary(filepath: str | Path, data: ParticleData, physics: str = "dem", *, record_kind: str = "input") -> None:
    """Write a strict v3 record; only fresh-input clump IDs may be defaulted."""
    if physics not in _PHYSICS_BY_NAME or record_kind not in _RECORD_BY_NAME:
        raise ValueError("physics must be dem/sph and record_kind input/output/checkpoint")
    if record_kind == "input" and data.lifecycle_state is not None:
        raise ValueError("lifecycle_state is runtime-only and cannot be written to a fresh input record")
    n = data.num_particles
    if n <= 0:
        raise ValueError("at least one particle is required")
    positions = _checked_array(data.positions, "positions", n, np.dtype("float64"), 3)
    global_id = _checked_array(data.global_id, "global_id", n, np.dtype("int32"), 1)
    material_id = _checked_array(data.material_id, "material_id", n, np.dtype("int32"), 1)
    if np.any(material_id < 0):
        raise ValueError("material_id values must be non-negative")
    if np.any(global_id < 0) or len(np.unique(global_id)) != n:
        raise ValueError("global_id values must be unique and non-negative")
    if data.clump_id is None:
        clump_id = np.zeros(n, dtype=np.int32)
    else:
        clump_id = _checked_array(data.clump_id, "clump_id", n, np.dtype("int32"), 1)
    if np.any(clump_id < -1):
        raise ValueError("clump_id values must be at least -1")
    if physics == "sph":
        _validate_sph_part_ids(clump_id)
    else:
        _validate_contiguous_clump_ids(clump_id)
    if data.lifecycle_state is None:
        lifecycle_state = np.zeros(n, dtype=np.uint8)
    else:
        lifecycle_state = _checked_array(data.lifecycle_state, "lifecycle_state", n, np.dtype("uint8"), 1)
    if np.any(lifecycle_state > 1):
        raise ValueError("lifecycle_state values must be 0 or 1")
    if record_kind == "checkpoint" and physics == "dem":
        clump_offset = _checked_array(data.clump_offset, "clump_offset", n, np.dtype("float32"), 3)
        if not np.isfinite(clump_offset).all():
            raise ValueError("clump_offset must be finite")
    velocities = np.zeros((n, 3), dtype=np.float64) if data.velocities is None else _checked_array(data.velocities, "velocities", n, np.dtype("float64"), 3)
    has_winding = _validate_winding_members((data.winding_x, data.winding_y, data.winding_z))
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("positions and velocities must be finite")
    field_ids = ([FIELD_MATERIAL_ID, 0x0011] if physics == "dem" else [FIELD_MATERIAL_ID, 0x0012, 0x0001])
    if record_kind == "input":
        # Fresh SPH input may carry an explicit per-particle smoothing length.
        # Keep the descriptor absent when no value was supplied so legacy
        # inputs retain the runner's YAML h fallback.
        optional = ([0x0010] if physics == "dem" else
                    [0x0004, 0x0005, 0x0020])
    else:
        optional = ([0x0010, 0x0012, 0x0013, 0x0014, 0x0015, 0x0016, 0x0017] if physics == "dem" else
                    [0x0004, 0x0002, 0x0003, 0x0005, 0x0014, 0x001E, 0x0020, 0x0021,
                     FIELD_FLAWS_NTOT, FIELD_FLAWS_STRAIN_MIN,
                     FIELD_FLAWS_STRAIN_MAX, FIELD_FLAWS_M0,
                     0x0030, 0x0031, 0x0032, 0x0033, 0x0034, 0x0035, 0x0040, 0x0041, 0x0042])
    if physics == "dem" and record_kind == "checkpoint":
        optional.append(0x001E)  # Retained slow-gravity cache for RESPA continuation.
    arrays = {"material_id": material_id, "clump_id": clump_id, "lifecycle_state": lifecycle_state, "clump_offset": data.clump_offset,
              "winding_x": data.winding_x, "winding_y": data.winding_y,
              "winding_z": data.winding_z,
              "radii": data.radii, "omega": data.omega, "mass": data.mass,
              "gravity_acceleration": data.gravity_acceleration,
              "moment_of_inertia": data.moment_of_inertia, "force": data.force, "torque": data.torque,
              "quaternion": data.quaternion, "num_contacts": data.num_contacts, "density": data.density,
              "smoothing_length": data.smoothing_length, "pressure": data.pressure, "c_sound": data.c_sound,
              "energy": data.energy, "damage": data.damage, "crack_strain": data.crack_strain,
              "flaws_ntot": data.flaws_ntot, "flaws_strain_min": data.flaws_strain_min,
              "flaws_strain_max": data.flaws_strain_max, "flaws_m0": data.flaws_m0,
              "S_xx": data.S_xx, "S_yy": data.S_yy, "S_zz": data.S_zz, "S_xy": data.S_xy,
              "S_xz": data.S_xz, "S_yz": data.S_yz, "distension": data.distension, "drhodt": data.drhodt, "dedt": data.dedt}
    if physics == "sph" and record_kind != "input":
        optional += list(range(0x0050, 0x0056))
        arrays.update({name: getattr(data, name) for name in ['energy_cold', 'energy_thermal_peak', 'pressure_peak', 'fracture_area', 'fragment_length', 'damping_energy']})
    if physics == "sph" and record_kind == "checkpoint":
        missing_flaw_metadata = [FIELD_REGISTRY[field_id][0] for field_id in FLAW_METADATA_FIELDS
                                 if arrays[FIELD_REGISTRY[field_id][0]] is None]
        if missing_flaw_metadata:
            raise ValueError("SPH checkpoints require compact flaw metadata: " +
                             ", ".join(missing_flaw_metadata))
    selected = []
    clump_fields = ([FIELD_CLUMP_ID] if record_kind != "input" or data.clump_id is not None else [])
    lifecycle_fields = ([FIELD_LIFECYCLE_STATE] if record_kind != "input" else [])
    if record_kind == "checkpoint" and physics == "dem":
        clump_fields.append(FIELD_CLUMP_OFFSET)
    winding_fields = list(WINDING_FIELDS) if has_winding else []
    for field_id in field_ids + optional + clump_fields + lifecycle_fields + winding_fields:
        name, input_dtype_value, count = FIELD_REGISTRY[field_id]
        exact_runtime = ((physics == "sph" and record_kind != "input" and field_id in SPH_RUNTIME_F64_FIELDS) or
                         (record_kind == "checkpoint" and field_id in (0x0014, 0x0015)))
        dtype_value = 1 if exact_runtime else input_dtype_value
        attr = _NAME_REMAP.get(name, name)
        value = arrays[attr]
        if field_id in field_ids or value is not None:
            if exact_runtime:
                runtime_value = np.asarray(value)
                expected_shape = (n,) if count == 1 else (n, count)
                if runtime_value.shape != expected_shape or runtime_value.dtype not in (
                        np.dtype("float32"), np.dtype("float64")):
                    raise ValueError(
                        f"{attr} must have shape {expected_shape} and a floating dtype, "
                        f"got {runtime_value.shape} and {runtime_value.dtype}")
                checked = runtime_value.astype(np.float64, copy=False)
            else:
                checked = _checked_array(
                    value, attr, n, DTYPE_MAP[dtype_value].newbyteorder("="), count)
            selected.append((field_id, dtype_value, checked))
    selected_values = {FIELD_REGISTRY[field_id][0]: arr for field_id, _, arr in selected}
    if physics == "dem":
        radius = selected_values["radius"]
        if not np.isfinite(radius).all() or np.any(radius <= 0):
            raise ValueError("radii must be finite and positive")
    else:
        for name in ("mass", "density"):
            value = selected_values[name]
            if not np.isfinite(value).all() or np.any(value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if "energy" in selected_values and not np.isfinite(selected_values["energy"]).all():
            raise ValueError("energy must be finite")
        if "damage" in selected_values:
            damage = selected_values["damage"]
            if not np.isfinite(damage).all() or np.any((damage < 0) | (damage > 1)):
                raise ValueError("damage must be finite and in [0, 1]")
    for name in ("omega", "force", "torque", "quaternion", "pressure",
                 "crack_strain", "drhodt", "dedt",
                 "S_xx", "S_yy", "S_zz", "S_xy", "S_xz", "S_yz"):
        if name in selected_values and not np.isfinite(selected_values[name]).all():
            raise ValueError(f"{name} must be finite")
    for name in ("mass", "moment_of_inertia"):
        if name in selected_values:
            value = selected_values[name]
            if not np.isfinite(value).all() or np.any(value <= 0):
                raise ValueError(f"{name} must be finite and positive")
    if "c_sound" in selected_values:
        value = selected_values["c_sound"]
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError("c_sound must be finite and nonnegative")
    if "smoothing_length" in selected_values:
        value = selected_values["smoothing_length"]
        if not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError("smoothing_length must be finite and positive")
    if "distension" in selected_values:
        value = selected_values["distension"]
        if not np.isfinite(value).all() or np.any(value < 1):
            raise ValueError("distension must be finite and at least one")
    if "num_contacts" in selected_values and np.any(selected_values["num_contacts"] < 0):
        raise ValueError("num_contacts must be nonnegative")
    offset = 0
    descriptors = []
    for field_id, dtype_value, arr in selected:
        _, _, count = FIELD_REGISTRY[field_id]
        descriptors.append((field_id, dtype_value, count, offset))
        offset += n * count * DTYPE_MAP[dtype_value].itemsize
    with Path(filepath).open("wb") as f:
        step = 0 if record_kind == "input" else data.current_step
        current_time = 0.0 if record_kind == "input" else data.current_time
        dt = 0.0 if record_kind == "input" else data.dt
        if step < 0 or not np.isfinite(current_time) or not np.isfinite(dt) or dt < 0:
            raise ValueError("step/time/timestep metadata is invalid")
        f.write(_HEADER.pack(MAGIC, HEADER_VERSION, HEADER_SIZE, n, step, current_time, dt,
                             len(descriptors), _PHYSICS_BY_NAME[physics], _RECORD_BY_NAME[record_kind],
                             0, 0, int(data.material_database_fingerprint),
                             int(data.clump_metadata_fingerprint)))
        for field_id, dtype_value, count, field_offset in descriptors:
            f.write(_DESCRIPTOR.pack(field_id, dtype_value, count, 0, field_offset))
        _write_soa(f, positions, 3)
        _write_soa(f, velocities, 3)
        _write_soa(f, global_id, 1)
        for (_, _, arr), (_, _, count, _) in zip(selected, descriptors):
            _write_soa(f, arr.astype(arr.dtype.newbyteorder("<"), copy=False), count)


def detect_file_type(filepath: str | Path) -> str:
    """Return the sole accepted runtime format after validating its header."""
    read_particle_binary(filepath)
    return "particle"


def ensure_particle_format(filepath: str | Path, output_dir: str | Path) -> str:
    """Validate an existing v3 file; conversion belongs to migrate_particle_v3.py."""
    read_particle_binary(filepath)
    return str(filepath)
