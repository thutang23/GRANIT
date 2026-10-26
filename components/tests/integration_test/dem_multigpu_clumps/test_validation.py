"""CPU checks for fixture geometry and strict artifact readers; no GPU runs."""
from pathlib import Path
import math
import struct
import tempfile
import unittest

import validate as v


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, name):
        work = self.work/name
        manifest = v.generate(work, name, count=27)
        return manifest, v.read_particles(work/'input.bin', False)

    def test_all_clump_has_no_normal_surrogate(self):
        m, p = self.fixture('all-clump')
        self.assertEqual((m['particles'], m['clumps'], m['normals']), (40, 10, 0))
        self.assertEqual(set(x[0] for x in p['fields'][0x18].values()), set(range(1, 11)))

    def test_contact_is_external_and_spin_is_rigid(self):
        _, p = self.fixture('contact-migration')
        groups = {}
        for gid, (cid,) in p['fields'][0x18].items():
            groups.setdefault(cid, []).append(gid)
        separation = min(math.dist(p['rows'][i][:3], p['rows'][j][:3])
                         for i in groups[1] for j in groups[2])
        self.assertLess(separation, .02)
        self.assertAlmostEqual(separation, math.sqrt(.010**2+.016**2))
        for cid in (1, 2):
            group = groups[cid]
            com = [sum(p['rows'][i][a] for i in group)/len(group) for a in range(3)]
            vel = [sum(p['rows'][i][a+3] for i in group)/len(group) for a in range(3)]
            for gid in group:
                spin = v.cross(p['fields'][0x10][gid], [p['rows'][gid][a]-com[a] for a in range(3)])
                self.assertLess(max(abs(p['rows'][gid][a+3]-vel[a]-spin[a]) for a in range(3)), 1e-12)

    def test_throughput_and_sphere_baseline_have_identical_nonoverlapping_geometry(self):
        v.generate(self.work/'bodies', 'throughput', count=27)
        v.generate(self.work/'spheres', 'throughput', count=27, spheres=True)
        bodies = v.read_particles(self.work/'bodies'/'input.bin', False)
        spheres = v.read_particles(self.work/'spheres'/'input.bin', False)
        self.assertEqual(bodies['rows'], spheres['rows'])
        self.assertTrue(all(cid == (0,) for cid in spheres['fields'][0x18].values()))
        rows = list(bodies['rows'].values())
        self.assertGreater(min(math.dist(a[:3], b[:3]) for i, a in enumerate(rows) for b in rows[i+1:]), .02)

    def test_inferred_spin_omits_omega_preserves_rotational_velocities(self):
        _, explicit = self.fixture('contact-migration')
        _, inferred = self.fixture('inferred-spin')
        self.assertNotIn(0x10, inferred['fields'])
        self.assertEqual(explicit['rows'], inferred['rows'])
        self.assertEqual(explicit['fields'][0x18], inferred['fields'][0x18])

    def test_parser_rejects_duplicate_missing_id(self):
        self.fixture('all-clump')
        path = self.work/'all-clump'/'input.bin'
        blob = bytearray(path.read_bytes())
        nf = v.HEADER.unpack_from(blob)[7]
        struct.pack_into('<i', blob, 72+16*nf+48*40, 1)
        path.write_bytes(blob)
        with self.assertRaises(AssertionError):
            v.read_particles(path, False)

    def test_parser_rejects_trailing_bytes_and_nonfinite_state(self):
        self.fixture('all-clump')
        path = self.work/'all-clump'/'input.bin'
        good = path.read_bytes()
        path.write_bytes(good+b'garbage')
        with self.assertRaises(AssertionError):
            v.read_particles(path, False)
        blob = bytearray(good)
        struct.pack_into('<d', blob, 72+16*4, float('nan'))
        path.write_bytes(blob)
        with self.assertRaises(AssertionError):
            v.read_particles(path, False)

    def test_parser_rejects_truncation(self):
        self.fixture('all-clump')
        path = self.work/'all-clump'/'input.bin'
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises((AssertionError, struct.error)):
            v.read_particles(path, False)


if __name__ == '__main__':
    unittest.main()
