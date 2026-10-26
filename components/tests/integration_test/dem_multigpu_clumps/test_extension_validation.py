"""CPU-only extension regressions: semantic history, shear images, global leaves."""
import copy
import math
from pathlib import Path
import struct
import tempfile
import unittest

import extension_validation as ext
import validate as base


def checkpoint_bytes(padding=b'\x00\x00', cache=1, spring=.001):
    n, stride = 2, 1
    fields = [(6, 2, 1, (0, 0)), (0x10, 1, 3, (0.,)*6),
        (0x11, 0, 1, (.01, .01)), (0x12, 0, 1, (1., 1.)),
        (0x13, 0, 1, (.1, .1)), (0x14, 0, 3, (0.,)*6),
        (0x15, 0, 3, (0.,)*6), (0x16, 1, 4, (1., 1., 0., 0., 0., 0., 0., 0.)),
        (0x17, 2, 1, (1, 1)), (0x18, 2, 1, (1, 2)),
        (0x19, 0, 3, (0.,)*6), (0x1d, 4, 1, (0, 0))]
    descriptors, payload, offset = [], [], 0
    for fid, dtype, count, values in fields:
        blob = struct.pack('<'+base.TYPES[dtype]*len(values), *values)
        descriptors.append(base.DESC.pack(fid, dtype, count, 0, offset))
        payload.append(blob); offset += len(blob)
    header = base.HEADER.pack(b'GRANIT\0\0', 3, 72, n, 2, 2e-5, 1e-5, len(fields), 1, 3, stride, 2, 123, 456)
    tier1 = struct.pack('<12d2i', -.1, .1, 0., 0., 0., 0., *(0.,)*6, 0, 1)
    history = []
    for gid in range(n):
        record = bytearray(ext.CONTACT.pack(1-gid, cache, 1, 0, spring, 0., 0., 0., spring, 0., 0., 0., spring))
        record[10:12] = padding
        history.append(record)
    return header+b''.join(descriptors)+tier1+b''.join(payload)+b''.join(history)


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def record(self, name, data):
        path = self.work/name
        path.write_bytes(data)
        return ext.read_record(path)

    def test_contact_history_semantics_ignore_padding_and_local_cache(self):
        a = self.record('a.bin', checkpoint_bytes())
        b = self.record('b.bin', checkpoint_bytes(b'\xaa\x7f', -999))
        result = ext.semantic_history_compare(a, b)
        self.assertTrue(result['pass'])
        self.assertEqual(result['nonzero_spring_contacts'], 2)
        self.assertEqual(a['material_fingerprint'], 123)
        self.assertEqual(a['clump_fingerprint'], 456)

    def test_contact_history_detects_one_float_change(self):
        a = self.record('a.bin', checkpoint_bytes())
        b = self.record('b.bin', checkpoint_bytes(spring=.0010000002))
        self.assertFalse(ext.semantic_history_compare(a, b)['pass'])

    def test_checkpoint_rejects_duplicate_id_and_truncated_history(self):
        blob = bytearray(checkpoint_bytes())
        nf = base.HEADER.unpack_from(blob)[7]
        struct.pack_into('<i', blob, 72+16*nf+48*2, 1)
        with self.assertRaises(AssertionError):
            self.record('duplicate.bin', blob)
        with self.assertRaises(AssertionError):
            self.record('truncated.bin', checkpoint_bytes()[:-1])

    def test_zero_active_contacts_accepts_omitted_or_full_inactive_blob(self):
        blob = bytearray(checkpoint_bytes())
        header = list(base.HEADER.unpack_from(blob)); header[11] = 0
        blob[:72] = base.HEADER.pack(*header)
        for index in range(2):
            blob[len(blob)-2*ext.CONTACT.size+index*ext.CONTACT.size+8] = 0
        full = self.record('inactive-full.bin', blob)
        omitted = self.record('inactive-omitted.bin', blob[:-2*ext.CONTACT.size])
        self.assertEqual(full['histories'], {})
        self.assertEqual(omitted['histories'], {})

    def config(self):
        return {'domain': {'box_min': [-1.]*3, 'box_max': [1.]*3},
                'shear_box': {'enabled': True, 'omega': .1, 'shear_parameter': 1.5,
                    'time_origin': 0., 'gravity_images': {'layers_x': 1, 'layers_y': 1, 'cutoff_radius': 0.}},
                'accelerations': {'self_gravity': {'unified_fmm': {'gravitational_constant': 1.}}}}

    def test_nearest_internal_copy_across_seam_is_not_central(self):
        result = ext.nearest_image((.995, 0., 0.), (-.995, 0., 0.), self.config(), 0.)
        self.assertEqual(result[1:3], (1, 0))
        self.assertAlmostEqual(result[0], .01)

    def test_finite_images_preserve_nonlocal_same_clump_replicas(self):
        config = self.config()
        record = {'time': 0., 'rows': {0: (.995, .03, .1, 0., 0., 0.), 1: (-.995, .03, .1, 0., 0., 0.)},
                  'fields': {0x18: {0: (1,), 1: (1,)}, 0x12: {0: (1.,), 1: (1.,)}}}
        forces, excluded = ext.finite_image_direct(record, config)
        self.assertEqual(excluded[(0, 1)], (1, 0))
        self.assertGreater(base.norm(forces[0]), 0.)
        self.assertLess(base.norm(forces[0]), 1.)  # erroneous internal copy would contribute10^4

    def test_shear_image_phase_and_velocity_shift(self):
        length, shift, speed = ext.shear_phase(self.config(), 1.)
        self.assertAlmostEqual(shift, .3)
        self.assertAlmostEqual(speed, .3)
        point = ext.image_position((-.995, .1, 0.), 1, 0, self.config(), 1.)
        self.assertAlmostEqual(point[0], 1.005)
        self.assertAlmostEqual(point[1], -.2)

    def test_full_inertia_hill_wrench_matches_point_quadrature(self):
        config = self.config()
        config['shear_box']['vertical_frequency'] = .07
        body = [0.]*45
        body[:6] = [.2, -.3, .1, .03, .04, -.02]
        body[6:10] = [math.cos(.3), 0., 0., math.sin(.3)]
        body[10:13] = [.7, -.9, 1.1]
        points = [(-.1, -.2, -.3), (-.1, .2, .3), (.1, -.2, .3), (.1, .2, -.3)]
        body[13] = len(points)
        tensor = [[sum((sum(x*x for x in p) if i == j else 0.)-p[i]*p[j] for p in points)
                   for j in range(3)] for i in range(3)]
        body[14:20] = [tensor[0][0], tensor[1][1], tensor[2][2], tensor[0][1], tensor[0][2], tensor[1][2]]
        analytic_force, analytic_torque = ext.hill_body_wrench(body, config)
        world_spin = ext.rotate(body[6:10], body[10:13])
        force, torque = [0.]*3, [0.]*3
        for point in points:
            arm = ext.rotate(body[6:10], point)
            spin = base.cross(world_spin, arm)
            x, y, z = [body[a]+arm[a] for a in range(3)]
            vx, vy, vz = [body[a+3]+spin[a] for a in range(3)]
            f = (2*.1*vy+2*1.5*.1**2*x, -2*.1*vx, -.07**2*z)
            moment = base.cross(arm, f)
            for a in range(3):
                force[a] += f[a]; torque[a] += moment[a]
        self.assertLess(max(abs(force[a]-analytic_force[a]) for a in range(3)), 1e-14)
        self.assertLess(max(abs(torque[a]-analytic_torque[a]) for a in range(3)), 1e-14)

    def test_global_occupancy_uses_sum_not_max_local(self):
        shards = [[(.1, .1, .1)]*80, [(.1, .1, .1)]*80]
        local = [max(ext.global_leaf_counts(s, (0.,)*3, 1., 2).values()) for s in shards]
        global_max = max(ext.global_leaf_counts([p for s in shards for p in s], (0.,)*3, 1., 2).values())
        self.assertEqual(local, [80, 80])
        self.assertEqual(global_max, 160)
        self.assertFalse(ext.admissible_depths([p for s in shards for p in s], (0.,)*3, 1., .01, 2, 2)[0]['pass'])

    def test_occupancy_and_full_diameter_can_have_no_joint_depth(self):
        points = [(.125+i*1e-6, .125, .125) for i in range(129)]
        rows = ext.admissible_depths(points, (0.,)*3, 1., .20, 2, 5)
        self.assertTrue(all(not row['pass'] for row in rows))
        self.assertTrue(any(row['global_max_leaf_count'] > 128 for row in rows))
        self.assertTrue(any(row['diameter'] >= row['leaf_width'] for row in rows))

    def test_every_fixture_prepares_without_gpu(self):
        for fixture in ext.FIXTURES:
            work = self.work/fixture
            manifest = ext.prepare(work, fixture)
            data = ext.read_record(work/'input.bin')
            self.assertEqual(data['n'], manifest['particles'])
            self.assertGreater(manifest['clumps'], 0)

    def test_combined_shear_adaptive_configuration_preserves_contact_geometry(self):
        import json
        work = self.work/'combined'
        ext.prepare(work, 'shear-adaptive-contact')
        config = json.loads((work/'config.json').read_text())
        adaptive = config['accelerations']['self_gravity']['unified_fmm']['adaptive_depth']
        self.assertEqual((adaptive['min_depth'], adaptive['max_depth']), (2, 6))
        self.assertEqual(adaptive['trial_interval_evaluations'], 2)
        self.assertEqual(adaptive['trial_warmup_evaluations'], 0)
        self.assertEqual(adaptive['trial_samples'], 1)
        source = ext.read_record(work/'input.bin')
        group1 = [gid for gid, cid in source['fields'][0x18].items() if cid == (1,)]
        group2 = [gid for gid, cid in source['fields'][0x18].items() if cid == (2,)]
        nearest = min(ext.nearest_image(source['rows'][a][:3], source['rows'][b][:3], config, 0.)[0]
                      for a in group1 for b in group2)
        self.assertLess(nearest, .02)

    def test_new_shear_near_control_is_contact_free_split_and_resolved(self):
        import json
        work = self.work/'near-control'
        ext.prepare(work, 'shear-near-gravity')
        config = json.loads((work/'config.json').read_text())
        data = ext.read_record(work/'input.bin')
        self.assertEqual(config['accelerations']['self_gravity']['unified_fmm']['tree_depth'], 2)
        for gid, row in data['rows'].items():
            wind = math.floor((row[0]+1.)/2.)
            data['rows'][gid] = (row[0]-wind*2., *row[1:])
        data['fields'][0x12] = {gid: (1000.*4*math.pi/3.*data['fields'][0x11][gid][0]**3,) for gid in data['rows']}
        direct, excluded = ext.finite_image_direct(data, config)
        self.assertGreater(sum(nx != 0 or ny != 0 for nx, ny in excluded.values()), 0)
        groups = [[gid for gid, cid in data['fields'][0x18].items() if cid == (index,)] for index in (1, 2)]
        distances = [ext.nearest_image(data['rows'][a][:3], data['rows'][b][:3], config, 0.)[0] for a in groups[0] for b in groups[1]]
        self.assertGreater(min(distances), .002)
        # At actual depth4 the h=.5 nearest interbody image is in the P2P
        # neighborhood, while the finite images at distance~2 still usefarwork.
        for a in groups[0]:
            cell = [math.floor((data['rows'][a][axis]+4.)/.5) for axis in range(3)]
            for b in groups[1]:
                point = ext.nearest_image(data['rows'][a][:3], data['rows'][b][:3], config, 0.)[3]
                other = [math.floor((point[axis]+4.)/.5) for axis in range(3)]
                self.assertLessEqual(max(abs(cell[axis]-other[axis]) for axis in range(3)), 1)
        torque = [0.]*3
        for gid in groups[0]:
            point = ext.nearest_image((.9995, .0003, -.0003), data['rows'][gid][:3], config, 0.)[3]
            moment = base.cross(tuple(point[a]-(.9995, .0003, -.0003)[a] for a in range(3)), direct[gid])
            for a in range(3):torque[a] += moment[a]
        self.assertGreater(base.norm(torque), 1e-15)

    def test_unclumped_causality_subtracts_only_exact_internal_copy(self):
        import json
        config = self.config()
        config['accelerations']['self_gravity']['unified_fmm']['softening_factor'] = 0.
        config['clumps'] = {'instances': [{'id': 1}]}
        records = {}
        for name, membership in [('clump', (1, 1)), ('sphere', (0, 0))]:
            work = self.work/name; (work/'output').mkdir(parents=True)
            blob = bytearray(checkpoint_bytes()[:-2*ext.CONTACT.size])
            header = list(base.HEADER.unpack_from(blob));header[4] = 0;header[5] = 0.;header[9] = 2;header[10] = header[11] = 0
            blob[:72] = base.HEADER.pack(*header)
            tier2 = 72+16*header[7]+52*header[3]
            for i in range(header[7]):
                fid, dtype, count, reserved, offset = base.DESC.unpack_from(blob,72+16*i)
                if fid == 0x18:struct.pack_into('<2i',blob,tier2+offset,*membership)
            path = work/'output/output_step_0000000000.bin';path.write_bytes(blob)
            records[name] = ext.read_record(path)
            cfg = copy.deepcopy(config)
            if name == 'sphere':cfg['clumps']['instances'] = []
            (work/'config.json').write_text(json.dumps(cfg))
        exact, excluded = ext.finite_image_direct(records['clump'], config)
        clump_force = {gid: tuple(row[a]+(.0001,.0002,-.0003)[a] for a in range(3)) for gid,row in exact.items()}
        sphere_force = {}
        for gid,row in records['clump']['rows'].items():
            other = records['clump']['rows'][1-gid]
            delta = tuple(other[a]-row[a] for a in range(3))
            sphere_force[gid] = tuple(clump_force[gid][a]+delta[a]/base.norm(delta)**3 for a in range(3))
        for name, forces in [('clump',clump_force),('sphere',sphere_force)]:
            record = records[name]
            path = Path(record['path']+'.gravity_force')
            path.write_bytes(struct.pack('<8sIiqdd',b'DEMOUT\0\0',1,2,record['step'],record['time'],record['dt'])+
                             struct.pack('<6d',*(forces[gid][a] for a in range(3) for gid in range(2))))
        result = ext.analyze_sphere_causality_control(self.work/'clump',self.work/'sphere')
        self.assertTrue(result['pass'])
        self.assertFalse(result['original_clump_vs_direct']['pass'])
        self.assertGreater(result['internal_force_l2_norm'], 1.)


if __name__ == '__main__':
    unittest.main()
