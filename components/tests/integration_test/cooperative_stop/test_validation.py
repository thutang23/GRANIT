"""CPU harness/oracle regressions. These are not CUDA solver validation."""
import importlib.util
import os
from pathlib import Path
import signal
import struct
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('cooperative_stop_validation', Path(__file__).with_name('validate.py'))
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
fixture_spec = importlib.util.spec_from_file_location('history_fixture', Path(__file__).parents[1]/'dem_multigpu_clumps/test_extension_validation.py')
fixture = importlib.util.module_from_spec(fixture_spec); fixture_spec.loader.exec_module(fixture)


class CooperativeStopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def record(self, name, padding=b'\x00\x00', cache=1, spring=.001):
        path = self.work/name
        path.write_bytes(fixture.checkpoint_bytes(padding, cache, spring))
        return v.read_checkpoint(path)

    def executable(self, body):
        path = self.work/'solver'
        path.write_text('#!'+sys.executable+'\n'+body)
        path.chmod(0o755)
        (self.work/'config.json').write_text('{}')
        return path

    def test_real_signal_sent_after_progress_and_bounded_exit(self):
        binary = self.executable('''import signal, time
from pathlib import Path
signal.signal(signal.SIGUSR1, lambda s, f: exit(0))
out = Path(__file__).parent/'output'; out.mkdir()
(out/'output_step_0000000004.bin').write_bytes(b'X'*72)
while True: time.sleep(.01)
''')
        result = v.launch(binary, self.work, 5, signal.SIGUSR1)
        self.assertEqual(result['returncode'], 0)
        self.assertEqual(result['signal_after_observed_step'], 4)
        self.assertLess(result['signal_to_exit_seconds'], 2)

    def test_exit_before_signal_is_failure(self):
        binary = self.executable('pass\n')
        with self.assertRaisesRegex(AssertionError, 'before requested signal'):
            v.launch(binary, self.work, 5, signal.SIGUSR1)

    def test_relative_work_fixture_uses_absolute_config_paths(self):
        path = Path(os.path.relpath(self.work/'relative-fixture', Path.cwd()))
        config = v.prepare_dem(path, 0, False)
        self.assertTrue(Path(config['material_database']).is_absolute())
        self.assertTrue(Path(config['io']['input']['particle_file']).is_absolute())
        self.assertNotIn('path', config['io']['checkpoint'])

    def test_runtime_variants_are_unclumped_and_keep_physical_settings_on_resume(self):
        for variant in ('yorp1', 'yorp3', 'yorp-adaptive', 'nonmorton-adaptive', 'morton-adaptive'):
            first = v.prepare_variant(self.work/variant, 2, False, variant=variant)
            resumed = v.prepare_variant(self.work/(variant+'-resume'), 2, True,
                                        self.work/'checkpoint.bin', 64*v.DEM_DT, variant)
            self.assertEqual(first['clumps']['instances'], [])
            self.assertEqual(first['accelerations'], resumed['accelerations'])
            self.assertFalse(first['io']['checkpoint']['enabled'])
            self.assertNotIn('path', first['io']['checkpoint'])

    def test_yorp_runtime_comparison_excludes_only_abi_padding(self):
        header = struct.pack('<8s4IqdQ', b'YORPRUN1', 1, 1, 296, 0, 8, 8*v.DEM_DT, 123)
        original = bytearray(header+bytes(296))
        changed = bytearray(original)
        for offset in (64+28, 160+28, 256+36):
            changed[48+offset:48+offset+4] = b'PAD!'
        a = self.work/'a'; b = self.work/'b'
        Path(str(a)+'.yorp_runtime').write_bytes(original)
        Path(str(b)+'.yorp_runtime').write_bytes(changed)
        self.assertEqual(v.sidecar_state(a, '.yorp_runtime'), v.sidecar_state(b, '.yorp_runtime'))
        struct.pack_into('<d', changed, 48+16, .125)  # next_time is meaningful.
        Path(str(b)+'.yorp_runtime').write_bytes(changed)
        self.assertNotEqual(v.sidecar_state(a, '.yorp_runtime'), v.sidecar_state(b, '.yorp_runtime'))

    def test_rank_diagnostic_audit_detects_missing_capacity_service_row(self):
        path = self.work/'output/adaptive_depth_rank_0/adaptive_fmm_depth.csv'
        path.parent.mkdir(parents=True)
        path.write_text('sequence,evaluation_counter\n1,1\n2,2\n')
        self.assertEqual(next(iter(v.audit_rank_diagnostics(self.work).values()))['rows'], 2)
        path.write_text('sequence,evaluation_counter\n1,1\n3,2\n')
        with self.assertRaisesRegex(AssertionError, 'gap or duplicate'):
            v.audit_rank_diagnostics(self.work)

    def test_nonmorton_state_ignores_restore_marker_but_keeps_epoch(self):
        blob = bytearray(128)
        struct.pack_into('<8sIIqdQ', blob, 0, b'AFMMDPTH', 4, 88, 8, 8*v.DEM_DT, 123)
        a = self.work/'a'; b = self.work/'b'
        Path(str(a)+'.adaptive_fmm_depth').write_bytes(blob)
        blob[88] = 1
        Path(str(b)+'.adaptive_fmm_depth').write_bytes(blob)
        self.assertEqual(v.sidecar_state(a, '.adaptive_fmm_depth'), v.sidecar_state(b, '.adaptive_fmm_depth'))
        blob[48] = 1  # Gravity epoch is substantive and must still differ.
        Path(str(b)+'.adaptive_fmm_depth').write_bytes(blob)
        self.assertNotEqual(v.sidecar_state(a, '.adaptive_fmm_depth'), v.sidecar_state(b, '.adaptive_fmm_depth'))

    def test_hung_child_is_killed_on_timeout(self):
        binary = self.executable('''import os, time
from pathlib import Path
(Path(__file__).parent/'pid').write_text(str(os.getpid()))
while True: time.sleep(.01)
''')
        with self.assertRaises(TimeoutError):
            v.launch(binary, self.work, .2)
        pid = int((self.work/'pid').read_text())
        with self.assertRaises(ProcessLookupError): os.kill(pid, 0)

    def test_meaningful_state_exact_ignores_contact_padding_and_cache(self):
        a = self.record('a.bin')
        b = self.record('b.bin', b'\xaa\x7f', -999)
        self.assertTrue(v.exact_state(a, b)['pass'])

    def test_one_spring_float_change_is_detected(self):
        a = self.record('a.bin')
        b = self.record('b.bin', spring=.0010000002)
        self.assertFalse(v.exact_state(a, b)['pass'])

    def test_truncated_checkpoint_is_rejected(self):
        path = self.work/'bad.bin'
        path.write_bytes(fixture.checkpoint_bytes()[:-1])
        with self.assertRaises(AssertionError): v.read_checkpoint(path)

    def test_alias_resolution_uses_immutable_generation_for_sidecars(self):
        generation = self.work/'.checkpoint-generation-test'; generation.mkdir()
        payload = generation/'checkpoint.bin'; payload.write_bytes(fixture.checkpoint_bytes())
        (generation/'COMPLETE').write_text('test manifest')
        checkpoints = self.work/'checkpoints'; checkpoints.mkdir()
        alias = checkpoints/'checkpoint_step_0000000002.bin'
        alias.symlink_to(payload)
        actual_alias, resolved, record = v.checkpoint(self.work)
        self.assertEqual(actual_alias, alias)
        self.assertEqual(resolved, payload)
        self.assertEqual(record['step'], 2)

    def test_legacy_f32_and_exact_f64_force_fields_both_parse(self):
        # Existing comparator now accepts both checkpoint ABIs; change only
        # descriptor/payload representation, preserving every physical value.
        blob = fixture.checkpoint_bytes()
        h = list(v.base.HEADER.unpack_from(blob)); n, nf = h[3], h[7]
        start = 72+16*nf; payload = start+52*n
        entries = []; chunks = []; offset = 0; old_end = 0
        for i in range(nf):
            fid, dtype, count, reserved, previous = v.base.DESC.unpack_from(blob, 72+16*i)
            size = struct.calcsize(v.base.TYPES[dtype])*n*count
            chunk = blob[payload+previous:payload+previous+size]
            old_end = max(old_end, previous+size)
            if fid in (0x14, 0x15):
                vals = struct.unpack('<'+v.base.TYPES[dtype]*(n*count), chunk)
                dtype = 1; chunk = struct.pack('<'+'d'*(n*count), *vals)
            entries.append(v.base.DESC.pack(fid, dtype, count, reserved, offset))
            chunks.append(chunk); offset += len(chunk)
        path = self.work/'f64.bin'
        path.write_bytes(blob[:72]+b''.join(entries)+blob[start:payload]+b''.join(chunks)+blob[payload+old_end:])
        new = v.ext.read_record(path)
        old = self.record('f32.bin')
        self.assertEqual(new['fields'][0x14], old['fields'][0x14])
        self.assertEqual(new['fields'][0x15], old['fields'][0x15])


if __name__ == '__main__': unittest.main()
