"""Python checkpoint ABI compatibility; run with numpy/PyYAML test environment."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, 'numpy is required for Python binary-I/O compatibility')
class PythonCheckpointIOTests(unittest.TestCase):
    def test_force_torque_checkpoint_roundtrip_retains_f64_and_reads_legacy(self):
        root = Path(__file__).resolve().parents[4]
        sys.path.insert(0, str(root/'src/plugins/scripts'))
        from cudadem_pipeline.common.io import read_particle_binary, write_particle_binary
        sys.path.insert(0, str(Path(__file__).parents[1]/'dem_multigpu_clumps'))
        spec = importlib.util.spec_from_file_location('io_contact_fixture', Path(__file__).parents[1]/'dem_multigpu_clumps/test_extension_validation.py')
        fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'legacy.bin'; path.write_bytes(fixture.checkpoint_bytes())
            data = read_particle_binary(path)
            self.assertEqual(data.force.dtype, np.dtype('float32'))
            self.assertEqual(data.torque.dtype, np.dtype('float32'))
            values = np.array([[1.+2.**-40, -2.+2.**-39, 3.-2.**-38]]*2, dtype=np.float64)
            data.force = values; data.torque = -values
            data.gravity_acceleration = values*.125
            output = Path(folder)/'exact.bin'
            write_particle_binary(output, data, physics='dem', record_kind='checkpoint')
            restored = read_particle_binary(output)
            self.assertEqual(restored.force.dtype, np.dtype('float64'))
            self.assertEqual(restored.torque.dtype, np.dtype('float64'))
            self.assertTrue(np.array_equal(restored.force, values))
            self.assertTrue(np.array_equal(restored.torque, -values))
            self.assertTrue(np.array_equal(restored.gravity_acceleration, values*.125))


if __name__ == '__main__': unittest.main()
