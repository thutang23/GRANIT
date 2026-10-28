import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import importlib.util
spec=importlib.util.spec_from_file_location("contact_history_validation",Path(__file__).with_name("validate.py"))
driver=importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)

class FixtureTests(unittest.TestCase):
    def test_overflow_geometry_has_17_center_neighbors_only(self):
        rows=list(driver.overflow_rows());self.assertEqual(len(rows),18)
        for i,a in enumerate(rows):
            for j,b in enumerate(rows[i+1:],i+1):
                distance=math.dist(a[0],b[0])
                self.assertEqual(distance<a[3]+b[3],i==0,(i,j,distance))
        self.assertEqual(len({r[4] for r in rows}),18)
    def test_churn_initially_contact_free(self):
        rows=list(driver.rows());self.assertEqual(len(rows),22)
        for i,a in enumerate(rows):
            for b in rows[i+1:]: self.assertGreater(math.dist(a[0],b[0]),a[3]+b[3])
        self.assertGreater(sum(abs(r[0][1])<r[3]+rows[0][3] for r in rows[1:]),16)
    def test_overflow_rejects_checkpoint_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            work=Path(d);self.assertTrue(driver.audit_overflow(work)["pass"])
            (work/"checkpoints").mkdir();(work/"checkpoints"/"unexpected.bin").touch()
            with self.assertRaisesRegex(AssertionError,"overflow emitted"):driver.audit_overflow(work)

    def test_transport_requires_actual_count_change(self):
        with tempfile.TemporaryDirectory() as d:
            work=Path(d);(work/"run.log").write_text("[clumps-contact] rank=0 step=1 remote_interclump_overlap_pairs=1\n[clumps-contact] rank=1 step=1 remote_interclump_overlap_pairs=1\n")
            with self.assertRaisesRegex(AssertionError,"member migration"):driver.audit_transport(work)

    def test_momentum_uses_omega_and_stored_inertia(self):
        record={"rows":{0:(2.,3.,4.,5.,6.,7.)},"fields":{0x12:{0:(2.,)},0x13:{0:(3.,)},0x10:{0:(11.,13.,17.)},0x15:{0:(101.,103.,107.)}}}
        # p=m*v; L=m*(r cross v)+I*omega. Torque is deliberately different.
        self.assertEqual(driver.momentum(record),([10.,12.,14.],[27.,51.,45.]))

    def test_audit_rejects_insufficient_births(self):
        with tempfile.TemporaryDirectory() as d:
            work=Path(d);(work/'checkpoints').mkdir();(work/'checkpoints'/'checkpoint_step_0000000001.bin').touch()
            history={(0,1):(True,True,1.,0.,0.,0.,0.,0.,0.,0.,0.),(1,0):(True,True,-1.,0.,0.,0.,0.,0.,0.,0.,0.)}
            with patch.object(driver.ext,'read_record',return_value={'histories':history}):
                with self.assertRaisesRegex(AssertionError,'>16'):driver.audit(work)

if __name__=='__main__':unittest.main()
