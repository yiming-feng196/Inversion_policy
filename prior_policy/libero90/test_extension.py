import fcntl
import json
from pathlib import Path
import tempfile
import unittest

from continue_extension import inspect_jobs, lock_is_free, validate_scope
from run_queue import job_disposition


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parent
        self.scope = json.loads((root / 'active_scope.json').read_text())
        self.registry = json.loads((root / 'tasks.json').read_text())

    def test_scope_preserves_original_and_protocol(self):
        ids = validate_scope(self.scope, self.registry)
        self.assertEqual(ids[:5], [35, 13, 8, 44, 73])
        self.assertEqual(len(ids), 15)
        self.assertNotIn('pi0', self.scope['model_families'])
        self.assertEqual(self.scope['model_families'], ['unet', 'dit'])
        self.assertEqual(self.scope['paused_model_families'], ['dp_unet'])

    def test_paused_diffusion_cannot_be_dispatched(self):
        scope = {**self.scope, 'model_families': ['unet', 'dit', 'dp_unet']}
        with self.assertRaises(ValueError):
            validate_scope(scope, self.registry)

    def test_protocol_mutation_rejected(self):
        scope = {**self.scope, 'source_stride': 16}
        with self.assertRaises(ValueError):
            validate_scope(scope, self.registry)

    def test_duplicate_rejected(self):
        scope = {**self.scope, 'task_ids': self.scope['task_ids'] + [0]}
        with self.assertRaises(ValueError):
            validate_scope(scope, self.registry)

    def test_original_order_is_preserved(self):
        scope = {**self.scope, 'task_ids': list(reversed(self.scope['task_ids']))}
        with self.assertRaises(ValueError):
            validate_scope(scope, self.registry)

    def test_busy_lock_prevents_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'worker.lock'
            with path.open('a') as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertFalse(lock_is_free(path))
            self.assertTrue(lock_is_free(path))

    def test_failures_stay_explicit(self):
        self.assertEqual(job_disposition('failed', False), 'raise_failed')
        self.assertEqual(job_disposition('failed', True), 'skip_failed')
        self.assertEqual(job_disposition('complete', True), 'skip_complete')
        self.assertEqual(job_disposition('paused', True), 'run')

    def test_manifest_inventory_does_not_drop_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'jobs/t035_unet/status.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'status': 'failed'}))
            result = inspect_jobs(root, self.scope)
            self.assertEqual(result['failed'], ['t035_unet'])
            self.assertEqual(len(result['pending']), 29)


if __name__ == '__main__':
    unittest.main()
