"""Exercise batch orchestration with a stand-in executable, never model inference."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_all.sh'
DATASETS = ('michaelov_2024', 'nieuwland_2018', 'szewczyk_2022', 'peelle')


class BatchScriptTests(unittest.TestCase):
    def test_all_combinations_and_stop_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data/processed').mkdir(parents=True)
            for name in DATASETS:
                (root / f'data/processed/{name}.csv').touch()
            binary = root / 'bin'
            binary.mkdir()
            python = binary / 'python'
            python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$NWP_BATCH_LOG"\nexit "${NWP_EXIT_CODE:-0}"\n')
            python.chmod(0o755)
            log = root / 'calls.log'
            env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'], NWP_BATCH_LOG=str(log))
            completed = subprocess.run(['bash', str(SCRIPT), '--device', 'cuda', '--dtype', 'bfloat16'], cwd=root, env=env, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            lines = log.read_text().splitlines()
            expected = {f'-m next_word_prediction --dataset data/processed/{d}.csv --model {m} --device cuda --dtype bfloat16'
                        for m in ('bert', 'deepseek', 'qwen', 'llama') for d in DATASETS}
            self.assertEqual(set(lines), expected)
            self.assertEqual(len(lines), 16)
            log.write_text('')
            env['NWP_EXIT_CODE'] = '1'
            completed = subprocess.run(['bash', str(SCRIPT)], cwd=root, env=env, capture_output=True)
            self.assertEqual(completed.returncode, 1)
            self.assertEqual(len(log.read_text().splitlines()), 1)

    def test_rejects_options_that_would_override_batch_selection(self):
        completed = subprocess.run(['bash', str(SCRIPT), '--model', 'bert'], capture_output=True)
        self.assertEqual(completed.returncode, 2)


if __name__ == '__main__':
    unittest.main()
