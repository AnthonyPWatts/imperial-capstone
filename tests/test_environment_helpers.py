"""Exercise the Windows helpers in standalone checkouts without downloads."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SHELLS = [path for name in ("powershell.exe", "pwsh.exe") if (path := shutil.which(name))]
BASE_PYTHON = sys._base_executable


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


@unittest.skipUnless(os.name == "nt" and SHELLS, "Requires Windows PowerShell or PowerShell 7")
class EnvironmentHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="capstone environment tests ")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.scratch = Path(cls.temporary.name)
        cls.foreign = cls.scratch / "other project" / ".venv"
        subprocess.run([BASE_PYTHON, "-I", "-m", "venv", "--without-pip", str(cls.foreign)],
                       check=True, capture_output=True, text=True)
        cls.foreign_config = (cls.foreign / "pyvenv.cfg").read_bytes()

    def setUp(self):
        self.temporary_repo = TemporaryDirectory(prefix="standalone capstone ", dir=self.scratch)
        self.addCleanup(self.temporary_repo.cleanup)
        self.repo = Path(self.temporary_repo.name)
        for name in ("setup-venv.ps1", "run-python.ps1"):
            shutil.copyfile(ROOT / name, self.repo / name)
        # Real venv and pip, but no modelling packages or network dependency.
        (self.repo / "requirements.txt").write_text("# No downloads in helper tests.\n")
        self.env = os.environ.copy()
        self.env["VIRTUAL_ENV"] = str(self.foreign)
        self.env["PATH"] = str(self.foreign / "Scripts") + os.pathsep + self.env["PATH"]
        self.env["PIP_NO_INDEX"] = "1"
        self.env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    def invoke(self, shell, script="setup-venv.ps1", arguments="", env=None):
        command = f"& {ps_quote(self.repo / script)} {arguments}"
        if script == "run-python.ps1":
            # -Command otherwise converts every non-zero script exit code to 1.
            command += "; exit $LASTEXITCODE"
        return subprocess.run(
            [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command",
             command],
            cwd=self.scratch, env=env or self.env, capture_output=True, text=True, timeout=60)

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_fresh_setup_and_reuse_ignore_active_foreign_environment(self):
        for shell in SHELLS:
            with self.subTest(shell=shell):
                # Separate clone for each shell so both exercise creation.
                original_repo = self.repo
                self.repo = original_repo / Path(shell).stem
                self.repo.mkdir()
                for name in ("setup-venv.ps1", "run-python.ps1", "requirements.txt"):
                    shutil.copyfile(original_repo / name, self.repo / name)
                try:
                    result = self.invoke(shell, arguments=f"-Python {ps_quote(BASE_PYTHON)}")
                    self.assert_success(result)
                    self.assertIn("Another environment is active", result.stdout)
                    config = (self.repo / ".venv" / "pyvenv.cfg").read_text()
                    self.assertIn("prompt = 'imperial-capstone'", config)
                    self.assertIn("include-system-site-packages = false", config)
                    marker = self.repo / ".venv" / "keep-me.txt"
                    marker.write_text("Preserve on reuse")
                    self.assert_success(self.invoke(shell))
                    self.assertEqual(marker.read_text(), "Preserve on reuse")
                    self.assertEqual((self.foreign / "pyvenv.cfg").read_bytes(), self.foreign_config)
                    # An unsatisfiable requirement must fail, never report Ready or clear the venv.
                    (self.repo / "requirements.txt").write_text("capstone-test-nonexistent-package==0\n")
                    failed = self.invoke(shell)
                    self.assertNotEqual(failed.returncode, 0)
                    self.assertIn("Dependency installation failed", failed.stdout + failed.stderr)
                    self.assertNotIn("Ready:", failed.stdout)
                    self.assertTrue(marker.exists())
                finally:
                    self.repo = original_repo

    def test_whatif_creates_nothing(self):
        for shell in SHELLS:
            with self.subTest(shell=shell):
                self.assert_success(self.invoke(shell, arguments=f"-Python {ps_quote(BASE_PYTHON)} -WhatIf"))
                self.assertFalse((self.repo / ".venv").exists())

    def test_foreign_venv_cannot_be_used_as_base(self):
        for shell in SHELLS:
            with self.subTest(shell=shell):
                result = self.invoke(shell, arguments=f"-Python {ps_quote(self.foreign / 'Scripts/python.exe')}")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("base 64-bit Python 3.14", result.stdout + result.stderr)
                self.assertFalse((self.repo / ".venv").exists())

    def test_incomplete_environment_is_preserved(self):
        (self.repo / ".venv").mkdir()
        marker = self.repo / ".venv" / "keep-me.txt"
        marker.write_text("Do not clear implicitly")
        for shell in SHELLS:
            with self.subTest(shell=shell):
                result = self.invoke(shell)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("existing .venv must be", result.stdout + result.stderr)
                self.assertEqual(marker.read_text(), "Do not clear implicitly")

    def test_recreate_refuses_active_target_environment(self):
        (self.repo / ".venv").mkdir()
        env = dict(self.env, VIRTUAL_ENV=str(self.repo / ".venv"))
        for shell in SHELLS:
            with self.subTest(shell=shell):
                result = self.invoke(shell, arguments="-Recreate", env=env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Deactivate this environment", result.stdout + result.stderr)
                self.assertTrue((self.repo / ".venv").exists())

    def test_runner_preserves_arguments_cwd_and_exit_code(self):
        subprocess.run([BASE_PYTHON, "-I", "-m", "venv", "--without-pip", str(self.repo / ".venv")],
                       check=True, capture_output=True, text=True)
        probe = "import json, os, sys; print(json.dumps([sys.executable, os.getcwd(), sys.argv[1:]]))"
        for shell in SHELLS:
            with self.subTest(shell=shell):
                result = self.invoke(shell, "run-python.ps1", f"-c {ps_quote(probe)} 'two words' --flag")
                self.assert_success(result)
                executable, cwd, arguments = json.loads(result.stdout)
                self.assertEqual(Path(executable), self.repo / ".venv/Scripts/python.exe")
                self.assertEqual(Path(cwd), self.scratch)
                self.assertEqual(arguments, ["two words", "--flag"])
                failed = self.invoke(shell, "run-python.ps1", "-c 'import sys; sys.exit(7)'")
                self.assertEqual(failed.returncode, 7, failed.stdout + failed.stderr)

    def test_runner_never_falls_back_when_local_environment_is_missing(self):
        for shell in SHELLS:
            with self.subTest(shell=shell):
                result = self.invoke(shell, "run-python.ps1", "--version")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("setup-venv.ps1", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
