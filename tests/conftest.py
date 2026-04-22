from pathlib import Path

import pytest

pytest_plugins = ["webgpu.testing"]

TESTS_DIR = Path(__file__).parent


@pytest.fixture(scope="session", autouse=True)
def _configure_dirs(webgpu_env):
    webgpu_env.output_dir = TESTS_DIR / "output"
    webgpu_env.baseline_dir = TESTS_DIR / "baselines"