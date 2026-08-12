import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: long-running physics/verification test "
        "(exclude with `-m 'not slow'`)")
