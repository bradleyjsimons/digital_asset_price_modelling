import pytest
import sys


if len(sys.argv) == 2 and sys.argv[1] == "all":
    pytest.main(["--cov=src", "--cov-report", "html", "--cov-config=.coveragerc"])
else:
    pytest.main(
        [
            "tests/learning/rl/test_environment.py",
            "--cov=src",
            "--cov-report",
            "html",
            "--cov-config=.coveragerc",
        ]
    )
