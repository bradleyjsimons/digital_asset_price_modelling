import pytest
import sys


if len(sys.argv) == 2 and sys.argv[1] == "all":
    pytest.main(["--cov=src", "--cov-report", "html", "--cov-config=.coveragerc"])
else:
    pytest.main(
        [
            "__tests__/data/test_data_controller.py",
            "--cov=src",
            "--cov-report",
            "html",
            "--cov-config=.coveragerc",
        ]
    )
