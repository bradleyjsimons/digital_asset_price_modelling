import pytest
import sys

if len(sys.argv) == 2 and sys.argv[1] == "all":
    pytest.main(["--cov=src", "--cov-report", "html", "--cov-config=.coveragerc"])
else:
    pytest.main(
        [
            "tests/unit/services/kraken/spot/test_kraken_spot_orders_service.py",
            "--cov=src",
            "--cov-report",
            "html",
            "--cov-config=.coveragerc",
        ]
    )
