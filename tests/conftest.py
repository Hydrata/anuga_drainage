"""pytest options shared by the culvert tests."""
import pathlib

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--results-dir", action="store", default=None,
        help="directory the unsteady culvert tests write their result CSVs and "
             "unsteady-summary.json to (default: a temp dir)")


@pytest.fixture(scope="session")
def results_dir(request, tmp_path_factory):
    given = request.config.getoption("--results-dir")
    path = pathlib.Path(given) if given else tmp_path_factory.mktemp("culvert-results")
    path.mkdir(parents=True, exist_ok=True)
    return path
