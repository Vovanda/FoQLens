"""The station's log takes a test session only when it ran a GPU test file."""

from types import SimpleNamespace

from conftest import ran_gpu_tests


def reporter(**stats: list[str]) -> SimpleNamespace:
    return SimpleNamespace(stats={k: [SimpleNamespace(nodeid=n) for n in ids] for k, ids in stats.items()})


def test_a_session_that_ran_a_gpu_file_is_logged():
    assert ran_gpu_tests(reporter(passed=["tests/test_bake_gpu.py::test_baked", "tests/test_io_unit.py::test_x"]))


def test_a_failed_gpu_test_counts_too():
    assert ran_gpu_tests(reporter(passed=["tests/test_io_unit.py::test_x"], failed=["tests/test_it_gpu.py::test_y"]))


def test_the_unit_suite_is_not_logged_even_where_a_unit_file_needs_cuda():
    assert not ran_gpu_tests(reporter(passed=["tests/test_precision_unit.py::test_bake[D4]",
                                              "tests/test_io_unit.py::test_x"]))


def test_a_gpu_word_in_a_test_name_is_not_a_gpu_file():
    assert not ran_gpu_tests(reporter(passed=["tests/test_gpu_share_unit.py::test_gpu_share_is_capped"]))
