"""The package imports and reports a version."""

import mariepy


def test_the_package_reports_a_version():
    assert isinstance(mariepy.__version__, str)
    assert mariepy.__version__
