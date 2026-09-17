"""Loading the compiled kernels that ship inside the wheel.

The package ships two extension modules, each built for every interpreter it
supports: ``mariepy._ext`` holds the package's own kernels, and ``mariepy._directfn`` the
DIRECTFN singular integrals, which carry their own licence. A kernel that will
not import therefore means a broken installation -- most often a wheel built
for one interpreter imported from another -- and never an optional extra a
caller may reasonably do without.

:func:`require` says so, naming the mismatch, rather than letting a caller
fall through to a slower Python path whose only symptom is unexplained
runtime.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any

__all__ = ["require"]

_DEFAULT_MODULE = "mariepy._ext"


def _installed_tags(module: str) -> list[str]:
    """Return the ABI tags of the extension builds present beside the package."""
    suffix = ".pyd" if sys.platform == "win32" else ".so"
    stem = module.rpartition(".")[2]
    package = import_module("mariepy")
    tags = set()
    for directory in getattr(package, "__path__", ()):
        for path in Path(directory).glob(f"{stem}.*{suffix}"):
            parts = path.name.split(".")
            if len(parts) >= 3:
                tags.add(parts[1])
    return sorted(tags)


def require(attribute: str | None = None, module: str = _DEFAULT_MODULE) -> Any:
    """Load the bundled accelerator, or raise naming the mismatch.

    Parameters
    ----------
    attribute
        Symbol the caller needs. Checked here, so a binary that predates the
        symbol fails at the load rather than at the call.
    module
        Which extension to load: ``mariepy._ext`` for the package's own kernels, or
        ``mariepy._directfn`` for the DIRECTFN singular integrals.

    Returns
    -------
    module or object
        The extension module, or the named attribute when one is given.

    Raises
    ------
    ImportError
        The accelerator is absent, was built for another interpreter, or does
        not provide ``attribute``.
    """
    try:
        loaded = import_module(module)
    except ImportError as error:
        installed = _installed_tags(module)
        found = ", ".join(installed) if installed else "none"
        raise ImportError(
            f"the bundled accelerator {module} did not load. Running "
            f"{sys.implementation.cache_tag}, builds present: {found}. "
            f"This package ships the kernels for every interpreter it "
            f"supports, so install the distribution matching this one."
        ) from error
    if attribute is None:
        return loaded
    try:
        return getattr(loaded, attribute)
    except AttributeError as error:
        raise ImportError(
            f"the bundled accelerator {module} does not provide "
            f"{attribute!r}; the installed binary predates it."
        ) from error
