"""Occupancy Networks package exports.

Some submodules (notably generation/config) depend on optional compiled
extensions. Keep model imports available even if those optional pieces are
missing in the current environment.
"""

from im2mesh.onet import models

try:
    from im2mesh.onet import config
except Exception:  # pragma: no cover
    config = None

try:
    from im2mesh.onet import generation
except Exception:  # pragma: no cover
    generation = None

try:
    from im2mesh.onet import training
except Exception:  # pragma: no cover
    training = None

__all__ = ["models", "config", "generation", "training"]
