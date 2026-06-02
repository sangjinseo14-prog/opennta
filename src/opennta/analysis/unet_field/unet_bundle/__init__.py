"""U-Net inference bundle (model, weights, preprocessing, visualization).

Re-exports the runner singleton so callers don't need to know which file
holds it. The class itself is heavy — only importing ``get_shared_runner``
is light; the runtime cost is deferred to the first call.
"""

from .inference import UNetInferenceRunner, clear_shared_runner, get_shared_runner

__all__ = ["UNetInferenceRunner", "clear_shared_runner", "get_shared_runner"]
