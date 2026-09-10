"""Keep client-only, stop-the-world cyclic GC outside a timed workload."""

import gc
from contextlib import contextmanager


@contextmanager
def defer_cyclic_gc():
    # CPython reference counting remains enabled. Only its occasional full
    # object-graph traversal is deferred; engine processes are unaffected.
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()
