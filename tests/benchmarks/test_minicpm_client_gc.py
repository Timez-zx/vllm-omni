import gc

import pytest

from benchmarks.minicpmo.client_gc import defer_cyclic_gc


def test_client_gc_restored_on_error():
    previous = gc.isenabled()
    try:
        gc.enable()
        with pytest.raises(RuntimeError), defer_cyclic_gc():
            assert not gc.isenabled()
            raise RuntimeError("test")
        assert gc.isenabled()
    finally:
        if not previous:
            gc.disable()


def test_already_disabled_gc_remains_disabled():
    previous = gc.isenabled()
    gc.disable()
    try:
        with defer_cyclic_gc():
            assert not gc.isenabled()
        assert not gc.isenabled()
    finally:
        if previous:
            gc.enable()
