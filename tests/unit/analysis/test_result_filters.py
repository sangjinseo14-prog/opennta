"""Unit test for ``opennta.analysis.result_filters.filter_by_r2``.

``filter_by_r2`` is the Qt-free domain rule shared by the Analysis and Batch
tabs. One mixed input exercises every gate of its predicate at once: each
particle id below encodes a distinct edge case, and the expected survivors pin
the whole contract in a single assertion.
"""
from __future__ import annotations

from opennta.analysis.result_filters import filter_by_r2


def test_filter_by_r2_predicate_gates():
    # threshold = 0.9; one particle per edge case.
    element_by_id = {
        1: 120.0,   # in range, positive            -> keep
        2: 80.0,    # r2 == threshold (lower bound)  -> keep
        3: 95.0,    # r2 == 1 (upper bound)          -> keep
        4: 10.0,    # r2 just below threshold        -> drop
        5: 10.0,    # r2 > 1 (non-physical clamp)    -> drop
        6: 20.0,    # r2 is NaN                       -> drop
        7: 0.0,     # element == 0                    -> drop
        8: -5.0,    # element negative                -> drop
        # id 9 deliberately absent from element map   -> .get sentinel -> drop
        None: 99.0,  # None particle id               -> skipped
        99: 30.0,   # present here but not in r2 map  -> never iterated
    }
    r2_by_id = {
        1: 0.95, 2: 0.90, 3: 1.0, 4: 0.8999, 5: 1.5,
        6: float("nan"), 7: 0.99, 8: 0.99, 9: 0.99, None: 0.99,
    }
    r2_snapshot, element_snapshot = dict(r2_by_id), dict(element_by_id)

    result = filter_by_r2(element_by_id, r2_by_id, r2_threshold=0.9)

    # Only the three in-range, positive-element particles survive, mapped to
    # their element value (diameter/D), not their R².
    assert result == {1: 120.0, 2: 80.0, 3: 95.0}
    # Inputs are read-only.
    assert r2_by_id == r2_snapshot
    assert element_by_id == element_snapshot
