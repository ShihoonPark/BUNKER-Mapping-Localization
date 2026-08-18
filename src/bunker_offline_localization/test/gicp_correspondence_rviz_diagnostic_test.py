import math

import generate_gicp_correspondence_rviz_diagnostic as diagnostic


def test_protected_input_fingerprints_are_stable_across_read_only_audit():
    before = diagnostic.protected_fingerprints()
    after = diagnostic.protected_fingerprints()
    assert before == after
    assert before["small_gicp"]["porcelain_status"] == ""


def test_latency_distribution_is_finite_and_complete():
    summary = diagnostic.distribution([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["mean"] == 2.5
    assert math.isfinite(summary["p95"])
    assert summary["max"] == 4.0
