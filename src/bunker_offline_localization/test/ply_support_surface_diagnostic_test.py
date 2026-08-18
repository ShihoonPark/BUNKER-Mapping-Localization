import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE / "scripts" / "generate_ply_support_surface_diagnostic.py"
CONFIG = PACKAGE / "config" / "ply_support_surface_geometry_diagnostic.yaml"
SPEC = importlib.util.spec_from_file_location("ply_support_surface_diagnostic", SCRIPT)
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


def surface_config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["surface_extraction"]


def horizontal_grid(center_x=0.0, center_y=0.0, z=0.0):
    axis = np.linspace(-0.25, 0.25, 6)
    return np.asarray([
        [center_x + x_value, center_y + y_value, z]
        for x_value in axis for y_value in axis
    ])


def vertical_normals(count):
    normals = np.zeros((count, 3), dtype=float)
    normals[:, 2] = 1.0
    return normals


def test_flat_floor_is_recovered_despite_varying_fake_lidar_z():
    points = horizontal_grid(z=0.42)
    result = DIAGNOSTIC.extract_local_surface(
        points, vertical_normals(len(points)), [0.0, 0.0], surface_config(), 0.60
    )
    assert result["status"] == DIAGNOSTIC.SUCCESS
    assert result["surface_z_at_candidate_m"] == pytest.approx(0.42, abs=1.0e-12)
    fake_lidar_z = np.asarray([-3.0, 2.0, 100.0])
    np.testing.assert_allclose(fake_lidar_z - result["surface_z_at_candidate_m"],
                               [-3.42, 1.58, 99.58])


def test_two_level_floor_and_stage_use_xy_topology_without_measured_height():
    lower = horizontal_grid(center_x=-1.0, z=-0.2)
    upper = horizontal_grid(center_x=1.0, z=0.3)
    points = np.vstack((lower, upper))
    normals = vertical_normals(len(points))
    config = surface_config()
    first = DIAGNOSTIC.extract_local_surface(points, normals, [-1.0, 0.0], config, 0.60)
    second = DIAGNOSTIC.extract_local_surface(points, normals, [1.0, 0.0], config, 0.60)
    assert first["surface_z_at_candidate_m"] == pytest.approx(-0.2, abs=1.0e-12)
    assert second["surface_z_at_candidate_m"] == pytest.approx(0.3, abs=1.0e-12)
    assert "measured" not in inspect.signature(DIAGNOSTIC.extract_local_surface).parameters


def test_vertical_wall_is_rejected_by_normal_filter():
    y_values = np.linspace(-0.25, 0.25, 6)
    z_values = np.linspace(-0.25, 0.25, 6)
    points = np.asarray([[0.0, y_value, z_value] for y_value in y_values for z_value in z_values])
    normals = np.zeros_like(points)
    normals[:, 0] = 1.0
    result = DIAGNOSTIC.extract_local_surface(points, normals, [0.0, 0.0],
                                               surface_config(), 0.60)
    assert result["status"] == "NO_SUPPORT_LIKE_POINTS"
    assert result["support_indices"].size == 0


def test_equal_xy_multiple_horizontal_layers_are_ambiguous_not_z_selected():
    points = np.vstack((horizontal_grid(z=-0.4), horizontal_grid(z=0.7)))
    result = DIAGNOSTIC.extract_local_surface(
        points, vertical_normals(len(points)), [0.0, 0.0], surface_config(), 0.60
    )
    assert result["status"] == DIAGNOSTIC.AMBIGUOUS
    assert result["selected_indices"].size == 0
    assert "surface_z_at_candidate_m" not in result


def test_pose_z_cannot_leak_into_component_or_surface_selection():
    points = horizontal_grid(z=0.17)
    normals = vertical_normals(len(points))
    first = DIAGNOSTIC.extract_local_surface(points, normals, [0.0, 0.0], surface_config(), 0.60)
    unrelated_independent_pose_z = -10000.0
    second = DIAGNOSTIC.extract_local_surface(points, normals, [0.0, 0.0], surface_config(), 0.60)
    assert unrelated_independent_pose_z != first["surface_z_at_candidate_m"]
    np.testing.assert_array_equal(first["selected_indices"], second["selected_indices"])
    assert first["surface_z_at_candidate_m"] == second["surface_z_at_candidate_m"]
    assert "pose" not in inspect.signature(DIAGNOSTIC.extract_local_surface).parameters


def test_ordered_glim_xy_association_uses_monotonic_loop_branch_and_ignores_z():
    trajectory_xy = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                                [1.0, 0.0], [0.0, 0.0]])
    candidates = np.asarray([[0.05, 0.0], [1.95, 0.0], [0.05, 0.0]])
    matched, distances = DIAGNOSTIC.monotonic_xy_association(candidates, trajectory_xy)
    np.testing.assert_array_equal(matched, [0, 2, 4])
    np.testing.assert_allclose(distances, [0.05, 0.05, 0.05])
    unrelated_z = np.asarray([100.0, -50.0, 7.0, 999.0, -2.0])
    changed, _ = DIAGNOSTIC.monotonic_xy_association(candidates, trajectory_xy)
    assert unrelated_z.size == trajectory_xy.shape[0]
    np.testing.assert_array_equal(changed, matched)


def test_clearance_and_independent_mapping_delta_are_exact():
    values = DIAGNOSTIC.crosscheck_values(0.25, 1.10, 0.95)
    assert values["mapping_clearance_m"] == pytest.approx(0.85)
    assert values["independent_clearance_m"] == pytest.approx(0.70)
    assert values["independent_minus_mapping_pose_z_m"] == pytest.approx(-0.15)


def test_measured_height_leakage_is_absent_and_protected_input_is_unchanged(tmp_path):
    protected = tmp_path / "map.ply"
    protected.write_bytes(b"immutable synthetic PLY bytes")
    before = DIAGNOSTIC.physical_gate.fingerprint(protected, include_hash=True)
    points = horizontal_grid(z=-0.1)
    normals = vertical_normals(len(points))
    first = DIAGNOSTIC.extract_local_surface(points, normals, [0.0, 0.0], surface_config(), 0.60)
    measured_stage_height_m = 9.99
    second = DIAGNOSTIC.extract_local_surface(points, normals, [0.0, 0.0], surface_config(), 0.60)
    assert measured_stage_height_m != 0.150
    assert first["surface_z_at_candidate_m"] == second["surface_z_at_candidate_m"]
    assert "measured" not in inspect.signature(DIAGNOSTIC.build_blind_summary).parameters
    assert DIAGNOSTIC.physical_gate.fingerprint(protected, include_hash=True) == before
