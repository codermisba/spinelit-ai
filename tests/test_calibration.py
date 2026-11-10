"""Calibration artifact loading / fallback behaviour (no network)."""

import pickle

import pytest


@pytest.fixture(autouse=True)
def _reset_global_calibrators(monkeypatch):
    import calibration
    monkeypatch.setattr(calibration, "_calibrators", None)
    yield


def test_default_is_raw_passthrough(monkeypatch, tmp_path):
    import calibration

    monkeypatch.setattr(calibration, "CALIB_ARTIFACT", tmp_path / "missing.pkl")
    monkeypatch.delenv("VISION_CHECKPOINT", raising=False)
    monkeypatch.delenv("MODEL_ID", raising=False)

    sev, prob = calibration.calibrate_ddd(3, 0.42)
    assert sev == "Moderate"
    assert prob == pytest.approx(0.42)          # no heuristic distortion
    assert calibration.get_calibrators().identity
    assert calibration.get_calibrators().source == "raw-passthrough"


def test_identity_artifact_serves_raw_probability(monkeypatch, tmp_path):
    import calibration

    path = tmp_path / "calibration.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"ddd": None, "identity": True,
                     "source": "raw-passthrough"}, fh)
    monkeypatch.setattr(calibration, "CALIB_ARTIFACT", path)

    cal = calibration.get_calibrators()
    assert cal.identity
    assert cal.source == "raw-passthrough"
    assert calibration.calibrate_ddd(1, 0.17)[1] == pytest.approx(0.17)


def test_fitted_artifact_is_applied(monkeypatch, tmp_path):
    import calibration
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip").fit(
        [0.0, 0.5, 1.0], [0.0, 0.9, 1.0])
    path = tmp_path / "calibration.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"ddd": iso, "identity": False,
                     "source": "fitted-heldout"}, fh)
    monkeypatch.setattr(calibration, "CALIB_ARTIFACT", path)

    sev, prob = calibration.calibrate_ddd(3, 0.5)
    assert sev == "Moderate"
    assert prob == pytest.approx(0.9, abs=1e-6)


def test_corrupt_artifact_degrades_to_passthrough(monkeypatch, tmp_path):
    import calibration

    path = tmp_path / "calibration.pkl"
    path.write_bytes(b"not a pickle")
    monkeypatch.setattr(calibration, "CALIB_ARTIFACT", path)

    cal = calibration.get_calibrators()
    assert cal.identity
    assert calibration.calibrate_ddd(2, 0.6)[1] == pytest.approx(0.6)


def test_missing_artifact_tries_hub_sidecar(monkeypatch, tmp_path):
    import calibration
    import weights

    monkeypatch.setattr(calibration, "CALIB_ARTIFACT", tmp_path / "missing.pkl")
    seen = {}

    def fake_sidecar(filename, repo_id=None):
        seen["filename"] = filename
        return None

    monkeypatch.setattr(weights, "download_sidecar", fake_sidecar)
    cal = calibration.get_calibrators()
    assert seen["filename"] == "calibration.pkl"
    assert cal.source == "raw-passthrough"


def test_hub_sidecar_artifact_is_loaded(monkeypatch, tmp_path):
    import calibration
    import weights

    path = tmp_path / "hub_calibration.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"ddd": None, "identity": True,
                     "source": "raw-passthrough"}, fh)
    monkeypatch.setattr(calibration, "CALIB_ARTIFACT", tmp_path / "missing.pkl")
    monkeypatch.setattr(
        weights, "download_sidecar", lambda filename, repo_id=None: path)

    cal = calibration.get_calibrators()
    assert cal.source == "raw-passthrough"


def test_vision_engine_applies_calibrator():
    import calibration
    import vision_engine

    assert vision_engine.calibrate_ddd is calibration.calibrate_ddd
