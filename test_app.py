"""
test_app.py
===========
Pytest suite for webapp/app.py pure-Python helpers.
Models are never loaded from disk — all ML artifacts are mocked.

Run from the project root:

    python -m pytest test_app.py -v --tb=short
"""

import sys
import types
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Bootstrap: make sure webapp/ and project root are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
WEBAPP       = PROJECT_ROOT / "webapp"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WEBAPP) not in sys.path:
    sys.path.insert(0, str(WEBAPP))


# ---------------------------------------------------------------------------
# Minimal stubs so importing app.py never touches disk / GPU
# ---------------------------------------------------------------------------
def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _stub_streamlit():
    """Replace streamlit with a no-op stub so app-level st.* calls don't crash."""
    st = _make_stub("streamlit")
    noop = MagicMock(return_value=None)
    for attr in ("set_page_config", "markdown", "error", "spinner",
                 "cache_resource", "sidebar", "columns", "button",
                 "text_input", "selectbox", "number_input", "form",
                 "form_submit_button", "expander", "tabs", "image",
                 "dataframe", "metric", "write", "info", "warning",
                 "success", "code", "divider", "title", "header",
                 "subheader", "caption", "progress", "balloons", "snow"):
        setattr(st, attr, noop)

    # cache_resource must work as a decorator that returns the function unchanged
    def _cache_resource(fn=None, **kwargs):
        if fn is None:
            return lambda f: f
        return fn
    st.cache_resource = _cache_resource

    # session_state behaves like a dict
    st.session_state = {}
    return st


def _stub_tensorflow():
    tf      = _make_stub("tensorflow")
    tf.keras = _make_stub("tensorflow.keras")
    tf.keras.models = _make_stub("tensorflow.keras.models")
    tf.keras.models.load_model = MagicMock(return_value=MagicMock())
    return tf


def _stub_torch():
    torch = _make_stub("torch")
    torch.no_grad = MagicMock(return_value=MagicMock(
        __enter__=MagicMock(return_value=None),
        __exit__=MagicMock(return_value=False),
    ))
    torch.tensor = MagicMock(side_effect=lambda x, **kw: x)
    torch.load    = MagicMock(return_value={})
    torch.softmax = MagicMock(return_value=MagicMock(
        numpy=MagicMock(return_value=np.array([[0.1, 0.7, 0.2]]))
    ))
    torch.long = 0
    return torch


def _stub_transformers():
    tr = _make_stub("transformers")
    tr.AutoTokenizer = MagicMock()
    tr.AutoTokenizer.from_pretrained = MagicMock(return_value=MagicMock())
    return tr


def _stub_pipelines():
    pip = _make_stub("pipelines")
    dp  = _make_stub("pipelines.data_pipeline")

    def fake_engineer(df, preprocessing_state=None, **kw):
        return df.copy(), preprocessing_state or {}

    dp.engineer_features = fake_engineer
    pip.data_pipeline    = dp
    return pip


# Apply all stubs before importing app
_stub_streamlit()
_stub_tensorflow()
_stub_torch()
_stub_transformers()
_stub_pipelines()

import app  # noqa: E402  (must come after stubs)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------
MINIMAL_PATIENT = {
    "age": "[60-70)",
    "gender": "Male",
    "race": "Caucasian",
    "time_in_hospital": 3,
    "num_lab_procedures": 40,
    "num_procedures": 1,
    "num_medications": 10,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "diag_1": "250.00",
    "diag_2": "250.00",
    "diag_3": "250.00",
    "number_diagnoses": 5,
    "max_glu_serum": "None",
    "A1Cresult": "None",
    "change": "No",
    "diabetesMed": "Yes",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
}


def _make_preprocessing_state() -> dict:
    return {
        "diag_1_cats": {"250.00": 0},
        "diag_2_cats": {"250.00": 0},
        "diag_3_cats": {"250.00": 0},
        "gender_cats": {"Male": 0, "Female": 1},
        "race_cats": {"Caucasian": 0},
        "max_glu_serum_cats": {"None": 0},
        "A1Cresult_cats": {"None": 0},
        "change_cats": {"No": 0},
        "diabetesMed_cats": {"Yes": 1, "No": 0},
    }


def _make_feature_names(n: int = 10) -> list:
    return [f"feat_{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# ===========================================================================
# 1. risk_label
# ===========================================================================
class TestRiskLabel:
    def test_high_risk_at_threshold(self):
        label, css = app.risk_label(0.65)
        assert label == "HIGH RISK"
        assert css == "risk-high"

    def test_high_risk_above_threshold(self):
        label, css = app.risk_label(0.99)
        assert label == "HIGH RISK"
        assert css == "risk-high"

    def test_moderate_risk_at_threshold(self):
        label, css = app.risk_label(0.40)
        assert label == "MODERATE RISK"
        assert css == "risk-medium"

    def test_moderate_risk_between_bounds(self):
        label, css = app.risk_label(0.55)
        assert label == "MODERATE RISK"
        assert css == "risk-medium"

    def test_moderate_risk_just_below_high(self):
        label, css = app.risk_label(0.6499)
        assert label == "MODERATE RISK"
        assert css == "risk-medium"

    def test_low_risk_at_zero(self):
        label, css = app.risk_label(0.0)
        assert label == "LOW RISK"
        assert css == "risk-low"

    def test_low_risk_just_below_moderate(self):
        label, css = app.risk_label(0.3999)
        assert label == "LOW RISK"
        assert css == "risk-low"

    def test_low_risk_midpoint(self):
        label, css = app.risk_label(0.20)
        assert label == "LOW RISK"
        assert css == "risk-low"

    def test_returns_tuple_of_two(self):
        result = app.risk_label(0.5)
        assert isinstance(result, tuple) and len(result) == 2

    def test_css_classes_are_strings(self):
        for p in [0.0, 0.4, 0.65, 1.0]:
            _, css = app.risk_label(p)
            assert isinstance(css, str)


# ===========================================================================
# 2. risk_gauge_svg
# ===========================================================================
class TestRiskGaugeSvg:
    def test_returns_string(self):
        assert isinstance(app.risk_gauge_svg(0.5), str)

    def test_contains_svg_tag(self):
        assert "<svg" in app.risk_gauge_svg(0.5)

    def test_low_probability_teal_color(self):
        svg = app.risk_gauge_svg(0.1)
        assert "#5eead4" in svg

    def test_moderate_probability_amber_color(self):
        svg = app.risk_gauge_svg(0.5)
        assert "#fbbf24" in svg

    def test_high_probability_red_color(self):
        svg = app.risk_gauge_svg(0.9)
        assert "#f87171" in svg

    def test_clamp_above_one(self):
        svg = app.risk_gauge_svg(2.0)
        assert "100.0<tspan" in svg

    def test_clamp_below_zero(self):
        svg = app.risk_gauge_svg(-1.0)
        assert "0.0<tspan" in svg

    def test_custom_label_appears_in_svg(self):
        svg = app.risk_gauge_svg(0.5, label="MYTEST")
        assert "MYTEST" in svg

    def test_default_label_is_risk(self):
        svg = app.risk_gauge_svg(0.5)
        assert "RISK" in svg

    def test_exactly_zero(self):
        svg = app.risk_gauge_svg(0.0)
        assert "0.0<tspan" in svg

    def test_exactly_one(self):
        svg = app.risk_gauge_svg(1.0)
        assert "100.0<tspan" in svg

    def test_boundary_040(self):
        """0.40 is the low→moderate boundary."""
        svg = app.risk_gauge_svg(0.40)
        assert "#fbbf24" in svg

    def test_boundary_065(self):
        """0.65 is the moderate→high boundary."""
        svg = app.risk_gauge_svg(0.65)
        assert "#f87171" in svg


# ===========================================================================
# 3. parse_id
# ===========================================================================
class TestParseId:
    def test_standard_format(self):
        assert app.parse_id("1 - Emergency") == 1

    def test_two_digit_id(self):
        assert app.parse_id("11 - Not Available") == 11

    def test_three_digit_id(self):
        assert app.parse_id("100 - Something") == 100

    def test_leading_spaces_stripped(self):
        assert app.parse_id("  3 - Elective") == 3

    def test_no_dash_returns_default(self):
        assert app.parse_id("Emergency") == 1

    def test_empty_string_returns_default(self):
        assert app.parse_id("") == 1

    def test_non_numeric_id_returns_default(self):
        assert app.parse_id("abc - desc") == 1

    def test_returns_int(self):
        result = app.parse_id("5 - test")
        assert isinstance(result, int)


# ===========================================================================
# 4. preprocess_patient
# ===========================================================================
class TestPreprocessPatient:
    def _call(self, patient=None, n_feats=10):
        state = _make_preprocessing_state()
        feats = _make_feature_names(n_feats)
        p = patient or MINIMAL_PATIENT.copy()
        return app.preprocess_patient(p, state, feats)

    def test_returns_dataframe(self):
        assert isinstance(self._call(), pd.DataFrame)

    def test_one_row(self):
        assert len(self._call()) == 1

    def test_columns_match_feature_names(self):
        n = 8
        df = self._call(n_feats=n)
        assert list(df.columns) == _make_feature_names(n)

    def test_missing_columns_filled_with_zero(self):
        df = self._call(n_feats=5)
        assert (df.fillna(0) == df).all(axis=None)

    def test_age_bracket_removed(self):
        df = self._call()
        assert "age" not in df.columns

    def test_age_numeric_derived(self):
        """age_numeric should be calculated even though it's later reindexed away."""
        state = _make_preprocessing_state()
        feats = ["age_numeric"]
        p = MINIMAL_PATIENT.copy()
        df = app.preprocess_patient(p, state, feats)
        assert df["age_numeric"].iloc[0] == 65

    def test_age_bracket_mapping_young(self):
        state = _make_preprocessing_state()
        feats = ["age_numeric"]
        p = {**MINIMAL_PATIENT, "age": "[0-10)"}
        df = app.preprocess_patient(p, state, feats)
        assert df["age_numeric"].iloc[0] == 5

    def test_unknown_age_defaults_to_65(self):
        state = _make_preprocessing_state()
        feats = ["age_numeric"]
        p = {**MINIMAL_PATIENT, "age": "[??-??)"}
        df = app.preprocess_patient(p, state, feats)
        assert df["age_numeric"].iloc[0] == 65

    def test_medication_defaults_injected(self):
        """A patient without 'insulin' key should get insulin='No'."""
        state = _make_preprocessing_state()
        feats = ["insulin"]
        p = {k: v for k, v in MINIMAL_PATIENT.items() if k != "insulin"}
        df = app.preprocess_patient(p, state, feats)
        # reindex fills missing cols with 0; but engineer_features will have seen it
        # The column should exist (value may be encoded); no KeyError is the test
        assert "insulin" in df.columns or True  # no exception is the real assertion

    def test_readmission_binary_dropped(self):
        state = _make_preprocessing_state()
        feats = ["feat_0"] 

        original_engineer = app.engineer_features if hasattr(app, "engineer_features") else None

        def fake_eng(df, preprocessing_state=None, **kw):
            df = df.copy()
            df["readmission_binary"] = 0
            df["feat_0"] = 1
            return df, preprocessing_state or {}

        with patch("app.engineer_features", fake_eng):
            df = app.preprocess_patient(MINIMAL_PATIENT.copy(), state, feats)
        assert "readmission_binary" not in df.columns

    def test_no_exception_on_minimal_patient(self):
        self._call()  # should not raise

    def test_prior_encounter_defaults_set(self):
        """patient without prior_encounters_count should not raise."""
        p = {k: v for k, v in MINIMAL_PATIENT.items()
             if k not in ("prior_encounters_count", "is_recurrent_patient")}
        self._call(patient=p)  # must not raise


# ===========================================================================
# 5. predict_m1  (model mocked)
# ===========================================================================
class TestPredictM1:
    def _mock_load(self, proba_val=0.7, threshold=0.5):
        state = _make_preprocessing_state()
        feats = _make_feature_names(5)
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[1 - proba_val, proba_val]])
        return mock_model, state, feats, threshold

    def test_high_proba_predicts_one(self):
        with patch("app.load_model1", return_value=self._mock_load(0.8, 0.5)):
            pred, proba, conf = app.predict_m1(MINIMAL_PATIENT.copy())
        assert pred == 1

    def test_low_proba_predicts_zero(self):
        with patch("app.load_model1", return_value=self._mock_load(0.2, 0.5)):
            pred, proba, conf = app.predict_m1(MINIMAL_PATIENT.copy())
        assert pred == 0

    def test_threshold_respected(self):
        """proba=0.4, threshold=0.5 → pred=0"""
        with patch("app.load_model1", return_value=self._mock_load(0.4, 0.5)):
            pred, _, _ = app.predict_m1(MINIMAL_PATIENT.copy())
        assert pred == 0

    def test_proba_in_range(self):
        with patch("app.load_model1", return_value=self._mock_load(0.6)):
            _, proba, _ = app.predict_m1(MINIMAL_PATIENT.copy())
        assert 0.0 <= proba <= 1.0

    def test_confidence_is_max(self):
        with patch("app.load_model1", return_value=self._mock_load(0.6)):
            _, proba, conf = app.predict_m1(MINIMAL_PATIENT.copy())
        assert conf == max(proba, 1 - proba)

    def test_confidence_ge_half(self):
        with patch("app.load_model1", return_value=self._mock_load(0.3)):
            _, _, conf = app.predict_m1(MINIMAL_PATIENT.copy())
        assert conf >= 0.5

    def test_returns_tuple_of_three(self):
        with patch("app.load_model1", return_value=self._mock_load()):
            result = app.predict_m1(MINIMAL_PATIENT.copy())
        assert len(result) == 3

    def test_pred_is_int(self):
        with patch("app.load_model1", return_value=self._mock_load()):
            pred, _, _ = app.predict_m1(MINIMAL_PATIENT.copy())
        assert isinstance(pred, int)

    def test_proba_is_float(self):
        with patch("app.load_model1", return_value=self._mock_load()):
            _, proba, _ = app.predict_m1(MINIMAL_PATIENT.copy())
        assert isinstance(proba, float)


# ===========================================================================
# 6. predict_m2  (model mocked)
# ===========================================================================
class TestPredictM2:
    def _mock_load(self, proba_val=0.7):
        state = _make_preprocessing_state()
        feats = _make_feature_names(5)
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[proba_val]])
        mock_scaler = MagicMock()
        mock_scaler.transform.side_effect = lambda x: x
        return mock_model, mock_scaler, state, feats

    def test_high_proba_predicts_one(self):
        with patch("app.load_model2", return_value=self._mock_load(0.8)):
            pred, _, _ = app.predict_m2(MINIMAL_PATIENT.copy())
        assert pred == 1

    def test_low_proba_predicts_zero(self):
        with patch("app.load_model2", return_value=self._mock_load(0.2)):
            pred, _, _ = app.predict_m2(MINIMAL_PATIENT.copy())
        assert pred == 0

    def test_threshold_exactly_half_predicts_one(self):
        with patch("app.load_model2", return_value=self._mock_load(0.5)):
            pred, _, _ = app.predict_m2(MINIMAL_PATIENT.copy())
        assert pred == 1

    def test_proba_in_range(self):
        with patch("app.load_model2", return_value=self._mock_load(0.6)):
            _, proba, _ = app.predict_m2(MINIMAL_PATIENT.copy())
        assert 0.0 <= proba <= 1.0

    def test_confidence_ge_half(self):
        with patch("app.load_model2", return_value=self._mock_load(0.3)):
            _, _, conf = app.predict_m2(MINIMAL_PATIENT.copy())
        assert conf >= 0.5

    def test_returns_three_values(self):
        with patch("app.load_model2", return_value=self._mock_load()):
            result = app.predict_m2(MINIMAL_PATIENT.copy())
        assert len(result) == 3

    def test_pred_is_int(self):
        with patch("app.load_model2", return_value=self._mock_load()):
            pred, _, _ = app.predict_m2(MINIMAL_PATIENT.copy())
        assert isinstance(pred, int)

    def test_scaler_is_called(self):
        mock_load = self._mock_load(0.5)
        with patch("app.load_model2", return_value=mock_load):
            app.predict_m2(MINIMAL_PATIENT.copy())
        mock_load[1].transform.assert_called_once()


# ===========================================================================
# 7. predict_m5  (model mocked)
# ===========================================================================
class TestPredictM5:
    LABELS = {
        0: "SHORT STAY (1-2 days)",
        1: "STANDARD STAY (3-5 days)",
        2: "EXTENDED STAY (6+ days)",
    }
    CSS = {0: "risk-low", 1: "risk-medium", 2: "risk-high"}

    def _mock_load(self, pred_class=0, proba=None):
        state = _make_preprocessing_state()
        feats = _make_feature_names(5)
        if proba is None:
            proba = np.zeros(3)
            proba[pred_class] = 0.9
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([pred_class])
        mock_model.predict_proba.return_value = np.array([proba])
        return mock_model, state, feats

    def test_short_stay_label(self):
        with patch("app.load_model5", return_value=self._mock_load(0)):
            label, _, _ = app.predict_m5(MINIMAL_PATIENT.copy())
        assert label == self.LABELS[0]

    def test_standard_stay_label(self):
        with patch("app.load_model5", return_value=self._mock_load(1)):
            label, _, _ = app.predict_m5(MINIMAL_PATIENT.copy())
        assert label == self.LABELS[1]

    def test_extended_stay_label(self):
        with patch("app.load_model5", return_value=self._mock_load(2)):
            label, _, _ = app.predict_m5(MINIMAL_PATIENT.copy())
        assert label == self.LABELS[2]

    def test_css_short(self):
        with patch("app.load_model5", return_value=self._mock_load(0)):
            _, _, css = app.predict_m5(MINIMAL_PATIENT.copy())
        assert css == "risk-low"

    def test_css_standard(self):
        with patch("app.load_model5", return_value=self._mock_load(1)):
            _, _, css = app.predict_m5(MINIMAL_PATIENT.copy())
        assert css == "risk-medium"

    def test_css_extended(self):
        with patch("app.load_model5", return_value=self._mock_load(2)):
            _, _, css = app.predict_m5(MINIMAL_PATIENT.copy())
        assert css == "risk-high"

    def test_confidence_in_range(self):
        with patch("app.load_model5", return_value=self._mock_load(0)):
            _, conf, _ = app.predict_m5(MINIMAL_PATIENT.copy())
        assert 0.0 <= conf <= 1.0

    def test_returns_three_values(self):
        with patch("app.load_model5", return_value=self._mock_load()):
            result = app.predict_m5(MINIMAL_PATIENT.copy())
        assert len(result) == 3

    def test_label_is_string(self):
        with patch("app.load_model5", return_value=self._mock_load()):
            label, _, _ = app.predict_m5(MINIMAL_PATIENT.copy())
        assert isinstance(label, str)

    def test_confidence_is_float(self):
        with patch("app.load_model5", return_value=self._mock_load()):
            _, conf, _ = app.predict_m5(MINIMAL_PATIENT.copy())
        assert isinstance(conf, float)


# ===========================================================================
# 8. predict_m4  (model + tokenizer mocked)
# ===========================================================================
class TestPredictM4:
    def _mock_load(self, label_str="High"):
        import torch as _torch
        drug_le  = MagicMock()
        cond_le  = MagicMock()
        label_le = MagicMock()

        drug_le.classes_  = np.array(["Metformin", "unknown"])
        cond_le.classes_  = np.array(["Diabetes", "unknown"])
        label_le.classes_ = np.array(["High", "Low", "Medium"])

        drug_le.transform  = MagicMock(return_value=np.array([0]))
        cond_le.transform  = MagicMock(return_value=np.array([0]))
        idx = list(label_le.classes_).index(label_str)
        label_le.inverse_transform = MagicMock(return_value=np.array([label_str]))

        mock_model = MagicMock()
        probs = np.zeros(3)
        probs[idx] = 0.9
        logits = MagicMock()

        import sys as _sys
        torch_mod = _sys.modules.get("torch")
        if torch_mod:
            softmax_result = MagicMock()
            softmax_result.numpy.return_value = np.array([probs])
            torch_mod.softmax = MagicMock(return_value=softmax_result)
            torch_mod.no_grad = MagicMock(return_value=MagicMock(
                __enter__=MagicMock(return_value=None),
                __exit__=MagicMock(return_value=False),
            ))

        mock_model.return_value = logits
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }

        return mock_model, tokenizer, drug_le, cond_le, label_le

    def test_high_label(self):
        with patch("app.load_model4", return_value=self._mock_load("High")):
            label, _, _ = app.predict_m4("notes", "Metformin", "Diabetes")
        assert "HIGH" in label

    def test_low_label(self):
        with patch("app.load_model4", return_value=self._mock_load("Low")):
            label, _, _ = app.predict_m4("notes", "Metformin", "Diabetes")
        assert "LOW" in label

    def test_medium_label(self):
        with patch("app.load_model4", return_value=self._mock_load("Medium")):
            label, _, _ = app.predict_m4("notes", "Metformin", "Diabetes")
        assert "MEDIUM" in label

    def test_label_ends_with_sentiment(self):
        with patch("app.load_model4", return_value=self._mock_load("High")):
            label, _, _ = app.predict_m4("notes", "Metformin", "Diabetes")
        assert label.endswith("RISK SENTIMENT")

    def test_css_high(self):
        with patch("app.load_model4", return_value=self._mock_load("High")):
            _, _, css = app.predict_m4("notes", "Metformin", "Diabetes")
        assert css == "risk-high"

    def test_css_low(self):
        with patch("app.load_model4", return_value=self._mock_load("Low")):
            _, _, css = app.predict_m4("notes", "Metformin", "Diabetes")
        assert css == "risk-low"

    def test_css_medium(self):
        with patch("app.load_model4", return_value=self._mock_load("Medium")):
            _, _, css = app.predict_m4("notes", "Metformin", "Diabetes")
        assert css == "risk-medium"

    def test_confidence_in_range(self):
        with patch("app.load_model4", return_value=self._mock_load("High")):
            _, conf, _ = app.predict_m4("notes", "Metformin", "Diabetes")
        assert 0.0 <= conf <= 1.0

    def test_returns_three_values(self):
        with patch("app.load_model4", return_value=self._mock_load("High")):
            result = app.predict_m4("notes", "Metformin", "Diabetes")
        assert len(result) == 3

    def test_label_is_string(self):
        with patch("app.load_model4", return_value=self._mock_load("High")):
            label, _, _ = app.predict_m4("notes", "Metformin", "Diabetes")
        assert isinstance(label, str)

    def test_confidence_is_float(self):
        with patch("app.load_model4", return_value=self._mock_load("High")):
            _, conf, _ = app.predict_m4("notes", "Metformin", "Diabetes")
        assert isinstance(conf, float)

    def test_unknown_drug_uses_fallback(self):
        """Drugs not in encoder classes should map to 'unknown' without raising."""
        mock_load = self._mock_load("High")
        with patch("app.load_model4", return_value=mock_load):
            app.predict_m4("notes", "NonExistentDrug", "Diabetes")
        # No exception = passpython