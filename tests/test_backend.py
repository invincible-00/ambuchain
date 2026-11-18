"""
Run from the project root:   python -m unittest discover -s tests -v
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from analysis import analytics, failure_detection as fd
from automata import dfa as d

FULL = list(d.EVENTS)


# --------------------------------------------------------------------------- #
# Generic DFA engine
# --------------------------------------------------------------------------- #
class TestGenericDFA(unittest.TestCase):
    def setUp(self):
        # Accepts strings over {a,b} containing an even number of 'a'
        self.even_a = d.DFA(
            states={"E", "O"}, alphabet={"a", "b"},
            transitions={("E", "a"): "O", ("O", "a"): "E", ("E", "b"): "E", ("O", "b"): "O"},
            start_state="E", accepting_states={"E"},
        )

    def test_accepts(self):
        self.assertTrue(self.even_a.accepts(["a", "b", "a"]))
        self.assertTrue(self.even_a.accepts([]))
        self.assertFalse(self.even_a.accepts(["a"]))

    def test_bad_definitions_raise(self):
        with self.assertRaises(ValueError):
            d.DFA({"A"}, {"x"}, {}, "Z", {"A"})            # start not in Q
        with self.assertRaises(ValueError):
            d.DFA({"A"}, {"x"}, {}, "A", {"B"})            # F not subset of Q
        with self.assertRaises(ValueError):
            d.DFA({"A"}, {"x"}, {("A", "y"): "A"}, "A", {"A"})  # symbol not in Sigma

    def test_complete_adds_trap(self):
        partial = d.DFA({"A", "B"}, {"x", "y"}, {("A", "x"): "B"}, "A", {"B"})
        total = partial.complete("DEAD")
        self.assertEqual(len(total.transitions), 3 * 2)
        self.assertEqual(total.next_state("B", "y"), "DEAD")
        self.assertEqual(total.next_state("DEAD", "x"), "DEAD")


# --------------------------------------------------------------------------- #
# Ambulance DFA
# --------------------------------------------------------------------------- #
class TestAmbulanceDFA(unittest.TestCase):
    def test_definition(self):
        spec = d.get_dfa_definition()
        self.assertEqual(len(spec["Q"]), 9)
        self.assertEqual(len(spec["Sigma"]), 8)
        self.assertEqual(len(spec["delta"]), 8)
        self.assertEqual(spec["q0"], "IDLE")
        self.assertEqual(spec["F"], ["CLOSED"])

    def test_full_sequence_accepted(self):
        res = d.validate_event_sequence(FULL)
        self.assertTrue(res.accepted)
        self.assertEqual(res.final_state, "CLOSED")

    def test_normalisation(self):
        self.assertTrue(d.is_valid_sequence([" call_received", "Ambulance Dispatched"] + FULL[2:]))

    def test_empty_sequence_is_incomplete(self):
        res = d.validate_event_sequence([])
        self.assertTrue(res.is_valid)
        self.assertFalse(res.is_complete)
        self.assertFalse(res.accepted)

    def test_prefix_is_valid_but_not_accepted(self):
        res = d.validate_event_sequence(FULL[:4])
        self.assertTrue(res.is_valid)
        self.assertFalse(res.accepted)
        self.assertEqual(res.final_state, "ON_SCENE")

    def test_skipped_event_detected(self):
        seq = FULL[:2] + FULL[3:]                      # EN_ROUTE missing
        res = d.validate_event_sequence(seq)
        self.assertFalse(res.is_valid)
        self.assertTrue(res.halted)
        bad = res.first_invalid
        self.assertEqual((bad.index, bad.symbol, bad.from_state), (2, "ARRIVED_SCENE", "UNIT_ASSIGNED"))
        self.assertEqual(bad.reason, d.UNDEFINED_TRANSITION)

    def test_unknown_event(self):
        res = d.validate_event_sequence(["CALL_RECEIVED", "SIREN_FAULT"])
        self.assertEqual(res.first_invalid.reason, d.UNKNOWN_SYMBOL)

    def test_recovery_mode_lists_all(self):
        seq = ["CALL_RECEIVED", "CALL_RECEIVED", "AMBULANCE_DISPATCHED", "AMBULANCE_DISPATCHED"]
        found = d.find_invalid_transitions(seq)
        self.assertEqual([f["index"] for f in found], [1, 3])

    def test_expected_event(self):
        self.assertEqual(d.get_expected_event("IDLE"), "CALL_RECEIVED")
        self.assertIsNone(d.get_expected_event("CLOSED"))

    def test_result_is_serialisable(self):
        import json
        json.dumps(d.validate_event_sequence(FULL[:3]).to_dict())


# --------------------------------------------------------------------------- #
# Failure classification
# --------------------------------------------------------------------------- #
class TestClassification(unittest.TestCase):
    def test_types(self):
        F = fd.FailureType
        self.assertEqual(fd.classify_invalid_transition("UNIT_ASSIGNED", "ARRIVED_SCENE")[0], F.MISSING_EVENT)
        self.assertEqual(fd.classify_invalid_transition("UNIT_MOVING", "EN_ROUTE")[0], F.DUPLICATE_EVENT)
        self.assertEqual(fd.classify_invalid_transition("TRANSPORTING", "PATIENT_LOADED")[0], F.OUT_OF_ORDER)
        self.assertEqual(fd.classify_invalid_transition("CLOSED", "CALL_RECEIVED")[0], F.OUT_OF_ORDER)
        self.assertEqual(fd.classify_invalid_transition("IDLE", "BANANA")[0], F.UNKNOWN_EVENT)


# --------------------------------------------------------------------------- #
# Sample-data integration tests
# --------------------------------------------------------------------------- #
class TestSampleData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = analytics.load_events()
        cls.failures = fd.detect_failures(cls.df)
        cls.metrics = analytics.compute_incident_metrics(cls.df)

    def types_for(self, incident):
        f = self.failures
        return set(f.loc[f["incident_id"] == incident, "failure_type"])

    def test_load(self):
        self.assertEqual(self.df["incident_id"].nunique(), 12)
        self.assertEqual(len(self.df), 95)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.df["timestamp"]))

    def test_missing_columns_rejected(self):
        with self.assertRaises(ValueError):
            analytics.prepare_events(pd.DataFrame({"incident_id": [1]}))

    def test_validation_table(self):
        v = fd.validate_incidents(self.df).set_index("incident_id")
        accepted = set(v.index[v["accepted"]])
        self.assertEqual(accepted, {"INC001", "INC002", "INC005", "INC007", "INC008", "INC010", "INC012"})
        self.assertEqual(v.loc["INC003", "first_invalid_index"], 2)
        self.assertFalse(v.loc["INC004", "is_complete"])

    def test_expected_failures_per_incident(self):
        expected = {
            "INC001": set(),
            "INC002": {"SLOW_RESPONSE"},
            "INC003": {"MISSING_EVENT"},
            "INC004": {"INCOMPLETE_SEQUENCE"},
            "INC005": {"SLOW_RESPONSE", "EVENT_GAP_EXCEEDED"},
            "INC006": {"DUPLICATE_EVENT"},
            "INC007": {"TIMESTAMP_REGRESSION"},
            "INC008": set(),
            "INC009": {"OUT_OF_ORDER"},
            "INC010": set(),
            "INC011": {"UNKNOWN_EVENT"},
            "INC012": set(),
        }
        for incident, types in expected.items():
            with self.subTest(incident=incident):
                self.assertEqual(self.types_for(incident), types)

    def test_failure_rate_and_summary(self):
        self.assertAlmostEqual(fd.failure_rate(self.df, self.failures), 8 / 12, places=3)
        summary = fd.summarize_failures(self.failures)
        self.assertEqual(int(summary["count"].sum()), len(self.failures))

    def test_no_failures_on_clean_data(self):
        clean = self.df[self.df["incident_id"].isin(["INC001", "INC008"])]
        self.assertTrue(fd.detect_failures(clean).empty)
        self.assertEqual(fd.failure_rate(clean, fd.detect_failures(clean)), 0.0)

    def test_metrics_values(self):
        m = self.metrics.set_index("incident_id")
        self.assertEqual(m.loc["INC001", "response_min"], 6.0)
        self.assertEqual(m.loc["INC001", "dispatch_min"], 1.0)
        self.assertEqual(m.loc["INC001", "travel_min"], 4.0)
        self.assertEqual(m.loc["INC001", "scene_min"], 9.0)
        self.assertEqual(m.loc["INC001", "transport_min"], 12.0)
        self.assertEqual(m.loc["INC001", "total_min"], 35.0)
        self.assertEqual(m.loc["INC005", "response_min"], 20.0)

    def test_negative_duration_becomes_nan(self):
        m = self.metrics.set_index("incident_id")
        self.assertTrue(math.isnan(m.loc["INC007", "response_min"]))

    def test_incomplete_incident_has_nan_totals(self):
        m = self.metrics.set_index("incident_id")
        self.assertTrue(math.isnan(m.loc["INC004", "total_min"]))
        self.assertEqual(m.loc["INC004", "response_min"], 9.0)

    def test_summary_stats(self):
        s = analytics.summarize_response_times(self.metrics)
        self.assertEqual(s["count"], 11)
        self.assertEqual(s["median"], 8.0)
        self.assertEqual(s["max"], 20.0)
        self.assertAlmostEqual(s["mean"], 9.64, places=2)

    def test_summary_empty(self):
        empty = pd.DataFrame({"response_min": [float("nan")]})
        self.assertEqual(analytics.summarize_response_times(empty)["count"], 0)

    def test_slow_flagging_is_strict(self):
        slow = analytics.flag_slow_responses(self.metrics, threshold_min=10.0)
        self.assertEqual(set(slow["incident_id"]), {"INC002", "INC005"})   # INC012 == 10.0, not slow
        self.assertEqual(
            set(analytics.flag_slow_responses(self.metrics, threshold_min=8.0)["incident_id"]),
            {"INC002", "INC005", "INC004", "INC008", "INC012"},
        )

    def test_ambulance_table(self):
        t = analytics.response_time_by_ambulance(self.metrics).set_index("ambulance_id")
        self.assertEqual(t.loc["AMB04", "mean"], 13.5)     # (20 + 7) / 2
        self.assertEqual(int(t.loc["AMB05", "measured_incidents"]), 1)  # INC007 is NaN
        self.assertEqual(int(t.loc["AMB05", "total_incidents"]), 2)

    def test_generate_report_keys(self):
        r = fd.generate_report(self.df)
        for key in ("total_incidents", "failure_rate", "validation", "failures",
                    "failure_summary", "metrics", "response_summary",
                    "ambulance_performance", "slow_responses"):
            self.assertIn(key, r)


if __name__ == "__main__":
    unittest.main()
