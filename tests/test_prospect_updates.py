import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service
from gtm_agent.gtm_agent import build_prospect_profile, score_prospect


class ProspectUpdateTests(unittest.TestCase):
    prospect_id = "LEAD-71001"

    def setUp(self):
        self.record = data_service.PROSPECTS[self.prospect_id]
        self.original_tech_stack = list(self.record["tech_stack"])
        self.had_profile = self.prospect_id in data_service._PROFILES
        self.original_profile = data_service._PROFILES.pop(self.prospect_id, None)

    def tearDown(self):
        self.record["tech_stack"] = self.original_tech_stack
        data_service._PROFILES.pop(self.prospect_id, None)
        if self.had_profile:
            data_service._PROFILES[self.prospect_id] = self.original_profile

    def test_update_persists_technology_and_avoids_duplicates(self):
        result = data_service.update_prospect_info(self.prospect_id, "Kafka")

        self.assertTrue(result["updated"])
        self.assertIn("Kafka", self.record["tech_stack"])
        self.assertEqual(result["tech_stack"], self.record["tech_stack"])

        duplicate_result = data_service.update_prospect_info(self.prospect_id, "Kafka")

        self.assertTrue(duplicate_result["updated"])
        self.assertEqual(self.record["tech_stack"].count("Kafka"), 1)

    def test_profile_read_after_update_rebuilds_cached_profile(self):
        initial = build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertNotIn("Kafka", initial["prospect_profile"]["tech_stack"])

        data_service.update_prospect_info(self.prospect_id, "Kafka")
        refreshed = build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        self.assertIn("Kafka", refreshed["prospect_profile"]["tech_stack"])

    def test_rescore_uses_persisted_technology(self):
        profile = build_prospect_profile.invoke({"prospect_id": self.prospect_id})[
            "prospect_profile"
        ]
        result = data_service.update_prospect_info(self.prospect_id, "Kafka")
        self.assertTrue(result["updated"])

        class FakeScorer:
            def invoke(self, messages):
                user_message = messages[1]["content"]
                profile_text = user_message.split("\n\nProspect profile:\n", 1)[1]
                scored_profile = json.loads(profile_text)
                missing = [
                    technology
                    for technology in data_service.OFFERINGS["OFFER-10005"][
                        "required_tech_stack"
                    ]
                    if technology not in scored_profile["tech_stack"]
                ]
                justification = (
                    f"Missing required technologies: {', '.join(missing)}."
                    if missing
                    else "No required technologies are missing."
                )
                return SimpleNamespace(
                    model_dump=lambda: {
                        "score": 85,
                        "max_score": 100,
                        "justification": justification,
                        "rubric_breakdown": {
                            "revenue_fit": 30,
                            "tech_stack_match": 30,
                            "segment_fit": 25,
                            "component_max": 100,
                        },
                    }
                )

        with patch("gtm_agent.gtm_agent._scoring_llm", FakeScorer()):
            score = score_prospect.invoke(
                {
                    "prospect_profile": profile,
                    "offering": data_service.get_offering("OFFER-10005"),
                }
            )

        self.assertNotIn("Kafka", score["justification"])

    def test_failed_write_does_not_claim_success_or_keep_stale_profile(self):
        class FailingRecord(dict):
            def __setitem__(self, key, value):
                if key == "tech_stack":
                    raise OSError("write failed")
                return super().__setitem__(key, value)

        prospect_id = "LEAD-FAIL"
        record = FailingRecord(tech_stack=["Snowflake"])
        records = {prospect_id: record}
        data_service._PROFILES[prospect_id] = {"tech_stack": ["Kafka"]}

        with patch.object(data_service, "PROSPECTS", records):
            result = data_service.update_prospect_info(prospect_id, "Kafka")

        self.assertFalse(result["updated"])
        self.assertTrue(result["found"])
        self.assertEqual(result["tech_stack"], ["Snowflake"])
        self.assertNotIn(prospect_id, data_service._PROFILES)
