import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ["OPENAI_API_KEY"] = "test"
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import gtm_agent


class ScoreProspectTests(unittest.TestCase):
    def setUp(self):
        self.scoring_result = {
            "score": 82.0,
            "max_score": 100,
            "justification": "Strong fit.",
            "rubric_breakdown": {
                "revenue_fit": 90.0,
                "tech_stack_match": 75.0,
                "segment_fit": 80.0,
                "component_max": 100,
            },
        }
        self.scoring_llm = patch.object(
            gtm_agent,
            "_scoring_llm",
            Mock(invoke=Mock(return_value=Mock(model_dump=Mock(return_value=self.scoring_result)))),
        )
        self.invoke_scoring_llm = self.scoring_llm.start().invoke
        self.addCleanup(self.scoring_llm.stop)

    def test_scores_using_server_loaded_records_and_id_only_schema(self):
        prospect_id = "LEAD-71001"
        offering_id = "OFFER-10005"
        gtm_agent.data_service._PROFILES.pop(prospect_id, None)
        self.addCleanup(gtm_agent.data_service._PROFILES.pop, prospect_id, None)
        with patch.object(
            gtm_agent,
            "_load_prospect_profile",
            wraps=gtm_agent._load_prospect_profile,
        ) as load_profile:
            result = gtm_agent.score_prospect.invoke(
                {"prospect_id": prospect_id, "offering_id": offering_id}
            )

        self.assertEqual(
            set(gtm_agent.score_prospect.args_schema.model_fields),
            {"prospect_id", "offering_id"},
        )
        load_profile.assert_called_once_with(prospect_id)
        self.invoke_scoring_llm.assert_called_once()
        user_message = self.invoke_scoring_llm.call_args.args[0][1]["content"]
        offering_text, profile_text = user_message.split("Offering:\n", 1)[1].split(
            "\n\nProspect profile:\n", 1
        )
        offering = json.loads(offering_text)
        profile = json.loads(profile_text)
        self.assertEqual(offering, gtm_agent.data_service.get_offering(offering_id))
        self.assertEqual(profile["prospect_id"], prospect_id)
        self.assertEqual(profile["name"], "Noah Williams")
        self.assertEqual(
            profile["tech_stack"],
            gtm_agent.data_service.fetch_tech_stack(prospect_id),
        )
        self.assertEqual(result, self.scoring_result)

    def test_validates_loaded_offering_before_scoring(self):
        offering = {"offering_id": "OFFER-10005", "required_tech_stack": []}
        with (
            patch.object(gtm_agent.data_service, "get_offering", return_value=offering),
            patch.object(gtm_agent, "_load_prospect_profile") as load_profile,
            patch.object(
                gtm_agent,
                "_offering_has_required_fields",
                wraps=gtm_agent._offering_has_required_fields,
            ) as validate_offering,
        ):
            result = gtm_agent.score_prospect.invoke(
                {"prospect_id": "LEAD-71001", "offering_id": "OFFER-10005"}
            )

        validate_offering.assert_called_once_with(offering)
        load_profile.assert_not_called()
        self.invoke_scoring_llm.assert_not_called()
        self.assertEqual(result, {"score": None, "error": "Cannot score without a valid offering."})

    def test_missing_or_invalid_loaded_records_do_not_score(self):
        valid_offering = {
            "required_tech_stack": ["Snowflake"],
            "min_annual_revenue": 1,
            "description": "Valid offering.",
        }
        cases = ((None, "LEAD-71001"), ({"name": "Missing id"}, "LEAD-71001"))
        for profile, prospect_id in cases:
            with self.subTest(profile=profile):
                with (
                    patch.object(
                        gtm_agent.data_service,
                        "get_offering",
                        return_value=valid_offering,
                    ),
                    patch.object(gtm_agent, "_load_prospect_profile", return_value=profile),
                ):
                    result = gtm_agent.score_prospect.invoke(
                        {"prospect_id": prospect_id, "offering_id": "OFFER-10005"}
                    )
            self.assertEqual(
                result,
                {"score": None, "error": "Cannot score without a valid prospect profile."},
            )
        self.invoke_scoring_llm.assert_not_called()

    def test_build_and_score_share_profile_loading(self):
        prospect_id = "LEAD-71001"
        gtm_agent.data_service._PROFILES.pop(prospect_id, None)
        self.addCleanup(gtm_agent.data_service._PROFILES.pop, prospect_id, None)
        with (
            patch.object(
                gtm_agent.data_service,
                "get_profile_from_db",
                wraps=gtm_agent.data_service.get_profile_from_db,
            ) as get_cached,
            patch.object(
                gtm_agent,
                "_load_prospect_profile",
                wraps=gtm_agent._load_prospect_profile,
            ) as load_profile,
        ):
            built = gtm_agent.build_prospect_profile.invoke({"prospect_id": prospect_id})
            profile = gtm_agent._load_prospect_profile(prospect_id)

        self.assertTrue(built["found"])
        self.assertEqual(profile, built["prospect_profile"])
        self.assertEqual(get_cached.call_count, 2)
        self.assertEqual(load_profile.call_count, 2)


if __name__ == "__main__":
    unittest.main()
