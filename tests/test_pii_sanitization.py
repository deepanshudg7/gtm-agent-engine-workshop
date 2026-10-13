import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gtm_agent import data_service
from gtm_agent.gtm_agent import build_prospect_profile, get_prospect, score_prospect


LEAD_ID = "LEAD-12853"
SENSITIVE_FIELD = "billing_qualification"
PROSPECT_RECORD = {
    "name": "Omar Okafor",
    "email": "omar.okafor@lakesideanalytics.com",
    "disqualified": False,
    SENSITIVE_FIELD: {"tax_id": "test-value"},
    "annual_revenue": 90000000,
}


class ProspectPIISanitizationTests(unittest.TestCase):
    def test_get_prospect_returns_only_contact_fields(self):
        with patch.object(data_service, "get_prospect_record", return_value=PROSPECT_RECORD):
            result = get_prospect.invoke({"prospect_id": LEAD_ID})

        self.assertEqual(
            result["prospect"],
            {
                "prospect_id": LEAD_ID,
                "name": "Omar Okafor",
                "email": "omar.okafor@lakesideanalytics.com",
                "disqualified": False,
            },
        )

    def test_new_profile_excludes_sensitive_fields_before_saving(self):
        with (
            patch.object(data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(data_service, "get_prospect_record", return_value=PROSPECT_RECORD),
            patch.object(data_service, "fetch_engagement_history", return_value=[]),
            patch.object(data_service, "fetch_account_details", return_value=[]),
            patch.object(data_service, "fetch_tech_stack", return_value=[]),
            patch.object(data_service, "save_profile_to_db") as save_profile,
        ):
            result = build_prospect_profile.invoke({"prospect_id": LEAD_ID})

        self.assertNotIn(SENSITIVE_FIELD, result["prospect_profile"])
        self.assertNotIn(SENSITIVE_FIELD, save_profile.call_args.args[1])

    def test_cached_profile_is_returned_as_sanitized_copy(self):
        cached_profile = {"prospect_id": LEAD_ID, **PROSPECT_RECORD}
        with patch.object(
            data_service,
            "get_profile_from_db",
            return_value={"prospect_profile": cached_profile},
        ):
            result = build_prospect_profile.invoke({"prospect_id": LEAD_ID})

        self.assertNotIn(SENSITIVE_FIELD, result["prospect_profile"])
        self.assertIn(SENSITIVE_FIELD, cached_profile)

    def test_scoring_llm_receives_sanitized_profile(self):
        profile = {"prospect_id": LEAD_ID, **PROSPECT_RECORD}
        offering = {
            "required_tech_stack": ["Kafka"],
            "min_annual_revenue": 1,
            "description": "Test offering",
        }
        with (
            patch.object(data_service, "fetch_tech_stack", return_value=["Kafka"]),
            patch("gtm_agent.gtm_agent._scoring_llm") as scoring_llm,
        ):
            scoring_llm.invoke.return_value = SimpleNamespace(model_dump=dict)
            score_prospect.invoke({"prospect_profile": profile, "offering": offering})

        user_message = scoring_llm.invoke.call_args.args[0][1]["content"]
        self.assertNotIn(SENSITIVE_FIELD, user_message)


if __name__ == "__main__":
    unittest.main()
