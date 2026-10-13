import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test")

agent_module = importlib.import_module("gtm_agent.gtm_agent")
SYSTEM_PROMPT = agent_module.SYSTEM_PROMPT
send_prospect_email = agent_module.send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {}})
        self.prospect = {
            "prospect_id": "LEAD-50001",
            "name": "Priya Nair",
            "email": "priya.nair@brightwaveapps.com",
        }
        self.from_rep = {"name": "Rep", "email": "rep@example.com"}

    def send(self, prospect=None):
        return send_prospect_email.func(
            prospect or self.prospect,
            "Subject",
            "Body",
            self.runtime,
            self.from_rep,
        )

    @patch("gtm_agent.gtm_agent.uuid.uuid4")
    @patch("gtm_agent.gtm_agent.data_service.get_prospect_record")
    def test_disqualified_prospect_is_blocked_without_sending(self, get_record, uuid4):
        get_record.return_value = {"disqualified": True}

        result = self.send()

        self.assertEqual(
            result,
            {"status": "blocked", "error": "Prospect is disqualified; email not sent."},
        )
        get_record.assert_called_once_with("LEAD-50001")
        uuid4.assert_not_called()

    @patch("gtm_agent.gtm_agent.uuid.uuid4")
    @patch("gtm_agent.gtm_agent.data_service.get_prospect_record")
    def test_qualified_prospect_can_be_emailed(self, get_record, uuid4):
        get_record.return_value = {"disqualified": False}
        uuid4.return_value.hex = "0123456789abcdef"

        result = self.send()

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["message_id"], "msg-0123456789ab")
        get_record.assert_called_once_with("LEAD-50001")
        uuid4.assert_called_once()

    @patch("gtm_agent.gtm_agent.uuid.uuid4")
    @patch("gtm_agent.gtm_agent.data_service.get_prospect_record", return_value=None)
    def test_missing_prospect_is_blocked_without_sending(self, get_record, uuid4):
        result = self.send()

        self.assertEqual(
            result,
            {"status": "blocked", "error": "Prospect record not found; email not sent."},
        )
        get_record.assert_called_once_with("LEAD-50001")
        uuid4.assert_not_called()

    def test_prompt_requires_explaining_a_disqualified_send_block(self):
        self.assertIn("disqualified", SYSTEM_PROMPT)
        self.assertIn("tell the rep the reason", SYSTEM_PROMPT)
        self.assertIn("never claim the email was sent", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
