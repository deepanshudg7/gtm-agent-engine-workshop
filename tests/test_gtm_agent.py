import os
import unittest
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

os.environ.setdefault("OPENAI_API_KEY", "test")

import gtm_agent.gtm_agent as gtm


REP = {"rep_id": "rep-test", "name": "Taylor Rep", "email": "rep@example.com"}
OFFERING = {
    "offering_id": "offer-test",
    "required_tech_stack": ["Kafka"],
    "min_annual_revenue": 1_000_000,
    "description": "Test offering",
}
RECORD = {
    "name": "Pat Prospect",
    "email": "prospect@example.com",
    "annual_revenue": 2_000_000,
}


class ScriptedModel(BaseChatModel):
    script: list[tuple[str, str, object]]
    bound_tool_names: list[str] = Field(default_factory=list)
    seen_messages: list[list[BaseMessage]] = Field(default_factory=list)
    called_tool_names: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen_messages.append(messages)
        kind, name, value = self.script.pop(0)
        if kind == "tool":
            self.called_tool_names.append(name)
            response = AIMessage(
                content="",
                tool_calls=[{
                    "name": name,
                    "args": value,
                    "id": f"call-{len(self.called_tool_names)}",
                    "type": "tool_call",
                }],
            )
        else:
            response = AIMessage(content=str(value))
        return ChatResult(generations=[ChatGeneration(message=response)])


class ScoreResult:
    def model_dump(self):
        return {
            "score": 82,
            "max_score": 100,
            "justification": "Strong fit",
            "rubric_breakdown": {
                "revenue_fit": 90,
                "tech_stack_match": 80,
                "segment_fit": 75,
                "component_max": 100,
            },
        }


class ScoringModel:
    def invoke(self, messages):
        return ScoreResult()


def tool_call(name, args=None):
    return ("tool", name, args or {})


def final_reply(content):
    return ("final", "", content)


class GTMAgentTests(unittest.TestCase):
    def make_agent(self, script):
        model = ScriptedModel(script=script)
        return model, gtm.create_gtm_agent(model)

    def mock_data(self, record=RECORD, tech_stack=None):
        rep_lookups = []

        def get_rep(rep_id):
            rep_lookups.append(rep_id)
            return REP

        self.enterContext(patch.object(gtm.data_service, "get_rep", side_effect=get_rep))
        self.enterContext(patch.object(gtm.data_service, "get_prospect_record", return_value=record))
        self.enterContext(patch.object(gtm.data_service, "get_offering", return_value=OFFERING))
        self.enterContext(patch.object(
            gtm.data_service, "get_profile_from_db", return_value={"prospect_profile": None}
        ))
        self.enterContext(patch.object(gtm.data_service, "fetch_engagement_history", return_value=[]))
        self.enterContext(patch.object(gtm.data_service, "fetch_account_details", return_value={}))
        self.enterContext(patch.object(
            gtm.data_service, "fetch_tech_stack", return_value=tech_stack or ["Kafka"]
        ))
        self.enterContext(patch.object(gtm.data_service, "save_profile_to_db", return_value={"saved": True}))
        self.enterContext(patch.object(
            gtm.data_service, "update_prospect_info", return_value={"updated": True, "found": True}
        ))
        self.enterContext(patch.object(gtm, "_scoring_llm", ScoringModel()))
        return rep_lookups

    def invoke_flow(self, model, agent, request):
        with patch.object(gtm, "gtm_agent", agent):
            result = gtm.run_agent(
                request,
                user_id="rep-test",
                environment="test",
                thread_id="thread-test",
            )
        return result

    def test_agent_uses_only_domain_prompt_and_tools(self):
        model, agent = self.make_agent([final_reply("ready")])

        result = agent.invoke({"messages": [{"role": "user", "content": "hello"}]})

        self.assertEqual(result["messages"][-1].content, "ready")
        self.assertEqual(model.bound_tool_names, [tool.name for tool in gtm.DOMAIN_TOOLS])
        self.assertEqual(len(model.seen_messages[0]), 2)
        self.assertEqual(model.seen_messages[0][0].content, gtm.SYSTEM_PROMPT)
        self.assertFalse({"write_todos", "ls", "read_file", "write_file", "task"} & set(model.bound_tool_names))
        self.assertNotIn("write_todos", model.seen_messages[0][0].content)
        self.assertNotIn("filesystem", model.seen_messages[0][0].content)
        self.assertNotIn("subagent", model.seen_messages[0][0].content)

    def test_run_agent_preserves_messages_input_and_runnable_config(self):
        class AgentSpy:
            def invoke(self, input, config):
                self.input = input
                self.config = config
                return {"messages": [AIMessage(content="done")]}

        spy = AgentSpy()
        with patch.object(gtm, "gtm_agent", spy):
            result = gtm.run_agent(
                "Score a prospect.",
                user_id="rep-test",
                environment="test",
                thread_id="thread-test",
            )

        self.assertEqual(spy.input, {"messages": [{"role": "user", "content": "Score a prospect."}]})
        self.assertEqual(spy.config["run_name"], "GTM Assistant")
        self.assertEqual(spy.config["metadata"], {
            "thread_id": "thread-test",
            "user_id": "rep-test",
            "environment": "test",
            "request_intent": "score_prospect",
        })
        self.assertEqual(result["reply"], "done")
        self.assertEqual(result["run_id"], str(spy.config["run_id"]))

    def test_email_flow_and_runtime_config(self):
        rep_lookups = self.mock_data()
        model, agent = self.make_agent([
            tool_call("get_current_rep"),
            tool_call("get_prospect", {"prospect_id": "lead-email"}),
            tool_call("send_prospect_email", {
                "prospect": {"prospect_id": "lead-email", **RECORD},
                "subject": "Demo",
                "body": "Let's schedule a demo.",
            }),
            final_reply("Email sent to Pat Prospect."),
        ])

        result = self.invoke_flow(model, agent, "Email lead-email about a demo.")

        self.assertEqual(model.called_tool_names, ["get_current_rep", "get_prospect", "send_prospect_email"])
        self.assertEqual(result["reply"], "Email sent to Pat Prospect.")
        self.assertIn("rep-test", rep_lookups)
        self.assertEqual(model.bound_tool_names, [tool.name for tool in gtm.DOMAIN_TOOLS])

    def test_score_flow(self):
        self.mock_data()
        profile = {
            "prospect_id": "lead-score",
            **RECORD,
            "engagement_history": [],
            "account_details": {},
            "tech_stack": ["Kafka"],
        }
        model, agent = self.make_agent([
            tool_call("get_current_rep"),
            tool_call("build_prospect_profile", {"prospect_id": "lead-score"}),
            tool_call("lookup_offering", {"offering_id": "offer-test"}),
            tool_call("score_prospect", {"prospect_profile": profile, "offering": OFFERING}),
            final_reply("Prospect score: 82."),
        ])

        result = self.invoke_flow(model, agent, "Score lead-score against offer-test.")

        self.assertEqual(model.called_tool_names, [
            "get_current_rep", "build_prospect_profile", "lookup_offering", "score_prospect"
        ])
        self.assertEqual(result["reply"], "Prospect score: 82.")

    def test_update_and_score_flow(self):
        self.mock_data(tech_stack=["Kafka"])
        profile = {
            "prospect_id": "lead-update",
            **RECORD,
            "engagement_history": [],
            "account_details": {},
            "tech_stack": ["Kafka"],
        }
        model, agent = self.make_agent([
            tool_call("get_current_rep"),
            tool_call("update_prospect_info", {"prospect_id": "lead-update", "technology": "Kafka"}),
            tool_call("build_prospect_profile", {"prospect_id": "lead-update"}),
            tool_call("lookup_offering", {"offering_id": "offer-test"}),
            tool_call("score_prospect", {"prospect_profile": profile, "offering": OFFERING}),
            final_reply("Updated the profile and scored it at 82."),
        ])

        result = self.invoke_flow(model, agent, "Add Kafka to lead-update and score against offer-test.")

        self.assertEqual(model.called_tool_names, [
            "get_current_rep", "update_prospect_info", "build_prospect_profile",
            "lookup_offering", "score_prospect",
        ])
        self.assertEqual(result["reply"], "Updated the profile and scored it at 82.")


if __name__ == "__main__":
    unittest.main()
