from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import unittest

from app.profiles import PROFILES, profile_prompt, role_packet


class ProfileTests(unittest.TestCase):
    def context(self, role, entities=None):
        return SimpleNamespace(role=role, route=SimpleNamespace(
            intent="billing", primary_agent="billing", entities=entities or {},
            jev={"choice_probabilities": {"billing": 0.91}},
        ))

    def test_contract_preserves_original_inputs_without_claiming_unavailable_data(self):
        profile = PROFILES["general"]
        self.assertIn("用户画像", profile.input_contract)
        prompt = profile_prompt(profile)
        self.assertIn(profile.mission, prompt)
        self.assertIn(" -> ".join(profile.workflow), prompt)
        self.assertIn("；".join(profile.output_contract), prompt)
        self.assertIn("当前未接入用户画像", prompt)
        self.assertIn("不得编造这些字段", prompt)
        with self.assertRaises(FrozenInstanceError):
            profile.max_tokens = 1

    def test_role_packet_missing_billing_fields_and_unavailable_urgency(self):
        packet = role_packet(self.context("billing", {"order_id": ["A1001"], "amount": [], "date": []}))
        self.assertEqual(packet["verification_fields"]["missing_fields"], ["支付金额"])
        self.assertEqual(packet["intent_confidence"], 0.91)
        self.assertIsNone(packet["urgency"])
        self.assertNotIn("user_profile", packet)
        other = role_packet(self.context("billing", {"amount": ["20元"]}))
        self.assertEqual(other["verification_fields"]["missing_fields"], ["订单号或交易号"])
        self.assertEqual(packet["verification_fields"]["order_id"], ["A1001"])

    def test_original_domain_packets_and_existing_shared_permissions(self):
        general = role_packet(self.context("general"))
        self.assertEqual(general["triage_targets"], ["technical", "billing", "escalation"])
        self.assertEqual(general["response_mode"], "answer_or_clarify")
        technical = role_packet(self.context("technical", {"error_code": ["401"]}))
        self.assertEqual(technical["diagnostic_fields"]["error_codes"], ["401"])
        for profile in PROFILES.values():
            self.assertIn("search_resolved_cases", profile.tool_scope)
            self.assertIn("create_handoff_summary", profile.tool_scope)
        self.assertNotIn("compare_amounts", PROFILES["technical"].tool_scope)
        self.assertIn("inspect_request_context", PROFILES["escalation"].tool_scope)


if __name__ == "__main__":
    unittest.main()
