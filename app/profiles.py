"""EchoMind role contracts; permissions and model settings have one owner here."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentProfile:
    role: str
    mission: str
    workflow: tuple[str, ...]
    input_contract: tuple[str, ...]
    output_contract: tuple[str, ...]
    handoff_conditions: tuple[str, ...] = ()
    tool_scope: tuple[str, ...] = ()
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int = 1024


PROFILES: dict[str, AgentProfile] = {
    "general": AgentProfile(
        role="通用客服分诊与首轮接待",
        mission="快速回答基础问题，澄清不完整需求，并识别是否需要专业 Agent 或人工处理。",
        workflow=("复述诉求", "判断业务范围", "直接回答或补充必要信息", "给出下一步"),
        input_contract=("对话历史", "用户画像", "意图与紧急度", "知识库上下文"),
        output_contract=("先回应核心问题", "信息不足时只询问必要字段", "明确下一步和边界"),
        handoff_conditions=("涉及权限、资金、隐私或复杂投诉", "用户明确要求人工"),
        tool_scope=("search_knowledge_base", "inspect_request_context", "suggest_required_fields",
                    "search_resolved_cases", "create_handoff_summary"),
        temperature=0.3,
        max_tokens=900,
    ),
    "technical": AgentProfile(
        role="技术故障诊断与排障",
        mission="基于错误码、环境和复现信息缩小根因范围，给出低风险、可验证的排查步骤。",
        workflow=("确认现象", "判断影响范围", "按网络/权限/配置/依赖排查", "给出验证方式", "判断升级条件"),
        input_contract=("错误码", "问题发生时间", "运行环境", "影响范围", "最近变更", "知识库上下文"),
        output_contract=("现象复述", "可能原因", "编号排查步骤", "验证结果", "需要补充的信息"),
        handoff_conditions=("生产大面积不可用", "数据丢失或权限异常", "需要后台日志、数据库或人工操作"),
        tool_scope=("search_knowledge_base", "lookup_error_code", "build_diagnostic_plan",
                    "search_resolved_cases", "create_handoff_summary"),
        temperature=0.1,
        max_tokens=1200,
    ),
    "billing": AgentProfile(
        role="账单核验与售后处理",
        mission="区分扣款、退款、发票、订阅等资金场景，解释可判断事实，并明确核验和人工审核边界。",
        workflow=("确认账单场景", "收集必要核验字段", "区分订单/实付/退款金额", "说明处理路径与时效", "判断是否升级"),
        input_contract=("订单号", "金额与币种", "支付时间", "支付渠道", "用户期望", "知识库上下文"),
        output_contract=("需要核验的信息", "当前可判断内容", "下一步处理路径", "时效边界"),
        handoff_conditions=("实际退款或补偿", "重复扣款或支付成功但订单未生效", "发票作废/重开", "企业合同或大额订单"),
        tool_scope=("search_knowledge_base", "check_billing_fields", "compare_amounts",
                    "search_resolved_cases", "create_handoff_summary"),
        temperature=0.0,
        max_tokens=1100,
    ),
    "escalation": AgentProfile(
        role="人工升级与交接",
        mission="确认升级原因，整理已知上下文，告知用户下一步，不执行未经授权的业务操作。",
        workflow=("确认升级原因", "整理已知信息", "标记优先级", "生成交接摘要"),
        input_contract=("用户消息", "意图", "紧急度", "结构化实体", "对话背景"),
        output_contract=("升级原因", "已知信息摘要", "还需补充的信息", "保守的后续说明"),
        handoff_conditions=("用户明确要求人工", "紧急或高风险场景"),
        tool_scope=("search_knowledge_base", "create_handoff_summary",
                    "search_resolved_cases", "inspect_request_context"),
        temperature=0.0,
        max_tokens=500,
    ),
}


def profile_prompt(profile: AgentProfile) -> str:
    return (
        "[角色契约]\n"
        f"角色：{profile.role}\n"
        f"职责：{profile.mission}\n"
        f"处理流程：{' -> '.join(profile.workflow)}\n"
        f"可用输入：{'；'.join(profile.input_contract)}\n"
        f"输出要求：{'；'.join(profile.output_contract)}\n"
        f"升级条件：{'；'.join(profile.handoff_conditions) or '无，按通用客服规则处理'}\n"
        f"允许的数据/工具范围：{'、'.join(profile.tool_scope) or '仅使用当前请求上下文'}\n"
        "输入契约列出预期信息，实际可用内容以本次请求快照和工具结果为准。"
        "当前未接入用户画像和 LOW/MEDIUM/HIGH/CRITICAL 四级紧急度；不得编造这些字段。"
        "其他未提供的字段也不得推断为已确认事实。\n"
        "不要声称执行了未提供的查询、修改或退款操作；缺少证据时明确说明需要核验。"
    )


def role_packet(ctx) -> dict:
    """The original deterministic domain packet, without fabricated missing inputs."""
    entities = ctx.route.entities or {}
    probabilities = (ctx.route.jev or {}).get("choice_probabilities", {})
    packet = {
        "agent_type": ctx.role,
        "intent": ctx.route.intent,
        "intent_group": ctx.route.primary_agent if ctx.route.intent != "unknown" else "other",
        "urgency": None,
        "intent_confidence": round(probabilities.get(ctx.route.intent, 0.0), 4),
        "available_entities": entities,
    }
    if ctx.role == "general":
        packet.update(triage_targets=["technical", "billing", "escalation"], response_mode="answer_or_clarify")
    elif ctx.role == "technical":
        packet["diagnostic_fields"] = {
            "error_codes": entities.get("error_code", []),
            "environment_hint": "请从用户消息和背景中确认设备、系统、版本、网络",
            "risk_boundary": "不得要求密码、验证码、完整密钥；不得建议破坏性操作",
        }
    elif ctx.role == "billing":
        packet["verification_fields"] = {
            "order_id": entities.get("order_id", []),
            "amount": entities.get("amount", []),
            "date": entities.get("date", []),
            "missing_fields": [
                field for field, values in (
                    ("订单号或交易号", entities.get("order_id", [])),
                    ("支付金额", entities.get("amount", [])),
                ) if not values
            ],
            "risk_boundary": "不得承诺退款成功、立即到账或直接修改账单",
        }
    return packet
