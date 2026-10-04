"""Business handlers reused from EchoMind; runtime injects identity, never the model."""
from types import SimpleNamespace
from typing import Annotated
from pydantic import Field
from langchain.tools import tool, ToolRuntime
from app.contracts import RunContext
from app.profiles import PROFILES
from app import business_tools as business


def request(runtime):
    ctx = runtime.context
    # Bridge new domain routing to the legacy deterministic field helper contract.
    domain = ctx.route.intent
    intent = {'technical': 'account', 'billing': 'payment_issue',
              'escalation': 'complaint', 'unknown': 'other'}.get(domain, 'other')
    if domain == 'general':
        if any(word in ctx.inp.message for word in ['物流', '快递', '配送', 'shipping']):
            intent = 'logistics'
        elif '订单' in ctx.inp.message or 'order' in ctx.inp.message.lower():
            intent = 'order_status'
    return SimpleNamespace(message=ctx.inp.message, request_id=ctx.request_id,
        context=ctx.inp.context, entities=ctx.route.entities,
        intent=SimpleNamespace(value=intent), intent_group=ctx.role,
        intent_confidence=max(ctx.route.routing_scores.values(), default=0),
        urgency=SimpleNamespace(name='HIGH' if ctx.route.escalated else 'NORMAL'))


def build_tools(index, cases):
    @tool
    async def search_knowledge_base(query: str, runtime: ToolRuntime[RunContext],
                                    top_k: Annotated[int, Field(ge=1, le=10)] = 5) -> dict:
        """检索正式政策、产品说明和标准流程；需要政策数字和条件时先调用。"""
        result = await index.search_knowledge(query, top_k)
        runtime.context.retrieved.extend(result.get('results', []))
        return result

    @tool
    async def search_resolved_cases(query: str, runtime: ToolRuntime[RunContext],
            top_k: Annotated[int, Field(ge=1, le=10)] = 5, product: str = '',
            version: str = '', error_code: str = '') -> dict:
        """检索本业务范围已发布、有效的解决经验。经验不能覆盖正式政策或证明个人订单状态。"""
        return {'success': True, 'cases': await cases.search(query, runtime.context.inp.scope,
            top_k, product=product, version=version, error_code=error_code)}

    @tool
    async def inspect_request_context(runtime: ToolRuntime[RunContext], focus: str = 'general') -> dict:
        """查看当前请求的意图和已有实体，不查询业务系统。"""
        return business.inspect_request_context(request(runtime), {'focus': focus})

    @tool
    async def suggest_required_fields(runtime: ToolRuntime[RunContext]) -> dict:
        """只询问当前诉求必要的缺失字段。"""
        return business.suggest_required_fields(request(runtime), {})

    @tool
    async def lookup_error_code(error_code: str, runtime: ToolRuntime[RunContext]) -> dict:
        """解释 HTTP 错误码及低风险排查方向，不读取服务端日志。"""
        return business.lookup_error_code(request(runtime), {'error_code': error_code})

    @tool
    async def build_diagnostic_plan(environment: str, reproduced: bool,
                                    runtime: ToolRuntime[RunContext]) -> dict:
        """根据环境及是否复现生成排障步骤，不执行修改。"""
        return business.build_diagnostic_plan(request(runtime), locals())

    @tool
    async def check_billing_fields(runtime: ToolRuntime[RunContext], payment_channel: str = '') -> dict:
        """检查账单核验字段，不查询订单支付或退款后台。"""
        return business.check_billing_fields(request(runtime), {'payment_channel': payment_channel})

    @tool
    async def compare_amounts(amount_a: Annotated[float, Field(allow_inf_nan=False)],
            amount_b: Annotated[float, Field(allow_inf_nan=False)], runtime: ToolRuntime[RunContext]) -> dict:
        """计算用户明确提供的金额差值，不做重复扣款结论，不执行退款。"""
        return business.compare_amounts(request(runtime), {'amount_a': amount_a, 'amount_b': amount_b})

    @tool
    async def create_handoff_summary(reason: str, runtime: ToolRuntime[RunContext]) -> dict:
        """生成交给人工客服的摘要，不会实际转接或创建工单。"""
        runtime.context.escalated = True
        return business.create_handoff_summary(request(runtime), {'reason': reason})

    tools = [search_knowledge_base, search_resolved_cases, inspect_request_context,
        suggest_required_fields, lookup_error_code, build_diagnostic_plan,
        check_billing_fields, compare_amounts, create_handoff_summary]
    return {t.name: t for t in tools}


SHARED = {'search_knowledge_base', 'search_resolved_cases', 'create_handoff_summary'}
TOOL_SCOPES = {role: set(profile.tool_scope) for role, profile in PROFILES.items()}
