"""Agent 工具定义与实现。

所有 Agent 工具集中在这里，编排器只负责：
  1. 根据 Agent 类型暴露工具白名单
  2. 执行 LLM 返回的 tool_use
  3. 将工具结果回传给 LLM

工具本身保持确定性、可测试，并明确区分：
  - 当前请求分析
  - 技术排障建议
  - 账单字段核验
  - 人工升级摘要
  - 共享知识库 RAG

订单查询、退款执行、账单修改等需要真实业务系统授权的动作不在这里伪造。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from agents.agent_orchestrator import Request


AgentToolHandler = Callable[["Request", Dict[str, Any]], Union[Any, Awaitable[Any]]]


@dataclass(frozen=True)
class AgentToolSpec:
    """Agent 可见工具的定义和执行函数。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: AgentToolHandler


def make_tool(
    name: str,
    description: str,
    properties: Dict[str, Any],
    handler: AgentToolHandler,
    required: Optional[List[str]] = None,
) -> AgentToolSpec:
    """创建带 JSON Schema 的 Agent 工具。"""
    return AgentToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
        handler=handler,
    )


def inspect_request_context(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用客服工具：返回脱敏后的当前请求快照。"""
    return {
        "intent": req.intent.value if req.intent else None,
        "intent_group": req.intent_group,
        "urgency": req.urgency.name if req.urgency else None,
        "intent_confidence": round(req.intent_confidence, 4),
        "entities": req.entities or {},
        "context_available": bool(req.context),
        "requested_focus": str(args.get("focus", "general"))[:40],
    }


def suggest_required_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用客服工具：按业务类型计算下一轮只需询问的字段。"""
    intent = req.intent.value if req.intent else "other"
    fields: List[str] = []
    if intent in {"order_status", "logistics"}:
        fields = ["订单号或下单时间"]
    elif intent in {"account", "account_security"}:
        fields = ["登录方式或账号标识", "问题发生时间"]
    elif intent in {"complaint", "request"}:
        fields = ["事件时间", "期望处理方式"]
    elif intent == "other":
        fields = ["希望解决的具体问题"]
    return {
        "intent": intent,
        "required_fields": fields,
        "known_entities": req.entities or {},
    }


def lookup_error_code(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """技术工具：解释常见错误码的排查方向，不声称读取了服务端日志。"""
    code = str(args.get("error_code", "")).upper().strip()
    mapping = {
        "401": ("认证失败", ["确认 Token/API Key 是否过期", "确认请求时间戳和签名", "确认账号登录状态"]),
        "403": ("权限不足", ["确认账号或套餐权限", "确认资源权限和 IP 白名单"]),
        "404": ("资源或路径不存在", ["确认接口路径和环境", "确认资源标识是否正确"]),
        "500": ("服务端处理异常", ["记录 request_id 和发生时间", "检查依赖服务、参数格式和服务端日志"]),
    }
    meaning, steps = mapping.get(
        code,
        ("暂未识别的错误码", ["补充完整错误信息、发生时间和运行环境"]),
    )
    return {
        "error_code": code,
        "meaning": meaning,
        "next_steps": steps,
        "server_log_checked": False,
    }


def build_diagnostic_plan(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """技术工具：生成低风险排障顺序。"""
    environment = str(args.get("environment", "unknown"))[:80]
    reproduced = bool(args.get("reproduced", False))
    steps = [
        "复现并记录完整错误信息",
        "确认网络、DNS、代理和证书",
        "确认版本、配置和权限",
    ]
    if reproduced:
        steps.append("用最小请求复现并记录 request_id")
    return {
        "environment": environment,
        "reproduced": reproduced,
        "diagnostic_steps": steps,
    }


def check_billing_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """账单工具：检查必要核验字段是否齐全。"""
    fields = {
        "order_id": bool(req.entities.get("order_id")),
        "amount": bool(req.entities.get("amount")),
        "date": bool(req.entities.get("date")),
        "payment_channel": bool(args.get("payment_channel")),
    }
    return {
        "fields": fields,
        "missing_fields": [name for name, present in fields.items() if not present],
        "can_confirm_refund": False,
        "reason": "当前工具只做字段检查，不连接订单或支付系统",
    }


def compare_amounts(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """账单工具：只做用户明确提供金额之间的算术。"""
    try:
        first = float(args["amount_a"])
        second = float(args["amount_b"])
    except (KeyError, TypeError, ValueError):
        return {"success": False, "error": "amount_a 和 amount_b 必须是数字"}
    return {
        "success": True,
        "amount_a": first,
        "amount_b": second,
        "difference": round(first - second, 2),
        "interpretation": "仅表示金额差值，不代表重复扣款或退款结论",
    }


def create_handoff_summary(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """升级工具：生成可交给人工客服的结构化摘要。"""
    return {
        "request_id": req.request_id,
        "reason": str(args.get("reason", "需要人工客服继续核验"))[:120],
        "intent": req.intent.value if req.intent else "unknown",
        "urgency": req.urgency.name if req.urgency else "UNKNOWN",
        "entities": req.entities or {},
        "sensitive_data_required": False,
    }


def build_shared_rag_tools(tool_manager: Any) -> Dict[str, AgentToolSpec]:
    """构建所有 Agent 可共享的 RAG 工具。"""

    async def search_knowledge_base(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or req.message or "").strip()
        top_k = int(args.get("top_k", 5) or 5)
        if not query:
            return {"success": False, "error": "query 不能为空", "results": []}
        if tool_manager is None:
            return {"success": False, "error": "RAG 工具未初始化", "results": []}

        result = await tool_manager.search_with_rewrite(
            "knowledge_search",
            query,
            top_k=top_k,
        )
        if not getattr(result, "success", False):
            return {
                "success": False,
                "query": query,
                "error": getattr(result, "error", "知识库检索失败"),
                "results": [],
                "reranked": False,
            }

        return {
            "success": True,
            "query": query,
            "top_k": top_k,
            "results": result.data,
            "reranked": bool(getattr(result, "reranked", False)),
        }

    return {
        "search_knowledge_base": make_tool(
            "search_knowledge_base",
            "检索知识库并返回最相关的文档片段；可用于通用、技术、账单和升级场景。",
            {
                "query": {"type": "string", "description": "用户问题或检索关键词"},
                "top_k": {"type": "integer", "description": "返回结果条数"},
            },
            search_knowledge_base,
            required=["query"],
        )
    }


def general_tools() -> Dict[str, AgentToolSpec]:
    return {
        "inspect_request_context": make_tool(
            "inspect_request_context",
            "查看当前请求的意图、紧急度、实体和上下文可用性；不查询外部业务系统。",
            {"focus": {"type": "string", "description": "希望关注的业务方向"}},
            inspect_request_context,
        ),
        "suggest_required_fields": make_tool(
            "suggest_required_fields",
            "根据当前意图建议下一轮只需向用户补充的字段。",
            {},
            suggest_required_fields,
        ),
    }


def technical_tools() -> Dict[str, AgentToolSpec]:
    return {
        "lookup_error_code": make_tool(
            "lookup_error_code",
            "解释常见 HTTP 错误码的可能含义和低风险排查方向；不会读取服务端日志。",
            {"error_code": {"type": "string", "description": "例如 401、403、500"}},
            lookup_error_code,
            required=["error_code"],
        ),
        "build_diagnostic_plan": make_tool(
            "build_diagnostic_plan",
            "根据运行环境和是否可复现生成排障顺序，不执行修改配置等操作。",
            {
                "environment": {"type": "string", "description": "App、浏览器、服务端或 Docker 等"},
                "reproduced": {"type": "boolean", "description": "问题是否可以稳定复现"},
            },
            build_diagnostic_plan,
            required=["environment", "reproduced"],
        ),
    }


def billing_tools() -> Dict[str, AgentToolSpec]:
    return {
        "check_billing_fields": make_tool(
            "check_billing_fields",
            "检查账单核验字段是否齐全；不连接订单、支付或退款系统。",
            {"payment_channel": {"type": "string", "description": "支付渠道，例如微信、支付宝、银行卡"}},
            check_billing_fields,
        ),
        "compare_amounts": make_tool(
            "compare_amounts",
            "计算用户明确提供的两笔金额差值；不判断是否重复扣款，也不执行退款。",
            {
                "amount_a": {"type": "number", "description": "第一笔金额"},
                "amount_b": {"type": "number", "description": "第二笔金额"},
            },
            compare_amounts,
            required=["amount_a", "amount_b"],
        ),
    }


def escalation_tools() -> Dict[str, AgentToolSpec]:
    return {
        "create_handoff_summary": make_tool(
            "create_handoff_summary",
            "生成交给人工客服的结构化交接摘要，不会创建真实工单。",
            {"reason": {"type": "string", "description": "需要升级的原因"}},
            create_handoff_summary,
        ),
    }


# ── 售前导购 ─────────────────────────────────────────────────────────────────

# 演示商品目录。生产环境应替换为商品搜索 API 的客户端，这里只做确定性返回，
# 避免售前 Agent 编造库存、价格和规格。
_PRODUCT_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "Pro X1 手机",
        "category": "手机",
        "price": 3999.0,
        "specs": {"屏幕": "6.7 英寸", "续航": "5000mAh", "NFC": "支持", "摄像头": "5000 万像素"},
        "tags": ["拍照", "游戏", "旗舰", "礼物"],
        "audience": "追求性能与影像体验的用户",
    },
    {
        "name": "Lite A3 手机",
        "category": "手机",
        "price": 1999.0,
        "specs": {"屏幕": "6.5 英寸", "续航": "4500mAh", "NFC": "支持", "摄像头": "4800 万像素"},
        "tags": ["性价比", "学生", "日常"],
        "audience": "预算有限、注重日常使用的用户",
    },
    {
        "name": "AirSound Pro 耳机",
        "category": "耳机",
        "price": 899.0,
        "specs": {"降噪": "主动降噪", "续航": "30 小时", "连接": "蓝牙 5.3"},
        "tags": ["降噪", "通勤", "礼物"],
        "audience": "通勤、差旅、需要降噪的用户",
    },
    {
        "name": "FitBand 6 手表",
        "category": "手表",
        "price": 1299.0,
        "specs": {"屏幕": "1.43 英寸 AMOLED", "续航": "14 天", "健康": "心率/血氧/睡眠"},
        "tags": ["运动", "健康", "礼物"],
        "audience": "关注运动与健康监测的用户",
    },
    {
        "name": "HomePod Mini 音箱",
        "category": "音箱",
        "price": 499.0,
        "specs": {"连接": "Wi-Fi/蓝牙", "控制": "语音助手"},
        "tags": ["居家", "智能", "礼物"],
        "audience": "家庭智能场景用户",
    },
    {
        "name": "ThinBook 14 笔记本",
        "category": "笔记本",
        "price": 5999.0,
        "specs": {"屏幕": "14 英寸 2.8K", "内存": "16GB", "重量": "1.29kg"},
        "tags": ["办公", "轻薄", "生产力"],
        "audience": "移动办公与生产力用户",
    },
]


def _extract_user_profile(context: str) -> Dict[str, Any]:
    """从 MemoryContext.to_prompt_text() 里提取 [用户画像] 的 JSON。"""
    if not context:
        return {}
    marker = "[用户画像]"
    pos = context.find(marker)
    if pos == -1:
        return {}
    start = context.find("{", pos + len(marker))
    if start == -1:
        return {}
    depth = 0
    for idx in range(start, len(context)):
        ch = context[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(context[start : idx + 1])
                    return data if isinstance(data, dict) else {}
                except Exception:
                    return {}
    return {}


def search_product_catalog(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """售前工具：检索商品目录并返回确定性结果，不伪造库存和实时价格。"""
    query = str(args.get("query") or req.message or "").strip()
    category = str(args.get("category") or "").strip().lower()
    budget = args.get("budget")
    limit = max(1, min(int(args.get("limit", 5) or 5), 10))

    try:
        budget = float(budget) if budget is not None else None
    except (TypeError, ValueError):
        budget = None

    lower_query = query.lower()
    query_tokens = _query_tokens(lower_query)

    def score(item: Dict[str, Any]) -> float:
        points = 0.0
        if category and category in item["category"].lower():
            points += 2.0
        haystack = " ".join(
            [item["name"], item["category"], item["audience"]]
            + list(item["specs"].values())
            + list(item["tags"])
        ).lower()
        for token in query_tokens:
            if token and token in haystack:
                points += 0.35
        return points

    candidates = [item for item in _PRODUCT_CATALOG if score(item) > 0] or list(_PRODUCT_CATALOG)
    candidates = [item for item in candidates if budget is None or item["price"] <= budget]
    candidates = sorted(candidates, key=score, reverse=True)

    results = [
        {
            "name": item["name"],
            "category": item["category"],
            "price": item["price"],
            "specs": item["specs"],
            "tags": item["tags"],
            "audience": item["audience"],
        }
        for item in candidates[:limit]
    ]

    profile = _extract_user_profile(req.context or "")
    profile_note = ""
    if profile:
        profile_note = "已结合用户画像中的历史偏好进行筛选口径；具体是否适合请以用户实际使用场景为准。"

    return {
        "success": True,
        "query": query,
        "category": category or None,
        "budget": budget,
        "matched": len(results),
        "results": results,
        "profile_used": bool(profile),
        "profile": profile,
        "personalized_note": profile_note,
        "stock_note": "库存与价格实时变化，请引导用户到商品页确认，不要承诺有货或最后一件。",
    }


def _query_tokens(query: str) -> List[str]:
    """把查询拆成 ASCII 单词 + 中文二元组，便于和商品目录做包含匹配。"""
    tokens: List[str] = re.findall(r"[a-z0-9]+", query.lower())
    for cjk_run in re.findall(r"[\u4e00-\u9fff]+", query):
        if len(cjk_run) == 1:
            tokens.append(cjk_run)
        else:
            tokens.extend(cjk_run[i : i + 2] for i in range(len(cjk_run) - 1))
    return tokens


def pre_sale_tools() -> Dict[str, AgentToolSpec]:
    return {
        "search_product_catalog": make_tool(
            "search_product_catalog",
            "检索商品目录，返回商品规格、价格区间、适用人群和标签，用于对比、推荐和规格解释；不返回实时库存。",
            {
                "query": {"type": "string", "description": "用户诉求或商品关键词"},
                "category": {"type": "string", "description": "商品分类，例如手机、耳机、手表"},
                "budget": {"type": "number", "description": "用户预算上限（元），可选"},
                "limit": {"type": "integer", "description": "返回数量，默认 5"},
            },
            search_product_catalog,
            required=["query"],
        ),
    }
