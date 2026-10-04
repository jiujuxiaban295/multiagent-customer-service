# 多agent编排客服

独立的 PyCharm / FastAPI 项目，只实现 **v3**：一次 Jev + 关键词路由，LangChain 主辅 Agent、本地人工交接节点，当前会话记忆和已解决案例检索。旧 EchoMind 项目、历史数据及评测结果保留。

```text
会话凭证 → 当前会话快照 → Jev + 本地规则 → 主辅角色并行执行
         → Agent 模型 / 工具循环或本地交接 → 必要时汇总 → 保存一组用户 / 助手消息

明确结案 → 验证用户解决证据与实际步骤 → 去身份化 → 发布共享案例
```

通用、技术、账单三个 LangChain Agent 图共用工厂，注入完整 `AgentProfile` 角色契约、角色输入包、Skills 和工具白名单；契约是提示约束，不是回答硬 schema。各技术/账单主辅分支失败后独立回退一次通用角色，并保留全部尝试轨迹。人工升级直接生成本地交接摘要，不调用生成模型；结构在 `extra.handoff_summaries`。四级紧急度尚未接入，该本地摘要标记为 `UNKNOWN`。正式知识库仍采用 bge-m3，保留查询改写、并行召回、片段去重和重排。没有模拟订单、支付或退款后台；工具不能查询个人交易状态或执行退款。

主角色沿用 Jev/关键词融合；辅助角色优先按粗意图与旧协作关键词选目标，无目标时要求分数至少 `0.45` 且至少为主角色分数的 `0.55` 倍。角色可用性只看注册集合，没有恢复 Monitor 在线降权。角色温度/最大输出分别为通用 `0.3/900`、技术 `0.1/1200`、账单 `0.0/1100`；Composer `0.1/1000`，强调主次、冲突需核验和通用 Skills 边界，失败时保留答案并加补充说明。本轮仅迁回角色契约、主辅选择/失败处理和人工升级，其他架构先保持现状。

## 在 PyCharm 中运行

打开项目目录 `/Users/yiyang/PycharmProjects/多agent编排客服`，解释器选择该目录下 `.venv/bin/python`，工作目录设为项目根目录，运行 `main.py`。

终端等价命令：

```bash
cd /Users/yiyang/PycharmProjects/多agent编排客服
.venv/bin/python main.py
```

入口：[接口文档](http://127.0.0.1:8010/docs)、[健康检查](http://127.0.0.1:8010/health)。本地入口仅监听 `127.0.0.1:8010`。

当前环境已安装依赖。需要重新创建时，使用 Python 3.12：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

启动前确认 `.env` 配置正确、Redis 可访问、LM Studio 已加载 bge-m3 并启用 HTTP 服务。`.env` 已忽略且权限为 `600`；不要把凭证写进代码、截图或提交记录。

| 配置 | 用途 |
|---|---|
| `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` / `ANTHROPIC_MODEL` | 业务生成模型；默认关闭 thinking |
| `JEV_API_KEY` / `JEV_MODEL` | Typesafe.ai 决策模型；当前固定版本 `jev-1.13.0` |
| `DEV_API_KEY` / `SESSION_SIGNING_KEY` | 本地开发会话创建密钥、会话凭证签名密钥 |
| `REDIS_URL` | 默认现有 Redis 的 DB 3；应用另加自身键前缀 |
| `ECHOMIND_EMBEDDING_BASE_URL` / `ECHOMIND_EMBEDDING_MODEL` | 本地 bge-m3；地址支持有或没有 `/v1`，统一调用 `/v1/embeddings` |
| `ECHOMIND_RAG_RECALL_K` / `RAG_CACHE_TTL` | 每个子查询召回数、召回缓存秒数；评测可设缓存为 `0` |
| `BUSINESS_SCOPE` | 服务端可信业务共享范围 |
| `MODEL_CALL_LIMIT` / `REQUEST_TIMEOUT` | 每个角色模型回合上限、路由与编排超时秒数 |

SQLite 原始会话及证据在 `data/records.sqlite3`；知识库和公开案例在本项目 `data/chroma/`，与旧项目隔离。正式知识库集合为 `knowledge_base__text_embedding_bge_m3`，案例集合为 `resolved_cases_v3`。修改默认目录可设置 `SQLITE_PATH` / `CHROMA_PATH`。

正式知识库已导入 26 篇；修订后重新导入：

```bash
.venv/bin/python -m scripts.import_kb data/kb/电商客服知识库.md
```

设置 `CHROMA_HOST` 时可只读使用现有远程知识库，导入命令会拒绝修改远程集合；案例仍保存在新项目本地。

## API 最小链路

先创建会话。以下密钥和 token 均为占位符：

```bash
curl -s http://127.0.0.1:8010/sessions \
  -H 'Content-Type: application/json' \
  -H 'X-Dev-Key: <.env 中的 DEV_API_KEY>' \
  -d '{"user_id":"demo-a"}'
```

保留响应中的 `conv_id` 和 `session_token`，后续请求携带 `Authorization: Bearer <session_token>`：

```bash
curl -s http://127.0.0.1:8010/chat \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <session_token>' \
  -d '{"message":"App 登录报错 401，怎么办？"}'
```

`conv_id` / `user_id` 可省略；身份取自签名凭证。若提交这两个字段，必须与凭证一致。`mode` 只接受 `v3`。`success` 表示回答链路执行成功，不能证明问题已经解决；`escalated` 表示路由的人工处理判断，或交接工具/本地节点已生成摘要，也不代表真实工单已经创建。 单分支失败回退时，`agent_type` 返回实际处理角色（如 `general`），`primary_agent` 保留原路由选择；分支详情见 `extra.role_results`。

真正执行步骤并确认解决后，才提交显式结案，例如：

```bash
curl -s http://127.0.0.1:8010/sessions/<conv_id>/close \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <session_token>' \
  -d '{"outcome":"resolved","confirmation":"我已在 App 重新登录，报错消失，问题已解决。","actual_steps":["重新登录"],"environment":"App"}'
```

这是请求格式示例，解决确认必须来自实际结果。每个 `actual_steps` 必须在用户解决确认原文中得到支持，`environment` 必须在用户记录中出现。只有关闭、明确解决且有充分公开内容的会话才发布案例。检查响应的 `outcome`、`publication_status` 和 `reason`；证据不足会拒绝发布。只需结束会话时可提交 `{"outcome":"unknown"}`。

| 接口 | 行为 |
|---|---|
| `GET /sessions/{conv_id}` | 当前会话状态、近期消息和摘要 |
| `POST /sessions/{conv_id}/close` | 显式结案；也支持已有用户消息的 `evidence_message_id` |
| `POST /sessions/{conv_id}/cases/retry` | 重试 `pending` 案例的索引写入 |
| `POST /sessions/{conv_id}/cases/revoke` | 撤销案例检索资格 |
| `POST /sessions/{conv_id}/reopen` | 先撤销旧案例，再重新打开会话 |
| `GET /trace/tool/{request_id}` | 查看同一会话凭证所属的工具轨迹 |
| `GET /skills` / `POST /skills/reload` | 查看或重新加载业务 Skills |
| `GET /metrics` | 完成请求数与 HTTP 耗时指标 |

除 `/sessions` 的 `X-Dev-Key`、公开 `/health` 和 `/metrics` 外，以上接口均需要 Bearer 凭证。关闭后不能继续聊天，必须明确重开或新建会话。另一用户只可经 Agent 的案例检索工具取得同一业务范围公开的解决经验，不能取得原始会话和证据。

## 验证与替代运行方式

```bash
.venv/bin/python -m unittest discover -s tests -v
```

目前 84 项自动测试通过，覆盖 Jev 优先/关键词兜底路由、角色契约和参数、主辅选择/注册过滤、实际 LangChain 图、主辅独立回退、本地交接、会话/案例生命周期、检索及 API。真实 DeepSeek 模型→工具→模型往返、78 条真实 Jev 路由校准与复测（多轮样例使用真实对话上下文）（见 [路由修改建议与实施记录](docs/路由修改建议.md)）、本地 bge-m3 检索和完整 API 结案闭环均已验证。真实 DeepSeek 改写、重排和 API 政策问答也有此前验证记录；本轮另通过3个真实 API 合成样例：技术工具链2次生成、技术+账单及汇总5次生成、本地人工交接0次生成，各样例Jev1次且只保存2条消息；见 [role-alignment-verification.json](docs/role-alignment-verification.json)。这些冒烟结果不代表所有失败场景已通过真实 provider 验收。复跑角色验收可执行 `.venv/bin/python -m scripts.verify_role_alignment`（会调用真实服务），脚本见 [verify_role_alignment.py](scripts/verify_role_alignment.py)。完整质量基准、路由阈值校准和生产认证尚未完成。详情见 [实现与验证](docs/实现与验证.md)。

另提供独立 Docker Compose 配置：

```bash
docker compose up --build -d
```

Compose 使用自身 Redis 和持久目录，访问宿主机 embedding 服务的地址为 `host.docker.internal`。该配置尚未实际构建验证；本次验收使用本地 `.venv`。
