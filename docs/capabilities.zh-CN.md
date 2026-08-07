# 能力清单

本清单承接 README 原有的详细实现状态。它陈述仓库已经实现的行为，不构成生产就绪认证。

## 入库与表示

- 统一的 `Document` 模型和 loader 接口通过同一入口覆盖 TXT、Markdown、PDF 与 Word `.docx`。
- TXT loader 会拒绝明显的二进制内容，并支持常见编码兜底。
- Markdown loader 解析 Front Matter、标题、链接、表格与 fenced code。代码正文默认进入可检索文本，同时保留显式提取占位符与删除两种兼容模式。
- PDF loader 能检测扫描页，并按门禁启用 OCR 与表格抽取；Word loader 保留段落和多个表格。
- 保守文本清洗器与普通/Parent-Child 分块器保留 metadata。每个输出 chunk 都是处理后文档的原文连续切片，并记录半开区间 `start_char`/`end_char`。
- Parent-Child 入库只索引子块，父块以确定性 `parent_id` 单独保存。

## 检索与生成漏斗

- Embedding provider 共用一套抽象。仓库提供确定性本地 Hash provider，以及支持 dimensions、批处理、重试、token 计数和 usage 日志的 OpenAI provider。
- 向量库共用一套抽象。内存实现用于确定性本地评估；Qdrant 适配器支持批量写入、metadata 过滤、payload index、重试和跨进程持久化。
- Dense 与 BM25 稀疏检索输出兼容的 record id；RRF 以可配置权重融合，并显式稳定处理同分项。
- 重排层可插拔：确定性词法重排用于离线测试，Cohere 用作外部 neural provider；失败时回退 dense 原序。
- Multi-Query 支持原始 query 加确定性 fixture 或 chat-backed 改写，随后进入跨 query RRF、既有重排和父块展开阶段。
- 父块展开只发生在最终子块选择之后，并按 parent id 折叠兄弟子块命中。
- `ContextPacker` 在生成输入侧执行整块装填、精确及可选近重复折叠、连续引用编号与 token-aware 预算。
- RAG pipeline 组装检索上下文，调用 OpenAI-compatible chat client，并返回带来源记录的答案。

## 服务、缓存与访问控制

- CLI 分离 ingest 与 query；Qdrant 是跨进程持久化路径。
- 异步 FastAPI 适配层提供 `/ingest`、`/query` 与 SSE `/query/stream`，将同步检索漏斗 offload 到线程池，统一脱敏异常，并在入库后失效缓存。
- Chat 与 OpenAI embedding client 使用 `httpx.AsyncClient`、异步重试退避和 OpenAI-compatible 流式 delta。
- Redis L1/L2/L3 装饰器分别缓存 query embedding、检索结果和完整答案。检索/答案 key 含持久 corpus version；Redis 故障时 fail-open 为 miss。
- Redis store 持有独立后台事件循环，避免同步 CLI 与服务线程跨已关闭 loop 复用 client。
- ACL/RBAC 在候选评分之前执行。MySQL resolver 提供 principal membership 与可选 document/chunk binding，向量 payload 保存检索时使用的 ACL 快照。
- 默认从受信上游 header 取得 query 身份；身份缺失或 ACL 解析失败时 fail-closed。客户端 metadata filter 只能收窄服务端生成的 ACL filter。
- Dense、Qdrant、BM25、Multi-Query 各变体与缓存 key 共用同一个有效 ACL filter。

## 评估、可观测性与部署

- 确定性 JSONL 评估报告 hit rate、MRR、recall、能力切片和负样本行为；CI 同时钉住 unittest 精确账目和四个检索 profile。
- 生成质量由独立 Ragas 轨评估。Live 判官必须显式开启；CI 无密钥回放冻结 verdict fixture。
- 纯 ASGI request id 能跨线程 offload 传播，并在不缓冲 SSE 的前提下写回 `X-Request-ID`。
- 脱敏 JSON-line 审计会散列 principal/query 标识，排除 source 正文、ACL subjects 与密钥；审计、指标和 alert hook 故障全部 fail-open。
- `/metrics` 暴露有界 label 的时延、token usage、检索分数、空结果与缓存计数。
- `/health` 是浅层存活探针；`/ready` 只检查已启用的必需后端，并以脱敏依赖名 fail-closed。
- Docker Compose 提供 Qdrant、Redis、MySQL 与 app，带健康检查和幂等 demo 初始化；运行时密钥只从环境注入。

## 明确缺口

内置认证、JWT 校验与 session 管理尚未实现，生产部署必须置于受信认证网关之后。冻结语料规模刻意保持很小，负样本弃答尚未实现，live provider 的质量与容量仍取决于具体部署。
