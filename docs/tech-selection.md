# 技术选型思考：面向企业的 RAG 知识库

本文记录当前项目为什么选择这些框架、模型、协议和工程边界。它不是安装手册，而是后续做架构复盘、替换组件和评估收益时的判断依据。

## 选型总原则

- **可替换**：embedding、vector store、reranker、query rewriter、cache 都通过抽象层接入，避免把供应商 SDK 写死在 RAG 主流程里。
- **可复现**：离线评估默认使用 `HashEmbeddingProvider` 和 deterministic fixture，先保证指标可逐位复现，再谈线上模型收益。
- **可降级**：网络 I/O、rerank、cache、query rewrite 失败时优先降级到已知稳定路径，而不是中断问答。
- **可观测**：关键阶段保留配置、来源、token、耗时、cache hit/miss 和错误分类，方便后续接入指标与审计。

## Document Loading：PyMuPDF + pdfplumber + RapidOCR 可选兜底

文档加载层优先做“入库诚实性”而不是盲目扩大格式能力。TXT、Markdown、PDF、Word 都先统一成 `Document(content, metadata)`，后续清洗、切分、embedding 和 ACL payload 写入只消费这个统一模型。

PDF 选择 PyMuPDF 作为主解析器，因为它对文本层和页级遍历足够稳定，也能在需要 OCR 时把单页渲染成图像。扫描页检测默认开启，只看页内文本层是否低于阈值；这一步不改变内容，只写 `scanned_page_count` / `ocr_page_count` 并打 warning，避免纯扫描 PDF 静默变成只有页眉骨架的“空知识”。OCR 选择 RapidOCR 作为可选兜底，原因是它本地运行、无需把文档图像送到外部 OCR 服务；但 OCR 默认关闭，只有显式开启且页面确实缺少文本层时才运行。

表格抽取继续使用 pdfplumber，但只在 `PDF_EXTRACT_TABLES=true` 时启用。实现上必须对整份 PDF 只打开一次 pdfplumber 句柄，再按页复用，避免页数增加时把全文件解析重复做成 O(pages) 次。表格输出保持原有 Markdown 形态，保证默认关和旧表格输出都能被字符级回归测试锁住。

TXT 加载器保留多编码探测，但在 latin-1 fallback 之前先拒绝明显二进制 payload。latin-1 可以解码任意字节，如果没有二进制 guard，损坏文件会被当作“合法文本”入库，后续检索指标也很难暴露这个错误。

## LLM API：DeepSeek 兼容聊天接口

当前聊天层通过 `api_client.py` 封装 DeepSeek 兼容接口。这样做的原因是：

- DeepSeek chat API 与 OpenAI chat completions 风格接近，便于复用通用请求/响应结构。
- 项目可以先稳定 prompt、retrieval 和 citation 逻辑，不把聊天模型选择耦合到检索漏斗。
- 聊天与 embedding 客户端已迁移到 `httpx.AsyncClient`，支持 async chat 和 streaming delta，服务层可以暴露真实 SSE。

工程边界：

- 上游 401、429、timeout、5xx 必须映射为结构化错误，HTTP 层不能靠消息子串猜状态。
- 同步 CLI 通过 `run_async_blocking()` 复用 async 实现，避免维护两套 retry 逻辑。
- 流式接口必须从 provider delta 逐步 yield，禁止先完整生成再切片。

## Embedding：OpenAI `text-embedding-3` + Hash 测试 Provider

生产 embedding 当前选择 OpenAI `text-embedding-3` 系列，测试使用 `HashEmbeddingProvider`。

选择 OpenAI embedding 的原因：

- `text-embedding-3` 系列支持 `dimensions` 参数，适合 Matryoshka 裁剪到 512 或 256 维，降低向量库成本。
- API 稳定、生态成熟，便于先把面向企业的 RAG 工程边界跑通。
- 与 `TokenCounter` 配合后可以做 token-aware batching，避免中文被旧 `len/4` 估算低估而超限。

保留 Hash provider 的原因：

- 不联网、不耗费 API 额度。
- 维度和输出确定，适合 CI、baseline 和重构防回归。
- 检索、重排、融合、查询改写和缓存基线的“逐位复现”都依赖它做稳定尺子。

边界提醒：

- Hash provider 不是语义模型，不能拿它的线上召回表现做生产结论。
- OpenAI provider 的 batch 构造必须走 `TokenCounter`，而不是字符数近似。
- query embedding 可以缓存，但 corpus embedding 不应在 query 热路径重算。

## Vector Store：Memory + Qdrant

项目保留两类向量库：

- `InMemoryVectorStore`：用于本地单测、离线 baseline 和快速验证。
- `QdrantVectorStore`：用于生产级持久化、跨进程 ingest/query 分离和 metadata filter。

选择 Qdrant 的原因：

- 原生向量检索能力成熟，支持 collection、payload、filter 和 scroll。
- 索引构建与查询分离要求 ingest 与 query 跨进程存活，进程内 dict 不能冒充持久化。
- 支持通过 payload 重建 BM25 corpus 和 parent store 所需文本，避免 query 时重 embed 语料。

边界提醒：

- `VectorRecord.id` 是 dense、BM25、RRF、rerank 之间对齐的核心。
- embedding model/dimension 必须在 ingest 与 query 之间一致；不一致要 raise，不能静默检索。
- memory store 只能证明职责分离或单元行为，不能证明持久化。

## Retrieval：Dense + BM25 + RRF

当前检索漏斗同时支持 dense、BM25 和 RRF 融合。

为什么引入 BM25：

- dense 检索擅长语义相似，但 exact name、编号、内部术语和长尾 query 经常需要词法匹配。
- 小型 golden set 中，Hash embedding 的 dense 路语义弱，BM25 可以清晰暴露稀疏召回的价值。
- BM25 不依赖外部网络，适合作为确定性 baseline 的一部分。

为什么选择 RRF：

- RRF 只依赖 rank，不强行比较 dense score 与 BM25 score 的数值尺度。
- 路径权重可配置，适合在 Hash dense 噪声较大时降低 dense 权重。
- 同一个 `ReciprocalRankFusion` 被 hybrid retrieval 和 multi-query retrieval 复用，减少融合器分叉。

边界提醒：

- BM25 输出 id 必须等于 `VectorRecord.id`，否则 RRF 会静默错配。
- Hybrid retrieval 的主评估必须在 reranker 关闭下测，否则 reranker 满分会掩盖融合贡献。

## Rerank：Deterministic + Cohere

Rerank 层分为离线确定性和生产 neural 两类。

- `DeterministicReranker`：读原文、零网络，用于 CI 门禁和冻结指标。
- `CohereReranker`：作为 neural provider，走 mock 和 gated 集成。

这个拆分避免把外部模型波动引入核心质量门禁。Rerank 层的重点是证明插入点、失败降级和指标口径正确，而不是把 CI 绑定到网络模型。

## Query Rewrite：Deterministic Fixture + Chat Provider

Query rewrite 是检索侧增强，允许指标变化，但必须防 query drift。

当前选择：

- `DeterministicQueryRewriter`：读取冻结 JSONL fixture，证明多查询 RRF 管线在已知优质改写下能工作。
- `ChatQueryRewriter`：生产态可注入聊天模型，失败时降级到原始 query。

红线：

- 原始 query 永远是 `q0`。
- `QUERY_REWRITE_WEIGHT_ORIGINAL >= QUERY_REWRITE_WEIGHT_VARIANT`。
- negative fixture 零覆盖时不能声称 multi-query-on-negative 安全。

## Service：FastAPI + SSE

选择 FastAPI 的原因：

- Pydantic 请求/响应模型天然适合做服务契约。
- 自动 OpenAPI 文档可作为 API 入口门禁。
- async endpoint 可以配合 `asyncio.to_thread` 保护事件循环，同时逐步把网络 I/O 迁到 async。

SSE 用于 `/query/stream`：

- 先发 `sources`，前端可以先展示引用。
- 再逐 token/delta 输出生成过程。
- L3 cache 命中时做 replay，并标注 `cached=true` 与 `stream_replay=true`，不冒充 live generation。

## Cache：Redis 三层缓存

Redis cache layer 使用 L1/L2/L3 三层设计：

- L1：query embedding，key 不含 `corpus_version`。
- L2：retrieval sources，key 含 Redis 持久化 `corpus_version`。
- L3：完整 `RAGResponse`，key 含 `corpus_version` 和生成配置。

选择 Redis 的原因：

- 跨进程、跨副本共享，比进程内 dict 更接近生产部署。
- `INCR` 可以作为 corpus version 的持久原子计数器。
- TTL 和 fail-open 策略能控制陈旧风险与服务可用性。
- `redis.asyncio` client 由 cache store 绑定到稳定后台 loop，既保留 async Redis I/O，又兼容同步 CLI、离线评估和服务 threadpool 调用。

边界提醒：

- Redis 挂了要降级为 miss，不能让 `/query` 500。
- 向量序列化用二进制 float64，不用 JSON float。
- 默认单测用 FakeRedis；live Redis smoke 和跨事件循环命中测试需要 `REDIS_URL` 显式开启。
- `redis.asyncio` 客户端不能作为普通进程单例在多个临时事件循环之间复用，store 需要持有稳定 loop 或改用同步 Redis client。
- 汇报缓存有效性时必须同时给出命中计数、上游调用计数和 Redis error 计数；冷/热指标不漂只能说明 fail-open 安全，不能单独证明缓存命中。

## 为什么暂未引入的组件

- **Redis 队列 / 后台任务队列**：`/ingest` 当前同步 offload，后台队列留到后续运维卡。
- **OpenTelemetry 与分布式指标汇聚**：当前已有有界标签 registry 与 Prometheus 文本端点，但多 worker 聚合、trace exporter 和外部告警路由仍属于部署增强。

## 为什么 ACL/RBAC 放在检索前过滤层

- 权限判断必须发生在 candidate scoring 之前，避免无权文档占据 top-k 或泄露存在性。
- MySQL 负责 principal membership 与 document/chunk binding 的真相记录，Qdrant payload 承担高性能执行快照。
- 服务只信任上游认证网关写入的 principal header，并把 principal 作为显式参数贯穿 ACL 解析链；客户端请求体和 metadata filter 只能收窄，不能提权。
- ACL 不混入缓存策略本身，但有效 ACL filter 必须进入 retrieval/answer cache key，避免高权结果被低权用户复用。

## 为什么审计与指标使用独立 fail-open 旁路

- 审计和指标不是安全决策输入。sink、registry 或 alert hook 故障不能改变查询结果，因此运行时语义必须与 ACL 的 fail-closed 刻意相反。
- JSON-line 审计使用有界后台队列和滚动文件，把文件 I/O 移出请求热路径；队列满或写入失败只留下稳定告警码。
- 指标 registry 只提供项目所需的 counter/histogram 与 Prometheus 文本导出，不把 principal、request ID、query 或 source ID 放入 label，避免无界时序数量。
- token usage 优先读取 chat/embedding provider 的累计 reported 计数，并用进程级 watermark 原子认领增量；上游不回 usage 时才使用 TokenCounter 并标记 `estimated`。
- request ID 使用纯 ASGI 中间件写入响应头，并通过 ContextVar 注入现有 logger；`asyncio.to_thread` 会传播该上下文，同时避免通用 HTTP middleware 对 SSE 产生缓冲或时序干扰。

## 部署镜像与一键起栈

- 服务单独维护一份精简的打包依赖清单，只装 serving 真正 import 的包（评估栈、重依赖不进 serving 镜像），并把版本钉到确切值：无上界的 `>=` 会让依赖解析在构建时爆炸，删掉不 import 的重依赖 + 钉版本是镜像可复现构建的前提。
- 镜像用多阶段构建、以非 root 用户运行；运行期需要的资源（如分词编码表）在构建期就固化进镜像，保证可断网、可离线、冷启动确定，不在首次使用时去公网拉取。
- 一键起栈用 compose 编排向量库、缓存、元数据库和应用四件套，后端之间用服务名互联而非 localhost；每个后端声明原生 healthcheck，应用用 `depends_on: service_healthy` 等后端就绪再起；应用启动跑一次幂等初始化，重复起栈不叠库。
- 密钥只从环境注入，不进镜像、编排文件和提交；缺生成密钥时仍允许起栈跑入库与检索（用不需要密钥的测试 embedding），这样部署冒烟不必先充值，但要诚实声明“能起栈”不等于“零密钥问答”。

## 健康门禁：liveness 与 readiness 分开

- 存活探针（liveness）只表示进程还活着、恒返回 200，语义偏宽容——进程没死就不该因为后端抖动被重启；容器编排的健康检查打这个探针。
- 就绪探针（readiness）表示能不能接流量，必须 fail-closed：真去探当前配置下启用的后端，任一必需后端不通就返回 503 并点名，交给编排层（如 k8s readinessProbe）决定是否导流量。这和可观测性那一层的 fail-open 语义刻意相反。
- 探针并发执行、每个有独立超时上界（防一个后端卡死拖挂整个就绪检查）、按配置条件探测（没启用的不探）、探向量库只做只读连通查询避免建集合副作用、探缓存走其绑定的稳定事件循环；就绪响应体不含连接串/密码/密钥。
- 结构性配置在应用创建时就 fail-fast（非法配置早失败好过首个请求才炸）；但“需要付费密钥”的生成校验不接进启动，缺密钥降级为就绪里的非阻断依赖，避免打穿无密钥起栈能力。
- 验证 fail-closed 只认真断后端（逐个停掉看就绪翻 503、存活仍 200、恢复回 200），mock 一个“假装挂了”的后端不作数。

## 生成质量评估：Ragas 双轨判官

Ragas 用来回答确定性检索指标回答不了的问题：答案是否有上下文支撑、是否回答了问题，以及供给上下文是否精确和完整。当前使用 Faithfulness、Answer Relevance、Context Precision、Context Recall 四维。

选型约束：

- Ragas 是外部 LLM 判官，不是真值。live 轨非确定、付费且涉及数据出境，只允许显式 gated 周期运行；每次 push 的 CI 只回放已审核 verdict。
- `ragas==0.4.3` 与 LangChain pre-1.0 兼容族共同 pin，避免无上界依赖解析到已删除兼容模块的版本。
- DeepSeek 通过 OpenAI-compatible endpoint 承担结构化判官；Answer Relevance 的 embedding 默认走本地 fastembed `BAAI/bge-small-en-v1.5`（384 维、不引 torch、无 key、无配额、可完全离线），Gemini `gemini-embedding-001` 原生 `batchEmbedContents` 作为可选 provider。判官与 embedding 的凭据、模型身份和失败域彼此隔离；这条 embedding 只给离线 answer-relevance 门禁打分，产品检索用的是另一条 embedding。
- live capture 的答案生成温度单独固定为 0，不改产品默认温度 0.7；判官温度与生成温度分别进入 provenance。温度为 0 仍不等于确定性。
- 每条正样本重复五次，case 中心估计使用 median，判官噪声 margin 使用 MAD。mean/stddev 保留用于诊断，不能再让单个伪零决定门禁。
- Faithfulness 使用版本化 statement-generation prompt：独立 claim 继续拆分，共同构成单一要求的联合谓语保持为一条；依赖 `because` 理由才能判定的关系 claim 必须保留具体证据。针对单句 `is relevant because` 另加窄确定性边界保护，避免概率抽取剥掉 NLI 所需的问题语境，但不改 NLI verdict。
- Faithfulness 还需要验证引用归属，因此单独消费 `[n] filename: content`；Context Precision/Recall 保持 content-only，避免标签改变相关性判断。
- 只有消费生成答案的 Faithfulness、Answer Relevance 和 negative 行为进入 gate。Context Precision/Recall 不消费答案，在当前确定性 hash 检索 profile 下只作为 reported-only 指标；检索输入漂移由 context hash 更确定地拦截。
- Answer Relevance 使用 cosine，相似度可能因浮点尾差略超 1。两条 embedding 路都先做 L2 归一化，live capture 再以 `1e-6` 紧容差 clamp；非有限值和真实越界仍失败。replay 不接受容差，保证 fixture 永远严格 in-band。
- judge model、Ragas 版本、重复次数、生成 prompt、statement prompt、Faithfulness context format 与 gating scope 都是基线协议。任一变化都要求重新校准，不能当作透明升级。

reported-only 不等于删除或隐藏：Context Precision/Recall 仍输出全量分布，只是不再用一个与生成无关的 harness 常量阻断生成质量基线。稳健聚合和 prompt/context 修复也只提高评估装置的可信度，不会把错误答案“调成通过”。

把 answer-relevance 的默认 embedding 选成本地模型，关键判断是这条闸只服务离线英文评测、不参与产品检索：本地推理确定性更好、无 API 配额、门禁可完全离线，within-case 离散度也比远程 embedding 更收紧，对依赖 MAD margin 的门禁更友好。生成提示按问题形状校准到“答案 + 最小支撑短语”的中间量后，已冻结一份通过门禁的正式 verdict fixture（schema `ragas-verdicts-v5`，pin 生成与抽取 prompt 版本、judge model、embedding 和 gating scope），`python -m eval.ragas_run replay` 离线自证门禁通过。
