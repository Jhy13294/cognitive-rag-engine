# 学习笔记：企业级 RAG 知识库

## 当前阶段

当前项目已经从“文档入库基础能力”推进到“可评估的高级 RAG 检索漏斗”阶段。现在不仅能完成文档入库、embedding、向量检索和生成回答，还能用同一套 golden set 评估 dense 检索、rerank 检索和 hybrid dense+BM25 融合检索的差异。

目前已经完成：

- API 客户端基础封装
- 配置和日志模块
- TXT、Markdown、PDF、Word 加载器
- 文本清洗器
- 文本切分器
- 统一文档加载入口
- 样例测试集和文档切分验证
- Embedding 抽象层
- OpenAI 生产级 Embedding Provider
- 本地确定性 Hash Embedding Provider
- 向量库抽象层
- 内存向量库
- Qdrant 生产级向量库适配器
- 最小 RAG 检索链路
- RAG 应用命令行入口
- 检索评估 golden set 和 Hash baseline
- Reranker 抽象层
- 确定性离线重排器
- Cohere Rerank Provider 的 mock 单测和可选 gated 集成入口
- 共享 `lexical/` 词法核心
- BM25 稀疏检索器
- RRF 融合器
- reranker 关闭条件下的 dense-only / bm25-only / fused 三路 hybrid 对照评估

## 为什么先做文档加载

RAG 系统的质量很大程度取决于入库前处理。如果文档解析、清洗、切分做得粗糙，后面的 embedding、向量库和 rerank 都只能在低质量文本上补救。

一个企业级知识库至少需要保留这些元数据：

- `source`：来源文件
- `file_type`：文件类型
- `chunk_index`：当前分块序号
- `total_chunks`：总分块数
- `start_char`：分块在原文中的起始位置
- `end_char`：分块在原文中的结束位置

这些字段后面会用于引用来源、权限过滤、问题排查和召回质量分析。

## 代码语言规范

项目代码内部统一使用英文，原因是：

- 方便阅读第三方库和官方文档。
- 方便后续接入 CI、日志平台和监控工具。
- 方便多人协作时保持统一风格。
- 让业务学习笔记和工程实现分开管理。

中文内容不放在代码注释里，而是放在中文 README 和学习笔记中。

## 下一步学习重点

建议按这个顺序继续：

1. 学习 embedding 的作用：理解“文本转向量”不是聊天模型的工作。
2. 区分测试 Provider 和生产 Provider：OpenAIEmbeddingProvider 用于真实语义向量，HashEmbeddingProvider 只用于本地验证。
3. 学习向量数据库：先理解向量记录、相似度分数、top-k 和 metadata filter。
4. 增加召回质量评估：准备问题集、期望来源和命中率指标。
5. 拆分索引和查询：当前已有 Qdrant 适配器，下一步重点是避免每次提问都重复全量入库。

## Embedding 抽象层的意义

Embedding 抽象层把“文本转向量”的能力从 RAG 主流程里拆出来。这样后续无论使用本地模型、云端 API，还是企业内部 embedding 服务，都只需要实现同一套接口。

当前的 `HashEmbeddingProvider` 是一个确定性测试工具。它的作用是：

- 不联网也能验证管线。
- 每次相同文本得到相同向量。
- 可以测试向量维度、归一化、元数据和批量处理。
- 可以提前打通后续向量库接口。

它不是语义 embedding 模型，不能用于真实检索质量评估。

当前的 `OpenAIEmbeddingProvider` 是生产级语义向量 Provider。它支持：

- OpenAI `text-embedding-3-small` 和 `text-embedding-3-large`。
- `dimensions` 参数，用于 Matryoshka 维度裁剪。
- 按 chunk 数量和近似 token 数分批。
- 429、5xx、超时和网络抖动重试。
- usage token 和请求耗时日志。
- mock 测试，不消耗真实 API 额度。

## 向量库抽象层的意义

向量库抽象层负责存储和检索已经生成的向量。当前的 `InMemoryVectorStore` 用于本地验证，不适合生产长期存储。项目已经新增 `QdrantVectorStore`，用于连接 Qdrant 做持久化向量存储。

它能验证这些能力：

- 向量记录入库。
- 按余弦相似度检索 top-k。
- 返回内容、分数和元数据。
- 支持简单 metadata 过滤。
- 验证向量维度一致性。
- 打通“文档加载 -> 切分 -> embedding -> 入库 -> 检索”的本地链路。

Qdrant 适配器保持同一套 `VectorStore` 接口，因此 `RAGPipeline` 不需要感知底层是内存库还是 Qdrant。

当前 Qdrant 适配器重点能力：

- collection 创建和维度校验。
- source、file_type、acl、chunk_index 的 payload index。
- 批量 upsert，避免逐条网络写入。
- metadata filter 翻译为 Qdrant 检索前过滤。
- 稳定业务 id 到 Qdrant UUID 的确定性映射。
- 429、5xx、超时和网络抖动重试。
- mock 单元测试和可选本地 Docker 集成测试。

## 检索评估基线

T04 的重点不是评估回答质量，而是先把“检索有没有找对资料”变成可复现数字。

当前评估基线遵守三条规则：

- golden set 使用 `relevant` 列表，而不是单个 expected source，这样才能计算 recall@k。
- 评估器只接收 `retrieve(question, top_k)` 可调用对象，不依赖完整 pipeline，也不调用聊天模型。
- 默认使用 `HashEmbeddingProvider`，确保离线、确定性、可在 CI 中复跑。

当前报告输出：

- hit_rate@3/5/10。
- MRR@3/5/10。
- recall@3/5/10。
- negative 样本的空召回率和误召回率。
- 按 capability 切片的指标。
- 每条低召回 case 的期望来源、实际 top-k、首个命中 rank 和 miss reason。

## Reranker 抽象层

T05 的核心不是“多接一个模型”，而是在 dense 召回之后增加一个可验证的重排阶段：

1. 先用 embedding 和向量库召回较大的候选池，例如 `fetch_k=30`。
2. reranker 读取 query 和候选 chunk 原文，重新计算相关性分数。
3. 按重排分取最终 top-k 进入上下文组装。

当前实现了两条路径：

- `DeterministicReranker`：离线、零网络、零模型下载，基于词项重叠和轻量 IDF 打分。它用于 CI 门禁和指标回归。
- `CohereReranker`：生产 neural rerank provider，网络调用通过 mock 单测覆盖，真实集成测试由环境变量 gated。

重要约束：

- `reranker=None` 时，`RAGPipeline.retrieve` 必须保持原 dense 路径行为不变。
- rerank 抛错、超时或服务不可用时，必须降级回 dense 原序，不能中断问答。
- T05 的 KPI 是 MRR@3 和 long_tail MRR，不是 recall@5；因为当前 golden set 的 recall@5 在 dense 基线中已经是 `1.0`。

当前 T05 离线评估结果：

- dense 基线 MRR@3：`0.677083`。
- deterministic rerank MRR@3：`1.000000`。
- dense long_tail MRR@3：`0.566667`。
- deterministic rerank long_tail MRR@3：`1.000000`。
- recall@5 保持 `1.000000`。
- negative_false_recall_rate 没有恶化，仍为当前 dense/rerank 都会返回非空结果的已知问题。

## Hybrid 检索与 RRF 融合

T06 的核心不是继续排序已有候选，而是解决“相关文档没有进入候选池”的召回问题。因此它和 T05 的职责不同：

- T05 `DeterministicReranker` 是排序器，只处理 dense 已经召回的候选。
- T06 `BM25Retriever` 是召回器，对全量 corpus 做稀疏检索。
- T06 `ReciprocalRankFusion` 是融合器，只根据各路 rank 融合，不依赖不同检索器的绝对分数。

本次把 T05 里已有的 token overlap、IDF 等词法 primitive 提炼到顶层 `lexical/` 包中。这样 T05 和 T06 复用同一套 tokenizer、normalization、IDF 和 BM25/lexical scoring 基础能力，但不共享业务入口类，避免 reranker 和 retriever 互相耦合。

T06 最重要的工程红线是 id 对齐：BM25 输出的 id 必须等于 dense 路里的 `VectorRecord.id`。RRF 是按 id 合并 rank 的，如果 dense 和 sparse 两路对同一个 chunk 产生不同 id，融合结果表面看起来正常，实际是在把两个不同候选当成两条记录，属于很隐蔽的召回质量 bug。

当前实现中，`eval.baseline.build_hash_retriever()` 先加载并切分一次文档，然后用同一批 embedded documents 构造 `VectorRecord` 列表；dense 向量库和 BM25Retriever 都使用这份相同 `VectorRecord` corpus。因此 dense 与 sparse 的 id 空间天然一致。

T06 的评估必须关闭 reranker。原因是当前 T05 确定性 reranker 已经把小型 golden set 的 MRR@3 顶到 `1.000000`，如果把 hybrid 放到 rerank 后面一起测，reranker 会掩盖 BM25/RRF 本身的贡献。正确方式是三路并报：

- dense-only：原始向量检索。
- bm25-only：纯稀疏检索。
- fused：dense 与 BM25 经过 RRF 融合。

当前 T06 离线评估结果：

- dense-only MRR@3：`0.677083`。
- bm25-only MRR@3：`1.000000`。
- fused MRR@3：`1.000000`。
- dense long_tail MRR@3：`0.566667`。
- fused long_tail MRR@3：`1.000000`。
- fused exact_name MRR@3：`1.000000`。

需要诚实理解这个结果：当前 dense 路使用 `HashEmbeddingProvider`，它是确定性测试 embedding，不是真实语义 embedding。因此 BM25 在这个小语料中非常强，fusion 的主要价值是证明架构、id 对齐、RRF 和评估口径都可靠；真正的生产收益需要后续用真实 embedding provider 和更大 golden set 复测。

RRF 当前默认配置为 `k=60`、dense weight `0.2`、sparse weight `1.0`。这样做是因为 hash dense 路语义噪声较大，如果等权融合，dense 噪声可能拖累 BM25；下调 dense 权重后，fusion 可以稳定不低于 bm25-only，并保留未来接入真实语义 embedding 时的双路融合空间。

## 最小 RAG 管线

当前的最小 RAG 管线包括四步：

1. 对用户问题生成 query embedding。
2. 用 query embedding 在向量库中检索 top-k chunk。
3. 将检索结果组装成带编号的上下文。
4. 调用聊天模型生成答案，并返回引用来源。

这个阶段的重点是链路闭环。现在已经可以使用 `OpenAIEmbeddingProvider` 生成真实语义向量；真正评估 RAG 效果时，还需要准备问题集和期望来源。

## RAG CLI 的作用

`rag_cli.py` 把已有模块串成一个可操作入口：

1. 指定文档或目录。
2. 自动加载、清洗、切分。
3. 使用本地测试 embedding provider 生成向量。
4. 写入内存向量库或 Qdrant 向量库。
5. 接收问题并调用 RAGPipeline。
6. 输出答案和引用来源。

它适合本地演示和端到端验证。当前 CLI 默认可通过 `EMBEDDING_PROVIDER=openai` 使用 OpenAI 生产级 embedding，并可通过 `VECTOR_STORE_PROVIDER=qdrant` 或 `--vector-store qdrant` 使用 Qdrant。

## 当前风险

当前项目已经有了最小测试体系、统一加载入口、OpenAI embedding provider、内存向量库、Qdrant 向量库适配器、最小 RAG 管线、RAG CLI、检索评估基线、rerank 漏斗和 hybrid 检索融合。下一步建议在两个方向中选择一个：工程化方向先回补索引构建/查询职责分离；检索质量方向继续做 parent-child 分块和上下文装填修复。同时继续补更多边界样例，例如扫描版 PDF、超长 Markdown、空文档、乱码文本、多表格 Word、以及带权限元数据的企业文档。
