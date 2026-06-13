# 学习笔记：企业级 RAG 知识库

## 当前阶段

当前项目处在“文档入库基础能力”阶段。这个阶段的目标不是马上接向量库，而是先让文档可以稳定进入系统，并形成统一的 `Document` 数据结构。

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

当前项目已经有了最小测试体系、统一加载入口、OpenAI embedding provider、内存向量库、Qdrant 向量库适配器、最小 RAG 管线和 RAG CLI，但测试样例还比较基础。下一步建议拆分索引构建和查询命令，并建立检索评估数据集；同时继续补更多边界样例，例如扫描版 PDF、超长 Markdown、空文档、乱码文本、多表格 Word、以及带权限元数据的企业文档。
