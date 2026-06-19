# 核心踩坑记录：排查与解决

本文记录项目推进中的关键事故、隐性 bug 和验收口径修正。目标是让后来者知道哪些“看起来能跑”的实现其实会污染指标、缓存或服务行为。

## 1. Parent Expansion Inflated Retrieval Metrics

**现象**

Parent-child 展开后，如果把展开后的 sources 直接喂给同一个评估器，MRR/recall 会上升。

**根因**

兄弟子块折叠成同一个父块后，top-k 窗口被压缩，低 rank 的异源相关文档可能被顶进 @k。评估器看到的是“折叠后的列表”，不是原始 ranked child list。

**修复**

- `eval.run --parent-child` 默认只评子块 ranked list。
- `--expand-parents` 只能做诊断，不用于声明检索提升。
- 后续应在评估器中对 expanded sources 打 warning 或拒绝评分。

**经验**

Parent expansion 是生成侧内容替换，不是检索排序能力。任何“父块展开拉高 MRR”的说法都应先视为测量口径错误。

## 2. Context Packing Must Not Leak Into Retrieval Evaluation

**现象**

引入整块装填、去重、引用重排后，如果把这些动作放进 `retrieve()`，MRR/recall 会跟着变化。

**根因**

Context packing 会跳块、去重和重排引用编号。这些行为属于最终 prompt 装配，不属于检索 ranked list。

**修复**

- `ContextPacker` 只在 `build_prompt` / `answer` 路径运行。
- `retrieve()` 返回的 sources 不做 context dedup。
- `RAGResponse.sources` 表示进入 prompt 的 used_sources，不能拿来替代检索评估 ranked list。

**经验**

Context packing 的收益是结构正确：不句中截断、不重复占预算、引用连号、token 预算准确。生成质量要留给 Ragas 这类答案质量评估。

## 3. Clause Separators Can Still Cut Inside Sentences

**现象**

早期 ContextPacker 的超预算单块截断使用了 TextSplitter 的全部 separators，包括逗号、分号和空格。测试只断言“不以半词结尾”，但实际可能截成 `...here. Gamma`，停在词边界而不是句末。

**根因**

“separator boundary”不等于“sentence boundary”。选择最靠后的分隔符会让空格或逗号盖过更早的句号。

**修复**

- 截断分隔符收窄到段落/句终边界。
- 测试断言升级为内容以句终标点或段落边界结束。
- metadata 标记为 `sentence_boundary`，避免宽泛表述。

**经验**

测试不能只证明“比旧实现好”，还要证明满足交底字面约束。

## 4. BM25 ID Mismatch Silently Breaks RRF

**现象**

Dense 和 BM25 各自返回结果都正常，但 fused 结果不稳定或没有预期收益。

**根因**

RRF 只按 id 合并 rank。如果 BM25 对同一 chunk 生成了不同 id，RRF 会把同一内容当成两条记录。

**修复**

- BM25Retriever 使用同一份 `VectorRecord` corpus。
- BM25 输出的 `RankedRecord.id` 必须等于 `VectorRecord.id`。
- eval 报告保留 source metadata id，方便抽样核对。

**经验**

多路召回的第一门槛不是分数调参，而是候选 id 空间一致。

## 5. Negative Query Bypass Is Not Multi-Query Safety

**现象**

Deterministic query rewrite fixture 中 negative 样本 top-k 与单路逐字节一致，容易被写成“negative 不退化”。

**根因**

4 条 negative query 根本没有 fixture rewrite，`DeterministicQueryRewriter` 返回 N=1，multi-query 整段旁路。它没有测试“negative 触发多查询后是否安全”。

**修复**

- README 改为“negative fixture 零覆盖，风险未在线下覆盖”。
- 离线涨幅表述从“生产下界”改为“机制受控演示”。
- 后续交给 threshold/abstain 门禁。

**经验**

没有触发的功能不能当作安全证明。尤其是 negative capability，不能被 overall 指标掩盖。

## 6. `async def` Does Not Guarantee Real Async Behavior

**现象**

服务端点写成 `async def` 后，仍可能因为内部同步检索、同步 chat 或 `time.sleep` 阻塞事件循环。

**根因**

Python 的 `async def` 只是函数形态。如果函数体内直接执行阻塞 I/O 或 CPU-heavy 同步流程，事件循环仍会被冻住。

**修复**

- 网络 I/O 迁到 `httpx.AsyncClient`。
- retry backoff 使用 `await asyncio.sleep`。
- 既有同步检索漏斗用 `asyncio.to_thread` offload。
- 用墙钟测试多个慢请求并发，总耗时应接近单请求。

**经验**

验收要看并发墙钟，不要只看函数签名。

## 7. Fake SSE Streaming

**现象**

一种常见错误是先调用 `answer()` 完整生成，再把答案切成 token 逐段 yield。前端看起来有 token event，但首 token 延迟等于完整答案延迟。

**根因**

流式接口没有消费 provider 的真实 streaming delta。

**修复**

- `/query/stream` 先发 `sources` event。
- 然后调用 `chat_client.stream_chat()` 逐 delta yield。
- 最后发 `done`，并断言 token 拼接等于完整答案。

**经验**

流式验收要看首 token 时间。`httpx.ASGITransport` 可能缓冲响应，时序测试需要直驱 generator 或真 uvicorn。

## 8. String-Based Exception Mapping Is Fragile

**现象**

服务层早期通过 `str(error)` 判断空库、维度不匹配、timeout 等错误，当前文案能命中，但未来改文案会静默退化。

**根因**

错误类型和错误消息混在一起。

**修复**

- 新增 typed RAG exceptions。
- 上游 API/embedding 错误带 `error_kind`。
- HTTP 层按类型和结构化 kind 映射状态码。
- 响应体脱敏，不泄露 key、header、traceback。

**经验**

异常映射是服务契约，不是日志字符串搜索。

## 9. Retry Should Not Recreate HTTP Clients Per Attempt

**现象**

功能上 retry 能成功，但每次 attempt 都创建新的 `httpx.AsyncClient`，连接池无法复用。

**根因**

client 生命周期放在 retry loop 内部。

**修复**

- 一次逻辑请求创建一个 async client。
- 多次 attempt 复用同一 client。
- 单测用 fake client create count 断言重试后仍只创建一次。

**经验**

生产优化不一定改变功能行为，但会明显影响连接管理、延迟和资源使用。

## 10. Redis Corpus Version Must Not Be In-Process State

**现象**

服务进程内的 `cache_generation` 可以让当前进程失效 pipeline cache，但 Redis 是跨进程缓存。服务重启后 generation 归零，旧 L2/L3 key 可能重新可达。

**根因**

进程内变量不能作为跨进程缓存版本。

**修复**

- `corpus_version` 存在 Redis 中。
- `/ingest` 成功后调用 Redis `INCR`。
- L2/L3 key 包含 Redis corpus version。
- `ServiceState.invalidate()` 同时清进程内 pipeline cache 和 bump Redis version。

**经验**

缓存版本必须和缓存存储拥有同样的生命周期。

## 11. Embedding Vectors Should Not Be Cached As JSON Floats

**现象**

用 JSON 存 embedding 向量看起来方便，但浮点文本化可能造成细微漂移，进而影响排序稳定性。

**根因**

相似度排序对浮点差异敏感，尤其是在分数接近时。

**修复**

- Redis 中使用二进制 float64 序列化。
- payload 带长度 header。
- 单测覆盖极大值、最小非规格化数、`-0.0`、π 和高精度小数。

**经验**

“近似相等”不够。缓存层必须保证 hit 路径和 cold 路径逐 float 一致。

## 12. FakeRedis Cannot Prove Live Redis Integration

**现象**

FakeRedis 单测可以证明装饰器、key 和 fail-open 逻辑，但不能证明 `.venv` 已安装 `redis` 包，也不能证明真实 `redis.asyncio.from_url`、TTL、INCR 和 TCP 连接可用。

**修复**

- `requirements.txt` 加 `redis>=5.0.0`。
- `.venv` 安装 `redis==8.0.0`。
- 新增 `REDIS_URL` gated live smoke，用于证明真实 `PING`、`INCR`、TTL、SET/GET 和二进制 payload 往返。
- 进一步新增跨事件循环 live regression，用于证明生产调用形态下第二次真的命中。

**经验**

报告时必须区分“替身测试通过”“单 loop live smoke 通过”和“跨事件循环生产形态通过”。这和 Qdrant gated 集成测试是同一类诚实边界。

## 13. redis.asyncio Clients Must Not Cross Event Loops

**现象**

真实 Redis 环境中，缓存看起来开启了，但 L1/L2/L3 几乎不命中。日志里能看到 `Event loop is closed` 和 `got Future attached to a different loop`，cache store fail-open 后退回冷路径，所以检索指标不漂但缓存价值消失。

**根因**

同步入口会通过阻塞桥多次创建并关闭事件循环。如果 `redis.asyncio` client 是进程单例，它会绑定第一次使用的 loop。第二次同步调用或服务线程池调用再复用这个 client 时，就会跨 loop 访问已关闭连接。

**修复**

- Redis store 创建并持有专属后台事件循环线程。
- `redis.asyncio.from_url()` 在该 store loop 内懒创建 client。
- 所有 Redis public methods 都调度到 store loop 执行。
- 增加真实 Redis 跨事件循环回归：同 query 第二次必须命中，Redis error 计数必须保持 0。

**验收口径**

- 同一文本第二次 query embedding 命中，provider 调用数不再增加。
- 同一检索第二次命中 L2，`similarity_search` 调用数不再增加。
- 同一问答第二次命中 L3，chat 调用数不再增加。
- 真 Redis eval 冷跑写入、热跑命中，`errors==0`。
- 指向不可用 Redis 时 fail-open 仍返回答案，且不会阻塞服务。

**经验**

异步客户端的生命周期必须和事件循环生命周期一致。FakeRedis 和单 loop live smoke 都抓不住这个问题，必须复刻生产调用形态。

## 14. Configuration Validators Must Run On Startup

**现象**

`Config.validate_cache()` 已定义 TTL/URL 守卫，但如果全仓无调用点，就是死代码。非法配置不会在启动时失败。

**修复**

- `create_app()` 调用 `Config.validate_cache()`。
- 服务测试覆盖非法 TTL 早失败。

**经验**

配置校验必须接入启动路径。否则它只是漂亮摆设。

## 后续仍需重点盯住

- negative query 的 threshold/abstain。
- ACL/RBAC 必须做检索前过滤。
- context packing 的生成质量收益要用 Ragas 测。
- Redis corpus version 当前是全局单键，安全但粗，会过度失效；后续可按 collection 细化。
- `/query/stream` L3 回填的 `raw_response` 是合成形，answer/sources 正确，但 raw 溯源语义不同。
