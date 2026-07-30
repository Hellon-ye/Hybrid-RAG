# llama.cpp RAG 调用链与架构决策

## 1. 参考版本

- llama.cpp 分支：hybrid-rag-migration
- 初始基线：c415d9e
- HybridRAG 参考版本：224c38b

## 2. llama-server 的两种运行模式

### 2.1 单模型模式

单个 llama-server 进程直接加载一个模型，并持有：

    llama_model
    llama_context
    server_context
    slot
    task queue

该模式适合普通 Completion、Embedding 或 Rerank 服务，但无法在一个请求中直接访问多个独立模型。

### 2.2 Router 模式

Router 主进程本身不加载推理模型。

每个模型由独立子进程加载，Router 根据请求 JSON 中的 model 字段选择对应子进程。

结构为：

    Router 主进程
    ├─ Expansion 模型子进程
    ├─ Embedding 模型子进程
    ├─ Reranker 模型子进程
    └─ Generation 模型子进程

Router 支持：

- 模型别名解析；
- 模型按需加载；
- 模型状态等待；
- 子进程端口管理；
- LRU 卸载；
- HTTP 请求代理。

## 3. Router 请求转发调用链

外部请求进入：

    tools/server/server.cpp

Completion、Embedding、Rerank 等路由在 Router 模式下被绑定到：

    server_models_routes::proxy_post

proxy_post 执行：

    解析请求 JSON
        -> 读取 model 字段
        -> router_validate_model()
        -> 解析模型别名
        -> 按需加载模型
        -> proxy_request()

router_validate_model() 负责：

- 检查 model 字段是否存在；
- 检查模型是否在 Router 配置中；
- 将别名转换为规范模型名；
- 根据 autoload 配置加载模型；
- 确认模型子进程处于运行状态。

ensure_model_ready() 负责：

- 检查模型当前状态；
- 未加载时调用 load()；
- 等待模型从 LOADING 进入 READY；
- 模型加载失败时抛出异常。

proxy_request() 负责：

- 获取模型子进程端口；
- 将请求路径、Header 和 Body 原样转发；
- 使用 CHILD_ADDR 访问子进程；
- 更新模型最后使用时间。

CHILD_ADDR 当前定义为：

    127.0.0.1

## 4. server_http_proxy 的定位

server_http_proxy 内部使用：

    httplib::ClientImpl
    pipe
    reader thread
    writer thread

它的用途是透明转发 HTTP 请求和响应，包括流式输出。

该抽象适合：

- 外部请求代理；
- Streaming Completion；
- 将子进程响应直接转交给原客户端。

它不适合作为 RAG 内部阶段调用的主要接口，因为 RAG Orchestrator 需要：

- 等待某个阶段完整完成；
- 获取完整 JSON；
- 解析 Embedding 向量或 Rerank 分数；
- 根据结果继续执行下一阶段。

## 5. 内部同步模型调用

server-models.cpp 已存在 Router 主进程直接使用 httplib::Client 请求模型子进程的实现，例如：

    httplib::Client cli(CHILD_ADDR, port);
    cli.Post(path, body, "application/json");

因此 RAG 第一版应新增内部同步接口：

    json server_models::request_model_json(
        const std::string & model,
        const std::string & path,
        const json & request_body);

该接口负责：

1. 解析并校验模型名称。
2. 调用 ensure_model_ready()。
3. 获取模型子进程端口。
4. 使用 httplib::Client 向子进程发送 POST。
5. 检查连接错误和 HTTP 状态码。
6. 解析并返回 JSON。
7. 更新模型最后使用时间。
8. 为错误添加模型名和请求路径信息。

第一版采用同步阻塞调用，以优先保证单请求 Sequential RAG 正确运行。

## 6. Embedding 调用链

HTTP 路由：

    /embedding
    /embeddings
    /v1/embeddings

请求处理入口：

    server_http_context::handle_embeddings_impl()

任务进入 server_context 后：

    tokenize
        -> 创建 llama_batch
        -> llama_set_embeddings()
        -> llama_decode()
        -> send_embedding()

结果提取：

    llama_get_embeddings_ith()
    llama_get_embeddings_seq()

RAG 映射：

    文档 Chunk
        -> Embedding 子进程 /v1/embeddings
        -> 文档向量

    Query 或子查询
        -> Embedding 子进程 /v1/embeddings
        -> Query 向量

## 7. Rerank 调用链

HTTP 路由：

    /rerank
    /reranking
    /v1/rerank
    /v1/reranking

Router 调用：

    server_http_context::post_rerank
        -> format_prompt_rerank()
        -> 创建 Rerank Task
        -> llama_decode()
        -> send_rerank()

Reranker Context 使用：

    LLAMA_POOLING_TYPE_RANK

结果通过：

    llama_get_embeddings_seq()
    llama_get_embeddings_ith()

提取 Rank Score。

RAG 映射：

    Query + Top-K Documents
        -> Reranker 子进程 /v1/rerank
        -> Top-N Documents

## 8. Generation 调用链

HTTP 路由：

    /completion
    /v1/completions
    /chat/completions
    /v1/chat/completions

主要流程：

    HTTP 请求
        -> Completion Task
        -> Slot 分配
        -> Tokenization
        -> llama_batch
        -> llama_decode() 执行 Prefill
        -> sampler 采样
        -> llama_decode() 逐 Token Decode
        -> HTTP Response

Query Expansion 与最终回答可以复用同一 Completion 调用链，但使用不同模型和 Prompt。

RAG 映射：

    原始 Query
        -> Expansion 模型 Completion
        -> 子查询列表

    Query + Top-N Context
        -> Generation 模型 Completion
        -> 最终 Answer

## 9. Hexagon Backend 调用链

Hexagon Backend 通过 ggml Backend Registry 注册：

    ggml/src/ggml-backend-reg.cpp
        -> register_backend(ggml_backend_hexagon_reg())

主要实现位于：

    ggml/src/ggml-hexagon/ggml-hexagon.cpp

关键职责：

- Hexagon Device 初始化；
- Session 和 Buffer 管理；
- 算子支持判断；
- Graph 优化；
- Graph Compute；
- HMX/HVX 路径选择；
- Host 与 Hexagon Buffer 复制。

上层 RAG Pipeline 不直接调用 Hexagon API。

Backend 路由通过 llama.cpp 模型与 Context 配置影响实际算子分配。

## 10. RAG 第一版架构

/v1/rag 只在 Router 模式下提供。

调用流程：

    POST /v1/rag
        -> RagRequest Parser
        -> RagOrchestrator
        -> request_model_json(expansion)
        -> request_model_json(embedding)
        -> FAISS Search
        -> request_model_json(reranker)
        -> Prompt Builder
        -> request_model_json(generation)
        -> RagResponse

RagOrchestrator 位于 Router 主进程，原因是：

- Router 能访问全部模型元数据；
- Router 能按需加载不同模型；
- 单个模型子进程只能访问自身模型；
- 检索和结果聚合不属于某个具体模型。

## 11. Prefill 和 Decode 的边界

Router 负责的是模型之间的编排：

    Expansion
    Embedding
    Reranker
    Generation

Router 不负责把同一次 Generation 的 Prefill 和 Decode 分发到不同子进程。

原因：

- KV Cache 位于 Generation 子进程的 llama_context；
- Prefill 和 Decode 必须共享同一逻辑序列状态；
- 跨进程传输 KV Cache 会显著增加实现复杂度。

因此异构 Generation 在 Generation 子进程内部完成：

    CPU Prefill + CPU Decode
    NPU Prefill + NPU Decode
    NPU Prefill + CPU Decode

其中 NPU Prefill 到 CPU Decode 需要单独实现 KV Bridge 或 Backend 切换能力。

## 12. 实现顺序

1. 在 server_models 中实现 request_model_json()。
2. 在 Router 中注册 /v1/rag。
3. 定义 RagRequest、RagResponse 和阶段指标。
4. 实现无 Query Expansion 的 CPU Sequential RAG。
5. 接入文档与 Query Embedding。
6. 接入 FAISS Top-K 检索。
7. 接入 Rerank Top-N。
8. 接入最终 Generation。
9. 增加 Query Expansion。
10. 增加 Backend 路由和 Prefill/Decode 拆分。