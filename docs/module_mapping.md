# HybridRAG 到 llama.cpp 模块映射

## 1. 参考版本

- HybridRAG：upstream/main@224c38b
- 目标 llama.cpp：hybrid-rag-migration，基线 commit c415d9e

主要依据：

- app/server/rag_pipeline.hpp
- HybridRAG 项目交接文档

## 2. HybridRAG 的主要执行入口

### 2.1 run_rag_sequential()

源码位置：

    app/server/rag_pipeline.hpp:689

职责：

- 按固定顺序执行完整 RAG 流程。
- 依次完成文档切分、Embedding、Query Expansion、检索、Reranking 和 Generation。
- 作为最基础的功能正确性基线。

迁移策略：

- 在 llama.cpp 中优先实现 CPU Sequential Pipeline。
- 第一版不依赖 Scheduler。
- 作为后续 NPU 和异构路径的功能基线。

### 2.2 run_rag_compute_carrier_baseline()

源码位置：

    app/server/rag_pipeline.hpp:943

职责：

- 按计算载体基线方式组织 RAG。
- 保留 Prefill、Decode 路由和相关指标。
- 用于比较不同计算载体。

迁移策略：

- 第一版不单独迁移成另一套 Pipeline。
- 将其路由和指标能力合并到统一的 Backend Router 和 Runtime 中。

### 2.3 run_rag_hetero_parallel()

源码位置：

    app/server/rag_pipeline.hpp:1390

职责：

- 构造异构 RAG DAG。
- 为节点指定 CPU 或 NPU 路由。
- 将 Generation 拆成 Prefill、Decode 和 Merge。
- 记录 Queue Wait、Execution Time 和 KV 状态。

主要 DAG 节点：

    indexing
    query_expand
    query_embedding_i
    searching_i
    reranking
    generation_prefill_i
    generation_decode_i
    generation_merge

依赖关系：

    query_expand
        -> query_embedding_i

    indexing + query_embedding_i
        -> searching_i

    indexing + all searching_i
        -> reranking

    reranking
        -> generation_prefill_i
        -> generation_decode_i

    all generation_decode_i
        -> generation_merge

迁移策略：

- CPU Sequential 跑通后再迁移异构 DAG。
- 第一版使用静态 DAG、固定路由和 FIFO。
- 暂不迁移 Critical-score、复杂任务窃取和激进并发策略。

## 3. 请求、响应和指标结构

### 3.1 RagRequest

主要字段：

    doc
    query
    mode
    enable_query_expansion

    generation_model
    embedding_model
    rerank_model
    expansion_model

    top_k
    top_n
    max_tokens
    generation_decode_steps
    generation_subquery_decode_steps
    generation_candidate_repeats

    generation_prefill_backend
    generation_decode_backend
    temperature

迁移策略：

- 在 llama.cpp Server 中定义对应的 RAG 请求结构。
- 第一版尽量保留现有字段名称，减少测试脚本修改。
- 模型逻辑名称和实际模型路径分离。
- Backend 字段统一使用 cpu、npu、auto。

### 3.2 RagStageMetrics

记录：

    indexing_ms
    query_expand_ms
    query_embedding_ms
    embedding_ms
    searching_ms
    reranking_ms
    generation_ms
    total_ms

迁移策略：

- Pipeline 层记录各 RAG 阶段总耗时。
- Runtime 层记录具体模型调用、Prefill、Decode 和 KV Bridge 耗时。

### 3.3 GenerationSubMetrics

记录：

    prefill_ms
    decode_ms
    prefill_sum_ms
    decode_sum_ms
    bridge_ms
    kv_snapshot_ms
    kv_restore_ms
    kv_snapshot_bytes

第一版至少实现：

    prefill_ms
    decode_ms
    bridge_ms

Snapshot、Restore 和 Base+Delta 在多 Candidate 阶段再处理。

### 3.4 RagResponse

主要输出：

    answer
    mode_requested
    mode_used
    query_used
    sub_queries

    context_chunks
    top_k_indices
    top_n_indices

    generation_prefill_backend_target
    generation_decode_backend_target
    generation_kv_bridge_available
    generation_route_note

    decode_task_summaries
    generation_sub_metrics
    metrics

迁移策略：

- /v1/rag 的响应结构尽量兼容 HybridRAG。
- 明确区分请求 Backend、目标 Backend、实际 Backend 和 fallback。
- fallback 时返回明确原因，不能静默回退。

## 4. Backend 路由模块

### 4.1 HybridRAG 主要函数

    normalize_backend_target()
    is_npu_available_in_binary()
    resolve_prefill_backend_target()
    resolve_decode_backend_target()
    is_kv_bridge_available_for_route()
    plan_generation_route()

### 4.2 当前路由语义

Prefill 使用 auto 时：

- NPU 可用则选择 NPU。
- NPU 不可用则选择 CPU。

Decode 使用 auto 时：

- 默认选择 CPU。

请求 NPU 但当前二进制不支持时：

- 回退到 auto。
- 在 route_note 中记录原因。

KV Bridge 可用条件：

- Prefill Backend 为 NPU。
- Decode Backend 为 CPU。
- 当前模型支持进程内 KV Bridge。

当前 PowerServe 中 Qwen3 的 KV Bridge 依赖：

    sync_qnn_kv_to_cpu()

### 4.3 llama.cpp 迁移方式

计划新增：

    BackendTarget
    GenerationRoutePlan
    BackendRouter

映射关系：

    PowerServe QNN/GGML 路由
        -> llama.cpp CPU/Hexagon Backend 路由

    PowerServe route_note
        -> llama.cpp fallback 和 debug 信息

    PowerServe Qwen3 KV Bridge 能力判断
        -> llama.cpp KV 导出和恢复能力判断

上层 RagPipeline 不直接操作 Backend 对象。

## 5. 文档切分和检索模块

### 5.1 HybridRAG 主要函数

    rag_trim()
    rag_first_line()
    rag_replace_fullwidth_semicolon()
    rag_split_sub_queries()
    rag_split_document()
    rag_search_faiss_ip()
    rag_merge_subquery_hits()

### 5.2 数据流程

    原始文档
        -> 文档切块
        -> 文档 Embedding

    Query Expansion 输出
        -> 解析子查询

    Query Embedding + Document Embedding
        -> FAISS Inner Product Search
        -> 每个查询 Top-K
        -> 多查询结果合并和去重

### 5.3 llama.cpp 迁移方式

计划新增：

    DocumentChunker
    QueryExpansionParser
    FaissRetriever
    SearchResultMerger

处理原则：

- 文档切分逻辑可以迁移 HybridRAG 现有实现。
- Query Expansion 输出解析可以迁移现有实现。
- FAISS 检索逻辑可以复用。
- Embedding 计算改为调用 llama.cpp Runtime。

## 6. 模型调用适配模块

### 6.1 HybridRAG 主要函数

    make_generation_input()
    apply_generation_route_to_input()
    make_embedding_input()
    make_rerank_input()
    build_generation_prompt()

### 6.2 迁移原则

PowerServe 的 ModelInput 不直接迁移。

计划定义框架无关接口：

    RagRuntime.embed()
    RagRuntime.rerank()
    RagRuntime.generate()
    RagRuntime.prefill()
    RagRuntime.decode()

映射关系：

    PowerServe embedding()
        -> llama.cpp Embedding Adapter

    PowerServe rerank()
        -> llama.cpp Rank Pooling / Rerank Adapter

    PowerServe blocking_inference() / generate()
        -> llama.cpp Generation Adapter

Prompt 构造属于 RAG Pipeline，不放入 llama.cpp 模型底层。

## 7. Generation 和 Candidate 模块

### 7.1 HybridRAG 主要函数

    build_generation_prompt()
    build_generation_segments()
    build_generation_decode_tasks()
    merge_generation_candidates_v1()

### 7.2 数据流程

    Query + Top-N Context
        -> Generation Prompt

    Generation Prompt
        -> 公共段和 Candidate 段

    Candidate
        -> Prefill Task
        -> Decode Task

    多个 Candidate
        -> Merge
        -> 最终 Answer

### 7.3 第一版迁移范围

第一版仅支持：

    单请求
    单 Candidate
    单次 Generation

暂不迁移：

    复杂 Candidate Merge
    多 Candidate Base + Delta KV
    generation_candidate_repeats > 1

接口中预留：

    PrefillHandle
    GenerationCandidate
    GenerationResult

## 8. Scheduler 模块

HybridRAG 当前包含：

    src/scheduler2/
    BackendRouter
    KVCacheManager
    CPU Worker
    NPU Worker
    Task DAG

llama.cpp 第一版计划实现：

    RagScheduler
    StaticDagTask
    CPU Worker
    NPU Worker
    Fixed Backend Route
    FIFO Ready Queue
    Failure Propagation

迁移顺序：

    CPU Sequential
        -> NPU Sequential
        -> Prefill/Decode Split
        -> NPU to CPU KV Bridge
        -> Static DAG Scheduler

## 9. 最终目标架构

    HTTP /v1/rag
        |
        v
    RagRequest Parser
        |
        v
    RagPipeline
        |- DocumentChunker
        |- QueryExpansionParser
        |- FaissRetriever
        |- PromptBuilder
        `- StageMetrics
        |
        v
    RagRuntime Interface
        |- embed()
        |- rerank()
        |- generate()
        |- prefill()
        `- decode()
        |
        v
    LlamaRagRuntime
        |- ModelManager
        |- EmbeddingAdapter
        |- RerankAdapter
        |- GenerationAdapter
        |- BackendRouter
        `- KVBridge
        |
        v
    llama.cpp
        |- llama_model
        |- llama_context
        |- tokenizer
        |- llama_batch
        |- sampler
        |- KV Cache
        |- CPU Backend
        `- Hexagon Backend

## 10. 不直接迁移的 PowerServe 内容

    PowerServe Model 基类
    PowerServe Qwen3 Forward 实现
    PowerServe Graph
    PowerServe Executor
    PowerServe GGML/QNN Backend 本体
    PowerServe ModelInput
    PowerServe TokenIterator

这些功能由 llama.cpp 自身实现替代。

## 11. 当前迁移结论

需要迁移或重写：

    RagRequest 和 RagResponse
    RAG Pipeline
    文档切分
    Query Expansion 解析
    FAISS 检索
    Reranking 调用组织
    Prompt 构造
    Backend 路由语义
    Generation Prefill/Decode 编排
    KV Bridge 控制逻辑
    最小 DAG Scheduler
    阶段指标和调试信息

优先复用 llama.cpp：

    模型加载
    Tokenizer
    Embedding 模型执行
    Reranker 模型执行
    Generation 模型执行
    Sampler
    llama_model
    llama_context
    llama_batch
    KV Cache 基础设施
    CPU Backend
    Hexagon Backend

第一版实现顺序：

1. 定义公共数据结构和 RagRuntime 接口。
2. 定位 llama.cpp 的 Embedding、Reranker 和 Generation 调用链。
3. 实现 CPU Runtime Adapter。
4. 实现 CPU Sequential RAG。
5. 增加 CPU/NPU Backend Router。
6. 显式拆分 Prefill 和 Decode。
7. 实现 NPU Prefill 到 CPU Decode 的 KV Bridge。
8. 接入最小静态 DAG Scheduler。
