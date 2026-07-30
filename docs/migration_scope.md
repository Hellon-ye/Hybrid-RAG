# HybridRAG 到 llama.cpp 迁移范围

## 1. 源项目基线

- 项目：HybridRAG
- 参考目录：`/data/junle/code/HybridRAG-reference`
- 上游分支：`upstream/main`
- Commit：`224c38b`
- 说明：该版本已包含 `new-scheduler` 分支的合并结果。

## 2. 目标项目基线

- 项目：llama.cpp
- 开发目录：`/data/junle/code/llama.cpp-rag`
- 开发分支：`hybrid-rag-migration`
- 基线 Commit：`c415d9e`
- 基线说明：包含 Snapdragon/Hexagon KV-cache matmul 的 HVX 路由修改。
- 不包含原实验工作区中尚未提交的 `semantic_align` 修改。

## 3. 迁移目标

在 llama.cpp 架构中实现一版可编译、可部署、可运行的异构 RAG 系统，复用 llama.cpp 已有的模型加载、Tokenizer、Embedding、Reranker、Generation、Sampler 和 Backend 能力。

目标 RAG 流程：

1. Document Chunking / Indexing
2. Document Embedding
3. Query Expansion
4. Query Embedding
5. FAISS Searching
6. Reranking
7. Prompt Construction
8. Generation

其中 Generation 后续需要支持：

- CPU Prefill + CPU Decode
- NPU Prefill + NPU Decode
- NPU Prefill + CPU Decode

## 4. 本次迁移需要完成的内容

- 新增 `/v1/rag` 请求入口。
- 迁移或重写 HybridRAG 六阶段 RAG Pipeline。
- 封装 llama.cpp 的 Embedding 接口。
- 封装 llama.cpp 的 Reranker 接口。
- 封装 llama.cpp 的 Generation 接口。
- 建立模型与 Context 生命周期管理。
- 建立 CPU/NPU Backend 路由及 fallback 记录。
- 显式拆分 Generation Prefill 与 Decode。
- 调查并实现 NPU KV Cache 到 CPU KV Cache 的交接。
- 接入最小可用 DAG Scheduler。
- 返回各阶段耗时、实际 Backend 和 fallback 信息。
- 提供编译、部署、启动和测试说明。

## 5. 第一版允许简化的内容

- 只支持单请求和单 Generation Candidate。
- Query Expansion 可以配置关闭。
- Scheduler 先采用静态 DAG、固定路由和 FIFO。
- KV Bridge 先保证正确性，不要求性能最优。
- 不要求第一版实现真正的 Prefill/Decode 并行重叠。
- 不要求复现原 HybridRAG 的历史最优性能。

## 6. 暂不处理的内容

- OpenCL Backend 合并。
- Critical-score 在线更新。
- 复杂任务窃取。
- 多 Candidate 的 Base + Delta KV 优化。
- 高并发多请求调度。
- NPU Embedding 混合路径的历史数值问题。
- Profiler 性能优化。
- 原工作区中未提交的 `semantic_align` 实验修改。

## 7. 最低验收条件

- 从干净工作区可以完成编译。
- 可以在目标 Android 设备上启动 Server。
- `/v1/rag` 可以返回有效答案。
- 文档切分、Embedding、检索、Reranking 和 Generation 均实际执行。
- CPU Sequential 模式可以稳定运行。
- NPU Prefill + CPU Decode 可以运行，或明确记录被哪个底层接口阻塞。
- 日志能显示目标 Backend、实际 Backend 和 fallback。
- 连续运行至少三次，不崩溃且不会发生请求状态污染。
- 提供可复现的测试请求、响应、日志和 README。

## 8. 待确认事项

- 最终主要测试设备是 8 Gen4、8 Gen5，还是两者都需要支持。
- 最终交付是否必须完整迁移 scheduler2。
- llama.cpp 当前 Snapdragon Backend 是否已经支持目标 Embedding 和 Reranker 模型。
- llama.cpp 当前是否存在可直接复用的跨 Backend KV 导出和恢复接口。
- 最终模型文件、量化格式和固定 workload。
