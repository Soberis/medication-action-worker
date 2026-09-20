# Qwen3.5-4B 配药视频识别优化研究

日期：2026-09-15
范围：当前 `medication-action-worker`，只讨论 Qwen 配药动作识别；不把机器人控制、MediaPipe 关键点纳入 Qwen 模型本身。

## 结论先行

目前最值得做的不是马上换模型或量化，而是把“标签定义—时间采样—评测—平台契约”固定下来，再优化推理内核。

按你已经跑出的基线，Qwen3.5-4B 在 4090 上处理 30 秒、1 fps 视频约为：模型加载 2.06 秒，准备 2.25 秒，推理 9.85 秒，总耗时（不含加载）12.11 秒，峰值显存约 11.86 GB。这个显存余量足够，量化不是当前第一优先级。

当前建议顺序：

1. **P0：先提高识别可信度**——冻结标签、修正提示词、规范时间段、建立金标准评测集。
2. **P0：再提高视频输入质量**——比较 fps、像素上限、分块边界和多摄像头融合。
3. **P1：再处理速度**——安装与当前 PyTorch 匹配的解码器和 Qwen3.5 专用 kernel；不要直接把最新 `flash-linear-attention` 装进正在工作的环境。
4. **P1：接入 platform**——让任务携带标签定义快照和 Qwen3.5 模型版本，而不是只传标签字符串。
5. **P2：数据足够后再 LoRA/QLoRA**——微调应建立在标签和评测稳定之后。

## 1. 标签和输出协议：对效果影响最大

### 建议的第一版标签集

平台当前已经发布 `p0-atomic-action.v4`，生产 worker 应保持这 9 个业务标签的契约；其中 6 个标签更偏向直接的液体/器械物理操作，适合另做低成本消融实验，但不应在 worker 内私自替换平台 schema：

| 层级 | label | 何时使用 |
|---|---|---|
| v4 | `acquire_item` | 明确取得将参与配药的容器或工具并带入操作区，不标持续持有或重复抓握 |
| v4 | `open_container` | 明确打开安瓿、瓶盖、药液容器或注射器保护结构 |
| v4 | `disinfect_access` | 明确擦拭瓶塞、入口或注射点，不能把普通擦拭都算进去 |
| v4 | `insert_needle` | 针头明确刺入药瓶、安瓿或指定入口 |
| v4 | `draw_liquid` | 注射器从容器中抽取液体，需看到活塞/液面或连续动作证据 |
| v4 | `transfer_liquid` | 将液体注入另一个容器，不要仅因针头靠近容器就判定 |
| v4 | `agitate_solution` | 摇匀、混匀、振荡或倒转混合药液 |
| v4 | `discard_sharps` | 将针头、注射器等锐器明确放入锐器盒/废弃容器 |
| v4 | `label_prepared_medication` | 明确把标签贴到已配制药液/成品上 |

`retrieve_medicine`、`check_info`、`place_item`、`seal_package` 不属于当前 v4 默认词表。`acquire_item` 虽然边界较宽，但已经是平台 v4 业务标签；应通过清晰的起止判据和按标签评测控制误报，而不是在 worker 侧删除它。

当前代码中要特别检查 `run_35.py`、`run_35_chunked.py` 和生产 worker 的提示词：示例、`DEFAULT_LABEL_DEFINITIONS`、platform 活跃标签和评测金标准必须来自同一个版本文件，不能由某个脚本单独维护另一套词表。

### 不要把 confidence 当成真实概率

当前结果多次给出 `0.98`，这更像模型自报置信度，不代表经过校准的 98% 正确率。建议：

- 输出 `confidence` 的同时输出 `evidence` 或 `evidence_level`：`clear`、`partial`、`occluded`；
- 对外阈值由验证集决定，例如按 label 分别设阈值，而不是统一使用 0.5；
- 如果暂时没有校准集，字段名可以改成 `model_confidence`，避免业务误读；
- 记录 `schema_digest`、`prompt_version`、`model_revision`，使每个结果可复现。

### JSON 需要两层防线

第一层是提示词要求只输出 JSON；第二层是代码使用固定 schema 做解析、字段类型检查、标签白名单检查、时间范围检查和排序/去重。解析失败时保留原始输出，但不要把失败结果写成成功候选。

如果未来改用 vLLM，可以使用其 structured outputs/JSON schema 能力约束结构；官方文档也说明旧的 `guided_json` 等字段正在被 `structured_outputs` 取代。见 [vLLM Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)。

## 2. 视频时序：1 fps 不一定覆盖短动作

配药动作中，`insert_needle`、`disinfect_access`、`discard_sharps` 可能持续时间很短。1 fps 适合先做基线，但不能默认是最优采样率。

建议做一个固定模型、固定 prompt、固定像素预算的网格实验：

| 实验组 | fps | 用途 |
|---|---:|---|
| A | 0.5 | 速度下限，观察长动作是否足够 |
| B | 1.0 | 当前基线 |
| C | 2.0 | 重点比较短动作召回率 |
| D | 4.0 | 只用于局部细化，不建议直接全视频长期运行 |

不要只看总耗时。每组同时记录每个 label 的 precision、recall、F1、边界误差和 JSON 合法率。Qwen3-VL 官方代码也把视频 `fps`、每帧像素范围、总像素数和帧数作为显式控制项；Qwen3.5 的 Transformers 文档要求通过 processor/chat template 正确传递视频输入。见 [Qwen3-VL 官方仓库](https://github.com/QwenLM/Qwen3-VL)、[Qwen3.5 Transformers 文档](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5)。

### 推荐“两阶段时序策略”

1. **粗检**：全视频按 1 fps 分块识别，找出动作变化点和疑似短动作。
2. **细检**：只对变化点前后约 2 秒按 2–4 fps 重采样，并用更紧的局部时间窗再次判断。
3. **合并**：只合并相邻且同 label 的片段；不同 label 不跨越合并。对每个片段保留粗检和细检来源。

这样通常比把整个长视频统一提高到 4 fps 更适合当前 4090，也更容易解释漏检来自哪里。

### 分块要处理边界

当前的“24 秒主窗口 + 3 秒上下文”方向是合理的，但需要明确：

- 上下文帧只用于判断边界，不应重复计入最终时长；
- 相邻 chunk 的同 label 片段要做时间合并；
- 输出统一使用原视频时间轴，不能使用 chunk 内相对秒数；
- 用真实视频时间戳而不是简单的帧序号推导时间；
- 在评测中单独统计“动作发生在 chunk 边界附近”的召回率。

## 3. 视频解码和像素预算：先稳定输入链路

你此前遇到过 `torchcodec`、PyAV 和 torchvision fallback 警告。Transformers 的 VideoProcessor 负责视频解码和模型相关的帧采样，官方文档说明其默认视频解码路径需要 `torchcodec`。见 [Transformers VideoProcessor](https://huggingface.co/docs/transformers/main/en/main_classes/video_processor)。

当前环境是 PyTorch 2.6.0，因此不要无条件安装 TorchCodec 最新版。TorchCodec 官方兼容表中，PyTorch 2.6 对应 `torchcodec 0.2` 系列；应在复制出的测试环境中安装并验证，不要直接覆盖已经能跑的生产环境。见 [TorchCodec 安装与兼容性](https://github.com/meta-pytorch/torchcodec#installing-torchcodec)。

建议的安全顺序：

```bash
# 在 action_poc35 的副本环境中测试，不要先改正在工作的环境
conda install ffmpeg -c conda-forge
pip install "torchcodec==0.2"
python -c "import torchcodec; print(torchcodec.__version__)"
```

然后用同一段 `test_30s.mp4` 比较解码耗时、帧数量、首帧时间戳和最终 JSON。若 `torchcodec==0.2` 与当前 Transformers 组合不兼容，保留现有可用解码路径，并单独升级整套环境；不要为了消除 warning 牺牲可复现性。

像素预算也要做小规模 sweep。保持 fps 不变，比较当前 `min_pixels/max_pixels` 的 0.5×、1×、2× 三档，重点查看针头、瓶塞、液面和标签是否仍可辨认。像素上限影响视觉 token 数和显存，必须和识别 F1 一起决定，不能只追求更小显存。

## 4. Qwen3.5 kernel 和 CUDA 加速：主要提升速度，不直接提升识别

你日志里的四类 warning 是 Qwen3.5 的 `causal_conv1d` 和 gated delta rule/linear attention 使用 PyTorch fallback。官方 Qwen3.5 文档明确说明，安装 `causal_conv1d`、`fla` 可使用更快 kernel；也提供了 `use_kernels=True` 的 Hub kernel 路径。见 [Qwen3.5 官方 Transformers 文档](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5)。

建议按这个顺序在独立环境 benchmark：

1. `causal-conv1d`；
2. 与当前 PyTorch/Transformers 匹配的 `flash-linear-attention` 或 Transformers Hub kernels；
3. 再考虑 FlashAttention-2。

关键版本风险：当前官方 `flash-linear-attention` 安装说明要求 PyTorch `>=2.7.0`、Triton `>=3.3`，而你的环境是 PyTorch 2.6.0。因此不要直接执行最新版本的 `pip install flash-linear-attention[cuda]`。应当二选一：

- 保持当前 PyTorch 2.6，在副本环境寻找并锁定兼容版本；或
- 新建完整的优化环境，一起升级 PyTorch、Transformers、Triton、FLA，再跑回归测试。

见 [flash-linear-attention 官方仓库](https://github.com/fla-org/flash-linear-attention)、[causal-conv1d 官方仓库](https://github.com/Dao-AILab/causal-conv1d)。

Transformers 的 kernel loader 支持 `pip install -U "transformers[kernels]"` 和 `use_kernels=True`，但官方 Qwen3.5 文档中的性能数据不能直接等同于 RTX 4090；必须用你自己的 30 秒视频和同一批次请求测量。见 [Transformers Kernel Loading](https://huggingface.co/docs/transformers/en/kernel_doc/loading_kernels)。

RTX 4090 属于 Ada，FlashAttention-2 的 CUDA 支持范围包含 Ada；FlashAttention-3 面向 Hopper/H100/H800，不适合拿来作为 4090 的首选。见 [FlashAttention 官方仓库](https://github.com/Dao-AILab/flash-attention)。此外，Qwen3.5 当前最明显的 fallback warning 来自线性注意力相关组件，所以 FlashAttention-2 不应排在 `causal-conv1d`/FLA 之前。

## 5. 推理参数和部署方式

### 当前 HF 直推保留即可

单卡、单 worker、低并发时，当前 Transformers 直推方式更容易调试。建议保持：

- `enable_thinking=False`；
- `do_sample=False`、`use_cache=True`；
- 把 `max_new_tokens` 从 512 做一次 128/256/512 对比；当前结果已证实密集窗口在 512 仍可能发生输出截断，因此生产代码必须严格拒绝不完整 JSON，并在必要时自动重试 1024；
- 删除无效的 `temperature/top_p/top_k` 生成参数，避免 warning；
- 生产进程常驻加载模型，视频任务只做 processor 和 inference。

### 已修复的长视频输出问题

旧逻辑曾在 JSON 未闭合时补 `}`/`]` 再解析。这会把模型输出上限触发造成的“只输出了前缀动作”伪装成成功结果。四个历史长视频共 35 个窗口中有 17 个 raw 窗口不完整，所以这些旧 JSON 不能作为准确率 benchmark。

现在的原则是：完整 JSON 才接受；疑似截断时最多自动用 1024 tokens 重试一次；仍不完整则任务失败，并保留已完成窗口的 `.raw.txt`，不再静默丢动作。

### 什么时候测试 vLLM

如果之后有多个摄像头任务并发、多个视频排队或需要结构化输出约束，再单独测试 vLLM。Qwen3.5 已在 vLLM 支持列表中作为 hybrid-only 模型列出；视频模式不能使用只支持文本的 `--language-model-only`。见 [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models/)。

测试 vLLM 时固定：`max_num_seqs=1`、相同 fps/像素预算、相同 prompt、相同 chunk 策略，并比较吞吐、单任务延迟、峰值显存和 JSON 合法率。对于本项目，vLLM 是并发/服务化优化，不是当前识别准确率的第一优化项。

## 6. 多摄像头和 MediaPipe 的协同边界

Qwen 和 MediaPipe 不需要合并成一个模型：

- Qwen：判断“做了什么”以及动作语义和时间段；
- MediaPipe：如果启用，提供手部关键点、手是否进入区域、遮挡/姿态等几何证据；
- platform：按同步时间轴和 `capture_group_id` 合并结果。

对当前 Qwen-only MVP，MediaPipe 可以先不参与 Qwen 推理。多摄像头建议先**每路独立识别，再按时间做跨视角去重/补证据**，不要把多个视角直接拼成一个长视频：拼接会增加视觉 token，且会破坏原始时间轴。每个候选至少携带：`camera_id`、`capture_group_id`、`start_ms`、`end_ms`、`source_view`。

如果同一动作在两个相机都看见，platform 应合并为一个业务事件，并保留多个来源视角；如果主视角被遮挡，则允许侧视角补充证据。后续再用 MediaPipe 作为规则/二阶段特征，不让它阻塞 Qwen-only 主链路。

## 7. platform 集成必须补的契约

当前 worker 能根据任务里的 `allowed_labels` 做白名单处理，但仅传字符串不足以让模型理解新标签。平台新增标签后，Qwen 不会因为标签库变化而“自动学会”；至少要把标签定义传给 worker，或者让 worker 使用与平台版本一致的固定 schema。

建议把 action job v2 设计成如下形态：

```json
{
  "model": {
    "algorithm_key": "QWEN3_5_ACTION",
    "model_name": "qwen3.5-4b",
    "model_revision": "<pinned-revision>"
  },
  "label_schema": {
    "version": "medication-v1",
    "digest": "<sha256>",
    "labels": [
      {
        "id": "draw_liquid",
        "description": "从药液容器中抽取液体",
        "positive_evidence": ["看到针头在容器内且液体进入注射器"],
        "negative_evidence": ["仅手持注射器但未抽液"]
      }
    ]
  }
}
```

结果也写回 `model_revision`、`prompt_version`、`label_schema.digest` 和 `source_camera_ids`。这样 platform 后续改标签时，可以知道旧结果使用的是什么语义，不会把不同版本混在一起评估。

生产与离线基线已全面统一收敛至 Qwen3.5-4B 架构，旧版 Qwen3-VL-8B 代码与离线对照脚本已彻底淘汰；任务租约、模型名称（`Qwen3.5-4B`）与平台端 `ActionCandidateJobService` 保持一致。

仓库不必合并：`medication-video-platform` 负责任务、标签版本、结果存储，`medication-action-worker` 负责 Qwen 推理；两者通过 action job/result contract 对接即可。

## 8. 评测和“强化”路线

### 先建评测集，不要用单个 30 秒视频决定模型

建议把数据划分为：

- smoke：当前 30 秒视频，用于检查环境和回归；
- development：至少覆盖多个护士/日期/药品摆放方式/摄像头视角；
- holdout：按护士或完整 session 划分，不能把同一视频切片随机分到训练和测试两边。

每个 gold event 记录 `label`、`start_ms`、`end_ms`、可见相机、遮挡状态和备注。至少统计：

- 每个 label 的 precision/recall/F1；
- temporal IoU（例如 0.25、0.5 两个阈值）；
- 起止边界误差的 median/P90；
- 每分钟误报数；
- JSON 合法率；
- 平均/ P95 延迟和峰值显存。

重点制作 hard negatives：拿起但未使用、针头靠近但未刺入、擦拭非入口、看见标签但没有核对、已完成动作后的停留，以及两个动作交接的边界片段。

### LoRA/QLoRA 放到第二阶段

当标签、时间定义和评测集稳定，并积累了足够多的真实仓内视频后，再做 Qwen3.5 的 LoRA/QLoRA：

1. 训练样本保留真实视频采样策略和时间戳；
2. prompt、label schema、JSON 输出格式与生产保持一致；
3. 先 LoRA 小规模验证是否改善短动作和遮挡，再决定是否全量训练；
4. 按护士/session 划分训练和验证，避免同一动作片段泄漏。

Qwen 官方训练 recipe 支持 LoRA/Q-LoRA，PEFT 文档也明确其目标是减少可训练参数和显存/计算开销。见 [Qwen 官方 fine-tuning recipe](https://github.com/QwenLM/Qwen/tree/main/recipes/finetune)、[Hugging Face PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora)。

## 推荐执行清单

### P0：现在就做

- [ ] 固定 `p0-atomic-action.v4` 九标签生产契约；另做六标签子集的离线消融，不改变生产 schema。
- [ ] 清理 `run_35.py`、`run_35_chunked.py` 中仍引用旧标签的示例。
- [ ] 固定 prompt、label schema、模型 revision，并保存 lock 文件。
- [ ] 做 0.5/1/2 fps 和像素预算的可重复 benchmark。
- [ ] 加入 JSON schema、时间范围、白名单、重复片段和跨 chunk 边界校验。
- [ ] 在独立环境验证 `torchcodec==0.2`；不改当前可用环境。
- [ ] 建立按护士/session/视角划分的首批 gold/holdout 评测集。

### P1：P0 稳定后

- [ ] 粗检 1 fps + 局部 2–4 fps 细检。
- [ ] 独立环境 benchmark `causal-conv1d`、匹配版本 FLA/Hub kernels。
- [ ] 更新 platform 的 Qwen3.5 model profile 和 label schema digest。
- [ ] 多摄像头按 `capture_group_id` 做去重和证据合并。
- [ ] 有并发需求时再比较 vLLM 与 HF 直推。

### P2：数据量达到后

- [ ] LoRA/QLoRA 微调。
- [ ] 置信度校准和自动 abstain/人工复核策略。
- [ ] 只有显存或吞吐出现实际瓶颈时，才评估量化。

## 参考资料

- [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Transformers Qwen3.5 documentation](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5)
- [Transformers VideoProcessor](https://huggingface.co/docs/transformers/main/en/main_classes/video_processor)
- [Qwen3-VL official repository](https://github.com/QwenLM/Qwen3-VL)
- [qwen-vl-utils video processing source](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py)
- [TorchCodec](https://github.com/meta-pytorch/torchcodec)
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [Transformers kernel loading](https://huggingface.co/docs/transformers/en/kernel_doc/loading_kernels)
- [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/)
- [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
- [Qwen fine-tuning recipes](https://github.com/QwenLM/Qwen/tree/main/recipes/finetune)
- [Hugging Face PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora)
