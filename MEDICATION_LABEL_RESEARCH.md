# 配药视频动作标签研究结论

日期：2026-09-14  
范围：用户提供的 `video2.mp4`、`video3.mp4`、`video4.mp4`，以及固定多摄像头配药仓的目标采集场景。

本文用于设计视频动作候选标签，不替代医院本地 SOP、药品说明书（SmPC/PIL）、处方审核或临床质量制度。视频模型只能输出画面支持的 Action Candidate，不能从视频直接判定药品、剂量、配伍、无菌结果或操作是否合规。

## 结论

建议建立 **9 个标签的 v4 业务词表**，但当前测试视频先只启用其中 **4 个已有清晰证据的动作**；真实配药仓采集到完整流程后，再扩展到 **6 个核心实体动作**，最后 3 个动作只有在对应机位能够持续提供清晰证据时才启用。

### 当前测试阶段：4 个动作

`video2`、`video3`、`video4` 当前能较稳定支持：`open_container`、`insert_needle`、`draw_liquid`、`transfer_liquid`。模型基准测试先只给这 4 个标签，能显著减少把拿取、持物和流程记忆误报成动作的情况。

### 完整采集后的 6 个核心动作

| label | 中文 | 正向证据 | 不应标注 |
| :--- | :--- | :--- | :--- |
| `open_container` | 开启容器 | 打开瓶盖、掰开安瓿或移除容器保护部件 | 只是拿起、触碰或把针头靠近容器 |
| `insert_needle` | 针头进入容器 | 清楚看到针头穿过瓶塞/接口并进入目标容器 | 针头悬停、遮挡或仅靠近接口 |
| `draw_liquid` | 抽取药液 | 针头已在容器内，活塞向后移动并出现抽液动作 | 只有针头插入，没有活塞/液体证据 |
| `transfer_liquid` | 转移药液 | 活塞向前移动，把液体推入另一个药瓶、袋或容器 | 面向患者的注射；无法确认目标容器时不标注 |
| `agitate_solution` | 混匀药液 | 明确摇晃、翻转或振荡装有药液的容器 | 仅拿着容器、移动容器或推断“已经混匀” |
| `discard_sharps` | 处置锐器 | 明确把针头、安瓿或其他锐器放入锐器盒 | 画面中只有锐器盒，或把普通耗材丢弃混为锐器处置 |

### 条件启用：3 个动作

| label | 中文 | 启用条件 | 边界 |
| :--- | :--- | :--- | :--- |
| `inspect_medication` | 查看药品信息 | 有标签/药品近景，能看出护士在查看药名、规格、效期或剂量信息 | 只能表示“发生查看”，不能表示核对正确或处方匹配 |
| `disinfect_access` | 消毒容器接口 | 近景能看到棉签/消毒用品实际擦拭瓶塞、安瓿颈或接口，并有足够时间确认 | 仅出现棉签、酒精或瓶塞不能推断已消毒 |
| `label_prepared_medication` | 给成品贴签 | 能看到标签实际贴到注射器、药瓶或成品输液容器 | 查看标签、拿着标签或成品包装不算贴签 |

`seal_package` 不放入第一版核心词表：国家静配流程确实包含成品包装/封签，但当前目标是仓内护士配药，除非实际采集明确包含封装交接环节，再单独加入。`hand_hygiene`、台面清洁、剂量计算、配伍判断、无菌是否达标属于安全/质量观察或结构化记录，不应让 VLM 用一个动作标签代替。

## 对当前测试视频的判断

- `video2`、`video3` 是竖屏近景，主要能观察药瓶/安瓿、注射器、针头和护士手部操作，适合验证 `open_container`、`insert_needle`、`draw_liquid`、`transfer_liquid`。
- `video4` 是横向较宽的操作台视角，能够同时提供操作区、容器和锐器盒的上下文，适合作为固定机位总览参考；针头与容器接口仍需要其他近景机位确认。
- 当前样本中棉签、标签、锐器盒等物品可见，但“实际擦拭”“实际贴签”“实际投入锐器盒”不能仅凭物品出现来判定。因此这三个标签先作为条件标签，不应在 1 fps 粗粒度测试中强行输出。
- 当前模型输出中反复出现的 `RETRIEVE_MEDICINE` 属于泛化取物，会把拿取/持续持物扩成大段区间；`CHECK_INFO` 把无法验证的认知结果混进视频动作。这两个标签应退出模型词表。

## 与官方流程的对应关系

国家卫生健康委发布的《静脉用药调配中心建设与管理指南（试行）》和相关操作规程把静配流程描述为：标签/摆药、核对、混合调配、成品核对、包装与封装等；详细调配步骤包括选择注射器、消毒输液袋或瓶的加药处、打开安瓿/西林瓶、消毒瓶塞或安瓿颈、抽取药液、注入目标容器、轻轻摇匀、再次核对并清场。见 [国家卫生健康委 2021 年指南](https://www.nhc.gov.cn/wjw/c100175/202112/9bd330edb1a34e5b87cb66d4f657074e.shtml) 和 [静脉用药集中调配质量管理规范及操作规程](https://www.nhc.gov.cn/zwgkzt/wsbysj/201004/46963.shtml)。

CDC 的注射安全要求强调：在指定清洁区域配药、采用无菌技术、在进入药瓶接口前消毒、使用新的针头和注射器，并按规定处置使用后的锐器和一次性容器。见 [CDC Injection Safety](https://www.cdc.gov/injection-safety/hcp/clinical-safety/index.html) 和 [CDC Core Infection Prevention Practices](https://www.cdc.gov/infection-control/hcp/core-practices/index.html)。这些要求支持 `disinfect_access`、`insert_needle`、`discard_sharps` 作为有明确物理证据时的标签，但不能让模型根据结果反推过程已经合规。

WHO 的注射相关最佳实践覆盖手卫生、适当使用手套、皮肤/接口准备和消毒；它支持把无菌准备作为流程边界，但手卫生和环境状态不一定适合当前视频动作候选。见 [WHO best practices for injections and related procedures toolkit](https://www.who.int/publications-detail-redirect/who-best-practices-for-injections-and-related-procedures-toolkit)。

NHS Specialist Pharmacy Service 将 injectable medicine 的 reconstitution 定义为溶解、分散、混合或稀释，并指出药品选择、计算、体积测量、操作方法和标签都可能产生错误；实际操作必须遵循药品说明书。见 [Introduction to aseptic manipulation of injectable medicines](https://sps.nhs.uk/articles/introduction-to-aseptic-manipulation-of-injectable-medicines/)。因此 `transfer_liquid` + `agitate_solution` 比一个含混的 `inject_mix` 更适合表达可观察过程，且不把“结果正确”写进标签。

## 标签设计原则

1. 标签描述一个可观察的物理动作，不描述意图、正确性或完整 SOP 步骤。
2. `insert_needle`、`draw_liquid`、`transfer_liquid` 分开；否则模型无法表达“针头已进入但尚未抽液”或“抽液结束后向目标容器转移”。
3. 同一动作在多个固定 Camera View 中只形成一条统一时间轴上的 Action Segment；其他机位作为证据，不重复计数。
4. 看不清、被手遮挡、只有物品出现或只能依靠流程推断时输出空列表/不标注。
5. 标签定义应包含正向证据和排除条件；药品名称、规格、批号、剂量等应作为对象/属性或平台元数据，而不是为每种药品创建新 label。

## 对平台和 Qwen Worker 的结论

平台新增标签后，Qwen 权重本身不需要重新训练；只要新标签的 ID、定义和可观察判据被传给 Worker，模型可以通过提示词进行 zero-shot 识别。但当前 `ai-action-job.v1` 只传 `allowed_labels` 字符串列表，不传标签描述；因此当前 Worker 对未知标签最多能接受 ID，却不能可靠理解其业务含义。

推荐后续使用新的事件契约版本，携带任务绑定的不可变词表快照，例如：

```json
{
  "schema_version": "ai-action-job.v2",
  "label_schema_id": "p0-atomic-action.v4",
  "label_schema_digest": "<sha256>",
  "label_definitions": [
    {
      "label_id": "insert_needle",
      "display_name": "针头进入容器",
      "positive_evidence": "针头穿过瓶塞或接口并进入目标容器",
      "exclusion_criteria": "针头仅靠近接口或被遮挡无法确认"
    }
  ]
}
```

不要修改已使用的 v3 词表；新增标签应发布 v4，旧任务、旧候选和已审核标注继续绑定旧 schema。Worker 按任务快照生成提示词，回传校验也应按该任务快照，而不是推理结束时的当前 active schema。
