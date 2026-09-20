# Medication Action Annotation Context

本项目描述的是使用 Qwen 从药品操作视频中提取“可观察动作时间段”的标注语义。动作候选由模型提出，最终语义由平台中的人工修订与审核确定。

## 采集范围

**Medication Preparation Cell**：用于仓内配药的封闭长方体工作空间，护士在其中完成药品和耗材准备，视频由多个固定机位持续采集。
_Avoid_: 把病人给药、输液治疗或移动拍摄场景混入配药数据集

**Preparation Session**：一次在 Medication Preparation Cell 内由护士完成的配药演示，目标是记录对药品容器和耗材的可观察物理操作及其时间顺序。
_Avoid_: 把同一 Session 的不同 Camera View 当成不同任务

**Preparation Workbench**：Medication Preparation Cell 内承载药品、耗材和废弃容器的固定操作台，是配药动作的主要空间边界。
_Avoid_: 把护士身体移动或仓外环境当成配药动作

**Camera View**：同一 Preparation Session 中某一个固定摄像头产生的视角视频；它提供观察证据，不独立产生一套动作真值。
_Avoid_: 每个机位各自计数一遍动作

**Cross-View Action Event**：同一物理动作在多个 Camera View 中的联合观察结果，最终只在一个统一媒体时间轴上保留一条动作语义。
_Avoid_: 把多机位观测简单拼接成重复动作

**Patient Administration**：面向患者的给药、输液或穿刺环节，当前仓内配药数据集不纳入该范围。
_Avoid_: 用 transfer_liquid 表示向患者给药

## 标注生命周期

**CaptureGroup**：平台中的采集归组事实，包含同一演示记录关联的视频资产和可对齐的机器/人工标注证据。
_Avoid_: 把单个模型窗口称为 CaptureGroup、把 HDF5 episode 当成业务主对象

**Action Candidate**：模型根据视频生成的、尚未经过人工确认的动作时间段假设。
_Avoid_: ground truth、最终标签

**AnnotationRevision**：人在平台中对动作标签、边界或属性进行修订后形成的不可变人工版本。
_Avoid_: 覆盖原始候选、把候选直接升级为正式标注

**Ground Truth**：经过人工审核、可作为评估或训练依据的 AnnotationRevision。
_Avoid_: 模型置信度高的结果

## 动作语义

**Observable Action**：能够由视频证据支持的外显操作，不包含无法从画面确认的意图、处方判断或流程推断。
_Avoid_: SOP step、意图标签

**Core Preparation Action**：当前统一业务词表中的可观察实体操作，包括拿取容器/工具、开启容器、消毒容器接口、针头进入容器、抽取药液、转移药液、混匀药液、处置锐器和给成品贴签；每个动作都必须由视频证据支持。
_Avoid_: 泛化拿取、持续持物、调整位置、等待、查看标签本身作为独立核心动作

**Acquire Item**：手明确到达并取得将参与配药的容器或工具，物体离开原稳定位置并进入操作区；它是一次取得事实，不覆盖后续持有或重复抓握。
_Avoid_: 只碰到物体、重新调整手中物体、持续拿着物体或拿取与配药无关的物品

**Conditional Preparation Action**：流程上有价值、但当前暂不启用、只有在对应机位能持续提供清晰证据并发布新词表后才纳入标注的动作，包括放置物品和查看药品信息。
_Avoid_: 仅凭标签、棉签或容器出现推断动作已发生或已正确完成

**Transfer Liquid**：使用注射器把液体从一个容器推入另一个目标容器的可观察操作；它不是面向患者的注射。
_Avoid_: 用 transfer_liquid 表示患者给药、根据流程推断发生了转移

**Prepared Medication Label**：可见把标签实际粘贴到注射器、药瓶或成品输液容器的操作；仅出现标签、包装或拿着成品不构成该动作。
_Avoid_: 把查看标签、核对正确性或持物标为贴签

**Action Segment**：在媒体时间轴上表示一个 Observable Action 的半开时间区间 `[start, end)`。
_Avoid_: 单帧标签、事件点

**No-Action Interval**：当前标签集合没有足够视频证据支持某个动作的时间区间；它不是一个正向动作类别。
_Avoid_: IDLE action、WAIT action

**Action Taxonomy**：带版本号的受控动作词汇表，定义每个标签的含义、可见证据、排除条件和可选属性。
_Avoid_: 临时 label list、模型自定义标签

**Current Business Action Labels**：当前统一使用的九个业务标签：`acquire_item`、`open_container`、`disinfect_access`、`insert_needle`、`draw_liquid`、`transfer_liquid`、`agitate_solution`、`discard_sharps`、`label_prepared_medication`。它们描述护士正在完成的配药任务，不描述机器人执行原语。
_Avoid_: 为左右手、工具或容器分别创建新标签；把 `move_to`、`grasp`、`release` 等机器人控制原语混入视频业务词表

**Deferred Action Label Candidates**：暂不启用的候选标签：`place_item`、`inspect_medication`。只有在新采集证据和验收规则明确后，才通过标签库发布新版本启用。
_Avoid_: 因为流程通常会发生，就在看不清动作时强行输出候选

**Action Attribute**：描述动作参与者和方式的结构化信息，例如工具、源容器、目标容器、开启动作或混匀方式；它不应被拼接进动作 ID。
_Avoid_: 为每种药瓶创建一个新标签

## 视频输入

**Clip Asset**：平台管理的原始或裁剪后视频及其稳定身份、时长和媒体时间基准。
_Avoid_: 把模型临时读取的本地文件路径当成业务身份
