# PU–MIRT / Gene2Wire 八份 Colab 实现审计与统一方案

审计日期：2026-09-03。本文比较的是用户提供的八份 notebook 的**实际代码、保存的执行状态与输出**；没有把 notebook 标题或注释当作实现事实，也没有重新跑完整实验。

## 证据等级

- **[E1 代码证据]**：直接从函数、配置、随机种子、mask、split、loss、optimizer 与 checkpoint 代码确认。
- **[E2 保存输出]**：来自 notebook 内已保存的表格、日志或图对应的数据；属于 descriptive result（描述性结果），不自动等于统计显著性。
- **[E3 建议]**：基于 [E1]/[E2] 的设计判断或迁移建议。

## 一页结论

**没有一份 notebook 可以原封不动成为所有数据的共同母版。** 它们虽然共享 `p = sigmoid(eta)` 和 PU detection likelihood（检测似然）的主线，但同名的 `MIRT`、`Hybrid`、`Joint` 在结构、正则化、初始化、优化器、校准和调参预算上并不等价。[E1]

| 数据组 | 当前最好用的 notebook | 结论边界 |
|---|---|---|
| SPIDER | **SPIDER（唯一版本）** | 数据处理、空间外层验证和 PU 概率合同可以保留；budget-6 调参把 rank、L2、learning rate 绑在一起，不能据此做 rank 或“共享结构更优”的强结论。[E1] |
| BARseq | **BARseq 2** | 是两份里可复现性更好的版本：完整运行且加入 Joint / PU-Joint。BARseq 1 的核心旧模型与 2 相同，但保存状态有中断和报错。BARseq 2 的 Joint 与旧模型并非公平的单因素比较。[E1] |
| Projection-TAGs | **Projection-TAGs 2，限于工程安全性** | rank-0 frozen fallback（冻结的直接模型回退）和 structured gate（结构模型门控）更安全；但两份 calibration seed 与 candidate seed 不同，连 PU-logistic baseline 都变了，所以不能把数值变化归因于新 tuner。做正式结论时，两份都不够，应补 strict inner-LOAO（严格内层 leave-one-animal-out）。[E1][E2] |
| Simulation | **没有单一赢家** | Simulation 1 的 10 repetitions 与 DGP 覆盖最好；Simulation 2 的非 oracle sensitivity estimator 和 checkpoint 最好；Simulation 3 正确使用 location，并把 two-stage rank→L2 写得较清楚，但预算不匹配。应组合优点重跑统一实验，而不是挑一个结果表。[E1][E3] |

**推荐总方案：**共同 core（模型、loss、tuning、metrics、checkpoint）＋四个 dataset adapter（数据适配器）＋ YAML config（超参数接口）。notebook 只负责选择 config 和启动 runner；模型代码里不出现 `if dataset == ...`。[E3]

## 八份 notebook 的实现差异矩阵

| Notebook | 数据、feature 与 mask | split / refit | 模型结构 | PU / calibration | optimizer 与 penalty | tuning | 执行与恢复状态 |
|---|---|---|---|---|---|---|---|
| **SPIDER** | 45,835 cells，36 slices，15 targets；32 gene＋3 location，经 spline 后 `X=50`；`W` 全为 true。[E1] | AP 方向 4 个 contiguous outer blocks，每块 9 slices；其余 27 slices 中固定选 4 个 inner-validation slices、23 个 train slices。preprocessing 只 fit train；所谓 final refit 仍保留 validation early stopping，未合并 train＋val。[E1] | Direct `eta=XC+b`；MIRT `eta=(XB)Aᵀ+b`；Hybrid `eta=(XB)Aᵀ+XC+b`。[E1] | `q=e*p`（代码中 `epsilon=0`）；20% paired calibration；有 no-hide、SCAR-80 与 Technical-SAR 20/40/60/80。[E1] | mini-batch Adam，batch 8192，clip 5，最多 70 epochs、patience 8；real-data penalty 写成 `l2*||theta||²`，gradient 是 `2*l2*theta`；Hybrid residual 有 L1＋L2。[E1] | 原池：Direct 6、MIRT 30、Hybrid 120；实际每模型 budget 6。rank、L2、LR 未正交覆盖。[E1] | 完整执行；无细粒度 checkpoint；主要保存 PNG，缺少统一结果表导出。[E1] |
| **BARseq 1** | 1,342 cells；23 genes＋location spline 后 `X=41`；A1 11 targets、M1 35 targets，分 panel 拟合；off-panel 由 `W=false` 排除；可加入 target metadata `Y` 项 `(XD)Yᵀ`。[E1] | 3 animals 内按 depth quarter 做 4 outer folds；3 inner depth folds；选参后在全部 outer-train 上按固定 epochs refit。[E1] | Direct / MIRT / sparse Hybrid，各有 non-PU、PU、`+Y` 版本。[E1] | 与 BARseq 2 的既有数据、hiding、paired calibration、metrics 代码相同。[E1] | 既有模型用 Adam；Hybrid residual 用 proximal L1；real-data L2 convention。[E1] | budget-12 broad screen → shortlist 3 → 3 inner folds × 2 restarts → one-SE selection。[E1] | 保存状态不一致：实验 cell `KeyboardInterrupt`，保存 cell `NameError`，后续还有未执行/陈旧输出；不宜作为交付基线。[E1] |
| **BARseq 2** | 与 BARseq 1 的数据、feature、mask、preprocessing 逐段相同。[E1] | 与 BARseq 1 相同。[E1] | 是 BARseq 1 的 strict superset：额外加入 dense Joint / PU-Joint，`eta=(XB)Aᵀ+XC+b`，SVD warm start。[E1] | 既有 PU calibration 与 BARseq 1 相同。[E1] | 旧模型仍为 Adam；新增 Joint 用 analytic-gradient full-batch L-BFGS，最多 1000 iterations。Joint 写 `0.5*l2*||theta||²`，所以同一数值 L2 只有旧 Adam convention 的一半强度。[E1] | 旧模型流程不变；Joint grid 中的 `learning_rate` 实际未被 L-BFGS 使用，却生成重复 candidates；所谓 restarts 对确定性 Joint 也是重复拟合。[E1] | 20 cells 顺序完整执行、无保存错误；有 model/panel/fold checkpoint。[E1] |
| **Projection-TAGs 1** | 37,827 modeled cells；32 genes＋origin，`X=33`；7 targets；standard detection 为训练标签；代码直接采用作者发布的 amplified-union column，并验证所有 standard positive 也为 union positive；union 只用于 paired calibration 与 evaluation；`W` 为 measured mask。[E1] | outer test animal pairs `{1,4}`、`{2,6}`、`{3,5}`；每 fold 实际是 3 train animals＋1 fixed validation animal＋2 test animals，并非标题暗示的完整 nested LOAO；没有用 4 development animals final refit。[E1] | Direct / MIRT / sparse Hybrid；low-rank 用 direct-SVD init；Hybrid 有 rank-0 direct fallback。[E1] | platform×target paired calibration；不足 10 positives 时 hard fallback；calibration seed 为 `fold_seed+205000`。[E1] | Adam；Hybrid residual 为 L1＋L2；uniform measured-entry loss。[E1] | matched budget 12；Direct pool 12、low-rank pool 56；Hybrid 分 6 个 rank/shared candidates＋6 个 residual candidates；candidate init 共用 seed。[E1] | 完整执行；输出可读，但设计仍是 single fixed validation animal。[E1] |
| **Projection-TAGs 2** | 数据与 preprocessing 与 1 相同。[E1] | 与 1 相同：仍是 3 train＋1 fixed val＋2 test。[E1] | Hybrid 改为 dense ridge residual；rank 0–7；structured candidate 只有通过 target-level one-SE / AUPRC non-inferiority gate 才接受，否则返回 prediction-identical 的 frozen direct parent。[E1] | calibration seed 改为 `fold_seed+205001`；candidate seed 也改为随 config 变化。[E1] | Adam；dense residual L2，不再是 1 的 sparse L1 Hybrid。[E1] | rank 0–7 加 4 个 ridge refinement；门控的 SE 来自 eligible targets，不是 animal-level uncertainty；SSp 不在 fixed validation animals 中，故不能参与 tuning。[E1] | 完整执行；回退保护比 1 强，但不是与 1 的 controlled comparison（受控比较）。[E1] |
| **Simulation 1** | `n=480`、24 targets、44 gene＋6 location，`X=50`，location 实际进入模型；4 DGP × 5 observation mechanisms；10 repetitions。[E1] | 4 outer folds、4 inner-validation slices；外层训练后 refit。[E1] | Logistic、PU、MIRT、PU-MIRT、Qiao-logit、Qiao-original；无 Hybrid / Joint。[E1] | SCAR / target-SCAR / Technical-SAR 等；Technical estimator 接收生成时真实 `sar_strength`，属于 oracle-like information（近似使用真值信息）。[E1] | analytic-gradient L-BFGS，`maxiter=100`；MIRT 从 Direct SVD warm start；penalty 为 `0.5*l2*||theta||²`。[E1] | rank `[1,2,3,4,6]` × 3 个 L2 的 15 组合中 coverage sample 6，不是 exhaustive grid。[E1] | 完整 10 repetitions；无 checkpoint；Qiao 路径收敛较差。[E1] |
| **Simulation 2** | active profile `n=300`、24 targets、24 genes；config 声明 6 location features，但实际 `x_raw=x_gene.copy()`，location 未进入模型；4 DGP × 5 mechanisms × 5 FNR × 4 repetitions。[E1] | 4 outer folds、4 inner-validation slices。[E1] | 10 models：Direct/PU、MIRT/PU-MIRT、Hybrid/PU-Hybrid、Joint/PU-Joint、两种 Qiao；其中 **Hybrid 是 direct 与 low-rank prediction/logit 的 alpha blend**，Joint 才是共同优化的 shared＋dense residual。[E1] | SCAR 用 Jeffreys pooling，target-SCAR 用 empirical-Bayes partial pooling，Technical-SAR 估计 slope＋target deviations，不读取真实 `sar_strength`；这是三份中最适合正式比较的 sensitivity estimator。[E1] | 多条路径：low-rank / Joint 以 analytic L-BFGS 为主；代码有多轮函数重定义，最终生效版本不易人工追踪。[E1] | 大 grid 仍以 matched budget 6 抽样；Hybrid 另有 alpha one-SE / cross-fit 选择路径。[E1] | 完成 400 setting units；有 model-fold 与 OOF checkpoint、较好的 config fingerprint；但 location bug 必须先修。[E1] |
| **Simulation 3** | `n=360`、24 targets、24 genes＋6 location，`X=30`，location 实际进入模型；3 shared-strength values × 3 mechanisms、无 bilinear DGP；实际 6 repetitions，日志标题却写“10-repetition”。[E1] | 4 outer folds、4 inner-validation slices。[E1] | 8 models：Direct/PU、MIRT/PU-MIRT、Joint/PU-Joint、两种 Qiao；无 blend Hybrid。[E1] | sensitivity estimator 回到 Simulation 1 的 known-slope / oracle-like 路径。[E1] | analytic-gradient L-BFGS，`maxiter=100`；`0.5*l2` convention。[E1] | MIRT/PU-MIRT 用 two-stage：8 ranks 固定 anchor L2，再对选中 rank 试其余 2 个 L2，共 10 fits；其他模型通常 budget 6，compute budget 不匹配。[E1] | 完成 270 setting units；只在 repetition×DGP×mechanism×FNR 整单元完成后 checkpoint，中途断开会丢失该单元全部工作；fingerprint 不含完整 source/data hash；Qiao 路径收敛较差。[E1] |

## 为什么同样叫 PU-MIRT / Hybrid，实际不是同一个模型

共同概率骨架是：

`p_it = sigmoid(eta_it)`，`q_it = e_it * p_it`，并在 `W_it=true` 的条目上最小化 `BCE(S_it, q_it)`。[E1]

但 `eta` 与训练方式不同：

| 名称 | 实际含义 | 不可忽略的差异 |
|---|---|---|
| Direct / Logistic | `eta=XC+b` | non-PU 令 `e=1`；PU 估计或校准 `e`。若 `e` 的来源不同，即使 `eta` 相同也不是同一 estimator。[E1] |
| “MIRT” | `eta=(XB)Aᵀ+b` | 这里是 reduced-rank multivariate logistic（降秩多任务 logistic），不是经典 IRT 中为每个 neuron 自由估计 latent trait 的模型。[E1] |
| Hybrid / Joint | 理想公式均可写为 `eta=(XB)Aᵀ+XC+b` | SPIDER/BARseq 旧 Hybrid 是 Adam＋稀疏 residual；PTAGs 2 是 dense ridge＋安全门控；Simulation 2 的 Hybrid 是后验 blend，不是联合优化；BARseq 2 / Simulation 的 Joint 才走 L-BFGS joint objective。[E1] |
| Qiao-logit / Qiao-original | 两种 bilinear baseline | 必须保留两个 registry name；它们不是一个模型的显示别名。[E1] |

另外，real-data Adam 代码的 penalty 是 `l2*||theta||²`，simulation / Joint 多为 `0.5*l2*||theta||²`。所以配置里同写 `l2=1e-3`，实际梯度强度相差 2 倍。[E1]

## 公平性与 scientific validity（科学有效性）问题

1. **SPIDER 的 rank 结论被搜索设计混杂。** budget-6 的六个 `(rank, l2, lr)` 是 `(4,1e-2,.005)`、`(12,1e-5,.01)`、`(2,1e-3,.01)`、`(8,1e-2,.01)`、`(15,1e-5,.005)`、`(2,1e-2,.005)`；rank 与 regularization / LR 同时变化。且 rank 15 等于 target 数，不能当作低秩共享优势。[E1]
2. **BARseq 2 的 Joint 对比同时改变 structure、initialization、optimizer 和 compute。** ignored learning-rate candidates 与 deterministic duplicate restarts 还会让 trial 数、SE 看起来比有效独立信息更多。[E1]
3. **Projection-TAGs 1→2 不只改 structured tuner。** seed 改动触发 “少于 10 positives” fallback：例如 fold 1 的 10X-RNA/MOp paired union 从 16（standard 0）变为 7（standard 0），估计的 sensitivity 从 `0.0294` 跳到 target-pooled `0.4769`。所以两本的整体 baseline 也不能直接比较。[E1]
4. **只有 6 animals 时，不应做 bootstrap / significance 的强宣传。** 推荐报告 animal-level OOF 点估计与透明的 fold/animal 分布；target-level SE 只能叫 tuning heuristic，不能叫 animal-level uncertainty。[E3]
5. **Simulation 的 estimator 信息量不一致。** Simulation 1/3 使用真实生成 slope，Simulation 2 从观测数据估计；这会系统性影响 PU 方法，不能合并排名。[E1]
6. **Simulation 的数据与预算也不一致。** repetitions、DGP、mechanisms、`n`、gene/location 维度及每模型 fits 都不同；尤其 Simulation 2 的 location 未使用，Simulation 3 的 MIRT 10 fits 对其他模型约 6 fits。[E1]
7. **所有正式比较应锁定同一 contract：**相同 outer splits、calibration rows、candidate budget、optimizer tolerance、restart 定义、refit data、metrics 与 seed family；oracle estimator 只能标成 oracle upper bound，不能进入主结论。[E3]

## 已保存结果：只能怎样解读

- SPIDER 在保存输出中，PU-MIRT 在各场景的 point estimate（点估计）总体最好；Technical-SAR 80% 时，macro AUPRC 为 **0.3413**，PU-logistic 为 **0.3361**；latent Brier 为 **0.0977** 对 **0.1096**；task recall 为 **0.3526** 对 **0.3464**。[E2] 这些差值是描述性的；单次运行、confounded tuner 和 rank-15 full-rank candidate 都不支持“显著优于”的说法。[E3]
- BARseq 2 在 Technical-SAR 80% 的预声明汇总中，PU-Joint 的 AUPRC 为 **0.3002**，PU-MIRT 为 **0.2662**，PU-Hybrid 为 **0.2637**；latent Brier 分别为 **0.1170 / 0.1216 / 0.1271**。[E2] 这是当前 notebook 内的最佳 observed run，但 Joint 同时更换 optimizer、warm start、residual 形式与有效 compute，且 L2 convention 不同，因此不能把差异单独归因于“Joint 结构”。[E1][E3]
- Projection-TAGs 的 PU-logistic 整体 AUPRC/AUROC 从 notebook 1 的 **0.1522 / 0.7278** 变为 notebook 2 的 **0.1557 / 0.7502**。[E2] 因为 direct baseline 都随 calibration/candidate seeds 改变，这组数最有价值的含义是“pipeline 没有被控制住”，而不是 notebook 2 的 structured model 更好。[E1][E3]
- 在各 Simulation 自己的 `shared strength = 1`、Technical-SAR 80% 条件内，保存的 mean AUPRC / latent Brier 为：Simulation 1，PU-logistic **0.2562 / 0.1452**，PU-MIRT **0.3314 / 0.1390**；Simulation 2，PU-logistic **0.2914 / 0.1395**，PU-MIRT **0.3105 / 0.1377**，PU-Joint **0.3737 / 0.1327**；Simulation 3，PU-logistic **0.2695 / 0.1442**，PU-MIRT **0.3376 / 0.1404**，PU-Joint **0.3565 / 0.1373**。[E2] 这些只能做 notebook 内的 descriptive comparison：三本的 sample size、features、estimator、DGP、预算、optimizer limit 和 repetitions 均不同，不能把数值横向排名。[E1][E3]
- 本审计不摘录 BARseq 的局部图形读数：当前没有一张跨模型、跨 panel、同口径且执行状态一致的主表。缺少可比口径时不制造排名。[E3]

## 推荐的统一接口

### 1. 固定数据合同（data contract）

```python
DatasetBundle(
    X_cell,          # cell / neuron features
    S_observed,      # observed positives/detections
    W_measured,      # measured-entry mask; W=False 永不进入 loss
    Y_target=None,   # optional target metadata
    Z_reference=None,# higher-sensitivity reference, not assumed ground truth
    reference_mask=None,
    cell_ids=None, target_ids=None, groups=None,
    feature_blocks=None, semantics=None, metadata=None,
)
```

loader 必须验证 `S_observed <= W_measured`、stable IDs（稳定 ID）、off-panel 不进 loss、test reference 对 tuner 不可见。[E3]

### 2. 用明确 model registry，禁止重载名称

建议名称：`direct_logistic`、`pu_direct`、`lowrank_logistic`、`pu_lowrank`、`joint_dense`、`pu_joint_dense`、`joint_sparse`、`pu_joint_sparse`、`blend_hybrid`、`qiao_logit`、`qiao_original`。每个 model 都使用同一 API：

```python
model = registry.create(model_name, model_config)
model.fit(design, labels, seed=seed)
p = model.predict_latent_proba(design)
q = model.predict_detection_proba(design, sensitivity)
```

### 3. YAML 超参数接口

```yaml
dataset:
  adapter: spider                 # spider | barseq | projection_tags | simulation
split:
  strategy: spatial_block         # animal_group | panel_depth | simulation_fold
observation:
  mechanism: technical_sar
  estimator: hierarchical         # main analysis; oracle 只能单独标注
model:
  names: [pu_direct, pu_lowrank, pu_joint_dense]
tuning:
  strategy: joint_grid            # final science default
  rank: [0, 1, 2, 4, 7]
  l2_shared: [1.0e-5, 1.0e-3, 1.0e-2]
  l2_residual: [1.0e-5, 1.0e-3, 1.0e-2]
  learning_rate: [0.005, 0.01]    # 仅对使用它的 optimizer 展开
  matched_fits_per_model: 12
  selection_metric: detection_log_loss
  tie_breaker: detection_auprc
regularization:
  convention: half_l2             # 统一为 0.5 * lambda * ||theta||^2
runtime:
  checkpoint_unit: outer_fold_model_trial
  seed: 42
```

正式主分析用 exhaustive / matched joint rank×L2（穷举或严格等预算联合搜索）；two-stage rank→L2 只作为 exploratory profile（探索配置），最后必须用同等 joint search confirmation。[E3]

### 4. checkpoint 与 fingerprint

每个 `condition × repetition × outer_fold × model × tuning_stage × trial` 原子保存；resume 后的结果必须与 uninterrupted run 相同。fingerprint 应包含 canonical config、input hashes、stable IDs、model version、seed map 与 source hash，但排除本机路径、hostname、GPU/CPU、worker 数和时间戳。[E3]

## 迁移顺序

1. **Freeze semantics（冻结语义）**：先统一 `S/W/Z/e/p/q`、penalty convention、metric baseline 和 model names；为现有八份输出记录 legacy profile。[E3]
2. **抽出 common core**：把 likelihood、Direct/Low-rank/Joint/Qiao、metrics、tuner、checkpoint 从 notebook 移到 package；notebook 只剩 config＋runner。[E3]
3. **逐数据集做 adapter**：SPIDER 保留 spatial blocks；BARseq 保留 panel mask 和 `Y_target`；Projection-TAGs 保留 animal grouping 且 union 只用于 calibration/evaluation；Simulation 用同一 model core 生成可知真值。[E3]
4. **加 invariant tests（不变量测试）**：`W=false` 零梯度、PU 在 `e=1` 时退化为 non-PU、target permutation equivariance、test label 改动不影响 tuning、preprocessing 只 fit train、rank-0 与 frozen direct prediction identical、resume 等于 uninterrupted。[E3]
5. **先 smoke test，再正式重跑**：小数据验证每条 code path；随后固定 outer splits、calibration subset、joint grid、repetitions 与计算预算，输出统一 long-format results、selected hyperparameters、checkpoint manifest 和 audit hash。[E3]

## 原始 Colab 索引

- [SPIDER](https://colab.research.google.com/drive/1Dh5Mk5CIdvU1TukzDllzZbuUn47hV4-H)
- BARseq：[1](https://colab.research.google.com/drive/1jkST6-sbdMfR_2Jbl_JjBxCRPC2NtHR0) · [2](https://colab.research.google.com/drive/19NzFvQyR6BIvZohgm3hnnhWoLUaOml9t)
- Projection-TAGs：[1](https://colab.research.google.com/drive/1pVU4UhICiI-qV0lU8KYkoMYv1GxuDCio) · [2](https://colab.research.google.com/drive/1FJ2hzw7ZOYtdpdLC1XSw0ePqibA9FY0l)
- Simulation：[1](https://colab.research.google.com/drive/1-pzTnQt77h-ftRZje2zOpvi5XmBJluCm) · [2](https://colab.research.google.com/drive/1Ju3TPTFxBjwcQVvdlPyOl2YK9ya9foH7) · [3](https://colab.research.google.com/drive/1myg-yr6C5dSieDTeo3LgHZoOJ1F1Na3f)

## 最终建议

短期可以把 **BARseq 2** 与 **Projection-TAGs 2** 当各自数据的工程起点，把 **Simulation 2 的 sensitivity/checkpoint**、**Simulation 1 的重复次数与 DGP 覆盖**、**Simulation 3 的 location 修正**移入共同 core；SPIDER 只保留 dataset-specific split 和 feature adapter。[E3]

正式论文或跨数据结论应来自统一 core 的重跑结果，不应把这八本现有 notebook 的模型排名直接合并。最稳妥的主比较是：同一 split、同一 calibration、同一 optimizer/penalty convention、同一有效 candidate 数、同一 refit 和同一 OOF metrics；任何 oracle 或 extra-compute 版本单列为 sensitivity analysis（敏感性分析）。[E3]
