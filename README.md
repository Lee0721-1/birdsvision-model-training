# BirdsVision Model Training

BirdsVision 的公开模型训练代码。第一阶段发布双视图 ConvNeXt-Tiny 分类训练、低学习率微调、受控融合实验、参数扫描、检查点恢复和合成数据测试。

一个父图由一张原图和若干已确认实例裁剪图组成。逻辑 batch 会展平成普通图片 batch，并且由同一个分类器只执行一次前向。裁剪 logits 在父图内等权平均，融合后才执行 softmax；没有裁剪时严格回退为原图结果。损失按父图平均。

## 不随仓库发布的内容

- 鸟类图片、复核事件、来源修正和下载工具；
- 类表、正式数据清单和冻结测试集；
- 训练检查点、最终权重和服务器私有配置。

仓库内的示例配置默认 `allow_training=False`，其中的路径和摘要均为占位内容。正式训练前必须自行准备有权使用的数据、冻结的数据合同和类表。

## 安装与验证

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python dual_view_classifier_train.py --synthetic-smoke
python dual_view_classifier_finetune.py --synthetic-smoke
```

## 训练入口

复制示例配置到仓库外的私有目录，填入真实路径和 SHA-256，完成只读预检后才把 `allow_training` 改为 `True`：

```bash
python dual_view_classifier_train.py --config /path/to/private_config.py
python dual_view_classifier_finetune.py --config /path/to/private_finetune_config.py
```

普通训练入口在参数解析和数据合同两层拒绝 `final_test`。参数扫描会记录各次运行的峰值 RSS、阶段峰值内存、运行时间、准确率、NLL、ECE、Brier 分数以及分组比较结果；合成扫描不会宣称产生最佳正式参数。

## SOYOL 单类鸟体定位

`soyol_export.py` 接收仓库外的已审计 A 层选择清单及其绑定报告，拒绝 B/C、未人工确认的框和封存 `final_test`，导出 Detect train/validation 文件。`soyol_train.py` 和 `soyol_validate.py` 是训练与 NMS 分支 validation 入口。`soyol_dataset.py` 在训练前校验文件摘要、分段、重复组、图片内容及标签；它会读取私有数据合同里的来源路径，但不会把路径或图片提交到本仓库。训练运行目录、权重和原图必须放在仓库外或被 `.gitignore` 排除的目录。训练脚本要求 CUDA，实际运行参数须由使用者按自有数据与设备确定。

```bash
python soyol_export.py --selection /private/selection/selection.jsonl \
  --output /private/soyol-data
python soyol_train.py --dataset /private/soyol-data --base /private/yolo26n.pt \
  --project /private/runs --name example-run --epochs 20 --imgsz 640 --batch 8
python soyol_validate.py --dataset /private/soyol-data \
  --best /private/runs/example-run/weights/best.pt \
  --last /private/runs/example-run/weights/last.pt \
  --output /private/reports/soyol-validation.json
```

训练入口只接受与脚本记录的 Ultralytics 官方 YOLO26n Detect 基础权重摘要一致的文件；不使用内部 TYLO Pose 权重。推理与 validation 显式选择 one-to-many 分支，经 NMS 后最多返回 10 框；真实人工标签不能因这个上限而删减。本仓库目前未提供 A 层数据的可公开派生清单、归属署名、模型权重或 `final_test` 验收材料，因此这组源码不表示 SOYOL 权重已可公开。

[SOYOL 模型卡准备记录](SOYOL_MODEL_CARD.md)列出当前训练事实和未完成事项。`soyol_attribution.py --selection /private/selection.jsonl --output /private/attribution.csv` 可在仓库外生成逐图署名表供人工复核；不要把未经复核的表直接发布。

## 许可证

源代码使用 [GNU Affero General Public License v3.0 only](LICENSE)。中文说明见 [LICENSE.zh-CN.md](LICENSE.zh-CN.md)。数据和模型权重不属于本仓库发布内容，也不因源代码许可证自动获得授权。
