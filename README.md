# BirdsVision Model Training

BirdsVision 的公开模型训练代码。第一阶段发布双视图 ConvNeXt-Tiny 分类训练、低学习率微调、受控融合实验、参数扫描、检查点恢复和合成数据测试。

一个父图由一张原图和若干已确认实例裁剪图组成。逻辑 batch 会展平成普通图片 batch，并且由同一个分类器只执行一次前向。裁剪 logits 在父图内等权平均，融合后才执行 softmax；没有裁剪时严格回退为原图结果。损失按父图平均。

## 项目官网

[鸟视 BirdsVision 官网](https://www.birdsvision.com.cn/)介绍 App、模型迭代进度、隐私说明与下载方式。本仓库提供训练源码；官网与本仓库均不提供训练图片或正式模型权重。

## 目录

- `convnext/`：双视图分类器训练、数据合同、参数扫描及示例配置；`tests/convnext/`：对应测试。
- `soyol/`：定位器数据导出、训练、验证与署名表工具；`tests/soyol/`：对应测试。
- 从仓库根目录使用下述 `python -m` 命令运行。

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
python -m convnext.dual_view_classifier_train --synthetic-smoke
python -m convnext.dual_view_classifier_finetune --synthetic-smoke
```

## 训练入口

复制示例配置到仓库外的私有目录，填入真实路径和 SHA-256，完成只读预检后才把 `allow_training` 改为 `True`：

```bash
python -m convnext.dual_view_classifier_train --config /path/to/private_config.py
python -m convnext.dual_view_classifier_finetune --config /path/to/private_finetune_config.py
```

普通训练入口在参数解析和数据合同两层拒绝 `final_test`。参数扫描会记录各次运行的峰值 RSS、阶段峰值内存、运行时间、准确率、NLL、ECE、Brier 分数以及分组比较结果；合成扫描不会宣称产生最佳正式参数。

## SOYOL 单类鸟体定位

SOYOL 取自 Student YOLO；内部教师模型 Teacher YOLO 简写为 TYLO。TYLO 是闭源内部模型，主要用于比对学生模型的效果。

`soyol/soyol_prepare_documented.py` 从仓库外的旧选择、逐图元数据和署名审计记录派生本轮排除清单；它不读取图片。本轮源码已用私有审计输入逐字节复现 1,316 张选择、179 张排除和逐图署名表。`soyol/soyol_export.py` 接收仓库外的已审计 A 层选择清单及其绑定报告，拒绝 B/C、未人工确认的框和封存 `final_test`，导出 Detect train/validation 文件。`soyol/soyol_train.py` 和 `soyol/soyol_validate.py` 是训练与 NMS 分支 validation 入口。`soyol/soyol_dataset.py` 在训练前校验文件摘要、分段、重复组、图片内容及标签；它会读取私有数据合同里的来源路径，但不会把路径或图片提交到本仓库。训练运行目录、权重和原图必须放在仓库外或被 `.gitignore` 排除的目录。训练脚本要求 CUDA，实际运行参数须由使用者按自有数据与设备确定。

```bash
python -m soyol.soyol_prepare_documented \
  --selection /private/old-selection/selection.jsonl \
  --attribution-dir /private/attribution-overlay \
  --metadata-dir /private/photo-metadata-audit \
  --output /private/documented-selection
python -m soyol.soyol_export --selection /private/documented-selection/selection.jsonl \
  --output /private/soyol-data
python -m soyol.soyol_train --dataset /private/soyol-data --base /private/yolo26n.pt \
  --project /private/runs --name example-run --epochs 20 --imgsz 640 --batch 8
python -m soyol.soyol_validate --dataset /private/soyol-data \
  --best /private/runs/example-run/weights/best.pt \
  --last /private/runs/example-run/weights/last.pt \
  --output /private/reports/soyol-validation.json
```

训练入口只接受与脚本记录的 Ultralytics 官方 YOLO26n Detect 基础权重摘要一致的文件；不使用内部 TYLO Pose 权重。推理与 validation 显式选择 one-to-many 分支，经 NMS 后最多返回 10 框；真实人工标签不能因这个上限而删减。`soyol/ATTRIBUTION_A_DOCUMENTED_20260926.csv` 是拟公开新权重所用 1,316 张照片的逐图署名与许可审阅表，包含改动说明；许可链接依据 iNaturalist 当前站点映射，平台条款仍待答复。仓库不提供训练原图、裁剪图、私有选择清单、模型权重或独立 `final_test` 结果，因此源码和署名表本身不表示权重已可公开。

[SOYOL 模型卡准备记录](SOYOL_MODEL_CARD.md)列出当前训练事实和未完成事项。`python -m soyol.soyol_attribution --selection /private/selection.jsonl --output /private/attribution.csv` 可在仓库外生成逐图署名表供人工复核；不要把未经复核的表直接发布。

## 许可证

源代码使用 [GNU Affero General Public License v3.0 only](LICENSE)。中文说明见 [LICENSE.zh-CN.md](LICENSE.zh-CN.md)。数据和模型权重不属于本仓库发布内容，也不因源代码许可证自动获得授权。
