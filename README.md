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

## 许可证

源代码使用 [GNU Affero General Public License v3.0 only](LICENSE)。中文说明见 [LICENSE.zh-CN.md](LICENSE.zh-CN.md)。数据和模型权重不属于本仓库发布内容，也不因源代码许可证自动获得授权。
