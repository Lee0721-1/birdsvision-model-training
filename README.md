# BirdsVision 分类器训练

> 本仓库是私密的 ConvNeXt 分类器训练项目。SOYOL 定位器已经迁往独立的 [SOYOL 仓库](https://github.com/Lee0721-1/birdsvision-soyol-locator)；两个项目分别维护。旧提交历史仍保留迁移前的 SOYOL 文件，因此本仓库不能直接切换 Public。

本仓库包含 BirdsVision 双视图 ConvNeXt-Tiny 分类训练代码，支持低学习率微调、受控融合实验、参数扫描、检查点恢复和合成数据测试。分类器源码、类表和正式权重均不属于 SOYOL 发布内容。

一个父图由一张原图和若干已确认实例裁剪图组成。逻辑 batch 会展平成普通图片 batch，并且由同一个分类器只执行一次前向。裁剪 logits 在父图内等权平均，融合后才执行 softmax；没有裁剪时严格回退为原图结果。损失按父图平均。

## 项目官网

[鸟视 BirdsVision 官网](https://www.birdsvision.com.cn/)介绍 App、模型迭代进度、隐私说明与下载方式。本仓库提供训练源码；官网与本仓库均不提供训练图片或正式模型权重。

## 目录

- `convnext/`：双视图分类器训练、数据合同、参数扫描及示例配置；`tests/convnext/`：对应测试。
- 从仓库根目录使用下述 `python -m` 命令运行。

## 私有材料

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

## 许可证

本仓库现有源文件保留其 [GNU Affero General Public License v3.0 only](LICENSE) 标记；中文说明见 [LICENSE.zh-CN.md](LICENSE.zh-CN.md)。仓库维持私密，且不作为 SOYOL 的源码发布入口。数据和模型权重不因本仓库源码许可证自动获得授权。
