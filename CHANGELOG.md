# Changelog

# Changelog

## [0.2.1] - 2024-XX-XX
- `exioml.datasets` 支持分类特征的 Leave-One-Out 编码，`prepare_dataset`/`build_preprocessor` 可直接处理混合类型。
- 新增 `LeaveOneOutEncoder` 公共导出，`frame_to_xy` 支持 `categorical_cols`/`as_frame` 参数，测试覆盖留一法编码与混合特征。

## [0.2.0] - 2024-XX-XX
- 新增 `exioml.preprocessing.prepare_regression_splits`，复现论文 64/16/20 划分、min-max 归一化与 Leave-One-Out 编码流程。
- `exioml` 包导出 `RegressionSplits` 元数据容器并更新 README 示例，方便在 PyPI 环境下直接复现实验设置。

## [0.1.0] - 2024-XX-XX
- 初始化 `exioml` PyPI 包框架，提供 `load_factor`、`list_regions`、`list_years` API。
- 新增 CLI：`python -m exioml --list-regions` 及 `exioml` 命令支持。
- 引入資料快取機制、可配置資料來源與測試樣例。
- 添加 `exioml.train` 轻量级 sklearn 训练接口与 `TrainingResult` 容器，并提供基础单元测试。
