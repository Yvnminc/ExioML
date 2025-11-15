# Changelog

## [0.1.0] - 2024-XX-XX
- 初始化 `exioml` PyPI 包框架，提供 `load_factor`、`list_regions`、`list_years` API。
- 新增 CLI：`python -m exioml --list-regions` 及 `exioml` 命令支持。
- 引入資料快取機制、可配置資料來源與測試樣例。
- 添加 `exioml.train` 轻量级 sklearn 训练接口与 `TrainingResult` 容器，并提供基础单元测试。
