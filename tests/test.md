# 测试结果

## 执行命令
`pytest -q`

## Pytest 输出节选
```text
............                                                             [100%]
12 passed in 2.60s
```

## 用例记录
### tests/test_factors.py::test_load_factor_filters_years_and_regions
- 覆盖函数：`exioml.load_factor`
- 断言：PxP 数据集在过滤 `years=[2010]`、`regions=["us"]` 后只返回美国 2010 年记录，且 `factor_value` 等于原生 `ghg_emissions`
- 结果节选：
  ```text
  schema region year factor_value ghg_emissions
  0    PxP     US 2010        100.0          100.0
  ```

### tests/test_factors.py::test_columns_argument_accepts_aliases
- 覆盖函数：`exioml.load_factor`
- 断言：`columns` 参数同时接受标准列名与别名；示例请求 `["value_added_meur", "Employment [1000 p.]"]` 后，返回列包含 `value_added_meur` 与别名映射出的 `employment_k`
- 结果节选：
  ```text
  ['value_added_meur', 'employment_k']
  ```

### tests/test_factors.py::test_listing_helpers
- 覆盖函数：`exioml.list_regions`、`exioml.list_years`
- 断言：分别按 schema 返回可用区域与年份；当前临时数据期望 `["CN", "US"]` 与 `[2015, 2018]`
- 结果节选：
  ```text
  list_regions("PxP") -> ['CN', 'US']
  list_years("IxI") -> [2015, 2018]
  ```

### tests/test_factors.py::test_invalid_schema_raises
- 覆盖函数：`exioml.load_factor`
- 断言：传入未知 schema 会抛出 `ValueError`
- 结果节选：
  ```text
  with pytest.raises(ValueError):
      load_factor(schema="invalid")
  ```

### tests/test_training.py::test_train_returns_training_result_for_hist_gbdt
- 覆盖函数：`exioml.train`
- 断言：以 `model="gdbt"` 训练后返回 `TrainingResult`，`test_score` 非负且 `predict` 输出与样本数一致
- 结果节选：
  ```text
  TrainingResult(metric_name='mse', feature_names=['value_added_meur', ...])
  ```

### tests/test_training.py::test_train_supports_custom_estimator_and_param_grid
- 覆盖函数：`exioml.train`
- 断言：传入 `RandomForestRegressor` 与 `param_grid` 能触发网格搜索，`best_params` 与 `cv_results` 均非空
- 结果节选：
  ```text
  {'max_depth': 2, 'n_estimators': 5}
  ```

### tests/test_datasets.py::test_frame_to_xy_validates_and_casts
- 覆盖函数：`exioml.frame_to_xy`
- 断言：默认 `dropna="any"` 会去除含 NaN 行并保持 `float32`，缺失列抛出 `ValueError`
- 结果节选：
  ```text
  X.shape -> (1, 2); y -> [3]
  ```

### tests/test_datasets.py::test_preprocess_xy_drop_strategy_aligns_y
- 覆盖函数：`exioml.preprocess_xy` + `build_preprocessor`
- 断言：`imputer="drop"` 可同步裁剪 X/y 后再做标准化；长度、顺序保持一致
- 结果节选：
  ```text
  len(y_proc) == 2  # 与去除 NaN 行后的样本数一致
  ```

### tests/test_datasets.py::test_split_xy_handles_stratify_fallback
- 覆盖函数：`exioml.split_xy`
- 断言：当分层抽样因类别过少失败时自动降级为非分层，样本总数保持 100%

### tests/test_datasets.py::test_prepare_dataset_with_load_factor
- 覆盖函数：`exioml.prepare_dataset`
- 断言：整合 `load_factor` 产出的 DataFrame，完成 X/y 提取、预处理与 50/25/25 划分，返回的特征 dtype 为 `float32`

## pymrio 深度调研（命令式探索）
### 执行命令
`python - <<'PY'`（内含 `pymrio.load_test()`、`calc_all(include_ghosh=True)`、特征表构造与聚合示例；详见输出摘要）

### 输出摘要
- 基础载入：`load_test()` 生成 6 区、8 部门、7 类最终需求的 MRIO，`Z (48, 48)`、`Y (48, 42)`。
- 核心计算：`calc_all(include_ghosh=True)` 后各区总产出 `x`（单位 indout）约 4.73e8–6.31e8，Leontief/Ghosh 前 3×3 子矩阵为：
  ```text
  L[:3,:3] ~ [[1.110, 0.001, 0.000],
              [0.002, 1.051, 0.000],
              [0.186, 0.135, 1.004]]
  G[:3,:3] ~ [[1.110, 0.000, 0.497],
              [0.009, 1.051, 0.442],
              [0.000, 0.000, 1.004]]
  ```
- 扩展计算（Emissions）：系数矩阵 `S` 与乘数 `M` 头部显示空气/水两类压力均成功传播；区域 PBA 与 CBA 总量：
  ```text
  D_pba_reg: reg1 2.19e8, reg2 1.32e8, reg3 9.14e8, reg4 5.53e8, reg5 5.82e8, reg6 1.08e9
  D_cba_reg: reg1 2.94e8, reg2 1.87e8, reg3 7.21e8, reg4 6.18e8, reg5 5.44e8, reg6 1.11e9
  D_cba_cap: reg1 0.266/0.111, reg2 0.031/0.019, …, reg6 0.970/0.341（空气/水 per capita）
  ```
- 贸易流：`get_gross_trade()` 得到跨区贸易矩阵（48×6），示例 `manufactoring` 对 reg2、reg5、reg6 的出口分别为 6.03e7、3.35e7、3.81e7；对应汇总表 `exports/imports` 覆盖各部门。
- 表征转换：构造 GWP100 因子（air=1, water=25 kg CO2e/kg）并通过 `Extension.characterize` 校验无单位/覆盖错误，得到新的 D_cba_reg 总量：
  ```text
  North America-like (reg1) 2.37e9, reg3 9.73e9, reg6 8.08e9 等
  ```
  乘数矩阵头部 `M`（GWP100）如：
  ```text
  impact=GWP100, region reg1, sectors [food, mining, manufactoring, electricity] ->
  [28.318, 40.368, 0.239, 144.108]
  ```
- 聚合：用 `aggregate(region_vector=['North','South'], sector_vector=['Primary','Industry','Infra','Services'])` 将 6→2 区、8→4 部门，得到 `Z (8, 8)`，区域总产出 North 1.77e9、South 1.56e9；聚合后的 CBA 排放 North 1.20e9、South 2.28e9。
- 搜索辅助：`mr.match('manu')` 返回各区“manufactoring”条目；`em.contains('water')` 可定位水体排放行，验证索引搜索工具可定位特定 compartment/stressor。
