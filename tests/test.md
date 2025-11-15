# 测试结果

## 执行命令
`pytest -q`

## Pytest 输出节选
```text
......                                                                   [100%]
6 passed in 4.90s
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
