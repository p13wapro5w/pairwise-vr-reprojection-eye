# pairwise-vr-reprojection-eye

## 领域

游戏、图形与多媒体

## 主题

VR 双目重投影

## 说明

姿态预测对齐、按眼历史与漂移校正。

初始状态下 `graphics/vr_reprojection_eye.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
