# 清扫机器人 PPO 训练

## 快速开始

```bash
pip install -r requirements.txt
python main.py          # 或 python train_sync.py
```

训练产物输出到 `artifacts/simple/checkpoints/<时间戳>/` 目录。

## 测试模型

训练过程中断点自动保存。随时可以运行 `test_model.py` 加载断点评估：

```bash
# 自动找最新训练 run 的最新 checkpoint
python test_model.py

# 最新 run 的指定步数断点
python test_model.py --step 4000

# 指定某次训练 run 的最新 checkpoint
python test_model.py --run 20260428_143052

# 指定 run 和步数
python test_model.py --run 20260428_143052 --step 4000

# 使用自定义配置文件
python test_model.py --config config/my_config.toml
```

评估产物输出到对应 checkpoint 的 `eval_<步数>/` 子目录，包含所有 episode 的 GIF + 日志 + 汇总文件。

## 目录结构

```
artifacts/
  └─ simple/
       └─ checkpoints/
            ├─ 20260428_143052/         ← 一次训练 run
            │    ├─ checkpoint_2000.pt
            │    ├─ checkpoint_4000.pt
            │    ├─ train.log
            │    │
            │    ├─ eval_2000/          ← checkpoint_2000 的评估产物
            │    │    ├─ ep01_score_180_step_200.gif
            │    │    ├─ ep01_score_180_step_200.log
            │    │    ├─ ...
            │    │    └─ summary.txt
            │    │
            │    └─ eval_4000/          ← checkpoint_4000 的评估产物
            │         ├─ ep01_score_220_step_180.gif
            │         ├─ ...
            │         └─ summary.txt
            │
            └─ 20260429_093015/         ← 另一次训练 run
                 └─ ...
```

## 环境配置

自定义地图和环境参数详见 [config/README.md](config/README.md)。
