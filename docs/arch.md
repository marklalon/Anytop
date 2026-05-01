训练:
  Truebones 干净 GT motion
      │
      ├─→ 渲染成多角度视频 → 跑 PCVG → 得到真实退化 → 退化后的 reference
      │         (深度偏差 / 子树遮挡 / 追踪抖动 / 物理违规)
      │                    │
      ▼                    ▼
  x_t (加噪的 GT)    reference_memory (退化版编码后)
      │                    │
      └──── AnyTop ────────┘  (cross-attention)
              │
              ▼
        预测噪声 → L = MSE(预测, 真实噪声)

推理:
  真实 PCVG 输出 (天然退化)
      │
      ▼
  编码 → reference_memory → AnyTop 扩散采样 → 干净 motion
