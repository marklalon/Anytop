# AnyTop 当前架构速览

> 详细说明与版本边界见 [anytop_model_architecture.md](anytop_model_architecture.md)。

```text
带噪动作 x_t [B,J,12,T]
  + rest pose（第 0 个时间 token）
  + T5 joint-name embedding
  + 13D joint structural embedding
  + absolute frame embedding
                    │
                    ▼
       GraphMotionDecoder × L
       ├─ graph-aware spatial attention
       ├─ full temporal attention
       ├─ cross-limb temporal block（配置的后 N 层）
       └─ FFN
                    │
                    ▼
             预测干净动作 x_0
```

每层都会重新注入同一个条件总线：

```text
diffusion timestep
  → species FiLM
  + resample-speed condition
  + canonical output-frame condition
  + loop condition
  + action-label condition
```

参考动作不经过 reference encoder 或 cross-attention。它只在采样器侧用于：

- `skip_timesteps` 的 img2img 初始化；
- 每步 clamp 已知区域的 inpainting；
- clamp 已有帧、生成新增帧的 outpainting；
- 必要时在采样前先 retarget 到目标骨架。

基础训练目标是 `MSE(predicted_x0, target_x0)`，不是噪声预测。当前 temporal attention 是
全窗口注意力；loop 使用条件 token、圆周相位、数据增强与闭合损失，不使用 temporal mask。
