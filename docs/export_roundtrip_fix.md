# 导出 Round-Trip 修复总结

## 背景

真实导出链路中曾经同时存在两类问题：

1. `BVH` 导入 Blender 后整体姿态不对。
2. 使用 `FBX` 格式时，Blender (bpy) 的 FBX 导入/导出无法做到无损 round-trip——即使导出参数完全一致，导回后的骨骼姿态仍会出现偏差。

**根本原因：Blender 的 FBX 导入器在每次导入时会重新计算骨骼局部轴系，且 FBX 二进制格式本身对四元数精度和骨骼层次的处理存在固有损失。**

因此最终方案是 **放弃 FBX，改用 GLB (glTF Binary)** 作为主要导出格式。GLB 在 Blender 中的导入/导出行为更加稳定，且 glTF 规范对骨骼动画的存储方式更加明确。

这次修复的目标不是“文件能导出”而已，而是让 **导出 -> 导回 Blender -> 骨骼姿态** 尽量与优化器内部的 FK 结果一致。

---

## 最终结论

### BVH

最终稳定方案分成两种模式：

1. **无 source rig 的纯骨架测试模式**
   - 继续使用 repo 内部的 `Y-up` 到 Blender/BVH 导入约定之间的基底变换。
   - 保持最小 round-trip 数值一致。

2. **有 source GLB rig 的真实导出模式**
   - 不再直接套用 meshless 的通用轴变换。
   - 改为读取 source GLB armature 的单位尺度，并按 source rig 的参考空间导出 BVH。
   - root 骨骼的 rest offset 不再写进 BVH 层级里的 root `OFFSET`，而是并入每帧 root translation 通道。

### GLB

最终稳定方案是：

1. **不要再只靠 armature object 运动 + pose bone quaternion 通道去表达优化器结果。**
2. **改为先用 FK 求出每帧每根骨骼的目标 pose matrix，再反推出 Blender 需要的 `matrix_basis`。**
3. armature object 保持 source rig 导入后的初始变换，只在 pose bones 上 bake 动画。
4. **使用 GLB (glTF Binary) 作为导出格式**，而非 FBX——因为 Blender 的 FBX 导入器无法做到无损 round-trip。

这样可以绕开 source rig 局部骨骼轴系复杂、`automatic_bone_orientation` 影响大、局部四元数难以直接映射的问题，同时避免 FBX 格式本身的精度损失。

---

## 症状与根因

## 1. BVH：导入 Blender 后整体绕 X 轴 90 度，头朝上，尺寸异常

### 现象

- 最小 synthetic skeleton round-trip 通过，但真实 dragon rig 导入 Blender 后仍有整体姿态和尺寸问题。
- `Head` 世界坐标和原始 source rig 差异很大。

### 根因

根因不是单一问题，而是三层叠加：

1. **meshless 测试模式和真实 source rig 模式混用了同一套 BVH 坐标变换。**
   - 最小骨架没有 source rig object transform。
   - 真实 dragon rig 来自 GLB，已经带有自己的单位尺度和参考空间。

2. **root rest offset 写在 BVH 层级 ` OFFSET` 中，但 Blender 的 BVH 导入并不会按我们预期把它当作真实 world 根位置。**
   - 导入后 root `CG` 被放在原点。
   - 下游所有骨骼位置整体偏掉。

3. **真实 source rig 的单位尺度来自 GLB armature object scale。**
   - dragon rig 在 Blender 导入后 armature scale 是 `0.01`。
   - 直接按 repo 内部单位写 BVH，导入后看起来会放大约 `100x`。

### 修复方案

对 BVH 导出做了模式分流：

1. `mesh_path is None`：走原来的 meshless 模式。
2. `mesh_path` 指向 `.glb`：临时导入 source GLB，读取 armature scale，作为 BVH 导出的单位尺度。不再对真实 rig 强制做 meshless 的 repo-to-BVH 全局基底转换。

同时对 root 写法做了关键修正：

```python
# root OFFSET 固定写 0
OFFSET 0 0 0

# root 的 rest offset 并入每帧 root translation
root_xyz = animated_root_translation + root_rest_offset
```

### 验证结果

- 仍然通过测试。
- 对真实 dragon rig 做 identity BVH 导出再导回后：
  - armature rotation = `(1, 0, 0, 0)`
  - armature scale = `(1, 1, 1)`
  - `CG / Pelvis / Spine / Head` 的世界坐标与 source rig 基本一致
  - `max_err ≈ 1e-4`

---

## 2. GLB：整体朝向/scale 正常，但动作完全不对

### 现象

- 真实导出文件导回 Blender 后，整体 object transform 看起来是对的。
- 但骨骼局部动作和 render/video 明显不一致。
- 早期版本还出现过导回后一帧整体滞后的问题。
- **此外，FBX 格式本身存在 round-trip 精度损失**：即使导出参数完全一致，Blender 的 FBX 导入器在重新计算骨骼局部轴系时仍会引入偏差。

### 根因

根因分两步定位出来：

1. **帧编号 off-by-one**
   - 使用 `scene.frame_start = 1` 且关键帧写在 `f + 1` 时，Blender 导回的 action range 会整体偏移。
   - 最小 generated-rig round-trip 明显出现一帧滞后。

2. **真正的主问题是：source rig 的 pose space 不能靠简单的 local quaternion 映射来恢复。**
   - 以前的思路是：

```python
pbone.rotation_quaternion = some_converted_quaternion
```

   - 这对最小 synthetic armature 可以成立。
   - 但对真实 dragon rig 不成立，因为：
     - bone local axes 复杂；
     - `automatic_bone_orientation=True` 会影响局部骨骼轴系；
     - 优化器的 joint quaternion 是 parent-space 语义；
     - Blender 真正吃的是 pose bone 在 rest pose 之上的 basis 变换。

换句话说：

**FK 的目标不是某个“局部四元数”，而是每根骨骼在该帧的完整目标 pose 矩阵。**

### 修复方案

把导出策略改成两阶段，并改用 GLB 格式：

1. 先在 PyTorch 里跑 FK；
2. 对每根 Blender pose bone：
   - 读取 source rig 的 `matrix_local` 作为 rest matrix；
   - 构造该帧的目标 pose matrix；
   - 反推出 `matrix_basis`；
   - 再从 basis 分解得到 `location / rotation_quaternion / scale` 并写 keyframe。
3. 使用 `bpy.ops.export_scene.gltf` 导出 GLB，而非 FBX。

核心思想是：

```python
desired_pose_matrix -> matrix_basis -> Blender keyframes -> GLB export
```

而不是：

```python
optimizer quaternion -> 猜一个 local quaternion -> Blender keyframes -> FBX export
```

另外还修了两个配套问题：

1. **帧编号改成 `scene.frame_start = 0`，关键帧写在 `frame = f`。**
2. **真实 CUDA 跑调试脚本时，FK 先在 skeleton 所在 device 上算，再把结果 `.cpu()` 给 Blender 用。**
   - 避免 device 不匹配错误。

### 验证结果

- 最终 `4 passed / 0 failed`
- generated-rig round-trip 的骨骼 head 误差为 `0.0`
- real dragon rig 的 identity rest-pose 误差为 `0.0`
- real dragon rig synthetic animation 在 source rig basis 下 round-trip 误差约 `1e-6`
- GLB round-trip 测试通过（`test_glb_npy_roundtrip.py`）

---

## 本次修复中最重要的经验

## 1. 不要把“最小骨架能对”误认为“真实 source rig 也会对”

最小 synthetic skeleton 只适合验证：

- 通道数量是否对齐
- 帧编号是否偏移
- 基础 round-trip 是否成立

但对真实 rig：

- bone local axes
- source armature object scale
- automatic bone orientation
- rest matrix

这些都会改变最终导入行为。

## 2. BVH 更像“文本格式 + 导入器约定问题”

BVH 的核心风险点在：

- Euler 顺序
- root channel 顺序
- root offset 的解释
- 单位尺度
- 导入器默认坐标约定

也就是说，BVH 的正确性高度依赖 **Blender 导入器的解释方式**，不是只看文件文本是否自洽。

## 3. GLB 导出更像"pose space / rest space 问题"

只要 source rig 不是你自己程序化创建的简单骨架，

```python
pbone.rotation_quaternion = converted_q
```

通常都不够稳。

更稳的做法是直接对齐：

```python
目标 pose matrix
```

再让 Blender 自己通过 basis 去解释。

## 4. FBX 不适合做精确 round-trip

Blender 的 FBX 导入器存在固有精度损失：
- 每次导入时重新计算骨骼局部轴系
- FBX 二进制格式对四元数精度的处理不够稳定
- 不同 Blender 版本的 FBX 导入行为可能不一致

因此改用 GLB (glTF Binary) 作为主要导出格式，其优势：
- glTF 规范对骨骼动画的存储方式更加明确
- Blender 的 glTF 导入/导出行为更加稳定
- 更适合需要精确数值一致性的场景

## 4. 最有价值的测试不是“能导出”，而是“导出再导回还能不能数值对上”

这次真正起作用的验证策略是 round-trip：

1. 程序化生成动画。
2. 导出 BVH/GLB。
3. 再导回 Blender。
4. 对比骨骼 head/tail 世界坐标与 FK 结果。

这比只检查：

- 文件是否生成
- action 是否存在
- keyframe 数量是否正确

更能抓住真实的姿态错误。

---

## 当前建议的回归检查

每次改导出器后，至少跑下面两类检查：

### 1. 最小 round-trip

```powershell
.\.venv\Scripts\python.exe .\Anytop\tests\test_bvh_roundtrip.py
.\Anytop\tests\test_glb_npy_roundtrip.bat
```

### 2. 真实导出链路

```powershell
.\.venv\Scripts\python.exe .\debug_phase2.py
```

然后把生成的：

- `outputs/debug_phase2/debug_phase2_animation.bvh`
- `outputs/debug_phase2/debug_phase2_animation.glb`

导回 Blender，快速检查：

1. armature rotation
2. armature scale
3. `Head` / `Spine` / `Pelvis` 首帧位置
4. 动作方向是否与 `phase2_final_render.mp4` 一致

---

## 关联文件

- `postprocessing/exporter.py` — BVH / GLB 导出主实现
- `Anytop/tests/test_bvh_roundtrip.py` — 最小 BVH round-trip 测试
- `Anytop/tests/test_glb_npy_roundtrip.py` — GLB round-trip 测试
- `debug_phase2.py` — 真实导出链路验证入口
- `pipeline.py` — 常规 pipeline 导出入口
- `docs/fbx_coordinate_and_lbs_fix.md` — 早期 FBX 坐标与 bind_matrices 修复记录（已弃用，改用 GLB）