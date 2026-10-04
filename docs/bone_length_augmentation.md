# 身体骨长比例增强

`train_all.bat` 通过 `--bone_length_aug_prob` 与 `--bone_length_aug` 启用，具体取值见该文件。
每个样本按 `--bone_length_aug_prob` 的概率尝试增强，选择一至两个身体组，各组倍率均匀采样于
`[1 - bone_length_aug, 1 + bone_length_aug]`，整段动作倍率固定。支持腿链、躯干、颈部、尾部；腿链与左右镜像
共同缩放，保留上下腿段原有长度比，肩/髋的附着 offset 不参与腿长缩放。
CLI 和通用 loader 默认关闭（概率为 0），以保持已有调用及验证集行为。

按人工核验要求，当前放开全部动作标签和骨架。没有解剖身体组的骨架改用无分叉
拓扑骨链，有镜像标注的链共同缩放；支持父节点不在子节点之前的关节顺序。
只有零 offset、没有实际骨长的骨架也导出，并在终端说明骨架未变化。
不因接触残差、穿地或缺少 IK 链拒绝增强，也不按失败结果缩小倍率或重抽。
训练仍由概率控制是否增强；val/test 保持原骨架，拒绝非零概率。

训练与预览共用 `truebones_utils/bone_length_aug.py`：直接在 RIC 位置中缩放被选
父子骨段向量（保留原动画平移和伸缩），并根据 rest 比例调整固定根高度。
旋转不变时，这与恢复局部平移再做 FK 等价；无需世界位置恢复、旋转格式转换和 FK。
接触 IK 已关闭，原始局部旋转通道逐值保留，不再通过锁脚目标改变膝肘姿态。
足端轨迹随新骨长自然变化。不运行接触检测、接触位移或穿地统计；
不增加基于动作的高度补偿，悬空/插地留给动作生成后的后处理。
无效父索引、循环拓扑、
非有限数值等无法计算的输入仍按错误报告，不用原动作掩盖错误。
随后重建位置/速度，以及 offsets、物理 rest、canonical rest、joint_struct；旋转直接复制。
速度由 RIC 位置差加原始根 XZ 位移得到，保留周期末帧速度约定。
后续 canonical 编码和 collate 从新 rest 重算尺寸 L；subset 标准化统计保持原值。
增强在 leaf-drop 后、时间增强前运行，不修改共享 cond 或缓存。
保持拓扑和名称，因此无需重新计算拓扑矩阵或名称 embedding，也不改变 checkpoint schema。

## 人工核验

从 `Anytop` 目录运行：

```powershell
..\.venv\Scripts\python.exe tools/sample_augmented_bvh.py --mode bone-length --objects-subset Horse --cond-path dataset/merged/cond.npy --output-dir outputs/bone_length_preview --seed 1234
```

此模式强制尝试增强，抽样池包括全部动作和骨架，关闭随机变速、leaf-drop 和
loop roll/tile。原始 loop 属性仍用于增强内部的周期末帧速度处理；与其他非 loop
预览模式相同，最终窗口按开放序列导出。保留已有自动长度适配、裁剪及真实时间导出。
随机裁剪在原始/增强两条路径上重放相同 RNG，使对照窗口一致。

每个成功样本输出：

- `*__original.bvh`：相同窗口与长度的原始动作。
- `*__bone-length+<组名><倍率>...bvh`：骨长增强动作，文件名包含实际组倍率。

不输出 JSON。实际倍率和 `contact_ik=off` 显示在终端；训练时每个样本是否增强进入 spike dump。

`--n` 是抽取的动作数，上限是当前子集的可加载动作数。有效输入不再因增强
适用性或质量检查被跳过；无实际骨长的样本原样导出并明确标记。
核验重点是脚接触、膝肘方向、动作幅度、
首尾姿态及是否出现新抖动。该增强不会清除源动作已有滑脚，也不是动力学模拟。
