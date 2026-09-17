新动作数据集处理流程：
1. 准备动作数据集（GLB, FBX），对齐目录结构、FPS、TPOSE、统一朝向、奇异骨骼名、Trim尾帧等；FBX全转为GLB；
2. 渲染gif（dataset/review/render_gifs.py，按 datasets.jsonl 对三个数据集统一渲染；有网格的 GLB 渲染皮肤，仅骨架的 GLB 渲染胶囊骨架），给llm自动标注species_tags.jsonl，action_labels.jsonl（action_group和action_label）;
3. 跑prefill_loop_flags.py工具脚本自动预填action_labels.jsonl中的is_loop;
4. 人工核验（借助dataset/review/serve.py + index.html);
5. 跑预处理preprocess_and_validate，生成npy和cond；
