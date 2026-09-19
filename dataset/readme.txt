新动作数据集处理流程：
1. 准备动作数据集（GLB, FBX），对齐目录结构、FPS、TPOSE、统一朝向、奇异骨骼名、Trim尾帧等；FBX全转为GLB；
2. 渲染gif（dataset/review/render_gifs.py，按 datasets.jsonl 对三个数据集统一渲染；有网格的 GLB 渲染皮肤，仅骨架的 GLB 渲染胶囊骨架），给llm自动标注species_tags.jsonl，action_labels.jsonl（action_group和action_label）;
3. 跑预填工具，从动作本身补 action_labels.jsonl 里漏标的槽（都默认 --dry-run，看完标定准确率再 --apply；
   写入的行标 reviewed:false 与 autofill:true，只补空槽、永不改写已有的词；不出页面，
   依据看控制台汇总和 --report 的 CSV）：
   - tools/prefill_loop_flags.py     预填 is_loop（预处理的硬前置）;
   - tools/prefill_hand_words.py     补 hand1/hand2（hands 槽空 = 空手，持物必须写）;
   - tools/prefill_direction_words.py 补方向词（方向槽空 = 任一方向，有单一主导侧向/朝向就要写）；
                                     不指向任何方向的动作（hurt / getup / idle / rest / stop /
                                     draw / sheathe / headbutt / bite，见 audit_action_labels.py
                                     的 NO_PLANAR_DIRECTION_WORDS）不写平面方向词，脚本跳过、
                                     R3/R4 也不要求；竖直词 up/down 不受影响;
4. 人工核验（借助dataset/review/serve.py + index.html，先过 reviewed:false 的行，
   预填写的行在页上标「自动补标」);
5. 跑预处理preprocess_and_validate，生成npy和cond；
