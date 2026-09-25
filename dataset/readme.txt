新动作数据集处理流程：
1. 准备动作数据集（GLB, FBX建议全转为GLB），检查目录结构、文件名规范、FPS、TPOSE、bind pose一致性、奇异骨骼名等；
   先跑 tools/precheck_dataset.py <原始数据集目录>（不需要 Blender，秒级），有 ERROR 必须先修；
   --joints 打印每个关节 原名 -> canonical -> T5 文本，--report 写 JSON（含改名建议）;
2. 渲染gif（dataset/review/render_gifs.py，按 datasets.jsonl 对所有数据集统一渲染；有网格的 GLB 渲染皮肤，仅骨架的 GLB 渲染胶囊骨架）；
3. 基于gif，让llm自动标注species_tags.jsonl，action_labels.jsonl（填action_group和action_label）;
   dataset/review/llm_annotate.py：actions / species 子命令出草稿到 <processed>/review/llm/（action_group 读行里已有的，不让 LLM 改），
   看完草稿再 apply（只写 reviewed:false 的行，写入标 reviewed:false + autofill:true；--what species 改 species_tags.jsonl 后要重建 cond）;
4. 跑预填工具，从动作本身补 action_labels.jsonl 里漏标的槽（默认 --dry-run，看完 --report 的 CSV 再 --apply；
   只补空槽、写入的行标 reviewed:false + autofill:true；reviewed:true 的行默认跳过，要一起判加 --include-reviewed）:
   - tools/prefill_loop_flags.py     预填 is_loop（预处理的硬前置）;
   - tools/prefill_direction_words.py 补方向词（方向槽空 = 任一方向，有单一主导侧向/朝向就要写）；
                                     不指向任何方向的动作（hurt / getup / idle / rest / stop /
                                     draw / sheathe / headbutt / bite，见 audit_action_labels.py
                                     的 NO_PLANAR_DIRECTION_WORDS）不写平面方向词；竖直词 up/down 不受影响;
5. 人工核验, 借助dataset/review/serve.py + index.html，先过 reviewed:false 的行;
6. 跑预处理preprocess_and_validate，生成npy和cond；
