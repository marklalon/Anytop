新动作数据集处理流程：
1. 准备原始数据集（建议全转 GLB），跑 tools/precheck_dataset.py <原始目录> 检查，有 ERROR 必须先修;
2. 渲染 gif：dataset/review/render_gifs.py（按 datasets.jsonl）;
3. LLM 标注：dataset/review/llm_annotate.py actions / species 出草稿，看完再 apply（写入标 reviewed:false）;
4. 预填 is_loop：tools/prefill_loop_flags.py（预处理的硬前置）;
5. 人工核验：dataset/review/serve.py + index.html，先过 reviewed:false 的行;
6. 预处理：preprocess_and_validate.py，生成 npy 和 cond;
7. 补方向词：tools/prefill_direction_words.py --cond-path <processed>/cond.npy（依赖预处理结果），
   看完 --report 再 --apply，核验后跑 preprocess_and_validate.py --regenerate-side-artifacts;
