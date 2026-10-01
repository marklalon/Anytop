新动作数据集处理流程：
1. 准备原始数据集（建议全转 GLB），跑 tools/precheck_dataset.py <原始目录> 检查，有 ERROR 必须先修;
   再让 LLM 分析 precheck 结果中 joint-names 一项（"blanked as non-anatomical (zero name embedding)" 的 INFO
   和 "blanked joint(s) that parent named anatomy" 的 WARN），找出被误判为非解剖结构的真实身体关节
  （拼音名、编号名、带奇怪前缀等），把这些关节在原始文件里改成规范的英文解剖名（留 .bak），再重跑 precheck 确认;
2. 渲染 gif：dataset/review/render_gifs.py（按 datasets.jsonl）;
3. LLM 标注：dataset/review/llm_annotate.py actions / species 出草稿，看完再 apply（写入标 reviewed:false）;
4. 预填 is_loop：tools/prefill_loop_flags.py（预处理的硬前置）;
5. 人工核验：dataset/review/serve.py + index.html，先过 reviewed:false 的行;
6. 预处理：preprocess_and_validate.py，生成 npy 和 cond;
7. 检查 loop 标记：tools/compute_loop_unclosure_error.py --data-root <processed>（依赖预处理结果），
   检查 is_loop 标记是否正确;
