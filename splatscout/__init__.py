"""SplatScout: query-driven, closed-loop referring segmentation in 2D Gaussian Splatting scenes.

A VLM controller drives three tool sets over the scene: the Navigator (explore / nearby), the Segmenter (segment / same,
SAM3-based) and the Lifter (lift), plus relate / answer / report. Entry points: run_bench (run queries), scene_prep
(per-scene preprocessing), local_vlm_server (serve an open VLM behind an OpenAI-compatible endpoint).
"""
