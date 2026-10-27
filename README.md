# SplatScout

Query-driven, closed-loop part-level referring segmentation in 2D Gaussian Splatting scenes, with the PartRef-ScanNet++ benchmark.
Paper: arXiv TBD.

```
splatscout/   method: VLM controller loop, Navigator / Segmenter / Lifter tools, SAM 3 wrapper, scene preprocessing, viewers
benchmark/    PartRef-ScanNet++: data, scene conversion, 2DGS training, ground truth, evaluator
```

## Install

```bash
conda env create -f environment.yml && conda activate splatscout
pip install git+https://github.com/facebookresearch/sam3.git
```

## Data

Download [ScanNet++](https://scannetpp.mpi-inf.mpg.de/) (scene ids in `benchmark/data/scenes.txt`; `scans/` and `dslr/` per scene).

```bash
export SCANNETPP_DATA=/path/to/scannetpp/data
export SPLATSCOUT_DATA=/path/to/workdir        # default ./data
export OPENAI_API_KEY=...                      # OPENAI_BASE_URL, AGENT_MODEL (default gpt-5.6) optional
```

## Prepare a scene

```bash
python benchmark/prep_scene.py --scene 578511c8a9
python benchmark/train_2dgs.py --scene 578511c8a9
python -m splatscout.scene_prep --scene 578511c8a9
```

2DGS training follows gsplat v1.5.1 `examples/simple_trainer_2dgs.py` (default recipe, normalization off, normal and distortion regularizers on), initialized from the mesh vertices, training frames only.

## Run

```bash
python -m splatscout.run_bench --scene 578511c8a9 --query "the pedestal fan standing beside the open lid of the big cardboard box"
python -m splatscout.run_bench --scene 578511c8a9 --tag v1          # all queries -> runs/578511c8a9/cold_given_v1/answers.json
python -m splatscout.run_bench --scene 578511c8a9 --live --port 8080  # viser viewer
```

`--backend chat --base-url http://host:port/v1` uses any OpenAI-compatible server (`splatscout/local_vlm_server.py` serves a Qwen-VL model).

## Evaluate

```bash
python benchmark/partref_gt.py --scene 578511c8a9
SCENE=578511c8a9 GS2D=1 python benchmark/partref_score.py --scene 578511c8a9 --answers runs/578511c8a9/cold_given_v1/answers.json --tag v1
python benchmark/partref_summary.py --tag v1
```

## License

Code: MIT. Benchmark data: CC BY-NC-SA 4.0.

## Citation

```bibtex
@article{splatscout2026,
  title   = {SplatScout: Query-Driven Part-Level Referring Segmentation in Gaussian Scenes},
  author  = {TODO},
  journal = {arXiv preprint arXiv:TODO},
  year    = {2026}
}
```
