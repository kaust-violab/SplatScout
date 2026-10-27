# PartRef-ScanNet++

261 referring expressions over 12 ScanNet++ scenes (object 78, part as target 92, part as discriminator 32, part as anchor 30,
multi-target 29), ground truth on the laser-scanned meshes, evaluation on 216 held-out test frames.

```
data/scenes.txt                        scene ids
data/queries/<scene>.json              qid, text, role, status, targets
data/anno/<scene>/objects.json         object instances (obj_id, label, status)
data/anno/<scene>/parts.json           part instances as mesh vertex lists (part_id, parent_obj, label, vert_indices)
data/anno/<scene>/part_verdicts.json   keep / drop decisions for ambiguous parts
data/test_frames/<scene>.txt           held-out test frames
```

A target is `{"kind": "obj", "obj_id": 40}` (vertices from the scene's `segments_anno.json`) or
`{"kind": "part", "part_id": "obj88_door_1"}` (vertices from `parts.json`); multi-target ground truth is the union.
Vertex indices refer to `scans/mesh_aligned_0.05.ply`.

## Submission

`answers.json` maps each `qid` to `{"gaussians": [...]}` (subset of the scene's 2DGS Gaussians), `{"weights": {"12": 0.9, ...}}`
(soft weights) or `{"vertices": [...]}` (mesh vertices). Missing or empty = IoU 0. `--ply <file>` scores your own 3DGS reconstruction.

## Metric

The predicted subset is rendered inside the complete scene (selected Gaussians 1, others 0, composited with the original
opacities); a pixel is foreground when membership share and accumulated opacity both exceed 0.5. Ground truth is the
first-hit ray tracing of the mesh. IoU is averaged over the test frames with at least 300 visible target pixels, then over
queries; Acc@0.25 / Acc@0.5 are the fractions of queries above the threshold (track A in the scorer output).

Commands: see the root README. License: CC BY-NC-SA 4.0.
