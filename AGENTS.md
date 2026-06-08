# AGENTS.md — Multi-Agent Multi-Modal Fusion System

## What This Project Is

A multi-agent sensor fusion system for autonomous driving perception. Radar and camera agents independently process raw sensor data from the RADIal dataset (Valeo, CVPR 2022), assess their own health and confidence, and publish structured reports. A fusion agent reasons over a knowledge graph to decide which data sources to combine, based on what's actually available and reliable — not by choosing a predefined fusion level.

The system runs locally: Ollama (llama3 + nomic-embed-text), LangGraph for orchestration, NetworkX for the knowledge graph.

---

## File Map

### Active (do not remove or rename)
- `radial_loader.py` — loads one RADIal sequence, runs radar signal processing via rpl.py, caches frames. Serves raw data to agents. Labels are separated as ground_truth for evaluation only.
- `radar_agent.py` — Stage A: real ADC → radar cube (early) → CFAR point cloud (mid) → DBSCAN clustering → object list (late) with per-cluster stats (doppler_std, range_spread, azimuth_spread). Stage B: motion_state (stationary/moving from doppler), confidence, self-assessment, trend.
- `camera_agent.py` — Stage A: real RGB image (early) → YOLO detection → bounding box list (late). Stage B: confidence, self-assessment, trend. Health uses Laplacian CV (std/mean of gradient magnitudes) as a noise- and blur-robust sharpness metric, compared against a running baseline via relative ratio.
- `fusion_knowledge_graph.py` — NetworkX MultiDiGraph with sensor, data, capability, and condition nodes connected by PROVIDES, DEGRADES, COMPENSATES, DERIVED_FROM edges
- `graph_query.py` — traverses knowledge graph given current sensor health, returns structured reasoning for the fusion agent. Note: has test code at the bottom that runs on import — move to separate test file if importing from other modules.
- `corruption_module.py` — corrupts raw sensor data at the earliest stage (radar ADC, camera RGB) for resilience testing. Supports Gaussian noise, blockage, interference, and misalignment for radar; Gaussian, rain, fog, and rain+fog for camera. Plugs into RadialLoader via optional `corruption_module` parameter.
- `fused_cross_modal_object_list.py` — cross-modal matching between radar and camera detections across late, mid, and early levels. Projects radar objects to camera pixels (using LiDAR-to-camera extrinsics), matches with YOLO bounding boxes, gathers evidence at each level, and composes fused confidence scores. Outputs a fused object list with evidence breakdown. **Step 6: Negotiation** — 2-round LLM debate on disputed radar-only objects. Deterministic raw data summaries (cube power stats, pixel patch metrics, evidence scores) are computed from sensor arrays and passed alongside agent text to the LLM judge. Judge uses chain-of-thought reasoning referencing actual values. Agent `argue()` methods receive raw sensor arrays (radar cube power slice, point cloud subset, camera 16×16 pixel patch) and format actual measurements for the LLM to reason from.
- `run_three.py` — standalone test script for 3 scenarios (healthy, camera rain+fog, radar interference). Imports agents + matcher + fusion graph, runs full pipeline including negotiation. Output redirection to `negotiation_report.txt` for inspection.
- `test_agents.py` — runs loader → agents → prints reports + visualization for verification. Supports CorruptionModule toggle for experiments.
- `labels_CVPR.csv` — RADIal ground truth labels (evaluation only, not agent input)

### Legacy (keep for reference, do not modify)
- `build_knowledge_base.py` — old ChromaDB vector RAG builder (replaced by knowledge graph)
- `fusion_graph.py` — old LangGraph experimentation (learning project — not part of current architecture)
- `orchestrator_agent.py` — old rule-based orchestrator
- `radar_agent_stub.py` — old hardcoded radar agent
- `camera_agent_stub.py` — old hardcoded camera agent

### Exploration / test scripts
- `test_loader.py`, `understandingRADIALdataset.py`

---

## Data Flow

```
RADIal dataset (1 sequence on disk)
    │
    ▼
radial_loader.py
    │  reads: raw ADC binary (4 chips), camera MJPEG, labels_CVPR.csv
    │  runs:  rpl.py / RadarSignalProcessing for radar cube + CFAR point cloud
    │  outputs per frame:
    │    radar{early, mid}    — radar cube + point cloud
    │    camera{early}        — RGB image
    │    ground_truth         — labels for evaluation only
    │    lidar                — LiDAR point cloud (reference)
    │
    ├──▶ radar_agent.py
    │      Stage A: receives frame["radar"] (early + mid)
    │               DBSCAN clusters point cloud → object list (late)
    │               Computes health from real radar cube power + detection count
    │               Estimates misalignment from stationary-object azimuth bias:
    │                 Filters low-doppler detections (stationary objects),
    │                 computes mean azimuth, expects ~0° for forward-facing radar.
    │                 Non-zero mean → physical rotation. EMA (α=0.3) across frames.
    │               Detects interference via two combined methods:
    │                 Dynamic: power ratio vs. adapting memory baseline (catches sudden jumps)
    │                 Absolute: power ratio vs. static clean baseline from first frame (catches sustained)
    │                 Final severity = max(dynamic, absolute)
     │      Stage B: motion_state (stationary/moving from doppler), computes confidence, self-assessment, trend
     │               Self-assessment follows physics-based dependency matrix:
     │                 interference → degrades velocity, range (moderate), angle, (class unavailable)
     │                 misalignment → degrades angle only (range/velocity unaffected)
     │                 blockage     → degrades range (velocity/angle estimates unaffected)
    │      Output: SensorReport dict
    │
    ├──▶ camera_agent.py
    │      Stage A: receives frame["camera"] (early only)
    │               YOLO detects objects → bounding box list with class + confidence (late)
    │               Computes health from real image brightness, sharpness, variance
    │      Stage B: computes confidence, self-assessment, trend
    │      Output: SensorReport dict
    │
    ▼
fused_cross_modal_object_list.py (CrossModalMatcher)
    │  Cross-modal matching, evidence scoring, and the negotiation-based fusion agent.
    │
    ├──▶ 1. Project radar objects to camera pixels
    │     Uses camera_calib.npy (LiDAR-to-camera extrinsics, verified against GT labels)
    │     Convention: pass (-y_agent, x_agent, z=0.5) to cv2.projectPoints
    │
    ├──▶ 2. Late-level matching: does projected pixel fall inside a camera bounding box?
    │
    ├──▶ 3. Mid-level evidence: radar cluster density (n_points) + Laplacian variance at patch
    │
    ├──▶ 4. Early-level evidence: cube power ratio (target/noise) + Canny edge density at patch
    │
    ├──▶ 5. Compose fused object list
    │     Confidence = mean of all available evidence scores,
    │     then multiplied by agent confidence (min of radar/camera for matched objects)
    │
    ├──▶ 6. Negotiation (Fusion Agent)
    │     For each disputed radar-only object (valid pixel, no camera match):
    │       Round 1: each agent argues from own raw data (LLM with cube power, pixel patches)
    │       Round 2: counter-argue referencing opponent's data
    │       Judge (knowledge-graph-aware LLM) evaluates:
    │         - Agent text (LLM-generated)
    │         - Deterministic raw data summaries (cube power stats, patch metrics, evidence scores)
    │         - Active knowledge graph conditions + sensor health
    │       Judge uses 5-step chain-of-thought reasoning referencing actual values
    │       → verdict: confirmed/rejected/unresolved per object
    │
    └──▶ Output: fused object list with negotiation verdicts
         Visualization colours: green=matched, yellow=radar-only, red=camera-only
```

---

## Knowledge Graph Structure (fusion_knowledge_graph.py)

Uses NetworkX MultiDiGraph (allows multiple edges between same node pair, e.g. rgb_image both PROVIDES and COMPENSATES classification).

Four node types connected by typed edges:

- **Sensor nodes** (2): radar, camera
- **Data nodes** (5): radar_cube, point_cloud, radar_object_list, rgb_image, camera_bboxes
- **Capability nodes** (6): range, velocity, angle, classification, depth, lateral_position
- **Condition nodes** (13): rain, fog, night, glare, interference, misalignment, radar_blockage, radar_hw_error, radar_calibration_error, lens_blockage, motion_blur, camera_hw_error, camera_calibration_error

Edge types:
- PRODUCES: sensor → first data node (radar → radar_cube, camera → rgb_image)
- DERIVED_FROM: data chain (point_cloud ← radar_cube, radar_object_list ← point_cloud, camera_bboxes ← rgb_image)
- PROVIDES: data → capability with quality (radar_cube → range: high, rgb_image → classification: high)
- DEGRADES: condition → data with severity (night → rgb_image: high, interference → radar_cube: medium)
- RESISTANT_TO: sensor → condition (radar → rain, fog, night, glare)
- COMPENSATES: cross-sensor compensation with reason (rgb_image → classification: "radar classification is rule-based")

Example traversal: "night DEGRADES rgb_image → classification affected → radar_object_list PROVIDES classification (low) as compensation"

---

## Graph Query (graph_query.py)

Replaces ChromaDB vector RAG. Given sensor reports, traverses the graph in 6 steps:

1. **Detect conditions**: maps health fields → active condition nodes
2. **Find degraded**: follows DEGRADES edges → which data nodes are affected
3. **Propagate via DERIVED_FROM**: if A DERIVED_FROM B and B is degraded, A is also degraded (e.g. interference → radar_cube → point_cloud → radar_object_list)
4. **Find affected capabilities**: follows PROVIDES edges from degraded data → which capabilities are lost
5. **Find reliable**: all data nodes NOT degraded
6. **Find compensations**: for each affected capability, finds reliable nodes that PROVIDE or COMPENSATE it

Output is structured text (not LLM-generated) that the fusion agent LLM uses to make its decision. Deterministic — same inputs always produce the same reasoning.

---

## Design Rules

1. **No early/mid/late decision.** The fusion agent does not categorize its strategy as "early fusion" or "mid fusion" or "late fusion." It reasons about which data sources to use based on availability and quality. The taxonomy is useful for humans describing the system — it is not the agent's decision space.

2. **Labels are ground truth, not input.** `labels_CVPR.csv` is for evaluation only. Agents produce their own detections: radar via DBSCAN clustering, camera via YOLO. The loader separates labels into `frame["ground_truth"]`.

3. **Radar is backbone sensor.** Radar carries higher default trust. When both sensors are healthy, both contribute. When camera degrades, the system leans on radar. This is encoded in the knowledge graph edge weights and RESISTANT_TO edges.

4. **Health and confidence from real data.** Radar health comes from actual radar cube power levels, detection counts, and azimuth bias estimates. Camera health comes from actual image brightness, sharpness, blockage estimates. No hardcoded values except where data can't provide the answer (calibration, hw_error).

5. **Agents are Stage A + Stage B.** Stage A = signal processing chain (data + health). Stage B = brain (classify, confidence, self-assessment, trend). Both stages must stay separate. Do not merge them.

6. **Common report structure.** Both agents output the same SensorReport format: modality, timestamp, data{early, mid, late}, health, confidence, self_assessment, features, trend. The fusion graph depends on this consistency.

7. **Safety agent is a separate certifiable layer.** When built, it must use explicit auditable rules with veto power — not learned behavior. Keep it separate from the fusion agent.

8. **LLM does reasoning with graph context; graph provides structured facts.** The knowledge graph traversal is deterministic and produces active conditions and their degradations. The LLM judge receives these facts and reasons from them. The graph ensures the LLM's reasoning is grounded in known degradation paths.
9. **Judge uses deterministic raw data, not agent paraphrases.** Pre-computed numeric summaries (cube power stats, pixel patch metrics, evidence scores) are passed alongside agent text. The COT prompt forces reference to these values — the judge's primary evidence is the deterministic data, with agent text as supplementary context.

---

## Corruption Module (corruption_module.py)

Plugs into `RadialLoader` via optional `corruption_module` parameter. Two corruption paths:

### Radar: ADC-level corruption
- Applied to raw 4-channel int16 ADC arrays *before* deinterleaving, RD transform, and CFAR
- Four corruption modes, set via `radar_corruption`:

#### Gaussian noise (`radar_corruption='gaussian'`)
- Adds complex noise to IQ samples: `adc += noise_std * signal_std * (randn + j*randn)`
- `noise_std` as fraction of signal std-dev (default 0.01, stress test uses 2.0)
- Noise cascades: ADC → cube → CFAR → point cloud → DBSCAN → health → confidence
- Dynamic interference detection compares against agent memory baseline (3+ frames)
- At `noise_std=2.0`, mean power spikes ~5x and interference triggers "medium" → confidence drops to 0.75

#### Blockage (`radar_corruption='blockage'`)
- Multiplies all 4 ADC arrays by attenuation factor `corruption_strength` in [0, 1]
- `corruption_strength=0.0` = complete blockage, `=1.0` (default) = transparent
- CFAR is extremely robust: even at 5% amplitude (strength=0.05), >99% of detections survive
- Health detects blockage when `n_detections < 10`

#### Interference (`radar_corruption='interference'`)
- Adds complex linear chirp to configurable number of chirps (default 50 out of 256)
- Chirp: `A * exp(j * π * (t - t0)^2 / (T/2)^2)` where A is relative to signal RMS
- `radar_interference_power` controls amplitude relative to signal RMS (default 3.0)
- Only affects radar cube; cascades through CFAR → point cloud → health
- Dynamic power baseline detects sudden spikes; sustained interference adapts away

#### Misalignment (`radar_corruption='misalignment'`)
- Applies progressive phase shift across 16 Rx channels *after* radar cube is formed
- Phase per channel rx: `exp(j * rx * π * sin(θ))` where θ = `radar_misalignment_deg`
- Models physical rotation of radar: d = λ/2 model at 77 GHz
- Affects angle-of-arrival estimation → **point cloud azimuths are shifted**
- Cascades: ADC → cube → *(phase shift)* → CFAR *(shifted azimuths)* → DBSCAN → health
- Health estimates misalignment from **stationary-object azimuth bias** (see radar_agent)
- Thresholds: >3° = medium deterioration, >6° = high
- At 10° injected, agent estimates ~6.1° after EMA convergence → confidence drops to 0.3

### Camera: albumentations-based rain + fog
- Applied to raw RGB image *before* YOLO detection
- Modes: `gaussian`, `rain`, `fog`, `rain_fog`
- Rain+fog drops sharpness (6.8 → 1.0), triggers motion_blur, frame_quality drops to 0.5
- YOLO detections drop (3 cars → 1 car), confidence drops to 0.51
- Knowledge graph detects `motion_blur` → camera data degraded → fusion falls back to radar

### Toggle via `test_agents.py`
```python
cm = CorruptionModule(radar_enabled=0, camera_enabled=0, radar_corruption='misalignment')
loader = RadialLoader(..., corruption_module=cm)
cm.enable_radar = 1   # toggle mid-run
cm.radar_misalignment_deg = 10.0
```

### Key findings
- Gaussian noise on camera *inflates* the Laplacian sharpness metric (noise = fake edges). A better metric would use Laplacian *variance* instead of mean — **now implemented**: Laplacian CV (std/mean), which drops for both noise and blur. Uses relative ratio against running baseline from agent memory, not hardcoded thresholds.
- Dynamic interference detection only catches *sudden* power jumps (baseline adapts to sustained noise). For sustained degradation, an absolute threshold should be added alongside the dynamic one — **now implemented**: static baseline from first clean frame, never adapts, combined via max(dynamic, absolute).
- Rain/fog realistically degrades camera health because it blurs (not sharpens) the image.
- CFAR is extremely robust: even at signal amplitude 5% of original (blockage 0.05) it still finds >99% of detections.
- Misalignment detection does not compare to clean data — it uses scene geometry (stationary objects' mean azimuth should be ~0° for forward-facing radar).
- EMA (α=0.3) converges the misalignment estimate over ~5-7 frames; single-frame estimate is noisy.

---

## What Works

- Full RADIal data pipeline: raw ADC → radar cube → CFAR point cloud (real signal processing via rpl.py)
- Camera pipeline: raw RGB from RADIal → YOLO object detection (pretrained yolov8n)
- Radar object detection: DBSCAN clustering on point cloud with pre-filtering (ego reflections, FOV limits)
- Both agents produce independent detections — no label dependency in the detection pipeline
- Both agents produce reports with real health metrics derived from actual sensor data
- Self-assessment is conditional on actual health values (not hardcoded templates)
- Radar interference detection uses dynamic baseline from agent memory
- Knowledge graph built with MultiDiGraph, queried, returns structured reasoning
- LangGraph chain runs end to end: agents → summarize → graph query → fusion decision
- LLM outputs which data sources to use (not a fusion level)
- Visualization: raw point cloud, DBSCAN clusters with ground truth overlay, camera image with YOLO bounding boxes
- **Corruption → health → graph query → fusion decision cascade confirmed working end to end** for both radar (ADC noise → interference "medium" → confidence 0.75) and camera (rain/fog → motion_blur → confidence 0.51)
- **Radar misalignment detection** from stationary-object azimuth bias: at 10° injected phase shift, agent estimates ~6.1° after EMA convergence → confidence drops to 0.3 → knowledge graph detects "misalignment (high)" → point_cloud degraded → fusion uses radar_cube + camera data instead
- **Cross-modal object matching** projects radar objects to camera pixels using LiDAR-to-camera extrinsics, matches with YOLO bboxes, gathers evidence at late/mid/early levels, outputs fused confidence scores
- **Absolute interference threshold** added alongside dynamic: static clean baseline from first frame never adapts, so sustained high power is detected across ALL frames (multi-frame test: conf=0.5 consistently, not just frame 0)
- **DERIVED_FROM propagation** in graph query: if radar_cube is degraded → point_cloud (cascade) → radar_object_list (cascade) are also flagged. Fusion correctly avoids corrupted object lists.
- **All three new radar corruption modes** (blockage, interference chirp, misalignment phase shift) confirmed cascading through CFAR → point cloud → health → confidence → knowledge graph → fusion decision
- Fusion decision correctly drops degraded sensor sources and uses only reliable ones
- **Negotiation system end-to-end**: 2-round LLM debate with raw sensor data. Radar agent reads cube power (mean/peak/active cells) + point cloud neighborhood. Camera agent reads 8×8 grayscale pixel patch + Sobel gradients + BGR colors. Both output structured claims. LLM judge evaluates each dispute per-object using knowledge graph conditions + sensor health — outputs reasoned verdict. Round 2 prompt enforces "counter with your own measurements, change only if opponent makes valid point."
- **Judge receives deterministic raw data alongside agent text**: `radar_det_summary` (cube peak, noise floor, active cells, cluster size, doppler stats, SNR) and `camera_det_summary` (gray mean/std, Sobel mean/max, BGR center, edge density) are pre-computed from arrays and formatted in the prompt as a `=== RAW SENSOR DATA (deterministic, not agent paraphrases) ===` section. Judge uses 5-step chain-of-thought reasoning that forces reference to specific values.
- **Camera noise scenario (noise_std=80)**: camera conf=0.31, sharpness=0.6, detections=0 → all 3 disputes confirmed (trust radar). Judge referenced cube_peak=1.4e+10 vs noise_floor=2.9e+07, point_score=1.000, patch_score=0.367. Fusion graph correctly dropped camera sources.
- **Radar interference scenario**: radar conf=0.50, interference=high → all 3 disputes confirmed (radar evidence still strong despite degradation). Judge referenced peak=1.1e+10 vs noise=3.1e+07, doppler=-1.0mps indicating motion. Fusion graph correctly dropped radar sources for overall fusion decision.
- **Radar motion_state** — radar object list includes `motion_state` per object (stationary/moving from doppler), available in negotiation data for the LLM to reference.
- **Per-cluster stats** — DBSCAN objects now include `doppler_std`, `range_spread`, `azimuth_spread_deg` computed from the raw point cloud, available as features for future classifiers.

## What's Stubbed or Missing

- **Concede mechanism** — agents should be able to concede when opponent's evidence is clearly stronger. Not yet implemented — both agents always contest every object.
- **Camera mid-level** — no CNN feature maps yet. Camera only has early (RGB) and late (YOLO bboxes).
- **Extract + fuse nodes** — fusion_graph decides which sources to use but never actually combines the data.
- **Safety agent** — designed but not implemented. Should have veto power with certifiable rules.
- **Adversarial agent** — designed but not implemented. Should corrupt raw data to test cascade.
- **Radar velocity conversion** — doppler bin to m/s needs RADIal chirp parameters for exact conversion. Currently uses bin-centered approximation.
- **Some health fields are stubbed** — calibration_valid (always True), hw_error (always False), occlusion_level (always "low"). These can't be derived from data alone.
- **DBSCAN tuning** — produces ~32 clusters per frame. Many are real static objects (parked cars, buildings). May need further refinement but not necessarily wrong.
- **Health metric distinguishes noise vs blur** — Laplacian CV drops for both Gaussian noise and motion blur, triggering `motion_blur` condition for either. This is safe (camera treated as degraded in both cases) but semantically incorrect.
- **Fusion graph LLM prompt parsing** — llama3 sometimes prefixes strategy output with "here is my response:" instead of clean comma-separated sources. Needs more explicit prompt formatting.
- **Radar has no object-level classification** — RADIal labels have no object classes (only detection quality flags: strong/weak/FP). Radar outputs `motion_state` (stationary/moving) and raw cluster stats instead. No "YOLO equivalent" exists for radar without proper training data.

## Next Steps

1. ~~**Agent disagreement detection + resolution** — radar sees 32 objects, camera sees 3 cars. Fusion agent must match and resolve conflicts using knowledge graph.~~ **Done via negotiation.**
2. ~~**Deterministic claim-counting judge replaced** — LLM judge now evaluates per-object with graph-aware reasoning and COT from raw data.~~ **Done.**
3. **Concede mechanism** — agents that lack evidence concede disputed objects instead of always contesting.
4. **Safety agent** — monitors fusion decisions, flags unresolved disagreements, veto power.
5. **Adversarial agent** — generates systematic corruption scenarios to find failure modes.
6. **Extract + fuse nodes** — actually combine sensor data based on the fusion decision.
7. **Fix fusion graph LLM preamble issue** — enforce strict 2-line output format.

---

## Environment

- Windows, Anaconda (fusion_env), Spyder IDE
- Ollama running locally: llama3, nomic-embed-text
- RADIal code cloned to C:\Users\sanhi\RADIal_code
- RADIal sequence: C:\Users\sanhi\Downloads\RECORD@2020-11-21_11.54.31
- Key libraries: numpy, scipy, matplotlib, opencv-python, mkl-fft, langchain, langchain-ollama, langgraph, networkx, scikit-learn, cantools, ultralytics (YOLO), albumentations
- Note: os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' required at top of scripts using mkl-fft (OpenMP conflict between mkl-fft and numpy)

---

## How To Work On This Project

- Always provide complete updated files, not partial diffs or instructions describing changes.
- Minimal targeted changes — do not refactor working code unnecessarily.
- Understand the data flow before editing. Changes to the loader affect both agents. Changes to report structure affect fusion_graph.
- Test with test_agents.py after any agent or loader change.
- Ollama must be running before executing fusion_graph.py.
- When adding a new sensor, add nodes + edges to the knowledge graph, create a new agent with Stage A + Stage B, and ensure it outputs SensorReport format.
