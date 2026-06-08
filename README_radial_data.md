# RADIal Data Loader — README

## Dataset: RADIal (Radar, Lidar et al.)
- **Source:** Valeo, CVPR 2022
- **Sequence used:** `RECORD@2020-11-21_11.54.31`
- **Duration:** ~45 seconds of driving
- **Total synchronized frames:** 965
- **Labelled frames:** 412 (753 vehicle labels total)

---

## What is a frame?

One frame = one synchronized snapshot across all sensors at the same moment in time.

```
Frame 0 (timestamp: 4523631)
├── Radar: 4 chips × 4 Rx × 256 chirps × 512 ADC samples (raw complex voltages)
├── Camera: 1080 × 1920 × 3 RGB image
├── LiDAR: ~13,000 points × 11 features
├── CAN: vehicle speed, steering, yaw
└── GPS: lat, lon
```

---

## Radar signal processing chain (inside one frame)

```
Raw ADC (4 × 1,048,576 int16 samples)
   │
   ├── Build complex frame: I + jQ → (512 samples, 256 chirps, 16 Rx)
   │
   ├── Range FFT (across 512 samples)     → HOW FAR
   │
   ├── Doppler FFT (across 256 chirps)    → HOW FAST
   │
   ├── [Output: Radar Cube (512, 256, 16) complex]  ← EARLY FUSION LEVEL
   │
   ├── CFAR detection (adaptive threshold)
   │
   ├── Angle estimation (calibration matrix × MIMO spectrum)
   │
   ├── [Output: Point Cloud (N, 4) float]            ← MID FUSION LEVEL
   │       columns: range_m, doppler_bin, azimuth_rad, elevation_rad
   │
   ├── Clustering (DBSCAN, future)
   │
   └── [Output: Object List]                          ← LATE FUSION LEVEL
           from labels_CVPR.csv for now
           future: derived from point cloud clustering
```

---

## Data at each fusion level

### Radar

| Level | Data | Shape | Source |
|-------|------|-------|--------|
| Early | Radar cube (Range × Doppler × Antenna) | (512, 256, 16) complex128 | `rpl.py` method='RD' |
| Mid   | Point cloud (CFAR detections) | (N, 4) float64 | `rpl.py` method='PC' |
| Late  | Object list (labelled vehicles) | list of dicts | `labels_CVPR.csv` |

**Point cloud columns:**
- `range_m` — distance to detection (0–103m)
- `doppler_bin` — velocity bin (0–255)
- `azimuth_rad` — horizontal angle (radians)
- `elevation_rad` — vertical angle (radians)

**Object list fields:**
- `x_m, y_m` — cartesian position in radar frame
- `range_m` — radial distance
- `azimuth_deg` — angle in degrees
- `doppler_mps` — velocity in m/s
- `power_db` — reflection power
- `annotation` — strong / weak / incomplete / FP

### Camera

| Level | Data | Shape | Source |
|-------|------|-------|--------|
| Early | Raw RGB image | (1080, 1920, 3) uint8 | `camera.mjpg` via DBReader |
| Mid   | CNN feature map | — | Future (requires backbone) |
| Late  | Bounding box list | list of dicts | `labels_CVPR.csv` |

**Bounding box fields:**
- `bbox` — [x1_pix, y1_pix, x2_pix, y2_pix] in image coordinates
- `annotation` — strong / weak / incomplete / FP

### LiDAR (ground truth reference)

| Data | Shape | Source |
|------|-------|--------|
| Point cloud | (~13000, 11) float | `scala.bin` via DBReader |

---

## Loader API

```python
from radial_loader import RadialLoader

loader = RadialLoader(
    seq_path="path/to/RECORD@...",
    calib_path="path/to/CalibrationTable.npy",
    labels_path="path/to/labels_CVPR.csv",
    radial_code_path="path/to/RADIal_code"
)

frame = loader.get_frame(index=0)

# Radar
frame["radar"]["early"]   # (512, 256, 16) complex — radar cube
frame["radar"]["mid"]     # (N, 4) float — point cloud
frame["radar"]["late"]    # list of dicts — object list

# Camera
frame["camera"]["early"]  # (1080, 1920, 3) uint8 — RGB image
frame["camera"]["late"]   # list of dicts — bounding boxes

# Metadata
frame["lidar"]            # (~13000, 11) float — LiDAR point cloud
frame["timestamp"]        # int — sync timestamp
frame["frame_index"]      # int — frame number
frame["has_labels"]       # bool — whether ground truth exists for this frame
```

---

## File structure

```
project/
├── radial_loader.py          ← data loader
├── radar_agent.py            ← radar perception agent
├── camera_agent.py           ← camera perception agent
├── fusion_graph.py           ← LangGraph fusion pipeline
├── labels_CVPR.csv           ← ground truth labels
└── understand_radial.py      ← exploration script
```

---

## Hardware in the RADIal dataset

| Sensor | Model | Key specs |
|--------|-------|-----------|
| Radar  | HD radar, 12 Tx × 16 Rx = 192 virtual antennas | 103m range, ±37° azimuth, ~0.4m range resolution |
| Camera | 5 Mpix RGB | 1920 × 1080, behind windshield |
| LiDAR  | 16-layer laser scanner | ~13,000 points per scan |

The three sensors point forward in the driving direction. Extrinsic calibration is provided with the dataset.

---

## Labels schema (labels_CVPR.csv)

| Column | Description |
|--------|-------------|
| numSample | Sync ID across sensors |
| x1_pix, y1_pix, x2_pix, y2_pix | Camera bounding box (pixels) |
| laser_X_m, laser_Y_m | LiDAR position (ground truth) |
| radar_X_m, radar_Y_m | Radar cartesian position (meters) |
| radar_R_m | Radar range (meters) |
| radar_A_deg | Radar azimuth (degrees) |
| radar_D_mps | Radar Doppler / velocity (m/s) |
| radar_P_db | Radar reflection power |
| dataset | Sequence name |
| index | Frame index within sequence |
| Annotation | strong / weak / incomplete / FP |
| Difficult | 0 or 1 |

**-1 in any field means no detection for that sensor.**

---

## What is NOT yet implemented

- Camera mid-level (CNN feature extraction)
- Radar late-level from data (clustering point cloud → objects, currently uses labels)
- Radar features in Stage B (CNN on radar cube)
- Health estimation from real sensor data (currently rule-based)
