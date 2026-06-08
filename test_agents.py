"""
test_agents.py
Tests radar and camera agents with real RADIal data.
Supports corruption via CorruptionModule.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
from radial_loader import RadialLoader
from radar_agent import RadarAgent
from camera_agent import CameraAgent
from corruption_module import CorruptionModule
from fusion_graph import app, FusionState
from fused_cross_modal_object_list import CrossModalMatcher


# --- Tee: print to both console and file ---
class Tee:
    def __init__(self, file):
        self.file = file
    def write(self, data):
        sys.__stdout__.write(data)
        self.file.write(data)
    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()

report_file = open("report.txt", "w", encoding="utf-8")
sys.stdout = Tee(report_file)


# --- Load data ---
SEQ = r"C:\Users\sanhi\Downloads\RECORD@2020-11-21_11.54.31"
CALIB = r"C:\Users\sanhi\RADIal_code\SignalProcessing\CalibrationTable.npy"
LABELS = r"C:\Users\sanhi\Downloads\labels_CVPR.csv"
RADIAL_CODE = r"C:\Users\sanhi\RADIal_code"
CAMERA_CALIB = r"C:\Users\sanhi\RADIal_code\DBReader\examples\camera_calib.npy"

cm = CorruptionModule(radar_enabled=0, camera_enabled=0, radar_corruption='interference')
cm.radar_interference_power = 10.0

loader = RadialLoader(SEQ, CALIB, LABELS, radial_code_path=RADIAL_CODE, corruption_module=cm)
matcher = CrossModalMatcher(CAMERA_CALIB)


def run_scenario(label, enable_radar, enable_camera, radar_type='interference', camera_type='rain_fog'):
    print("\n" + "=" * 70)
    print(f"  SCENARIO: {label}")
    print("=" * 70)

    cm.enable_radar = 0; cm.enable_camera = 0
    cm.radar_corruption = radar_type
    cm.camera_corruption = camera_type
    if radar_type == 'interference':
        cm.radar_interference_power = 10.0

    ra = RadarAgent(memory_size=10)
    ca = CameraAgent(memory_size=10)
    for i in range(3):
        fw = loader.get_frame(index=i)
        ra.get_report(fw["radar"]); ca.get_report(fw["camera"])
    cm.enable_radar = int(enable_radar); cm.enable_camera = int(enable_camera)

    frame = loader.get_frame(index=0)
    rr = ra.get_report(frame["radar"])
    cr = ca.get_report(frame["camera"])

    print(f"RADAR: conf={rr['confidence']:.2f} trend={rr['trend']}  "
          f"health: n_det={rr['health']['n_detections']} "
          f"interference={rr['health']['interference_level']} "
          f"misalignment={rr['health']['misalignment_deg']:.1f}deg")
    print(f"CAMERA: conf={cr['confidence']:.2f} trend={cr['trend']}  "
          f"health: brightness={cr['health']['brightness']:.1f} "
          f"sharpness={cr['health']['sharpness']:.1f} "
          f"quality={cr['health']['frame_quality']}")
    print(f"Radar objects: {len(rr['data']['late'])}  Camera detections: {len(cr['data']['late'])}  "
          f"Ground truth: {len(frame['ground_truth'])}")

    # Fused object list
    fused = matcher.fuse(rr, cr)
    matched = sum(1 for o in fused if o["source"] == "matched")
    radar_only = sum(1 for o in fused if o["source"] == "radar")
    cam_only = sum(1 for o in fused if o["source"] == "camera")
    confs = [o["confidence"] for o in fused if o["source"] != "camera"]
    avg_conf = sum(confs) / len(confs) if confs else 0
    print(f"Fused: {len(fused)} total ({matched} matched, {radar_only} radar, {cam_only} cam)  "
          f"mean conf={avg_conf:.3f}")
    matcher.visualize(frame["camera"]["early"], fused, f"fused_{label}.png")

    # Fusion graph
    state = {"radar_report": rr, "camera_report": cr, "situation_summary": None,
             "retrieved_knowledge": None, "fusion_strategy": None,
             "decision_explanation": None, "iteration_count": 0, "should_retry_retrieval": False}
    res = app.invoke(state)
    print(f"[FUSION DECISION]\n{res['fusion_strategy']}")
    return rr, cr, fused, res


# ===================== RUN 3 SCENARIOS =====================
r1, c1, f1, d1 = run_scenario("A_healthy",        enable_radar=0, enable_camera=0)
r2, c2, f2, d2 = run_scenario("B_camera_rainfog",  enable_radar=0, enable_camera=1)
r3, c3, f3, d3 = run_scenario("C_radar_interference", enable_radar=1, enable_camera=0)

# ===================== COMPARISON TABLE =====================
print("\n" + "=" * 70)
print("  COMPARISON SUMMARY")
print("=" * 70)
rows = [
    ("METRIC", "HEALTHY", "CAMERA RAIN+FOG", "RADAR INTERFERENCE"),
    ("Radar objects",      len(r1['data']['late']), len(r2['data']['late']), len(r3['data']['late'])),
    ("Camera detections",  len(c1['data']['late']), len(c2['data']['late']), len(c3['data']['late'])),
    ("Radar confidence",   f"{r1['confidence']:.2f}", f"{r2['confidence']:.2f}", f"{r3['confidence']:.2f}"),
    ("Camera confidence",  f"{c1['confidence']:.2f}", f"{c2['confidence']:.2f}", f"{c3['confidence']:.2f}"),
    ("Fused objects total", len(f1), len(f2), len(f3)),
    ("  Matched", sum(1 for o in f1 if o['source']=='matched'), sum(1 for o in f2 if o['source']=='matched'), sum(1 for o in f3 if o['source']=='matched')),
    ("  Radar-only", sum(1 for o in f1 if o['source']=='radar'), sum(1 for o in f2 if o['source']=='radar'), sum(1 for o in f3 if o['source']=='radar')),
    ("  Camera-only", sum(1 for o in f1 if o['source']=='camera'), sum(1 for o in f2 if o['source']=='camera'), sum(1 for o in f3 if o['source']=='camera')),
    ("Mean fused conf (radar)", np.mean([o['confidence'] for o in f1 if o['source']!='camera']), np.mean([o['confidence'] for o in f2 if o['source']!='camera']), np.mean([o['confidence'] for o in f3 if o['source']!='camera'])),
]
for cells in rows:
    print(f"{cells[0]:<25} {str(cells[1]):<22} {str(cells[2]):<22} {str(cells[3]):<22}")
print(f"\nFilenames: fused_A_healthy.png, fused_B_camera_rainfog.png, fused_C_radar_interference.png")

sys.stdout = sys.__stdout__
report_file.close()
print("\nDone.")
