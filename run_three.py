import os; os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
from radial_loader import RadialLoader
from radar_agent import RadarAgent
from camera_agent import CameraAgent
from corruption_module import CorruptionModule
from fusion_graph import app
from fused_cross_modal_object_list import CrossModalMatcher
from fusion_knowledge_graph import build_fusion_graph

SEQ = r"C:\Users\sanhi\Downloads\RECORD@2020-11-21_11.54.31"
CALIB = r"C:\Users\sanhi\RADIal_code\SignalProcessing\CalibrationTable.npy"
LABELS = r"C:\Users\sanhi\Downloads\labels_CVPR.csv"
RADIAL_CODE = r"C:\Users\sanhi\RADIal_code"
CAMERA_CALIB = r"C:\Users\sanhi\RADIal_code\DBReader\examples\camera_calib.npy"

fusion_graph_kg = build_fusion_graph()


def run_scenario(label, er, ec, corruption_kwargs):
    cm = CorruptionModule(radar_enabled=0, camera_enabled=0)
    for k, v in corruption_kwargs.items():
        setattr(cm, k, v)
    loader = RadialLoader(SEQ, CALIB, LABELS, radial_code_path=RADIAL_CODE, corruption_module=cm)
    matcher = CrossModalMatcher(CAMERA_CALIB)

    print("=" * 70)
    print(f"SCENARIO: {label}")
    print("=" * 70)

    cm.enable_radar = 0; cm.enable_camera = 0
    ra = RadarAgent(memory_size=10)
    ca = CameraAgent(memory_size=10)
    for i in range(3):
        fw = loader.get_frame(index=i)
        ra.get_report(fw["radar"]); ca.get_report(fw["camera"])

    cm.enable_radar = er; cm.enable_camera = ec
    frame = loader.get_frame(index=0)
    rr = ra.get_report(frame["radar"])
    cr = ca.get_report(frame["camera"])

    print(f"Radar: conf={rr['confidence']:.2f} trend={rr['trend']}  "
          f"interference={rr['health']['interference_level']}  "
          f"mis={rr['health']['misalignment_deg']:.1f}deg  "
          f"n_det={rr['health']['n_detections']}  "
          f"objects={len(rr['data']['late'])}")
    print(f"Camera: conf={cr['confidence']:.2f} trend={cr['trend']}  "
          f"brightness={cr['health']['brightness']:.1f}  "
          f"sharpness={cr['health']['sharpness']:.1f}  "
          f"detections={len(cr['data']['late'])}")

    fused = matcher.fuse(rr, cr, radar_agent=ra, camera_agent=ca,
                         fusion_graph=fusion_graph_kg)
    matched = sum(1 for o in fused if o['source'] == 'matched')
    radar_only = sum(1 for o in fused if o['source'] == 'radar')
    cam_only = sum(1 for o in fused if o['source'] == 'camera')
    confs_radar = [o['confidence'] for o in fused if o['source'] != 'camera']
    print(f"Fused: {len(fused)} ({matched} matched, {radar_only} radar, {cam_only} cam)  "
          f"mean conf={sum(confs_radar)/len(confs_radar):.3f}")
    matcher.visualize(frame["camera"]["early"], fused, f"fused_{label}.png")

    state = {"radar_report": rr, "camera_report": cr, "situation_summary": None,
             "retrieved_knowledge": None, "fusion_strategy": None,
             "decision_explanation": None, "iteration_count": 0, "should_retry_retrieval": False}
    result = app.invoke(state)
    print(f"\nFUSION GRAPH OUTPUT:")
    print(f"[STRATEGY]\n{result['fusion_strategy']}\n")
    print(f"[EXPLANATION]\n{result['decision_explanation']}")


scenario = sys.argv[1] if len(sys.argv) > 1 else "all"

if scenario in ("all", "camera_noise"):
    run_scenario(
        label="A_camera_noise",
        er=0, ec=1,
        corruption_kwargs={"camera_corruption": "gaussian", "camera_noise_std": 80.0},
    )

if scenario in ("all", "radar_interference"):
    run_scenario(
        label="B_radar_interference",
        er=1, ec=0,
        corruption_kwargs={"radar_corruption": "interference", "radar_interference_power": 10.0},
    )
