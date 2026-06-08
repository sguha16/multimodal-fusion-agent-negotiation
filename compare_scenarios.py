"""
compare_scenarios.py
Runs the full pipeline across multiple frames and scenarios,
saving cross-modal fused visualizations for comparison.

Usage:
    python compare_scenarios.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from radial_loader import RadialLoader
from radar_agent import RadarAgent
from camera_agent import CameraAgent
from corruption_module import CorruptionModule
from fused_cross_modal_object_list import CrossModalMatcher

SEQ = r"C:\Users\sanhi\Downloads\RECORD@2020-11-21_11.54.31"
CALIB = r"C:\Users\sanhi\RADIal_code\SignalProcessing\CalibrationTable.npy"
LABELS = r"C:\Users\sanhi\Downloads\labels_CVPR.csv"
RADIAL_CODE = r"C:\Users\sanhi\RADIal_code"
CAMERA_CALIB = r"C:\Users\sanhi\RADIal_code\DBReader\examples\camera_calib.npy"


def build_scenario_name(label, frame_idx):
    return f"{label}_frame{frame_idx}".replace(" ", "_").lower()


def run_scenario(label, radar_corruption, camera_corruption,
                 enable_radar, enable_camera, loader, matcher,
                 frame_indices=(0,), radar_interference_power=10.0):
    """
    Run one scenario across multiple frames.
    Returns list of (frame_idx, fused_objects) tuples.
    """
    print("\n" + "=" * 70)
    print(f"SCENARIO: {label}")
    print("=" * 70)

    # Reset corruption module
    loader.corruption_module.enable_radar = 0
    loader.corruption_module.enable_camera = 0
    loader.corruption_module.radar_corruption = radar_corruption
    loader.corruption_module.camera_corruption = camera_corruption
    if radar_corruption == 'interference':
        loader.corruption_module.radar_interference_power = radar_interference_power
    if radar_corruption == 'misalignment':
        loader.corruption_module.radar_misalignment_deg = 10.0

    # Fresh agents for each scenario (clean memory)
    radar_agent = RadarAgent(memory_size=10)
    camera_agent = CameraAgent(memory_size=10)

    # Warm-up: 3 clean frames (no corruption) for power baselines
    for i in range(3):
        fw = loader.get_frame(index=i)
        radar_agent.get_report(fw["radar"])
        camera_agent.get_report(fw["camera"])
    print(f"  [Warm-up complete — 3 clean frames]")

    # Enable corruption for this scenario
    if enable_radar:
        loader.corruption_module.enable_radar = 1
        print(f"  [Radar corruption: {radar_corruption}]")
    if enable_camera:
        loader.corruption_module.enable_camera = 1
        print(f"  [Camera corruption: {camera_corruption}]")

    results = []
    for idx in frame_indices:
        frame = loader.get_frame(index=idx)
        radar = radar_agent.get_report(frame["radar"])
        camera = camera_agent.get_report(frame["camera"])
        fused = matcher.fuse(radar, camera)

        # Save visualization
        save_name = build_scenario_name(label, idx)
        save_path = f"fused_{save_name}.png"
        matcher.visualize(frame["camera"]["early"], fused, save_path)

        # Print summary
        matched = sum(1 for o in fused if o["source"] == "matched")
        radar_only = sum(1 for o in fused if o["source"] == "radar")
        cam_only = sum(1 for o in fused if o["source"] == "camera")
        high_conf = sum(1 for o in fused if o["confidence"] > 0.5)
        mid_conf = sum(1 for o in fused if 0.3 < o["confidence"] <= 0.5)
        low_conf = sum(1 for o in fused if o["confidence"] <= 0.3)
        confs = [o["confidence"] for o in fused if o["source"] != "camera"]

        print(f"\n  Frame {idx}:")
        print(f"    GT objects: {len(frame['ground_truth'])}")
        print(f"    Radar objects: {len(radar['data']['late'])}")
        print(f"    Camera detections: {len(camera['data']['late'])}")
        print(f"    Fused objects: {len(fused)} ({matched} matched, "
              f"{radar_only} radar-only, {cam_only} camera-only)")
        print(f"    Confidence bands: >0.5={high_conf}  0.3-0.5={mid_conf}  <0.3={low_conf}")
        if confs:
            print(f"    Radar-origin conf: min={min(confs):.3f}  "
                  f"max={max(confs):.3f}  mean={sum(confs)/len(confs):.3f}")
        if matched > 0:
            match_confs = [o["confidence"] for o in fused if o["source"] == "matched"]
            print(f"    Matched confs: {[round(c, 3) for c in sorted(match_confs, reverse=True)]}")
        print(f"    Agent health: radar_conf={radar['confidence']} "
              f"camera_conf={camera['confidence']} "
              f"radar_trend={radar['trend']}")

        results.append((idx, fused))
        print(f"    Saved: {save_path}")

    # Return raw reports too for scenario-level summary
    return results


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    # Shared corruption module + loader (expensive init, do once)
    cm = CorruptionModule(radar_enabled=0, camera_enabled=0,
                          radar_corruption='interference')
    cm.radar_interference_power = 10.0

    loader = RadialLoader(SEQ, CALIB, LABELS,
                          radial_code_path=RADIAL_CODE,
                          corruption_module=cm)
    matcher = CrossModalMatcher(CAMERA_CALIB)

    # =====================================================================
    # PART 1: Same scenario across DIFFERENT FRAMES
    # =====================================================================
    print("\n\n")
    print("#" * 70)
    print("# PART 1: NO DEGRADATION — ACROSS 5 FRAMES")
    print("#" * 70)

    run_scenario(
        label="no_degradation",
        radar_corruption='gaussian',
        camera_corruption='rain_fog',
        enable_radar=False, enable_camera=False,
        loader=loader, matcher=matcher,
        frame_indices=(0, 10, 20, 30, 40),
    )

    # =====================================================================
    # PART 2: THREE SCENARIOS — FRAME 0
    # =====================================================================
    print("\n\n")
    print("#" * 70)
    print("# PART 2: THREE SCENARIOS COMPARISON")
    print("#" * 70)

    # Scenario A: No degradation
    run_scenario(
        label="A_healthy",
        radar_corruption='gaussian',
        camera_corruption='rain_fog',
        enable_radar=False, enable_camera=False,
        loader=loader, matcher=matcher,
        frame_indices=(0,),
    )

    # Scenario B: Fog + rain on camera
    run_scenario(
        label="B_rainfog",
        radar_corruption='gaussian',
        camera_corruption='rain_fog',
        enable_radar=False, enable_camera=True,
        loader=loader, matcher=matcher,
        frame_indices=(0,),
    )

    # Scenario C: High radar interference
    run_scenario(
        label="C_interference",
        radar_corruption='interference',
        camera_corruption='rain_fog',
        enable_radar=True, enable_camera=False,
        loader=loader, matcher=matcher,
        frame_indices=(0,),
    )

    print("\n\nDone. Generated files:")
    print("  fused_no_degradation_frame0.png   fused_no_degradation_frame10.png  ...")
    print("  fused_A_healthy_frame0.png")
    print("  fused_B_rainfog_frame0.png")
    print("  fused_C_interference_frame0.png")
