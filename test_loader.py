# -*- coding: utf-8 -*-
"""
test_loader.py
Quick test: read one frame from RADIal, print shapes at all levels.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from radial_loader import RadialLoader

SEQ = r"C:\Users\sanhi\Downloads\RECORD@2020-11-21_11.54.31"
CALIB = r"C:\Users\sanhi\RADIal_code\SignalProcessing\CalibrationTable.npy"
LABELS = r"C:\Users\sanhi\Downloads\labels_CVPR.csv"
RADIAL_CODE = r"C:\Users\sanhi\RADIal_code"

loader = RadialLoader(SEQ, CALIB, LABELS, radial_code_path=RADIAL_CODE)

frame = loader.get_frame(index=0)

print("\n=== FRAME 0 ===")
print(f"Timestamp: {frame['timestamp']}")
print(f"Has labels: {frame['has_labels']}")

print(f"\n--- RADAR ---")
print(f"Early (radar cube):  shape={frame['radar']['early'].shape}, dtype={frame['radar']['early'].dtype}")
print(f"Mid (point cloud):   shape={frame['radar']['mid'].shape}, dtype={frame['radar']['mid'].dtype}")
print(f"Late (object list):  {len(frame['radar']['late'])} objects")
for obj in frame['radar']['late']:
    print(f"  id={obj['id']} range={obj['range_m']:.1f}m azimuth={obj['azimuth_deg']:.1f}deg doppler={obj['doppler_mps']:.1f}mps")

print(f"\n--- CAMERA ---")
print(f"Early (RGB image):   shape={frame['camera']['early'].shape}, dtype={frame['camera']['early'].dtype}")
print(f"Late (bbox list):    {len(frame['camera']['late'])} objects")
for obj in frame['camera']['late']:
    print(f"  id={obj['id']} bbox={obj['bbox']}")

if frame['lidar'] is not None:
    print(f"\n--- LIDAR ---")
    print(f"Point cloud: shape={frame['lidar'].shape}")

print(f"\nTotal frames available: {loader.get_frame_count()}")
