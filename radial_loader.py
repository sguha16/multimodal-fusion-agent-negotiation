# -*- coding: utf-8 -*-
"""
radial_loader.py
Loads one synchronized frame from RADIal dataset.
Serves radar and camera data at all three fusion levels.

Usage:
    loader = RadialLoader(seq_path, calib_path, labels_path)
    frame = loader.get_frame(index=0)
    frame["radar"]["early"]   → radar cube (512, 256, 16)
    frame["radar"]["mid"]     → point cloud (N, 4): range_m, doppler_bin, azimuth_rad, elevation_rad
    frame["radar"]["late"]    → object list from labels
    frame["camera"]["early"]  → RGB image (1080, 1920, 3)
    frame["camera"]["late"]   → bounding box list from labels
    frame["lidar"]            → LiDAR point cloud (ground truth)
    frame["timestamp"]        → sync timestamp
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import csv
import numpy as np


class RadialLoader:
    def __init__(self, seq_path, calib_path, labels_path, radial_code_path=None, max_frames=50, corruption_module=None):
        """
        Parameters:
        -----------
        seq_path       : path to RECORD@... folder
        calib_path     : path to CalibrationTable.npy
        labels_path    : path to labels_CVPR.csv
        radial_code_path : path to cloned RADIal repo (for imports)
        max_frames     : max frames to cache (default 50, prevents memory issues)
        """
        self.seq_path = seq_path
        self.calib_path = calib_path

        # Add RADIal code to path if provided
        if radial_code_path:
            sys.path.insert(0, radial_code_path)

        # Import RADIal libraries
        from DBReader.DBReader import SyncReader
        from SignalProcessing.rpl import RadarSignalProcessing

        # Initialize reader
        print("Loading sequence...")
        self.reader = SyncReader(seq_path, tolerance=20000)

        # Cache frames — limited to avoid memory issues
        print(f"Caching frames (max {max_frames})...")
        self.frames = []
        for sample in self.reader:
            self.frames.append(sample)
            if len(self.frames) >= max_frames:
                break
        print(f"Cached frames: {len(self.frames)}")

        # Initialize signal processing — one instance per method
        print("Initializing signal processing...")
        self.rsp_rd = RadarSignalProcessing(calib_path, method='RD', device='cpu', lib='CuPy')
        self.rsp_pc = RadarSignalProcessing(calib_path, method='PC', device='cpu', lib='CuPy')

        # Optional corruption module — corrupts raw data before processing
        self.corruption_module = corruption_module

        # Load labels for this sequence
        self.seq_name = os.path.basename(seq_path)
        self.labels = self._load_labels(labels_path)
        print(f"Labels loaded: {sum(len(v) for v in self.labels.values())} labels across {len(self.labels)} frames")

    def _load_labels(self, labels_path):
        """Load labels_CVPR.csv, filter to this sequence, group by frame index."""
        labels_by_frame = {}
        with open(labels_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['dataset'] != self.seq_name:
                    continue
                idx = int(row['index'])
                if idx not in labels_by_frame:
                    labels_by_frame[idx] = []
                labels_by_frame[idx].append({
                    "bbox": [int(row['x1_pix']), int(row['y1_pix']),
                             int(row['x2_pix']), int(row['y2_pix'])],
                    "radar_X_m": float(row['radar_X_m']),
                    "radar_Y_m": float(row['radar_Y_m']),
                    "radar_R_m": float(row['radar_R_m']),
                    "radar_A_deg": float(row['radar_A_deg']),
                    "radar_D_mps": float(row['radar_D_mps']),
                    "radar_P_db": float(row['radar_P_db']),
                    "laser_X_m": float(row['laser_X_m']),
                    "laser_Y_m": float(row['laser_Y_m']),
                    "annotation": row['Annotation'],
                    "difficult": int(float(row['Difficult'])),
                })
        return labels_by_frame

    def get_frame_count(self):
        return len(self.frames)

    def get_frame(self, index=0):
        """
        Returns all data for one synchronized frame at all fusion levels.
        """
        if index >= len(self.frames):
            raise IndexError(f"Frame {index} out of range (cached: {len(self.frames)})")

        sample = self.frames[index]

        # ----- RADAR -----
        adc0 = sample['radar_ch0']['data']
        adc1 = sample['radar_ch1']['data']
        adc2 = sample['radar_ch2']['data']
        adc3 = sample['radar_ch3']['data']

        # Corrupt raw ADC before signal processing (if enabled)
        if self.corruption_module is not None:
            adc0, adc1, adc2, adc3 = self.corruption_module.corrupt_radar_adc(
                adc0, adc1, adc2, adc3
            )

        # Early: Range-Doppler-Antenna cube
        radar_cube = self.rsp_rd.run(adc0, adc1, adc2, adc3)

        # Mid: Point cloud from CFAR
        point_cloud = self.rsp_pc.run(adc0, adc1, adc2, adc3)

        # Late: Object list from labels (ground truth)
        frame_labels = self.labels.get(index, [])
        
        # ----- CAMERA -----
        camera_rgb = sample['camera']['data']

        # Corrupt raw image before YOLO (if enabled)
        if self.corruption_module is not None:
            camera_rgb = self.corruption_module.corrupt_camera_image(camera_rgb)


        # ----- LIDAR (ground truth) -----
        lidar = sample.get('scala', {}).get('data', None)

        # ----- TIMESTAMP -----
        timestamp = sample['radar_ch0'].get('timestamp', 0)

        return {
            "radar": {
                "early": radar_cube,        # (512, 256, 16) complex
                "mid": point_cloud,          # (N, 4) float: range, doppler, azimuth, elevation
            },
            "camera": {
                "early": camera_rgb,         # (1080, 1920, 3) uint8
            },
            "ground_truth": frame_labels,    # add this line (labels for evaluation only)
            "lidar": lidar,
            "timestamp": timestamp,
            "frame_index": index,
            "has_labels": len(frame_labels) > 0,
        }
