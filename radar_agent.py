# -*- coding: utf-8 -*-
"""
radar_agent.py
Radar perception agent with real RADIal data.

Stage A: signal processing chain (data from radial_loader)
    Output: data dict with early/mid/late + health dict
Stage B: brain (unchanged from original)
    Output: confidence, self_assessment, features
Report: common output structure for fusion agent
"""
import time
import numpy as np
from sklearn.cluster import DBSCAN


class RadarAgent:
    def __init__(self, memory_size=10):
        self.memory = []
        self.memory_size = memory_size
        self._misalignment_estimate = 0.0  # running estimate in degrees
        self._clean_power_baseline = None  # static baseline from first clean frame, never adapts

    def get_report(self, frame_data):
        """
        Entry point. Runs A then B, assembles report, stores in memory.
        
        Parameters:
        -----------
        frame_data : dict from radial_loader with keys "early", "mid", "late"
            frame_data["early"] = radar cube (512, 256, 16) complex
            frame_data["mid"]   = point cloud (N, 4) float
            frame_data["late"]  = object list from labels
        """
        # STAGE A — signal processing chain
        data, health = self._signal_chain(frame_data)

        # STAGE B — brain: interpret A output, compute confidence, self-assess
        confidence, self_assessment, features = self._brain(data, health)

        # REPORT — structured output to fusion agent
        report = {
            "modality": "radar",
            "timestamp": time.time(),
            "data": data,
            "health": health,
            "confidence": confidence,
            "self_assessment": self_assessment,
            "features": features,
            "trend": self.confidence_trend()
        }

        self._store(report)
        return report

    # -------------------------------------------------------------------------
    # STAGE A — signal processing chain
    # Input: real RADIal data from radial_loader
    # Output: data dict with all three fusion levels + health dict
    # -------------------------------------------------------------------------
    def _signal_chain(self, frame_data):

        # --- Early: radar cube (already computed by loader via rpl.py) ---
        radar_cube = frame_data["early"]  # (512, 256, 16) complex

        # --- Mid: point cloud (already computed by loader via rpl.py CFAR) ---
        point_cloud_raw = frame_data["mid"]  # (N, 4): range, doppler_bin, azimuth_rad, elevation_rad
        point_cloud = []
        for i in range(point_cloud_raw.shape[0]):
            point_cloud.append({
                "id": i,
                "range_m": float(point_cloud_raw[i, 0]),
                "doppler_bin": float(point_cloud_raw[i, 1]),
                "azimuth_rad": float(point_cloud_raw[i, 2]),
                "elevation_rad": float(point_cloud_raw[i, 3]),
            })

        # --- Late: DBSCAN clustering on point cloud → object list ---
        object_list = self._cluster_to_objects(point_cloud_raw)

        data = {
            "early": radar_cube,
            "mid": point_cloud,
            "late": object_list,  # late — also used by Stage B for classification
        }

        # --- Health: derived from data characteristics ---
        # future: replace with real hardware diagnostics
        health = self._assess_health(radar_cube, point_cloud_raw)

        return data, health
    def _cluster_to_objects(self, point_cloud_raw):
        """
        DBSCAN clustering on point cloud to produce object list.
        Converts polar (range, azimuth) to cartesian (x, y) for distance.
        Points close in space AND similar velocity → same object.
        """
        if point_cloud_raw.shape[0] == 0:
            return []

        # Pre-filter: remove noise before clustering
        point_cloud_raw = self._prefilter(point_cloud_raw)
        
        if point_cloud_raw.shape[0] == 0:
            return []

        ranges = point_cloud_raw[:, 0]
        doppler = point_cloud_raw[:, 1]
        azimuth = point_cloud_raw[:, 2]

        # Polar to cartesian
        x = ranges * np.cos(azimuth)
        y = ranges * np.sin(azimuth)

        # Scale doppler to similar range as spatial coords
        doppler_scaled = doppler * (103.0 / 256.0)

        # Cluster on (x, y, doppler)
        features = np.column_stack([x, y, doppler_scaled])
        clustering = DBSCAN(eps=3.0, min_samples=3).fit(features)
        labels = clustering.labels_

        # Build object list from clusters (ignore noise = -1)
        unique_labels = set(labels)
        unique_labels.discard(-1)

        object_list = []
        for obj_id, label in enumerate(unique_labels):
            mask = labels == label
            cluster_points = point_cloud_raw[mask]

            mean_range = float(np.mean(cluster_points[:, 0]))
            mean_doppler = float(np.mean(cluster_points[:, 1]))
            mean_azimuth = float(np.mean(cluster_points[:, 2]))

            obj_x = float(mean_range * np.cos(mean_azimuth))
            obj_y = float(mean_range * np.sin(mean_azimuth))

            # Doppler bin to velocity: center bin 128 = 0 velocity
            # New: wrap around 256, bins near 0 = stationary
            if mean_doppler > 128:
                doppler_centered = mean_doppler - 256.0  # 255 → -1, 254 → -2
            else:
                doppler_centered = mean_doppler           # 0 → 0, 1 → 1
            velocity_mps = float(doppler_centered)        # bin units for now, not m/s

            # Per-cluster stats for classification
            cluster_doppler = cluster_points[:, 1].copy()
            cluster_doppler[cluster_doppler > 128] -= 256
            dop_std = float(np.std(cluster_doppler))
            range_spread = float(np.std(cluster_points[:, 0]))
            az_spread = float(np.std(cluster_points[:, 2]))

            object_list.append({
                "id": obj_id,
                "x_m": round(obj_x, 2),
                "y_m": round(obj_y, 2),
                "range_m": round(mean_range, 2),
                "azimuth_deg": round(float(np.degrees(mean_azimuth)), 2),
                "velocity_mps": round(velocity_mps, 2),
                "n_points": int(np.sum(mask)),
                "doppler_std": round(dop_std, 2),
                "range_spread": round(range_spread, 2),
                "azimuth_spread_deg": round(float(np.degrees(az_spread)), 2),
                "class": None,
            })

        return object_list
    
    def _prefilter(self, point_cloud_raw):
        """
        Remove obvious non-target detections before clustering.
        - Range < 2m: ego-vehicle reflections
        - Range == 0: invalid detections
        - Azimuth beyond ±35 deg: outside useful FOV
        """
        ranges = point_cloud_raw[:, 0]
        azimuth = point_cloud_raw[:, 2]

        mask = (
            (ranges > 2.0) &                          # remove ego reflections
            (ranges < 100.0) &                         # remove out-of-range
            (np.abs(azimuth) < np.radians(35))         # remove outside FOV
        )

        filtered = point_cloud_raw[mask]
        return filtered

    def _assess_health(self, radar_cube, point_cloud):
        power = np.mean(np.abs(radar_cube) ** 2)
        
        # Dynamic threshold: compare to own history
        if len(self.memory) >= 3:
            historical_power = [r['health']['mean_power'] for r in self.memory]
            baseline = np.median(historical_power)
            if power > baseline * 5:
                dynamic_interference = "high"
            elif power > baseline * 2:
                dynamic_interference = "medium"
            else:
                dynamic_interference = "low"
        else:
            dynamic_interference = "low"

        # Absolute threshold: static baseline from first frame, never adapts.
        # Catches sustained interference that the dynamic check (which adapts) misses.
        if self._clean_power_baseline is None and len(self.memory) >= 1:
            first_power = self.memory[0]['health']['mean_power']
            if first_power > 0:
                self._clean_power_baseline = first_power

        if self._clean_power_baseline is not None:
            ratio = power / self._clean_power_baseline
            if ratio > 5:
                absolute_interference = "high"
            elif ratio > 2:
                absolute_interference = "medium"
            else:
                absolute_interference = "low"
        else:
            absolute_interference = "low"

        # Combine: take the more severe of dynamic and absolute
        severity_order = {"low": 0, "medium": 1, "high": 2}
        if severity_order.get(absolute_interference, 0) > severity_order.get(dynamic_interference, 0):
            interference = absolute_interference
        else:
            interference = dynamic_interference

        blockage = point_cloud.shape[0] < 10

        # Estimate misalignment from stationary-object azimuth bias
        misalignment = self._estimate_misalignment_from_data(point_cloud)
    
        return {
            "misalignment_deg": misalignment,
            "blockage": blockage,
            "interference_level": interference,
            "calibration_valid": True,
            "hw_error": False,
            "n_detections": point_cloud.shape[0],
            "mean_power": float(power),
        }

    def _estimate_misalignment_from_data(self, point_cloud):
        """
        Estimate radar misalignment angle by tracking the mean azimuth
        of all valid detections over time. In forward-facing automotive,
        the azimuth distribution should be centered near 0°.
        A biased mean indicates the radar is physically rotated.
        Uses exponential moving average across frames.
        """
        if point_cloud.shape[0] < 20:
            return round(self._misalignment_estimate, 1)

        azimuths = point_cloud[:, 2]
        # Remove outliers: only consider detections within ±60 deg FOV
        valid_mask = np.abs(azimuths) < np.radians(60)
        valid_az = azimuths[valid_mask]

        if len(valid_az) < 10:
            return round(self._misalignment_estimate, 1)

        # Mean absolute azimuth — a biased distribution indicates misalignment
        mean_az_deg = abs(float(np.degrees(np.mean(valid_az))))

        # Exponential moving average across frames
        alpha = 0.3
        self._misalignment_estimate = (
            alpha * mean_az_deg + (1 - alpha) * self._misalignment_estimate
        )

        return round(self._misalignment_estimate, 1)

    # -------------------------------------------------------------------------
    # STAGE B — brain
    # Unchanged from original: classify, confidence, self-assessment
    # future: replace with ML classifier + learned confidence model
    # -------------------------------------------------------------------------
    def _brain(self, data, health):

        # --- classify objects ---
        for obj in data["late"]:
            obj["motion_state"] = self._motion_state(obj)

        # --- compute confidence from data quality + health ---
        confidence = self._compute_confidence(data, health)

        # --- features: future CNN feature maps for mid-level fusion ---
        features = []

        # --- self assessment: physics-based capability degradation ---
        # Each capability degraded by the sensor faults that actually affect it:
        #   velocity: chirp interference creates false Doppler signatures
        #   range:    interference raises noise floor, weak targets lost
        #   angle:    systematic misalignment shift + interference noise
        #   class:    relies on Doppler + position, both corrupted by interference
        if health["hw_error"]:
            vel_assess = "unreliable, hardware error"
            rng_assess = "unreliable, hardware error"
            ang_assess = "unreliable, hardware error"
            cls_assess = "unreliable, hardware error"
        else:
            # --- velocity ---
            if health["interference_level"] in ("medium", "high"):
                vel_assess = f"degraded, interference level is {health['interference_level']}"
            else:
                vel_assess = "reliable, RF front end nominal"

            # --- range ---
            if health["blockage"]:
                rng_assess = "degraded, blockage detected"
            elif health["interference_level"] in ("medium", "high"):
                rng_assess = f"moderate, interference level is {health['interference_level']}"
            else:
                rng_assess = "reliable, RF front end nominal"

            # --- angle ---
            if health["misalignment_deg"] > 6:
                ang_assess = f"unreliable, misalignment is {health['misalignment_deg']} deg"
            elif health["misalignment_deg"] > 3:
                ang_assess = f"degraded, misalignment is {health['misalignment_deg']} deg"
            elif health["interference_level"] in ("medium", "high"):
                ang_assess = f"degraded, interference level is {health['interference_level']}"
            else:
                ang_assess = "reliable, misalignment within tolerance"

            # --- classification (not available — radar has no object-level classifier) ---
            cls_assess = "not available, radar has no object-level classification capability"

        self_assessment = {
            "velocity": vel_assess,
            "range": rng_assess,
            "angle": ang_assess,
            "class": cls_assess,
        }

        return confidence, self_assessment, features

    def _motion_state(self, obj):
        """Return 'moving' or 'stationary' based on doppler velocity."""
        if abs(obj.get("velocity_mps", 0)) < 1.0:
            return "stationary"
        return "moving"

    def _compute_confidence(self, data, health):
        """
        Computes confidence from health state and detection quality.
        future: replace with learned confidence model.
        """
        confidence = 1.0

        if health["interference_level"] == "high":
            confidence *= 0.5
        elif health["interference_level"] == "medium":
            confidence *= 0.75

        if health["misalignment_deg"] > 6.0:
            confidence *= 0.3

        if not health["calibration_valid"]:
            confidence *= 0.5

        if health["blockage"] or health["hw_error"]:
            confidence = 0.0

        if len(data["late"]) == 0:
            confidence *= 0.6

        return round(confidence, 2)

    # -------------------------------------------------------------------------
    # MEMORY
    # -------------------------------------------------------------------------
    def _store(self, report):
        self.memory.append(report)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def confidence_trend(self):
        """
        Returns 'degrading', 'improving', or 'stable' based on memory.
        Orchestrator uses this to distinguish sudden vs gradual faults.
        """
        if len(self.memory) < 2:
            return "stable"

        confidences = [r["confidence"] for r in self.memory]
        delta = confidences[-1] - confidences[0]

        if delta < -0.1:
            return "degrading"
        elif delta > 0.1:
            return "improving"
        else:
            return "stable"

    # -------------------------------------------------------------------------
    # NEGOTIATION — argue for/against disputed objects
    # -------------------------------------------------------------------------
    def argue(self, disputed_objects, opponent_arguments=None, round_num=1):
        """
        Given a list of disputed objects with raw radar data, generate structured
        claims by reasoning from actual measurements.

        Parameters
        ----------
        disputed_objects : list of dict
            Each has: object_id, range_m, n_points, velocity_mps, azimuth_deg,
            range_slice (256x16 power matrix), nearby_points (list of point dicts)
        opponent_arguments : str or None
            Camera agent's previous argument text (for round 2+)
        round_num : int (1 or 2)

        Returns
        -------
        dict: {object_id: {"claims": {...}, "text": str}}
        """
        from langchain_ollama import OllamaLLM
        if not hasattr(self, '_argue_llm'):
            self._argue_llm = OllamaLLM(model="llama3")
        llm = self._argue_llm

        # Build per-object data strings with raw measurements
        obj_lines = []
        for obj in disputed_objects:
            parts = [f"Object {obj['object_id']}: range={obj['range_m']:.1f}m, "
                     f"azimuth={obj['azimuth_deg']:.1f}deg, "
                     f"velocity={obj['velocity_mps']:.1f}mps, "
                     f"motion_state={obj.get('motion_state', 'unknown')}"]

            # Radar cube power at this range bin
            rs = obj.get("range_slice")
            if rs is not None and rs.size > 0:
                rs_mean = float(np.mean(rs))
                rs_peak = float(np.max(rs))
                rs_noise = float(np.median(rs))
                active = int(np.sum(rs > rs_noise * 2))
                total = rs.size
                parts.append(f"Cube power at this range: mean={rs_mean:.1e}, "
                             f"peak={rs_peak:.1e}, noise_floor={rs_noise:.1e}, "
                             f"active cells={active}/{total}")
                # Top active cells
                flat_idx = np.argsort(rs.ravel())[-5:]  # top 5
                top_cells = []
                for idx in reversed(flat_idx):
                    chirp = idx // rs.shape[1]
                    rx = idx % rs.shape[1]
                    top_cells.append(f"(chirp={chirp}, rx={rx}, power={rs.ravel()[idx]:.1e})")
                if top_cells:
                    parts.append("Top active cells: " + ", ".join(top_cells))

            # Nearby point cloud
            nearby = obj.get("nearby_points", [])
            if nearby:
                dop = [p.get("velocity_mps", 0) for p in nearby]
                snr = [p.get("snr", 0) for p in nearby]
                az = [p.get("azimuth_deg", 0) for p in nearby]
                dop_str = ", ".join(f"{d:.1f}" for d in dop[:10])
                snr_str = ", ".join(f"{s:.1f}" for s in snr[:10])
                az_str = ", ".join(f"{a:.1f}" for a in az[:10])
                parts.append(f"Nearby points ({len(nearby)} within 2m):")
                parts.append(f"  doppler: [{dop_str}]")
                parts.append(f"  SNR: [{snr_str}]")
                parts.append(f"  azimuth: [{az_str}]")
            else:
                parts.append("Nearby points: none")

            parts.append(f"DBSCAN cluster size: {obj.get('n_points', 0)} points")
            obj_lines.append("\n".join(parts))

        objects_str = "\n\n".join(obj_lines)

        # Build opponent context
        self_cheader = "=== YOUR RADAR DATA (UNCHANGED ACROSS ROUNDS) ==="
        opponent_str = ""
        if opponent_arguments:
            opponent_str = (
                f"\n=== OPPONENT'S ARGUMENTS ===\n"
                f"Camera agent's previous argument:\n{opponent_arguments}\n"
            )

        # Build prompt body — different instructions per round
        if round_num == 1:
            instructions = (
                f"Disputed objects (your data):\n"
                f"{objects_str}\n\n"
                f"{opponent_str}"
                f"Round 1 of 2. For each object output a JSON object with:\n"
                f"- \"object_id\": the integer ID\n"
                f"- \"claims\": dict with boolean values for: present (object is real based on radar evidence), range_reliable (range estimate is trustworthy), velocity_reliable (doppler/velocity is trustworthy), class_is_vehicle (this looks like a vehicle, not clutter or stationary structure like a building)\n"
                f"- \"text\": brief 2-3 sentence argument embedding the actual numerical measurements from your radar data — cube peak power, active cell count, noise floor, doppler spread, cluster size, SNR values, azimuth spread, motion_state. Reference specific numbers to support each claim.\n\n"
                f"Output a JSON array of these objects, nothing else."
            )
        else:
            instructions = (
                f"{self_cheader}\n"
                f"{objects_str}\n\n"
                f"{opponent_str}"
                f"=== INSTRUCTIONS ===\n"
                f"Your data has NOT changed. Counter the camera agent's argument using your specific measurements. If the camera agent makes a valid point that your data cannot refute, you may change your claim.\n\n"
                f"Round 2 of 2. For each object output a JSON object with:\n"
                f"- \"object_id\": the integer ID\n"
                f"- \"claims\": dict with boolean values for: present, range_reliable, velocity_reliable, class_is_vehicle\n"
                f"- \"text\": brief 2-3 sentence argument embedding the actual numerical measurements from your radar data — cube peak power, active cell count, noise floor, doppler spread, cluster size, SNR values, azimuth spread, motion_state. Reference specific numbers to support each claim.\n\n"
                f"Output a JSON array of these objects, nothing else."
            )

        prompt = (
            f"You are the radar sensor agent in an autonomous driving fusion system. "
            f"You are in a negotiation with the camera agent about whether certain "
            f"radar-detected objects are real. You must argue based ONLY on the radar data below.\n\n"
            f"{instructions}"
        )

        raw = llm.invoke(prompt).strip()
        # Parse JSON from LLM output
        import re
        import json
        try:
            json_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(raw)
            if not isinstance(parsed, list):
                parsed = [parsed]
        except (json.JSONDecodeError, ValueError):
            parsed = [{"object_id": o["object_id"],
                       "claims": {"present": True, "range_reliable": True,
                                    "velocity_reliable": True, "class_is_vehicle": False},
                       "text": "Unable to parse LLM output. Defaulting to assertion based on CFAR detection."
                       } for o in disputed_objects]

        # Build structured output
        result = {}
        for entry in parsed:
            oid = entry.get("object_id")
            if oid is None:
                continue
            result[oid] = {
                "claims": entry.get("claims", {}),
                "text": entry.get("text", ""),
            }

        # Ensure all disputed objects have an entry
        for obj in disputed_objects:
            oid = obj["object_id"]
            if oid not in result:
                result[oid] = {
                    "claims": {"present": True, "range_reliable": True,
                                "velocity_reliable": True, "class_is_vehicle": False},
                    "text": "No argument generated, defaulting to assertion based on CFAR detection.",
                }

        return result
