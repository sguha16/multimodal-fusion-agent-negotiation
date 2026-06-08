# -*- coding: utf-8 -*-
"""
camera_agent.py
Camera perception agent with real RADIal data.

Stage A: signal processing chain (data from radial_loader)
    Output: data dict with early/late + health dict
Stage B: brain (unchanged from original)
    Output: confidence, self_assessment, features
Report: common output structure for fusion agent
"""
import time
import numpy as np
import cv2


class CameraAgent:
    def __init__(self, memory_size=10):
        self.memory = []
        self.memory_size = memory_size

    def get_report(self, frame_data):
        """
        Entry point. Runs A then B, assembles report, stores in memory.
        
        Parameters:
        -----------
        frame_data : dict from radial_loader with keys "early", "late"
            frame_data["early"] = RGB image (1080, 1920, 3) uint8
            frame_data["late"]  = bounding box list from labels
        """
        # STAGE A — signal processing chain
        data, health = self._signal_chain(frame_data)

        # STAGE B — brain: interpret A output, compute confidence, self-assess
        confidence, self_assessment, features = self._brain(data, health)

        # REPORT — structured output to fusion agent
        report = {
            "modality": "camera",
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
    # Input: real RADIal camera data from radial_loader
    # Output: data dict with fusion levels + health dict
    # -------------------------------------------------------------------------
    def _signal_chain(self, frame_data):

        # --- Early: raw RGB image ---
        image = frame_data["early"]  # (1080, 1920, 3) uint8

        # --- Late: simple contour-based detection from image ---
        object_list = self._detect_objects(image)

        data = {
            "early": image,
            "mid":None,
            "late": object_list,  # late — also used by Stage B
        }

        # --- Health: derived from image characteristics ---
        health = self._assess_health(image)

        return data, health
    
    def _detect_objects(self, image):
        """
        YOLO-based object detection.
        Returns bounding boxes with class labels and confidence.
        """
        from ultralytics import YOLO
        
        if not hasattr(self, '_yolo_model'):
            self._yolo_model = YOLO("yolov8n.pt")  # loads once, reuses
        
        results = self._yolo_model(image, verbose=False)
        
        object_list = []
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            object_list.append({
                "id": i,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "class": results[0].names[int(box.cls[0])],
                "yolo_confidence": round(float(box.conf[0]), 2),
            })
        
        return object_list
    
    # def _detect_objects(self, image):
    #     """
    #     Simple contour-based object detection.
    #     Background subtraction → threshold → contours → bounding boxes.
    #     Not accurate — just provides independent camera detections.
    #     future: replace with YOLO or proper detector.
    #     """
    #     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #     # Background = heavily blurred image
    #     background = cv2.GaussianBlur(gray, (51, 51), 0)

    #     # Foreground = difference from background
    #     diff = cv2.absdiff(gray, background)

    #     # Threshold
    #     _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    #     # Cleanup
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    #     thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #     thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    #     # Find contours
    #     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     # Filter by size
    #     h, w = image.shape[:2]
    #     min_area = (h * w) * 0.002
    #     max_area = (h * w) * 0.5

    #     object_list = []
    #     for i, contour in enumerate(contours):
    #         area = cv2.contourArea(contour)
    #         if area < min_area or area > max_area:
    #             continue

    #         x, y, bw, bh = cv2.boundingRect(contour)
    #         object_list.append({
    #             "id": i,
    #             "bbox": [x, y, x + bw, y + bh],
    #             "area": int(area),
    #             "class": None,
    #         })

    #     return object_list

    def _compute_frame_quality(self, image,brightness,gray,laplacian,sharpness):
        """
        Estimates frame quality from image statistics.
        future: replace with learned quality model.
        """

        quality = 1.0
        # Penalize very dark or very bright
        if brightness < 40:
            quality *= 0.4
        elif brightness < 80:
            quality *= 0.7
        elif brightness > 220:
            quality *= 0.6

        # Penalize blur via sharpness drop vs running baseline
        baseline = np.median([r['health']['sharpness'] for r in self.memory]) if self.memory else 0
        if baseline > 0:
            ratio = sharpness / baseline
            if ratio < 0.3:
                quality *= 0.5
            elif ratio < 0.7:
                quality *= 0.75

        return round(float(quality), 2)

    def _assess_health(self, image):
        """
        Estimates camera health from actual image characteristics.
        future: replace with real hardware diagnostics.
        """
        brightness = np.mean(image)

        # Lighting assessment
        if brightness < 40:
            lighting = "night"
        elif brightness < 100:
            lighting = "poor"
        else:
            lighting = "good"

        # Lens blockage: uniform low-variance image suggests blockage
        variance = np.var(image)
        lens_blockage = variance < 100

        # Motion blur: sharpness drop relative to running baseline
        gray = np.mean(image, axis=2)
        laplacian = np.abs(np.diff(gray, axis=0)[:, :-1]) + np.abs(np.diff(gray, axis=1)[:-1, :])
        sharpness_mean = np.mean(laplacian)
        sharpness_std = np.std(laplacian)
        sharpness = sharpness_std / max(sharpness_mean, 1e-6)  # CV: high for natural images, low for noise/blur
        if len(self.memory) >= 1:
            past_sharp = [r['health']['sharpness'] for r in self.memory]
            baseline = np.median(past_sharp)
            motion_blur = (sharpness / max(baseline, 1e-6)) < 0.3
        else:
            motion_blur = False
        
        # --- Frame quality from image statistics ---
        frame_quality = self._compute_frame_quality(image,brightness,gray,laplacian,sharpness)

        return {
            "lighting": lighting,
            "occlusion_level": "low",  # cannot estimate from single frame
            "lens_blockage": lens_blockage,
            "motion_blur": motion_blur,
            "calibration_valid": True,  # assume valid unless told otherwise
            "hw_error": False,
            "brightness": round(float(brightness), 1),
            "sharpness": round(float(sharpness), 1),
            "frame_quality": frame_quality
        }

    # -------------------------------------------------------------------------
    # STAGE B — brain
    # Unchanged from original: confidence, self-assessment
    # future: replace with learned confidence model
    # -------------------------------------------------------------------------
    def _brain(self, data, health):

        # --- compute confidence from data quality + health ---
        confidence = self._compute_confidence(data, health)

        # --- features: future CNN feature maps for mid-level fusion ---
        features = []

        # --- self assessment: camera knows its own weaknesses ---
        # self_assessment = {
        #     "velocity": "unreliable, monocular camera cannot measure velocity directly",
        #     "range": f"moderate, monocular depth estimation, lighting is {health['lighting']}",
        #     "angle": "reliable, pixel space gives good angular resolution",
        #     "class": f"reliable, detection model output, occlusion is {health['occlusion_level']}",
        # }
        
        self_assessment = {
            "velocity": "unreliable, monocular camera cannot measure velocity directly",
            "range": "moderate, monocular depth estimation" if health["lighting"] == "good" else f"degraded, lighting is {health['lighting']}",
            "angle": "reliable, pixel space gives good angular resolution" if not health["motion_blur"] else "degraded, motion blur affecting pixel accuracy",
            "class": "reliable, detection model output" if health["lighting"] == "good" and health["occlusion_level"] == "low" else f"degraded, lighting={health['lighting']}, occlusion={health['occlusion_level']}",
        }

        return confidence, self_assessment, features

    def _compute_confidence(self, data, health):
        """
        Computes confidence from health state and detection quality.
        future: replace with learned confidence model.
        """
        confidence = 1.0

        if health["lighting"] == "night":
            confidence *= 0.4
        elif health["lighting"] == "poor":
            confidence *= 0.65

        if health["occlusion_level"] == "high":
            confidence *= 0.4
        elif health["occlusion_level"] == "medium":
            confidence *= 0.7

        if health["lens_blockage"] or health["hw_error"]:
            confidence = 0.0

        if health["motion_blur"]:
            confidence *= 0.6

        if not health["calibration_valid"]:
            confidence *= 0.5

        if health["frame_quality"] < 0.5:
            confidence *= 0.6
        elif health["frame_quality"] < 0.75:
            confidence *= 0.85

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
        Given a list of disputed objects with raw camera patch data, generate
        structured claims by reasoning from actual pixel measurements.

        Parameters
        ----------
        disputed_objects : list of dict
            Each has: object_id, u, v (pixel coords), patch (16x16 uint8 BGR)
        opponent_arguments : str or None
            Radar agent's previous argument text (for round 2+)
        round_num : int (1 or 2)

        Returns
        -------
        dict: {object_id: {"claims": {...}, "text": str}}
        """
        from langchain_ollama import OllamaLLM
        if not hasattr(self, '_argue_llm'):
            self._argue_llm = OllamaLLM(model="llama3")
        llm = self._argue_llm

        obj_lines = []
        for obj in disputed_objects:
            parts = [f"Object {obj['object_id']}: pixel=({obj['u']},{obj['v']})"]
            parts.append(f"YOLO at this pixel: none")

            patch = obj.get("patch")
            if patch is not None and patch.size > 0:
                # Convert to grayscale for display
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch

                # Show 8x8 center grayscale pixel values (4 rows x 8 cols)
                cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
                patch_sample = gray[max(0, cy-2):cy+2, max(0, cx-4):cx+4]
                rows_str = []
                for row_idx in range(patch_sample.shape[0]):
                    row_vals = " ".join(f"{patch_sample[row_idx, c]:3d}" for c in range(patch_sample.shape[1]))
                    rows_str.append(f"    [{row_vals}]")
                parts.append("Grayscale patch center 8x8:")
                parts.extend(rows_str)

                # Sobel gradient magnitudes
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
                mag = np.sqrt(sobelx**2 + sobely**2)
                parts.append(f"Sobel gradient: min={float(np.min(mag)):.1f}, "
                             f"max={float(np.max(mag)):.1f}, "
                             f"mean={float(np.mean(mag)):.1f}, "
                             f"std={float(np.std(mag)):.1f}")

                # Color channels at center 3x3 (only if BGR)
                if patch.ndim == 3 and patch.shape[2] == 3:
                    cy, cx = patch.shape[0] // 2, patch.shape[1] // 2
                    center = patch[max(0, cy-1):cy+2, max(0, cx-1):cx+2]
                    b_str = "; ".join(" ".join(f"{center[r,c,0]:3d}" for c in range(center.shape[1])) for r in range(center.shape[0]))
                    g_str = "; ".join(" ".join(f"{center[r,c,1]:3d}" for c in range(center.shape[1])) for r in range(center.shape[0]))
                    r_str = "; ".join(" ".join(f"{center[r,c,2]:3d}" for c in range(center.shape[1])) for r in range(center.shape[0]))
                    parts.append(f"Color B at center 3x3: [{b_str}]")
                    parts.append(f"Color G at center 3x3: [{g_str}]")
                    parts.append(f"Color R at center 3x3: [{r_str}]")
            else:
                parts.append("Patch: none (object projects outside image bounds)")

            obj_lines.append("\n".join(parts))

        objects_str = "\n\n".join(obj_lines)

        opponent_str = ""
        if opponent_arguments:
            opponent_str = (
                f"\n=== OPPONENT'S ARGUMENTS ===\n"
                f"Radar agent's previous argument:\n{opponent_arguments}\n"
            )

        # Build prompt body — different instructions per round
        self_cheader = "=== YOUR CAMERA DATA (UNCHANGED ACROSS ROUNDS) ==="
        if round_num == 1:
            instructions = (
                f"Disputed objects (your image data):\n"
                f"{objects_str}\n\n"
                f"{opponent_str}"
                f"Round 1 of 2. For each object output a JSON object with:\n"
                f"- \"object_id\": the integer ID\n"
                f"- \"claims\": dict with boolean values for: present (is there really an object visible in this image patch?), class_is_vehicle (if present, does it look like a vehicle?), texture_consistent (does the patch have edge/texture structure consistent with a real object, or is it uniform noise?)\n"
                f"- \"text\": brief 2-3 sentence argument embedding the actual numerical measurements from your camera data — grayscale pixel values, mean/std, Sobel gradient min/max/mean, BGR color values at center. Reference specific numbers to support each claim.\n\n"
                f"Output a JSON array of these objects, nothing else."
            )
        else:
            instructions = (
                f"{self_cheader}\n"
                f"{objects_str}\n\n"
                f"{opponent_str}"
                f"=== INSTRUCTIONS ===\n"
                f"Your data has NOT changed. Counter the radar agent's argument using your specific measurements. If the radar agent makes a valid point that your data cannot refute, you may change your claim.\n\n"
                f"Round 2 of 2. For each object output a JSON object with:\n"
                f"- \"object_id\": the integer ID\n"
                f"- \"claims\": dict with boolean values for: present, class_is_vehicle, texture_consistent\n"
                f"- \"text\": brief 2-3 sentence argument embedding the actual numerical measurements from your camera data — grayscale pixel values, mean/std, Sobel gradient min/max/mean, BGR color values at center. Reference specific numbers to support each claim.\n\n"
                f"Output a JSON array of these objects, nothing else."
            )

        prompt = (
            f"You are the camera sensor agent in an autonomous driving fusion system. "
            f"You are in a negotiation with the radar agent about whether certain "
            f"radar-detected objects are real. The radar claims there is an object at "
            f"a given pixel location. You must check your image data at that location "
            f"and argue whether the object is really there.\n\n"
            f"At the disputed pixel locations, YOLO did NOT detect any object. However, "
            f"you can examine the raw image patch \u2014 there might be a real object "
            f"that YOLO missed, or the patch might just be noise/clutter.\n\n"
            f"{instructions}"
        )

        raw = llm.invoke(prompt).strip()
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
                       "claims": {"present": False, "class_is_vehicle": False,
                                   "texture_consistent": False},
                       "text": "Unable to parse LLM output. Defaulting to rejection."
                       } for o in disputed_objects]

        result = {}
        for entry in parsed:
            oid = entry.get("object_id")
            if oid is None:
                continue
            result[oid] = {
                "claims": entry.get("claims", {}),
                "text": entry.get("text", ""),
            }

        for obj in disputed_objects:
            oid = obj["object_id"]
            if oid not in result:
                result[oid] = {
                    "claims": {"present": False, "class_is_vehicle": False,
                                "texture_consistent": False},
                    "text": "No argument generated, defaulting to rejection.",
                }

        return result
