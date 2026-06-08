import time


class CameraAgent:
    def __init__(self, memory_size=10):
        self.memory = []
        self.memory_size = memory_size

    def get_report(self):
        """
        Entry point. Runs A then B, assembles report, stores in memory.
        """
        # STAGE A — signal processing chain
        data, health = self._signal_chain()

        # STAGE B — brain: interpret A output, compute confidence, self-assess
        confidence, self_assessment, features = self._brain(data, health)

        # REPORT — structured output to orchestrator
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
    # ISP pipeline: debayering, noise reduction, white balance, tone mapping
    # then object detection (e.g. YOLO) on clean RGB frame
    # future: replace with real ISP + detection model output
    # -------------------------------------------------------------------------
    def _signal_chain(self):
        data = {
            "object_list": [
                {"id": 1, "bbox": [120, 200, 300, 400], "distance_m": 14.8, "class": "car"},
                {"id": 2, "bbox": [400, 150, 480, 380], "distance_m": 8.0,  "class": "pedestrian"},
            ],
            # note: class comes from detection model in Stage A (unlike radar where B classifies)
            "frame_quality": 0.95,  # ISP quality check output
        }

        health = {
            "lighting": "poor",        # good / poor / night
            "occlusion_level": "low",  # low / medium / high
            "lens_blockage": True,
            "motion_blur": False,
            "calibration_valid": True,
            "hw_error": False,
        }

        return data, health

    # -------------------------------------------------------------------------
    # STAGE B — brain
    # future: replace with learned confidence model
    # -------------------------------------------------------------------------
    def _brain(self, data, health):

        # --- compute confidence from data quality + health ---
        confidence = self._compute_confidence(data, health)

        # --- features: future CNN feature maps for mid-level fusion ---
        features = []
        #classify is missing in B. it is done in A. Refining of classification for camera can be considered in B.

        # --- self assessment: camera knows its own weaknesses ---
        self_assessment = {
            "velocity": "unreliable, monocular camera cannot measure velocity directly",
            "range": f"moderate, monocular depth estimation, lighting is {health['lighting']}",
            "angle": "reliable, pixel space gives good angular resolution",
            "class": f"reliable, detection model output, occlusion is {health['occlusion_level']}",
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
            
        if data["frame_quality"] < 0.5:
            confidence *= 0.6
        elif data["frame_quality"] < 0.75:
            confidence *= 0.85

        if len(data["object_list"]) == 0:
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

