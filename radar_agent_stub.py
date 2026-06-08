import time


class RadarAgent:
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
            "modality": "radar",
            "timestamp": time.time(),
            "data": data,
            "health": health,
            "confidence": confidence,
            "self_assessment": self_assessment,
            "trend": self.confidence_trend()
        }

        self._store(report)
        return report

    # -------------------------------------------------------------------------
    # STAGE A — signal processing chain
    # future: replace with real CFAR, angle estimation, clustering etc.
    # -------------------------------------------------------------------------
    def _signal_chain(self):
        data = {
            "object_list": [
                {"id": 1, "distance_m": 15.2, "velocity_mps": -5.0, "azimuth_deg": 2.1, "class":None},
                {"id": 2, "distance_m": 40.0, "velocity_mps": 0.0,  "azimuth_deg": -8.3, "class":None},
            ],
            "point_cloud": [], # future: raw point cloud
        }

        health = {
            "misalignment_deg": 6,     # ISO 26262 threshold: 6 deg
            "blockage": False,
            "interference_level": "low", # low / medium / high
            "calibration_valid": True,
            "hw_error": False,
        }

        return data, health

    # -------------------------------------------------------------------------
    # STAGE B — brain
    # future: replace with ML classifier + learned confidence model
    # -------------------------------------------------------------------------
    def _brain(self, data, health):

        # --- classify objects ---
        for obj in data["object_list"]:
            obj["class"] = self._classify(obj)
            
        # --- compute confidence from data quality + health ---
        confidence = self._compute_confidence(data, health)
        features=[]  # future: feature maps for mid-level fusion-wrong pos maybe

        # --- self assessment: radar knows its own weaknesses ---
        self_assessment = {
        "velocity": "reliable, RF front end nominal",
        "range": "reliable, RF front end nominal",
        "angle": f"degraded, misalignment is {health['misalignment_deg']} deg",
        "class": f"unreliable, interference level is {health['interference_level']}",
        }

        return confidence, self_assessment, features

    def _classify(self, obj):
        """
        Simple rule-based classification from radar detection.
        future: replace with ML classifier.
        """
        if abs(obj["velocity_mps"]) > 1.0:
            return "moving_vehicle"
        else:
            return "static_object"

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

        if len(data["object_list"]) == 0:
            confidence *= 0.6  # no detections reduces trust

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
