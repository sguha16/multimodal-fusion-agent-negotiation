import time
import json
import os
import requests

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01",
}


class OrchestratorAgent:
    def __init__(self, memory_size=10):
        self.memory = []
        self.memory_size = memory_size

    def run(self, radar_report, camera_report, radar_trend, camera_trend):
        """
        Entry point. Runs A then B, assembles decision, stores in memory.
        """
        # STAGE A — collect and format reports from all agents
        situation = self._collect(radar_report, camera_report, radar_trend, camera_trend)

        # LLM — translate situation into structured summary for brain
        situation_summary = self._llm_translate(situation)

        # STAGE B — brain: decides fusion strategy
        # future: replace with RL policy
        fusion_strategy = self._brain(situation)

        # LLM — translate decision into natural language for logging + ASPICE
        decision_explanation = self._llm_explain(fusion_strategy, situation)
        # MEMORY
        self._store(situation, fusion_strategy, decision_explanation)

        return {
            "fusion_strategy": fusion_strategy,
            "situation_summary": situation_summary,
            "decision_explanation": decision_explanation,
        }

    # -------------------------------------------------------------------------
    # STAGE A — collect and format agent reports
    # -------------------------------------------------------------------------
    def _collect(self, radar_report, camera_report, radar_trend, camera_trend):
        return {
            "radar": {
                "confidence": radar_report["confidence"],
                "trend": radar_report["trend"],
                "self_assessment": radar_report["self_assessment"],
                "health": radar_report["health"],
            },
            "camera": {
                "confidence": camera_report["confidence"],
                "trend": camera_report["trend"],
                "self_assessment": camera_report["self_assessment"],
                "health": camera_report["health"],
            },
        }

    # -------------------------------------------------------------------------
    # LLM — translate situation into natural language summary
    # sits at communication boundary, not the brain
    # -------------------------------------------------------------------------
    def _llm_translate(self, situation):
        prompt = f"""
You are a sensor fusion system monitor.
Given the following sensor situation report, write a concise 2-3 sentence 
natural language summary of the current sensor state for a safety engineer to read.

Situation:
{json.dumps(situation, indent=2)}

Be specific about confidence levels, trends, and any issues flagged in self assessments.
"""
        try:
             response = requests.post(
                 "http://localhost:11434/api/chat",
                 json={
                     "model": "llama3",
                     "stream": False,
                     "messages": [{"role": "user", "content": prompt}]
                 }
             )
             data = response.json()
             return data["message"]["content"].strip()
            
        except Exception as e:
            return f"LLM translation failed: {e} | response: {response.json() if 'response' in dir() else 'no response'}"


    # -------------------------------------------------------------------------
    # STAGE B — brain: decides fusion strategy
    # future: replace with RL policy
    # for now: rule-based dummy that always picks mid level
    # -------------------------------------------------------------------------
    def _brain(self, situation):
        radar_conf = situation["radar"]["confidence"]
        camera_conf = situation["camera"]["confidence"]
        radar_trend = situation["radar"]["trend"]
        camera_trend = situation["camera"]["trend"]
        #for future the LLM can be used as a prior for reinforcement learning. 
        #LLM — domain knowledge, initialization, validation
        #RL brain — learns from experience, makes actual decisions
        
        # both sensors healthy — use mid level fusion (default)
        if radar_conf > 0.7 and camera_conf > 0.7:
            return "mid"

        # radar degraded — fall back to late fusion, rely on camera classification
        if radar_conf < 0.4:
            return "late"

        # camera degraded — fall back to late fusion, rely on radar detections
        if camera_conf < 0.4:
            return "late"

        # both degrading — safety fallback
        if radar_trend == "degrading" and camera_trend == "degrading":
            return "late"

        # default
        return "mid"
    
    # -------------------------------------------------------------------------
    # LLM — explain brain decision in natural language
    # -------------------------------------------------------------------------
    def _llm_explain(self, fusion_strategy, situation):
        prompt = f"""
You are a sensor fusion system monitor.
The fusion brain decided: {fusion_strategy} fusion.
Radar confidence: {situation['radar']['confidence']}, trend: {situation['radar']['trend']}
Camera confidence: {situation['camera']['confidence']}, trend: {situation['camera']['trend']}
Radar self assessment: {situation['radar']['self_assessment']}
Camera self assessment: {situation['camera']['self_assessment']}

In 1-2 sentences explain why this fusion strategy was selected given the situation.
"""
        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "llama3",
                    "stream": False,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            data = response.json()
            return data["message"]["content"].strip()
           
        except Exception as e:
            return f"LLM explanation failed: {e} | response: {response.json() if 'response' in dir() else 'no response'}"

    # -------------------------------------------------------------------------
    # MEMORY
    # -------------------------------------------------------------------------
    def _store(self, situation, fusion_strategy, decision_explanation):
        entry = {
            "timestamp": time.time(),
            "fusion_strategy": fusion_strategy,
            "decision_explanation": decision_explanation,
            "radar_confidence": situation["radar"]["confidence"],
            "camera_confidence": situation["camera"]["confidence"],
            "radar_trend": situation["radar"]["trend"],
            "camera_trend": situation["camera"]["trend"],
            "fused_classification": None,
            "aspice_feedback": None,
        }
        self.memory.append(entry)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)