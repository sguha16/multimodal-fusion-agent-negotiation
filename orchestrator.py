# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:35:21 2026

@author: sanhi
"""

from radar_agent import RadarAgent
from camera_agent import CameraAgent
from orchestrator_agent import OrchestratorAgent

def run():
    print("=" * 50)
    print("MULTI-AGENT FUSION SYSTEM — v0.1")
    print("=" * 50)

    radar_agent = RadarAgent(memory_size=10)
    camera_agent = CameraAgent(memory_size=10)
    orchestrator = OrchestratorAgent(memory_size=10)

    radar = radar_agent.get_report()
    camera = camera_agent.get_report()

    print(f"\n[RADAR]  confidence: {radar['confidence']}  trend: {radar_agent.confidence_trend()}")
    print(f"[CAMERA] confidence: {camera['confidence']}  trend: {camera_agent.confidence_trend()}")

    print(f"\n[RADAR  SELF ASSESSMENT] {radar['self_assessment']}")
    print(f"\n[CAMERA SELF ASSESSMENT] {camera['self_assessment']}")

    result = orchestrator.run(
        radar_report=radar,
        camera_report=camera
    )

    print(f"\n[SITUATION SUMMARY]\n{result['situation_summary']}")
    print(f"\n[FUSION STRATEGY] {result['fusion_strategy']}")
    print(f"\n[DECISION EXPLANATION]\n{result['decision_explanation']}")
    print("=" * 50)

if __name__ == "__main__":
    run()