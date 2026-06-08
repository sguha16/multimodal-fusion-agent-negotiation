# -*- coding: utf-8 -*-
"""
graph_query.py
Queries the fusion knowledge graph given current sensor health.
Replaces ChromaDB vector retrieval with structured graph traversal.

Usage:
    from fusion_knowledge_graph import build_fusion_graph
    from graph_query import query_fusion_graph
    
    G = build_fusion_graph()
    reasoning = query_fusion_graph(G, radar_report, camera_report)
    # reasoning is a structured string the LLM can use to make a decision
"""
import networkx as nx


def query_fusion_graph(G, radar_report, camera_report):
    """
    Given current sensor reports, traverse the knowledge graph to produce
    structured reasoning for the fusion decision.
    
    Steps:
        1. Detect active conditions from health reports
        2. Find which data nodes are degraded
        3. Find which capabilities are affected
        4. Find which data nodes are still reliable
        5. Find compensation paths
        6. Assemble structured reasoning
    
    Returns: string with structured reasoning (replaces ChromaDB output)
    """
    # Step 1: detect active conditions from health
    active_conditions = _detect_conditions(radar_report, camera_report)

    # Step 2: find degraded data nodes
    degraded = _find_degraded(G, active_conditions)

    # Step 3: find affected capabilities
    affected_capabilities = _find_affected_capabilities(G, degraded)

    # Step 4: find reliable data nodes (not degraded)
    all_data_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "data"]
    reliable = []
    for node in all_data_nodes:
        if node not in degraded:
            reliable.append(node)

    # Step 5: find compensation paths
    compensations = _find_compensations(G, affected_capabilities, reliable)

    # Step 6: assemble reasoning
    reasoning = _assemble_reasoning(
        active_conditions, degraded, affected_capabilities,
        reliable, compensations, radar_report, camera_report
    )

    return reasoning


def _detect_conditions(radar_report, camera_report):
    """Map health fields to condition nodes in the graph."""
    active = []

    # --- Radar conditions ---
    radar_health = radar_report["health"]

    if radar_health.get("interference_level") == "high":
        active.append(("interference", "high"))
    elif radar_health.get("interference_level") == "medium":
        active.append(("interference", "medium"))

    if radar_health.get("misalignment_deg", 0) > 3:
        active.append(("misalignment", "high" if radar_health["misalignment_deg"] > 6 else "medium"))

    if radar_health.get("blockage"):
        active.append(("radar_blockage", "high"))

    if radar_health.get("hw_error"):
        active.append(("radar_hw_error", "critical"))

    if not radar_health.get("calibration_valid", True):
        active.append(("radar_calibration_error", "high"))

    # --- Camera conditions ---
    camera_health = camera_report["health"]

    if camera_health.get("lighting") == "night":
        active.append(("night", "high"))
    elif camera_health.get("lighting") == "poor":
        active.append(("night", "medium"))

    if camera_health.get("lens_blockage"):
        active.append(("lens_blockage", "critical"))

    if camera_health.get("motion_blur"):
        active.append(("motion_blur", "medium"))

    if camera_health.get("hw_error"):
        active.append(("camera_hw_error", "critical"))

    if not camera_health.get("calibration_valid", True):
        active.append(("camera_calibration_error", "high"))

    # Weather — inferred from camera health
    # frame_quality < 0.5 + good lighting could indicate rain/fog
    frame_quality = camera_health.get("frame_quality", 1.0)
    if frame_quality < 0.5 and camera_health.get("lighting") == "good":
        active.append(("rain", "medium"))

    return active


def _find_degraded(G, active_conditions):
    """Find all data nodes degraded by active conditions."""
    degraded = {}  # node_name -> list of (condition, severity)

    for condition, severity in active_conditions:
        if condition not in G.nodes:
            continue
        # Follow DEGRADES edges from this condition
        for _, target, attrs in G.edges(condition, data=True):
            if attrs.get("relation") == "DEGRADES":
                if target not in degraded:
                    degraded[target] = []
                degraded[target].append({
                    "condition": condition,
                    "severity": severity,
                    "effect": attrs.get("effect", "")
                })

    # Propagate through DERIVED_FROM chain:
    # If A DERIVED_FROM B and B is degraded, then A is also degraded
    # (e.g. interference → radar_cube (direct) → point_cloud (cascade) → radar_object_list (cascade))
    changed = True
    while changed:
        changed = False
        for node in list(degraded.keys()):
            for src, dst, attrs in G.edges(data=True):
                if attrs.get("relation") == "DERIVED_FROM" and dst == node:
                    if src not in degraded:
                        degraded[src] = [entry.copy() for entry in degraded[node]]
                        changed = True

    return degraded


def _find_affected_capabilities(G, degraded):
    """Find capabilities that are affected because their data sources are degraded."""
    affected = {}  # capability -> list of (data_node, quality_when_healthy)

    for data_node in degraded:
        if data_node not in G.nodes:
            continue
        for _, target, attrs in G.edges(data_node, data=True):
            if attrs.get("relation") == "PROVIDES":
                if target not in affected:
                    affected[target] = []
                affected[target].append({
                    "data_node": data_node,
                    "quality_when_healthy": attrs.get("quality", "unknown"),
                    "degraded_by": degraded[data_node]
                })

    return affected


def _find_compensations(G, affected_capabilities, reliable):
    """Find reliable data nodes that can compensate for affected capabilities."""
    compensations = []

    for capability in affected_capabilities:
        # Find all data nodes that PROVIDE this capability and are reliable
        for data_node in reliable:
            if data_node not in G.nodes:
                continue
            for _, target, attrs in G.edges(data_node, data=True):
                if attrs.get("relation") == "PROVIDES" and target == capability:
                    compensations.append({
                        "capability": capability,
                        "compensated_by": data_node,
                        "quality": attrs.get("quality", "unknown"),
                    })

        # Also check COMPENSATES edges
        for data_node in reliable:
            if data_node not in G.nodes:
                continue
            for _, target, attrs in G.edges(data_node, data=True):
                if attrs.get("relation") == "COMPENSATES" and target == capability:
                    compensations.append({
                        "capability": capability,
                        "compensated_by": data_node,
                        "quality": "direct compensation",
                        "reason": attrs.get("reason", ""),
                    })

    return compensations


def _assemble_reasoning(active_conditions, degraded, affected_capabilities,
                        reliable, compensations, radar_report, camera_report):
    """Build structured reasoning string for the LLM."""
    lines = []

    lines.append("=== GRAPH-BASED SENSOR ANALYSIS ===")
    lines.append("")

    # Current state
    lines.append(f"Radar confidence: {radar_report['confidence']}, trend: {radar_report['trend']}")
    lines.append(f"Camera confidence: {camera_report['confidence']}, trend: {camera_report['trend']}")
    lines.append("")

    # Active conditions
    if active_conditions:
        lines.append("ACTIVE CONDITIONS:")
        for condition, severity in active_conditions:
            lines.append(f"  - {condition} (severity: {severity})")
    else:
        lines.append("ACTIVE CONDITIONS: None — both sensors nominal")
    lines.append("")

    # Degraded data
    if degraded:
        lines.append("DEGRADED DATA SOURCES:")
        for node, issues in degraded.items():
            for issue in issues:
                effect = f" — {issue['effect']}" if issue['effect'] else ""
                lines.append(f"  - {node}: degraded by {issue['condition']} ({issue['severity']}){effect}")
    else:
        lines.append("DEGRADED DATA SOURCES: None — all data sources healthy")
    lines.append("")

    # Affected capabilities
    if affected_capabilities:
        lines.append("AFFECTED CAPABILITIES:")
        for cap, sources in affected_capabilities.items():
            source_names = [s["data_node"] for s in sources]
            lines.append(f"  - {cap}: affected via {', '.join(source_names)}")
    else:
        lines.append("AFFECTED CAPABILITIES: None")
    lines.append("")

    # Reliable data
    lines.append("RELIABLE DATA SOURCES:")
    if reliable:
        for node in reliable:
            lines.append(f"  - {node}")
    else:
        lines.append("  WARNING: No reliable data sources available")
    lines.append("")

    # Compensations
    if compensations:
        lines.append("COMPENSATION PATHS:")
        for comp in compensations:
            reason = f" — {comp['reason']}" if comp.get('reason') else ""
            lines.append(f"  - {comp['capability']} can be covered by {comp['compensated_by']} (quality: {comp['quality']}){reason}")
    lines.append("")

    # Recommendation
    lines.append("RECOMMENDATION:")
    if not active_conditions:
        lines.append("  All sensors nominal. Use all available data sources for maximum information.")
    elif not reliable:
        lines.append("  CRITICAL: No reliable data sources. System should fall back to safe state.")
    elif compensations:
        comp_sources = set(c["compensated_by"] for c in compensations)
        lines.append(f"  Use reliable sources: {', '.join(reliable)}")
        lines.append(f"  Key compensations available via: {', '.join(comp_sources)}")
    else:
        lines.append(f"  Use remaining reliable sources: {', '.join(reliable)}")
        lines.append(f"  WARNING: No compensation available for affected capabilities")

    return "\n".join(lines)


# =========================================================================
# TESTS
# =========================================================================
from fusion_knowledge_graph import build_fusion_graph

G = build_fusion_graph()

if __name__ == '__main__':
    # Test 1: both sensors healthy
    print("=" * 60)
    print("TEST 1: Both sensors healthy")
    print("=" * 60)
    radar_healthy = {
        "confidence": 1.0, "trend": "stable",
        "health": {"interference_level": "low", "misalignment_deg": 0,
                    "blockage": False, "hw_error": False, "calibration_valid": True,
                    "n_detections": 664, "mean_power": 1e6},
        "self_assessment": {"velocity": "reliable", "range": "reliable",
                            "angle": "reliable", "class": "unreliable"}
    }
    camera_healthy = {
        "confidence": 1.0, "trend": "stable",
        "health": {"lighting": "good", "occlusion_level": "low",
                    "lens_blockage": False, "motion_blur": False,
                    "calibration_valid": True, "hw_error": False,
                    "brightness": 131.0, "sharpness": 6.8, "frame_quality": 0.75},
        "self_assessment": {"velocity": "unreliable", "range": "moderate",
                            "angle": "reliable", "class": "reliable"}
    }
    print(query_fusion_graph(G, radar_healthy, camera_healthy))

    # Test 2: camera degraded (night)
    print("\n" + "=" * 60)
    print("TEST 2: Camera degraded — night driving")
    print("=" * 60)
    print("Edges from rgb_image:", list(G.edges("rgb_image", data=True)))
    camera_night = {
        "confidence": 0.4, "trend": "degrading",
        "health": {"lighting": "night", "occlusion_level": "low",
                    "lens_blockage": False, "motion_blur": False,
                    "calibration_valid": True, "hw_error": False,
                    "brightness": 30.0, "sharpness": 4.0, "frame_quality": 0.4},
        "self_assessment": {"velocity": "unreliable", "range": "unreliable",
                            "angle": "degraded", "class": "degraded"}
    }
    print(query_fusion_graph(G, radar_healthy, camera_night))

    # Test 3: radar degraded (interference + misalignment)
    print("\n" + "=" * 60)
    print("TEST 3: Radar degraded — interference + misalignment")
    print("=" * 60)
    radar_degraded = {
        "confidence": 0.3, "trend": "degrading",
        "health": {"interference_level": "high", "misalignment_deg": 8,
                    "blockage": False, "hw_error": False, "calibration_valid": True,
                    "n_detections": 50, "mean_power": 5e8},
        "self_assessment": {"velocity": "degraded", "range": "degraded",
                            "angle": "unreliable", "class": "unreliable"}
    }
    print(query_fusion_graph(G, radar_degraded, camera_healthy))

    