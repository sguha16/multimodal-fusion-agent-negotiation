# -*- coding: utf-8 -*-
"""
fusion_knowledge_graph.py
Knowledge graph encoding sensor capabilities, degradation relationships,
and fusion reasoning for radar-camera fusion system.

Node types:
    sensor       — radar, camera
    data         — radar_cube, point_cloud, object_list, rgb_image, bboxes
    capability   — range, velocity, angle, classification, depth, lateral_position
    condition    — rain, fog, night, glare, interference, misalignment, blockage, hw_error

Edge types:
    PROVIDES     — data node provides a capability (with quality attribute)
    DERIVED_FROM — data node is derived from another data node
    PRODUCES     — sensor produces a data node
    DEGRADES     — condition degrades a data node or capability
    RESISTANT_TO — sensor is resistant to a condition
    COMPENSATES  — one capability source can compensate for another when degraded
"""
import networkx as nx


def build_fusion_graph():
    #G = nx.DiGraph()
    G = nx.MultiDiGraph()
    # =====================================================================
    # SENSOR NODES
    # =====================================================================
    G.add_node("radar", type="sensor")
    G.add_node("camera", type="sensor")

    # =====================================================================
    # DATA NODES
    # =====================================================================
    # Radar data chain
    G.add_node("radar_cube", type="data", level="early",
               description="Range-Doppler-Antenna tensor (512x256x16)")
    G.add_node("point_cloud", type="data", level="mid",
               description="CFAR detections (N x 4: range, doppler, azimuth, elevation)")
    G.add_node("radar_object_list", type="data", level="late",
               description="Clustered objects with position, velocity, class")

    # Camera data chain
    G.add_node("rgb_image", type="data", level="early",
               description="Raw camera frame (1080x1920x3)")
    G.add_node("camera_bboxes", type="data", level="late",
               description="Detected bounding boxes with class")

    # =====================================================================
    # CAPABILITY NODES — what the system can measure
    # =====================================================================
    G.add_node("range", type="capability", description="Distance to target")
    G.add_node("velocity", type="capability", description="Speed of target")
    G.add_node("angle", type="capability", description="Azimuth/elevation direction")
    G.add_node("classification", type="capability", description="Object type: car, pedestrian, etc.")
    G.add_node("depth", type="capability", description="Distance estimation from camera")
    G.add_node("lateral_position", type="capability", description="Left/right position of target")

    # =====================================================================
    # CONDITION NODES — things that affect sensors
    # =====================================================================
    # Weather
    G.add_node("rain", type="condition", category="weather")
    G.add_node("fog", type="condition", category="weather")
    G.add_node("night", type="condition", category="weather")
    G.add_node("glare", type="condition", category="weather")

    # Radar faults
    G.add_node("interference", type="condition", category="radar_fault",
               description="RF interference, high noise floor, weak detections lost")
    G.add_node("misalignment", type="condition", category="radar_fault",
               description="Azimuth and elevation positions degraded/shifted")
    G.add_node("radar_blockage", type="condition", category="radar_fault",
               description="Missing detections from blocked area")
    G.add_node("radar_hw_error", type="condition", category="radar_fault",
               description="Temperature/voltage affecting chip, defects in chirp")
    G.add_node("radar_calibration_error", type="condition", category="radar_fault",
               description="Effects similar to misalignment")

    # Camera faults
    G.add_node("lens_blockage", type="condition", category="camera_fault",
               description="Dirt, water, ice on lens")
    G.add_node("motion_blur", type="condition", category="camera_fault",
               description="Fast movement causing image smear")
    G.add_node("camera_hw_error", type="condition", category="camera_fault")
    G.add_node("camera_calibration_error", type="condition", category="camera_fault")

    # =====================================================================
    # EDGES: PRODUCES — sensor produces data
    # =====================================================================
    G.add_edge("radar", "radar_cube", relation="PRODUCES")
    G.add_edge("camera", "rgb_image", relation="PRODUCES")

    # =====================================================================
    # EDGES: DERIVED_FROM — data chain
    # =====================================================================
    G.add_edge("point_cloud", "radar_cube", relation="DERIVED_FROM")
    G.add_edge("radar_object_list", "point_cloud", relation="DERIVED_FROM")
    G.add_edge("camera_bboxes", "rgb_image", relation="DERIVED_FROM")

    # =====================================================================
    # EDGES: PROVIDES — data provides capability (with quality)
    # =====================================================================
    # Radar cube
    G.add_edge("radar_cube", "range", relation="PROVIDES", quality="high")
    G.add_edge("radar_cube", "velocity", relation="PROVIDES", quality="high")
    G.add_edge("radar_cube", "angle", relation="PROVIDES", quality="medium")

    # Point cloud
    G.add_edge("point_cloud", "range", relation="PROVIDES", quality="high")
    G.add_edge("point_cloud", "velocity", relation="PROVIDES", quality="high")
    G.add_edge("point_cloud", "angle", relation="PROVIDES", quality="medium")

    # Radar object list
    G.add_edge("radar_object_list", "range", relation="PROVIDES", quality="high")
    G.add_edge("radar_object_list", "velocity", relation="PROVIDES", quality="high")
    G.add_edge("radar_object_list", "classification", relation="PROVIDES", quality="low")

    # RGB image
    G.add_edge("rgb_image", "classification", relation="PROVIDES", quality="high")
    G.add_edge("rgb_image", "angle", relation="PROVIDES", quality="high")
    G.add_edge("rgb_image", "lateral_position", relation="PROVIDES", quality="high")
    G.add_edge("rgb_image", "depth", relation="PROVIDES", quality="low")

    # Camera bboxes
    G.add_edge("camera_bboxes", "classification", relation="PROVIDES", quality="high")
    G.add_edge("camera_bboxes", "lateral_position", relation="PROVIDES", quality="high")
    G.add_edge("camera_bboxes", "angle", relation="PROVIDES", quality="high")

    # =====================================================================
    # EDGES: DEGRADES — conditions degrade data/capabilities
    # =====================================================================
    # Weather → camera
    G.add_edge("rain", "rgb_image", relation="DEGRADES", severity="high")
    G.add_edge("rain", "camera_bboxes", relation="DEGRADES", severity="high")
    G.add_edge("fog", "rgb_image", relation="DEGRADES", severity="high")
    G.add_edge("fog", "camera_bboxes", relation="DEGRADES", severity="medium")
    G.add_edge("night", "rgb_image", relation="DEGRADES", severity="high")
    G.add_edge("night", "camera_bboxes", relation="DEGRADES", severity="high")
    G.add_edge("glare", "rgb_image", relation="DEGRADES", severity="medium")

    # Radar faults → radar data
    G.add_edge("interference", "radar_cube", relation="DEGRADES", severity="medium",
               effect="high noise floor, weak detections lost")
    G.add_edge("interference", "point_cloud", relation="DEGRADES", severity="medium",
               effect="fewer detections, false alarms")
    G.add_edge("misalignment", "point_cloud", relation="DEGRADES", severity="high",
               effect="azimuth and elevation shifted")
    G.add_edge("misalignment", "angle", relation="DEGRADES", severity="high")
    G.add_edge("radar_blockage", "radar_cube", relation="DEGRADES", severity="high",
               effect="missing detections from blocked area")
    G.add_edge("radar_blockage", "point_cloud", relation="DEGRADES", severity="high")
    G.add_edge("radar_hw_error", "radar_cube", relation="DEGRADES", severity="critical",
               effect="defects in chirp: power, noise, phase")
    G.add_edge("radar_calibration_error", "point_cloud", relation="DEGRADES", severity="high",
               effect="similar to misalignment")

    # Camera faults → camera data
    G.add_edge("lens_blockage", "rgb_image", relation="DEGRADES", severity="critical")
    G.add_edge("lens_blockage", "camera_bboxes", relation="DEGRADES", severity="critical")
    G.add_edge("motion_blur", "rgb_image", relation="DEGRADES", severity="medium",
               effect="smeared image, bbox and classification degraded")
    G.add_edge("motion_blur", "camera_bboxes", relation="DEGRADES", severity="medium")
    G.add_edge("camera_hw_error", "rgb_image", relation="DEGRADES", severity="critical")
    G.add_edge("camera_calibration_error", "camera_bboxes", relation="DEGRADES", severity="high",
               effect="pixel positions dont map correctly to world coordinates")

    # =====================================================================
    # EDGES: RESISTANT_TO — radar resists weather
    # =====================================================================
    G.add_edge("radar", "rain", relation="RESISTANT_TO")
    G.add_edge("radar", "fog", relation="RESISTANT_TO")
    G.add_edge("radar", "night", relation="RESISTANT_TO")
    G.add_edge("radar", "glare", relation="RESISTANT_TO")

    # =====================================================================
    # EDGES: COMPENSATES — cross-sensor compensation
    # =====================================================================
    # Radar compensates for camera weaknesses
    G.add_edge("radar_cube", "depth", relation="COMPENSATES",
               reason="radar range is direct measurement, camera depth is estimation")
    G.add_edge("point_cloud", "depth", relation="COMPENSATES",
               reason="radar range is direct measurement")
    G.add_edge("radar_object_list", "velocity", relation="COMPENSATES",
               reason="camera cannot measure velocity directly")

    # Camera compensates for radar weaknesses
    G.add_edge("rgb_image", "classification", relation="COMPENSATES",
               reason="radar classification is rule-based and unreliable")
    G.add_edge("camera_bboxes", "classification", relation="COMPENSATES",
               reason="camera detection model provides reliable class")

    return G


def detect_active_conditions(radar_health, camera_health):
    """Map agent health dicts to knowledge graph condition nodes with severity."""
    conditions = {}

    # Radar conditions
    if_level = radar_health.get("interference_level", "low")
    if if_level in ("medium", "high"):
        conditions["interference"] = if_level
    mis = radar_health.get("misalignment_deg", 0)
    if mis > 3:
        conditions["misalignment"] = "high" if mis > 6 else "medium"
    if radar_health.get("blockage"):
        conditions["radar_blockage"] = "high"
    if radar_health.get("hw_error"):
        conditions["radar_hw_error"] = "critical"

    # Camera conditions
    if camera_health.get("motion_blur"):
        conditions["motion_blur"] = "medium"
    if camera_health.get("lens_blockage"):
        conditions["lens_blockage"] = "critical"
    if camera_health.get("hw_error"):
        conditions["camera_hw_error"] = "critical"
    lighting = camera_health.get("lighting")
    if lighting == "night":
        conditions["night"] = "high"
    elif lighting == "poor":
        conditions["night"] = "medium"

    return conditions


def check_claim(claim, active_conditions, G):
    """
    Check if a structured claim is supported by the knowledge graph.

    Parameters
    ----------
    claim : dict with keys:
        - agent: "radar" or "camera"
        - capability: e.g. "range", "velocity", "angle", "classification"
        - value: "reliable", "degraded", "unreliable", or "present"
    active_conditions : dict {condition_name: severity}
    G : NetworkX MultiDiGraph

    Returns
    -------
    bool — True if the claim is supported (graph does NOT contradict it)
    """
    capability = claim.get("capability")
    value = claim.get("value")
    agent = claim.get("agent")

    if not capability or not value or not agent:
        return True  # don't penalize malformed claims

    # Determine which data nodes this agent uses for this capability
    if agent == "radar":
        producer_nodes = ["radar_cube", "point_cloud", "radar_object_list"]
    else:
        producer_nodes = ["rgb_image", "camera_bboxes"]

    # Find which data nodes can provide this capability
    providing_nodes = []
    for node in producer_nodes:
        if node not in G:
            continue
        for _, cap_node, edge_data in G.out_edges(node, data=True):
            if edge_data.get("relation") == "PROVIDES" and cap_node == capability:
                providing_nodes.append(node)

    if not providing_nodes:
        return True  # no graph info for this capability-agent combo

    # Check if any active condition degrades the providing nodes
    for cond_name, severity in active_conditions.items():
        if cond_name not in G:
            continue
        for _, data_node, edge_data in G.out_edges(cond_name, data=True):
            if edge_data.get("relation") != "DEGRADES":
                continue
            if data_node in providing_nodes:
                # Active condition degrades a provider of this capability
                if value in ("reliable", "present", "true"):
                    return False  # claim contradicted
                else:
                    return True   # claim supported (capability IS degraded)

    # No contradiction found
    if value in ("reliable", "present", "true"):
        return True   # claim supported
    else:
        return False  # agent claims degradation but no active condition found


def evaluate_claims(all_claims, active_conditions, G):
    """
    Evaluate all claims from both agents.

    Parameters
    ----------
    all_claims : list of dict, each with keys:
        - agent, object_id, capability, value

    Returns
    -------
    dict: {object_id: {"radar": N_supported, "camera": N_supported,
                        "total_radar": N_total, "total_camera": N_total,
                        "verdict": "confirmed"|"rejected"|"unresolved"}}
    """
    by_object = {}
    for cl in all_claims:
        oid = cl.get("object_id")
        if oid not in by_object:
            by_object[oid] = {"radar": [], "camera": []}
        agent = cl.get("agent")
        if agent in ("radar", "camera"):
            supported = check_claim(cl, active_conditions, G)
            by_object[oid][agent].append(supported)

    results = {}
    for oid, groups in by_object.items():
        radar_supported = sum(groups["radar"])
        radar_total = len(groups["radar"])
        camera_supported = sum(groups["camera"])
        camera_total = len(groups["camera"])

        if radar_supported > camera_supported:
            verdict = "confirmed"
        elif camera_supported > radar_supported:
            verdict = "rejected"
        else:
            verdict = "unresolved"

        results[oid] = {
            "radar_supported": radar_supported,
            "radar_total": radar_total,
            "camera_supported": camera_supported,
            "camera_total": camera_total,
            "verdict": verdict,
        }
    return results


def llm_judge_dispute(object_id,
                      radar_claims_r1, radar_text_r1,
                      radar_claims_r2, radar_text_r2,
                      camera_claims_r1, camera_text_r1,
                      camera_claims_r2, camera_text_r2,
                      active_conditions, radar_health, camera_health,
                      radar_confidence, camera_confidence, G,
                      radar_det_summary=None, camera_det_summary=None,
                      mid_evidence=None, early_evidence=None):
    """
    LLM-based judge: given both agents' claims from 2 rounds of debate,
    the active knowledge graph conditions, and sensor health,
    produce a reasoned verdict per disputed object.

    Returns dict: {"object_id": int, "reasoning": str, "verdict": "confirmed"|"rejected"|"unresolved"}
    """
    from langchain_ollama import OllamaLLM

    # Gather graph facts about active conditions
    condition_lines = []
    for cond_name, severity in active_conditions.items():
        if cond_name not in G:
            continue
        degrades = []
        for _, target, attrs in G.edges(cond_name, data=True):
            if attrs.get("relation") == "DEGRADES":
                degrades.append(f"  - {cond_name} ({severity}) DEGRADES {target} — {attrs.get('effect', 'general degradation')}")

        if degrades:
            condition_lines.append(f"Condition: {cond_name} (severity: {severity})")
            condition_lines.extend(degrades)

    condition_str = "\n".join(condition_lines) if condition_lines else "No active conditions degrading any data source."

    # Gather what each sensor provides (PROVIDES edges from their data nodes)
    radar_provides = []
    camera_provides = []
    for node in ["radar_cube", "point_cloud", "radar_object_list"]:
        if node not in G:
            continue
        for _, cap, attrs in G.edges(node, data=True):
            if attrs.get("relation") == "PROVIDES":
                radar_provides.append(f"  {node} → {cap} (quality: {attrs.get('quality', 'unknown')})")
    for node in ["rgb_image", "camera_bboxes"]:
        if node not in G:
            continue
        for _, cap, attrs in G.edges(node, data=True):
            if attrs.get("relation") == "PROVIDES":
                camera_provides.append(f"  {node} → {cap} (quality: {attrs.get('quality', 'unknown')})")

    provides_str = "Radar provides:\n" + "\n".join(radar_provides) if radar_provides else ""
    provides_str += "\n\nCamera provides:\n" + "\n".join(camera_provides) if camera_provides else ""

    # Build RAW SENSOR DATA section from deterministic summaries
    rd = radar_det_summary or {}
    cd = camera_det_summary or {}
    me = mid_evidence or {}
    ee = early_evidence or {}

    radar_data_parts = [
        f"range={rd.get('range_m', 0):.1f}m",
        f"vel={rd.get('velocity_mps', 0):.1f}m/s",
        f"motion={rd.get('motion_state', '?')}",
        f"cluster={rd.get('n_points', 0)}pts",
    ]
    if "cube_peak" in rd:
        radar_data_parts.append(f"cube_peak={rd['cube_peak']:.1e}")
        radar_data_parts.append(f"noise_floor={rd['cube_noise_floor']:.1e}")
        radar_data_parts.append(f"active_cells={rd.get('cube_active_cells', 0)}/{rd.get('cube_total_cells', 0)}")
    if "nearby_count" in rd:
        radar_data_parts.append(f"nearby={rd['nearby_count']}pts")
        radar_data_parts.append(f"doppler=[{rd['doppler_min']:.1f},{rd['doppler_max']:.1f}]")
        radar_data_parts.append(f"SNR=[{rd['snr_min']:.1f},{rd['snr_max']:.1f}]")
    radar_data_str = " | ".join(radar_data_parts)

    camera_data_parts = [f"pixel=({cd.get('pixel_u', 0)},{cd.get('pixel_v', 0)})"]
    if "gray_mean" in cd:
        camera_data_parts.append(f"gray_mean={cd['gray_mean']:.1f}")
        camera_data_parts.append(f"gray_std={cd['gray_std']:.1f}")
        camera_data_parts.append(f"sobel_mean={cd['sobel_mean']:.1f}")
        camera_data_parts.append(f"sobel_max={cd['sobel_max']:.1f}")
    if "bgr_center" in cd:
        camera_data_parts.append(f"BGR={cd['bgr_center']}")
    camera_data_parts.append("YOLO=no_detection")
    camera_data_str = " | ".join(camera_data_parts)

    ev_parts = []
    if me:
        ev_parts.append(f"point_score={me.get('point_score', 0):.3f}")
        ev_parts.append(f"patch_score={me.get('patch_score', 0):.3f}")
    if ee:
        ev_parts.append(f"cube_power_ratio={ee.get('cube_power_ratio', 0):.3f}")
        ev_parts.append(f"edge_density={ee.get('edge_density', 0):.3f}")
    ev_str = " | ".join(ev_parts) if ev_parts else "N/A"

    prompt = f"""You are the impartial judge in an autonomous driving sensor fusion system. Two sensor agents (radar and camera) are in a 2-round debate about whether a radar-detected object is real.

OBJECT {object_id}

=== RAW SENSOR DATA (deterministic, not agent paraphrases) ===
Radar: {radar_data_str}
Camera: {camera_data_str}
Evidence scores: {ev_str}

=== ACTIVE KNOWLEDGE GRAPH CONDITIONS ===
{condition_str}

=== SENSOR CAPABILITIES ===
{provides_str}

=== SENSOR HEALTH ===
Radar confidence: {radar_confidence:.2f}
Radar health: interference_level={radar_health.get('interference_level', 'low')}, misalignment={radar_health.get('misalignment_deg', 0):.1f}deg, n_detections={radar_health.get('n_detections', 0)}

Camera confidence: {camera_confidence:.2f}
Camera health: brightness={camera_health.get('brightness', 0):.1f}, sharpness={camera_health.get('sharpness', 0):.1f}, motion_blur={camera_health.get('motion_blur', False)}, frame_quality={camera_health.get('frame_quality', 1.0):.2f}

=== AGENT ARGUMENTS (LLM-generated) ===
=== ROUND 1 ===
Radar argument: "{radar_text_r1}"
Radar claims: {radar_claims_r1}

Camera argument: "{camera_text_r1}"
Camera claims: {camera_claims_r1}

=== ROUND 2 ===
Radar argument: "{radar_text_r2}"
Radar claims: {radar_claims_r2}

Camera argument: "{camera_text_r2}"
Camera claims: {camera_claims_r2}

=== YOUR ROLE: CHAIN-OF-THOUGHT REASONING ===
For this disputed object, reason step by step:

1. What does radar's RAW DATA show? Reference specific values from the deterministic data above.
   Is the cube peak well above the noise floor? Is the cluster size meaningful? Are dopplers consistent?
   Use the point_score and cube_power_ratio to support your assessment.

2. What does camera's RAW DATA show? Reference specific values from the deterministic data above.
   Is there edge structure (sobel_mean) or is the patch uniform (gray_std)?
   Are colors consistent with a vehicle? Use the patch_score and edge_density.

3. Which active conditions affect which sensor? Check against the knowledge graph conditions above.
   Example: interference DEGRADES radar_cube → radar's evidence is less reliable.
   Example: fog DEGRADES rgb_image → camera's "no detection" is less reliable.

4. Given the active conditions and sensor health, which agent's evidence is more trustworthy?
   A degraded sensor's conflicting evidence should be weighted less.
   Radar confidence: {radar_confidence:.2f} | Camera confidence: {camera_confidence:.2f}

5. Output your verdict.

Output a JSON object with:
- "object_id": {object_id}
- "reasoning": 3-5 sentences following the steps above, referencing specific values from the RAW DATA section
- "verdict": "confirmed" (trust radar), "rejected" (trust camera), or "unresolved" (cannot determine)

JSON object only, nothing else."""

    if not hasattr(llm_judge_dispute, '_llm'):
        llm_judge_dispute._llm = OllamaLLM(model="llama3")
    llm = llm_judge_dispute._llm

    raw = llm.invoke(prompt).strip()
    import re
    import json
    try:
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = json.loads(raw)
        return {
            "object_id": parsed.get("object_id", object_id),
            "reasoning": parsed.get("reasoning", ""),
            "verdict": parsed.get("verdict", "unresolved"),
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "object_id": object_id,
            "reasoning": "Judge LLM output not parseable.",
            "verdict": "unresolved",
        }
    """Print a summary of the knowledge graph."""
    node_types = {}
    for node, attrs in G.nodes(data=True):
        t = attrs.get("type", "unknown")
        if t not in node_types:
            node_types[t] = []
        node_types[t].append(node)

    edge_types = {}
    for u, v, attrs in G.edges(data=True):
        r = attrs.get("relation", "unknown")
        if r not in edge_types:
            edge_types[r] = []
            edge_types[r].append((u, v, attrs))
    print("=" * 60)
    print("FUSION KNOWLEDGE GRAPH SUMMARY")
    print("=" * 60)

    for t, nodes in node_types.items():
        print(f"\n{t.upper()} NODES ({len(nodes)}):")
        for n in nodes:
            print(f"  {n}")

    print(f"\nTOTAL NODES: {G.number_of_nodes()}")
    print(f"TOTAL EDGES: {G.number_of_edges()}")

    for r, edges in edge_types.items():
        print(f"\n{r} ({len(edges)}):")
        for u, v, attrs in edges:
            extra = ""
            if "quality" in attrs:
                extra = f" [quality: {attrs['quality']}]"
            elif "severity" in attrs:
                extra = f" [severity: {attrs['severity']}]"
            print(f"  {u} → {v}{extra}")


if __name__ == "__main__":
    G = build_fusion_graph()
    print_graph_summary(G)

#VISUAL GRAPH
import matplotlib.pyplot as plt

def visualize_graph(G):
    colors = {
        "sensor": "red",
        "data": "dodgerblue",
        "capability": "green",
        "condition": "orange"
    }
    node_colors = [colors.get(G.nodes[n].get("type", ""), "gray") for n in G.nodes]
    
    # Manual positions: sensors top, data middle, capabilities bottom, conditions on sides
    pos = {}
    # Sensors
    pos["radar"] = (-2, 4)
    pos["camera"] = (2, 4)
    # Data — radar left, camera right
    pos["radar_cube"] = (-4, 2)
    pos["point_cloud"] = (-2, 2)
    pos["radar_object_list"] = (0, 2)
    pos["rgb_image"] = (2, 2)
    pos["camera_bboxes"] = (4, 2)
    # Capabilities — bottom center
    pos["range"] = (-4, 0)
    pos["velocity"] = (-2, 0)
    pos["angle"] = (0, 0)
    pos["classification"] = (2, 0)
    pos["depth"] = (4, 0)
    pos["lateral_position"] = (5, 0)
    # Conditions — sides
    pos["interference"] = (-6, 3)
    pos["misalignment"] = (-6, 1)
    pos["radar_blockage"] = (-6, 2)
    pos["radar_hw_error"] = (-6, 0)
    pos["radar_calibration_error"] = (-6, -1)
    pos["rain"] = (6, 4)
    pos["fog"] = (6, 3)
    pos["night"] = (6, 2)
    pos["glare"] = (6, 1)
    pos["lens_blockage"] = (6, 0)
    pos["motion_blur"] = (6, -1)
    pos["camera_hw_error"] = (6, -2)
    pos["camera_calibration_error"] = (6, -3)

    plt.figure(figsize=(22, 14))
    nx.draw(G, pos, with_labels=True, node_color=node_colors,
            node_size=2500, font_size=8, font_weight="bold",
            arrows=True, arrowsize=15, edge_color="gray", alpha=0.9)

    edge_labels = {(u, v): d.get("relation", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)

    plt.title("Fusion Knowledge Graph", fontsize=16)
    plt.tight_layout()
    plt.savefig("fusion_graph.png", dpi=150, bbox_inches='tight')
    print("Saved to fusion_graph.png")
#visualize_graph(G)