"""
fused_cross_modal_object_list.py
Cross-modal matching between radar and camera detections at all available levels.
Projects radar objects into camera image space, matches with bounding boxes,
gathers mid and early level evidence, and composes fused confidence scores.

Usage:
    matcher = CrossModalMatcher(camera_calib_path)
    fused = matcher.fuse(radar_report, camera_report)
    matcher.visualize(camera_image, fused, save_path="fused_detections.png")
"""
import numpy as np
import cv2


class CrossModalMatcher:
    """Matches radar and camera detections across late, mid, and early levels."""

    def __init__(self, camera_calib_path, image_width=1920, image_height=1080,
                 projection_z=0.5):
        """
        Parameters
        ----------
        camera_calib_path : str
            Path to camera_calib.npy (LiDAR-to-camera extrinsics).
        image_width, image_height : int
            Camera image dimensions.
        projection_z : float
            Height (m) above ground for radar point projection. Verified at 0.5m
            against ground truth labels.
        """
        calib = np.load(camera_calib_path, allow_pickle=True).item()
        self.camera_matrix = calib['intrinsic']['camera_matrix']
        self.dist_coeffs = calib['intrinsic']['distortion_coefficients']
        self.rvec = calib['extrinsic']['rotation_vector']
        self.tvec = calib['extrinsic']['translation_vector']
        self.image_width = image_width
        self.image_height = image_height
        self.projection_z = projection_z

        # Evidence normalizers — tunable from data
        self.max_points_for_density = 30.0   # cluster with 30+ points = saturated
        self.max_lapvar_for_texture = 300.0  # Laplacian variance cap
        self.patch_size_mid = 32             # pixels, for mid-level texture check
        self.patch_size_early = 48           # pixels, for early-level edge check

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fuse(self, radar_report, camera_report,
             radar_agent=None, camera_agent=None, fusion_graph=None):
        """
        Main entry point. Returns fused object list with evidence-based confidence.

        Parameters
        ----------
        radar_report : dict
            From RadarAgent.get_report(). Contains data.early (radar cube),
            data.mid (point cloud list), data.late (object list).
        camera_report : dict
            From CameraAgent.get_report(). Contains data.early (RGB image),
            data.late (bounding box list).
        radar_agent : RadarAgent instance or None
            If provided, runs negotiation on disputed objects.
        camera_agent : CameraAgent instance or None
            If provided, runs negotiation on disputed objects.
        """
        radar_objects = radar_report["data"]["late"]
        camera_bboxes = camera_report["data"]["late"]
        radar_point_cloud = radar_report["data"]["mid"]
        camera_image = camera_report["data"]["early"]
        radar_cube = radar_report["data"]["early"]

        # Step 1: project each radar object to camera pixel coordinates
        radar_with_pixels = self._project_radar_objects(radar_objects)

        # Step 2: match at late level (projected pixel vs camera bboxes)
        match_results = self._match_late(radar_with_pixels, camera_bboxes)

        # Step 3: mid-level evidence for each radar object
        mid_ev = self._analyze_mid(
            radar_objects, radar_with_pixels, camera_image
        )

        # Step 4: early-level evidence
        early_ev = self._analyze_early(
            radar_objects, radar_cube, radar_with_pixels, camera_image
        )

        # Step 5: compose fused object list
        fused = self._compose_fused(
            radar_objects, camera_bboxes, match_results, mid_ev, early_ev,
            radar_report.get("confidence", 1.0),
            camera_report.get("confidence", 1.0),
        )

        # Step 6: negotiate disputed objects (if agents provided)
        if radar_agent is not None and camera_agent is not None:
            fused = self._negotiate(
                fused, radar_objects, radar_with_pixels,
                camera_bboxes, mid_ev, early_ev,
                radar_agent, camera_agent,
                radar_report, camera_report,
                fusion_graph,
            )

        return fused

    def visualize(self, camera_image, fused_objects, save_path="fused_detections.png"):
        """(unchanged)"""
        img = camera_image.copy()

        for obj in fused_objects:
            src = obj["source"]
            conf = obj["confidence"]

            if src == "matched":
                color = (0, 255, 0)
            elif src == "radar":
                color = (0, 200, 200)
            else:
                color = (0, 0, 255)

            if obj["bbox_camera"] is not None:
                x1, y1, x2, y2 = obj["bbox_camera"]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{obj['class']} {conf:.2f} ({src})"
                if "negotiation" in obj:
                    v = obj["negotiation"]["verdict"]
                    label += f" [{v}]"
                cv2.putText(img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if obj["position_radar"] is not None and obj["bbox_camera"] is None:
                u, v = self._project_point(
                    obj["position_radar"]["x_m"],
                    obj["position_radar"]["y_m"]
                )
                cv2.circle(img, (u, v), 5, color, -1)
                label = f"R{obj['id']} {conf:.2f}"
                if "negotiation" in obj:
                    verdict = obj["negotiation"]["verdict"]
                    label += f" [{verdict}]"
                cv2.putText(img, label, (u + 6, v),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Saved fused visualisation to {save_path}")

    # ------------------------------------------------------------------
    # Step 6: Negotiation
    # ------------------------------------------------------------------
    def _negotiate(self, fused, radar_objects, radar_with_pixels,
                   camera_bboxes, mid_ev, early_ev,
                   radar_agent, camera_agent,
                   radar_report, camera_report, fusion_graph):
        """
        Run 2-round LLM debate on disputed radar-only objects,
        then judge claims via knowledge graph.
        """
        from fusion_knowledge_graph import detect_active_conditions, llm_judge_dispute
        if fusion_graph is None:
            print("  [Negotiation skipped — no fusion graph provided]")
            return fused

        # Identify disputed objects: radar-only with valid pixel,
        # max 3 per scene for LLM speed
        max_disputes = 3
        disputed_obj_ids = set()
        for obj in fused:
            if obj["source"] != "radar":
                continue
            if len(disputed_obj_ids) >= max_disputes:
                break
            oid = obj["id"]
            # Find matching radar object with valid pixel
            for robj in radar_objects:
                if robj["id"] != oid or "_pixel" not in robj:
                    continue
                u, v = robj["_pixel"]
                if u <= 0 or u >= self.image_width - 1 or v <= 0 or v >= self.image_height - 1:
                    continue
                disputed_obj_ids.add(oid)
                break

        if not disputed_obj_ids:
            print("  [No disputed objects — nothing to negotiate]")
            return fused

        print(f"  [Negotiation] disputing {len(disputed_obj_ids)} radar-only objects")

        # Extract raw radar cube and point cloud from reports
        radar_cube = radar_report["data"]["early"]
        point_cloud = radar_report["data"]["mid"]
        camera_image = camera_report["data"]["early"]
        max_range = 100.0
        n_range = radar_cube.shape[0] if radar_cube.ndim >= 3 else 1

        # Build per-object data dicts for each agent
        radar_dispute_data = []
        camera_dispute_data = []
        for robj in radar_objects:
            if robj["id"] not in disputed_obj_ids:
                continue
            rid = robj["id"]

            # Radar: extract cube power slice at object's range bin
            range_idx = int(robj["range_m"] / max_range * n_range)
            range_idx = max(0, min(range_idx, n_range - 1))
            range_slice = np.abs(radar_cube[range_idx]) ** 2  # (256, 16)

            # Point cloud: nearby points within 2m
            nearby = []
            if point_cloud is not None and len(point_cloud) > 0:
                xm, ym = robj.get("x_m", 0), robj.get("y_m", 0)
                for pc in point_cloud:
                    dx = pc.get("x_m", 0) - xm
                    dy = pc.get("y_m", 0) - ym
                    if dx*dx + dy*dy < 4.0:
                        nearby.append(pc)

            radar_dispute_data.append({
                "object_id": rid,
                "range_m": robj.get("range_m", 0),
                "n_points": robj.get("n_points", 0),
                "velocity_mps": robj.get("velocity_mps", 0),
                "azimuth_deg": robj.get("azimuth_deg", 0),
                "motion_state": robj.get("motion_state", "unknown"),
                "range_slice": range_slice,
                "nearby_points": nearby,
            })

            # Camera: extract 16x16 patch at projected pixel
            u, v = robj["_pixel"]
            h, w = camera_image.shape[:2]
            half_p = 8
            y1, y2 = max(0, v - half_p), min(h, v + half_p)
            x1, x2 = max(0, u - half_p), min(w, u + half_p)
            patch = camera_image[y1:y2, x1:x2].copy()

            camera_dispute_data.append({
                "object_id": rid,
                "u": u, "v": v,
                "patch": patch,
                "yolo_class": None,
                "yolo_conf": 0.0,
            })

        def claims_to_text(claims_dict, agent_name):
            """Generate a sentence from structured claims."""
            parts = []
            for k, v in claims_dict.items():
                parts.append(f"{k}={v}")
            return f"[{agent_name}] " + ", ".join(parts)

        def join_agent_output(result_dict, agent_name):
            lines = []
            for oid in sorted(result_dict.keys()):
                d = result_dict[oid]
                text = claims_to_text(d["claims"], agent_name)
                lines.append(f"Object {oid}: {text}")
            return "\n".join(lines)

        # Round 1
        radar_result_r1 = radar_agent.argue(radar_dispute_data, round_num=1)
        radar_text_r1 = join_agent_output(radar_result_r1, "Radar")

        camera_result_r1 = camera_agent.argue(
            camera_dispute_data,
            opponent_arguments=f"Radar Round 1:\n{radar_text_r1}",
            round_num=1,
        )
        camera_text_r1 = join_agent_output(camera_result_r1, "Camera")

        # Round 2
        radar_result_r2 = radar_agent.argue(
            radar_dispute_data,
            opponent_arguments=f"Camera Round 1:\n{camera_text_r1}",
            round_num=2,
        )
        radar_text_r2 = join_agent_output(radar_result_r2, "Radar")

        camera_result_r2 = camera_agent.argue(
            camera_dispute_data,
            opponent_arguments=f"Radar Round 2:\n{radar_text_r2}",
            round_num=2,
        )
        camera_text_r2 = join_agent_output(camera_result_r2, "Camera")

        # Print full per-object conversation
        print("  [Negotiation transcript]")
        for oid in sorted(disputed_obj_ids):
            rr1_t = radar_result_r1.get(oid, {}).get("text", "")
            rr1_c = radar_result_r1.get(oid, {}).get("claims", {})
            cr1_t = camera_result_r1.get(oid, {}).get("text", "")
            cr1_c = camera_result_r1.get(oid, {}).get("claims", {})
            rr2_t = radar_result_r2.get(oid, {}).get("text", "")
            rr2_c = radar_result_r2.get(oid, {}).get("claims", {})
            cr2_t = camera_result_r2.get(oid, {}).get("text", "")
            cr2_c = camera_result_r2.get(oid, {}).get("claims", {})

            print(f"\n{'='*50}")
            print(f"  OBJECT {oid}")
            print(f"{'='*50}")
            print(f"  [R1] Radar:  claims={rr1_c}")
            print(f"    \"{rr1_t}\"")
            print(f"  [R1] Camera: claims={cr1_c}")
            print(f"    \"{cr1_t}\"")
            print(f"  [R2] Radar:  claims={rr2_c}")
            print(f"    \"{rr2_t}\"")
            print(f"  [R2] Camera: claims={cr2_c}")
            print(f"    \"{cr2_t}\"")

        # Gather all structured claims from both rounds
        all_claims = []
        for oid, rdata in radar_result_r1.items():
            for cap, val in rdata["claims"].items():
                all_claims.append({
                    "agent": "radar",
                    "object_id": oid,
                    "capability": cap,
                    "value": "reliable" if val else "unreliable",
                })
        for oid, rdata in radar_result_r2.items():
            for cap, val in rdata["claims"].items():
                all_claims.append({
                    "agent": "radar",
                    "object_id": oid,
                    "capability": cap,
                    "value": "reliable" if val else "unreliable",
                })
        for oid, rdata in camera_result_r1.items():
            for cap, val in rdata["claims"].items():
                all_claims.append({
                    "agent": "camera",
                    "object_id": oid,
                    "capability": cap,
                    "value": "reliable" if val else "unreliable",
                })
        for oid, rdata in camera_result_r2.items():
            for cap, val in rdata["claims"].items():
                all_claims.append({
                    "agent": "camera",
                    "object_id": oid,
                    "capability": cap,
                    "value": "reliable" if val else "unreliable",
                })

        # Per-object LLM judge
        from fusion_knowledge_graph import detect_active_conditions, llm_judge_dispute
        active_conditions = detect_active_conditions(
            radar_report.get("health", {}),
            camera_report.get("health", {}),
        )

        print(f"  [Judge] active conditions: {dict(active_conditions)}")
        # Look up dispute data and evidence for each object
        radar_summary_lookup = {}
        camera_summary_lookup = {}
        for d in radar_dispute_data:
            radar_summary_lookup[d["object_id"]] = self._summarize_radar_data(d)
        for d in camera_dispute_data:
            camera_summary_lookup[d["object_id"]] = self._summarize_camera_data(d)

        verdicts = {}
        for oid in sorted(disputed_obj_ids):
            rr1 = radar_result_r1.get(oid, {})
            rr2 = radar_result_r2.get(oid, {})
            cr1 = camera_result_r1.get(oid, {})
            cr2 = camera_result_r2.get(oid, {})

            result = llm_judge_dispute(
                object_id=oid,
                radar_claims_r1=rr1.get("claims", {}),
                radar_text_r1=rr1.get("text", ""),
                radar_claims_r2=rr2.get("claims", {}),
                radar_text_r2=rr2.get("text", ""),
                camera_claims_r1=cr1.get("claims", {}),
                camera_text_r1=cr1.get("text", ""),
                camera_claims_r2=cr2.get("claims", {}),
                camera_text_r2=cr2.get("text", ""),
                active_conditions=active_conditions,
                radar_health=radar_report.get("health", {}),
                camera_health=camera_report.get("health", {}),
                radar_confidence=radar_report.get("confidence", 1.0),
                camera_confidence=camera_report.get("confidence", 1.0),
                G=fusion_graph,
                radar_det_summary=radar_summary_lookup.get(oid, {}),
                camera_det_summary=camera_summary_lookup.get(oid, {}),
                mid_evidence={k: round(v, 3) for k, v in mid_ev.get(oid, {}).items()},
                early_evidence={k: round(v, 3) for k, v in early_ev.get(oid, {}).items()},
            )
            verdicts[oid] = result
            print(f"    Object {oid}: verdict={result.get('verdict', 'unknown')}")
            print(f"      {result.get('reasoning', '')}")

        # Build full transcript
        transcript_parts = []
        transcript_parts.append("=== ROUND 1 ===")
        transcript_parts.append("--- Radar ---")
        transcript_parts.append(radar_text_r1)
        transcript_parts.append("--- Camera ---")
        transcript_parts.append(camera_text_r1)
        transcript_parts.append("=== ROUND 2 ===")
        transcript_parts.append("--- Radar ---")
        transcript_parts.append(radar_text_r2)
        transcript_parts.append("--- Camera ---")
        transcript_parts.append(camera_text_r2)
        full_transcript = "\n".join(transcript_parts)

        # Enrich fused objects
        for obj in fused:
            oid = obj["id"]
            if isinstance(oid, str) and oid.startswith("cam_"):
                continue
            if oid not in disputed_obj_ids:
                continue
            v = verdicts.get(oid, {})
            verdict = v.get("verdict", "unresolved")

            # Build per-object claims list
            obj_claims = []
            for cl in all_claims:
                if cl["object_id"] == oid:
                    obj_claims.append({
                        "agent": cl["agent"],
                        "capability": cl["capability"],
                        "value": cl["value"],
                    })

            obj["negotiation"] = {
                "verdict": verdict,
                "reasoning": v.get("reasoning", ""),
                "claims": obj_claims,
                "transcript": full_transcript,
            }

        # Summary
        confirmed = sum(1 for o in fused if o.get("negotiation", {}).get("verdict") == "confirmed")
        rejected = sum(1 for o in fused if o.get("negotiation", {}).get("verdict") == "rejected")
        unresolved = sum(1 for o in fused if o.get("negotiation", {}).get("verdict") == "unresolved")
        print(f"  [Negotiation complete] {confirmed} confirmed, {rejected} rejected, {unresolved} unresolved")

        return fused

    # ------------------------------------------------------------------
    # Step 1: Radar projection to camera pixel coordinates
    # ------------------------------------------------------------------
    def _project_point(self, x_m, y_m):
        """Single radar (x,y) -> pixel (u,v)."""
        world_pt = np.array([[-y_m, x_m, self.projection_z]], dtype=np.float32)
        imgpts, _ = cv2.projectPoints(
            world_pt, self.rvec, self.tvec,
            self.camera_matrix, self.dist_coeffs
        )
        u = int(round(imgpts[0][0][0]))
        v = int(round(imgpts[0][0][1]))
        u = max(0, min(u, self.image_width - 1))
        v = max(0, min(v, self.image_height - 1))
        return u, v

    def _project_radar_objects(self, radar_objects):
        """Add '_pixel' key to each radar object."""
        for obj in radar_objects:
            u, v = self._project_point(obj["x_m"], obj["y_m"])
            obj["_pixel"] = (u, v)
        return radar_objects

    # ------------------------------------------------------------------
    # Step 2: Late-level matching
    # ------------------------------------------------------------------
    def _match_late(self, radar_objects, camera_bboxes):
        """Match radar objects to camera bboxes by pixel overlap."""
        matches = {}  # radar_id -> camera_id or None
        for robj in radar_objects:
            u, v = robj["_pixel"]
            matched = None
            for cobj in camera_bboxes:
                x1, y1, x2, y2 = cobj["bbox"]
                if x1 <= u <= x2 and y1 <= v <= y2:
                    matched = cobj["id"]
                    break
            matches[robj["id"]] = matched
        return matches

    # ------------------------------------------------------------------
    # Step 3: Mid-level evidence
    # ------------------------------------------------------------------
    def _analyze_mid(self, radar_objects, radar_with_pixels, camera_image):
        """
        For each radar object, compute:
        - point_score : normalised DBSCAN cluster size
        - patch_score : Laplacian variance at projected pixel (texture = object)
        """
        evidence = {}
        for robj in radar_objects:
            point_score = min(robj["n_points"] / self.max_points_for_density, 1.0)

            patch_score = 0.0
            if "_pixel" in robj:
                patch = self._extract_patch(
                    camera_image, robj["_pixel"][0], robj["_pixel"][1],
                    self.patch_size_mid
                )
                if patch is not None:
                    lap_var = float(cv2.Laplacian(patch, cv2.CV_64F).var())
                    patch_score = min(lap_var / self.max_lapvar_for_texture, 1.0)

            evidence[robj["id"]] = {
                "point_score": point_score,
                "patch_score": patch_score,
            }
        return evidence

    # ------------------------------------------------------------------
    # Step 4: Early-level evidence
    # ------------------------------------------------------------------
    def _analyze_early(self, radar_objects, radar_cube,
                       radar_with_pixels, camera_image):
        """
        For each radar object, compute:
        - cube_power_ratio : radar cube power at target range vs noise floor
        - edge_density     : Canny edge fraction at projected pixel
        """
        evidence = {}
        noise_floor = float(np.median(np.abs(radar_cube) ** 2))
        n_range = radar_cube.shape[0]      # 512
        max_range = 100.0                   # approximate max unambiguous range

        for robj in radar_objects:
            range_m = robj["range_m"]
            range_idx = int(range_m / max_range * n_range)
            range_idx = max(0, min(range_idx, n_range - 1))

            cube_slice = np.abs(radar_cube[range_idx, :, :]) ** 2
            target_power = float(np.mean(cube_slice))
            cube_ratio = min(target_power / max(noise_floor, 1e-6), 1.0)

            edge_density = 0.0
            if "_pixel" in robj:
                patch = self._extract_patch(
                    camera_image, robj["_pixel"][0], robj["_pixel"][1],
                    self.patch_size_early
                )
                if patch is not None:
                    edges = cv2.Canny(patch, 50, 150)
                    edge_density = float(np.mean(edges > 0))

            evidence[robj["id"]] = {
                "cube_power_ratio": cube_ratio,
                "edge_density": edge_density,
            }
        return evidence

    # ------------------------------------------------------------------
    # Step 5: Compose fused object list
    # ------------------------------------------------------------------
    def _compose_fused(self, radar_objects, camera_bboxes,
                       match_results, mid_ev, early_ev,
                       radar_agent_conf=1.0, camera_agent_conf=1.0):
        """
        Build fused objects. Confidence = mean of evidence scores,
        then multiplied by the originating agent's confidence.
        """
        fused = []

        # --- Process each radar object ---
        for robj in radar_objects:
            rid = robj["id"]
            cam_id = match_results.get(rid)
            mid = mid_ev.get(rid, {"point_score": 0.0, "patch_score": 0.0})
            early = early_ev.get(rid, {"cube_power_ratio": 0.0, "edge_density": 0.0})

            # Gather all non-null evidence scores
            scores = []
            if cam_id is not None:
                scores.append(1.0)                          # late match
            scores.append(mid["point_score"])                # radar cluster
            if mid["patch_score"] > 0:
                scores.append(mid["patch_score"])            # camera texture
            if early["cube_power_ratio"] > 0:
                scores.append(early["cube_power_ratio"])     # radar cube power
            if early["edge_density"] > 0:
                scores.append(early["edge_density"])         # camera edges

            conf = float(np.mean(scores)) if scores else 0.0
            # Scale by originating agent's confidence
            if cam_id is not None:
                conf *= min(radar_agent_conf, camera_agent_conf)
            else:
                conf *= radar_agent_conf

            # Get camera class + confidence if matched
            cam_class = None
            cam_conf = 0.0
            cam_bbox = None
            if cam_id is not None:
                for cobj in camera_bboxes:
                    if cobj["id"] == cam_id:
                        cam_class = cobj.get("class")
                        cam_conf = cobj.get("yolo_confidence", 0.0)
                        cam_bbox = cobj["bbox"]
                        break

            fused.append({
                "id": rid,
                "source": "matched" if cam_id is not None else "radar",
                "class": cam_class if cam_id is not None else robj.get("class"),
                "confidence": round(conf, 3),
                "radar_confidence": round(conf, 3),          # fused confidence
                "camera_confidence": round(cam_conf, 3),
                "bbox_camera": cam_bbox,
                "position_radar": {
                    "x_m": robj["x_m"],
                    "y_m": robj["y_m"],
                    "range_m": robj["range_m"],
                    "azimuth_deg": robj["azimuth_deg"],
                },
                "velocity_mps": robj["velocity_mps"],
                "n_radar_points": robj["n_points"],
                "evidence_levels": {
                    "late_match": cam_id is not None,
                    "mid_point_score": round(mid["point_score"], 3),
                    "mid_patch_score": round(mid["patch_score"], 3),
                    "early_cube_power": round(early["cube_power_ratio"], 3),
                    "early_edge_density": round(early["edge_density"], 3),
                },
            })

        # --- Add camera-only objects (no radar match) ---
        matched_radar_ids = {v for v in match_results.values() if v is not None}
        for cobj in camera_bboxes:
            if cobj["id"] not in matched_radar_ids:
                fused.append({
                    "id": f"cam_{cobj['id']}",
                    "source": "camera",
                    "class": cobj.get("class"),
                    "confidence": cobj.get("yolo_confidence", 0.0),
                    "radar_confidence": 0.0,
                    "camera_confidence": cobj.get("yolo_confidence", 0.0),
                    "bbox_camera": cobj["bbox"],
                    "position_radar": None,
                    "velocity_mps": None,
                    "n_radar_points": 0,
                    "evidence_levels": {
                        "late_match": False,
                        "mid_point_score": 0.0,
                        "mid_patch_score": 0.0,
                        "early_cube_power": 0.0,
                        "early_edge_density": 0.0,
                    },
                })

        fused.sort(key=lambda o: o["confidence"], reverse=True)
        return fused

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_patch(image, u, v, size):
        """Square patch centred at (u,v), returns None if too close to edge."""
        half = size // 2
        h, w = image.shape[:2]
        x1 = max(0, u - half)
        x2 = min(w, u + half)
        y1 = max(0, v - half)
        y2 = min(h, v + half)
        if x2 - x1 < size // 2 or y2 - y1 < size // 2:
            return None
        return image[y1:y2, x1:x2]

    @staticmethod
    def _summarize_radar_data(rd):
        """Convert raw radar dispute data dict into flat numeric summary for judge."""
        summary = {
            "range_m": rd.get("range_m", 0),
            "velocity_mps": rd.get("velocity_mps", 0),
            "azimuth_deg": rd.get("azimuth_deg", 0),
            "motion_state": rd.get("motion_state", "unknown"),
            "n_points": rd.get("n_points", 0),
        }
        rs = rd.get("range_slice")
        if rs is not None and rs.size > 0:
            noise_floor = float(np.median(rs))
            summary["cube_peak"] = float(np.max(rs))
            summary["cube_noise_floor"] = noise_floor
            summary["cube_mean"] = float(np.mean(rs))
            summary["cube_active_cells"] = int(np.sum(rs > noise_floor * 2))
            summary["cube_total_cells"] = rs.size
        nearby = rd.get("nearby_points", [])
        if nearby:
            dop = [p.get("velocity_mps", 0) for p in nearby]
            snr = [p.get("snr", 0) for p in nearby]
            summary["nearby_count"] = len(nearby)
            summary["doppler_min"] = min(dop)
            summary["doppler_max"] = max(dop)
            summary["snr_min"] = min(snr)
            summary["snr_max"] = max(snr)
        return summary

    @staticmethod
    def _summarize_camera_data(cd):
        """Convert raw camera dispute data dict into flat numeric summary for judge."""
        summary = {
            "pixel_u": cd.get("u", 0),
            "pixel_v": cd.get("v", 0),
            "yolo": False,
        }
        patch = cd.get("patch")
        if patch is not None and patch.size > 0:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
            summary["gray_mean"] = float(np.mean(gray))
            summary["gray_std"] = float(np.std(gray))
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            mag = np.sqrt(sobelx ** 2 + sobely ** 2)
            summary["sobel_mean"] = float(np.mean(mag))
            summary["sobel_max"] = float(np.max(mag))
            if patch.ndim == 3 and patch.shape[2] == 3:
                cy, cx = patch.shape[0] // 2, patch.shape[1] // 2
                summary["bgr_center"] = (
                    int(patch[cy, cx, 0]),
                    int(patch[cy, cx, 1]),
                    int(patch[cy, cx, 2]),
                )
        return summary
