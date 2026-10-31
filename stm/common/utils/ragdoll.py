import json
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
POSE_REPO = "Tau-J/RTMPose"
DETECTOR = REPO_ROOT / "models/Tau-J/RTMPose/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx"
POSE_MODEL = REPO_ROOT / "models/Tau-J/RTMPose/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx"
POSE_FILES = {
    DETECTOR: "rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
    POSE_MODEL: "rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
}
RIG_PARENT = {
    "forearm-l": "upper-arm-l", "hand-l": "forearm-l",
    "forearm-r": "upper-arm-r", "hand-r": "forearm-r",
    "calf-l": "thigh-l", "foot-l": "calf-l",
    "calf-r": "thigh-r", "foot-r": "calf-r",
}

def _ensure_model(path):
    path = Path(path)
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    archive = hf_hub_download(POSE_REPO, POSE_FILES[path])
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        member = next(name for name in zf.namelist() if name.endswith("end2end.onnx"))
        with zf.open(member) as source, path.with_suffix(".tmp").open("wb") as target:
            shutil.copyfileobj(source, target)
    path.with_suffix(".tmp").replace(path)
    return path


def detect_pose(src_dir, detector=DETECTOR, pose_model=POSE_MODEL):
    import onnxruntime as ort
    from rtmlib import Wholebody, draw_skeleton

    detector = _ensure_model(detector)
    pose_model = _ensure_model(pose_model)
    src_dir = Path(src_dir)
    rgba = cv2.imread(str(src_dir / "src_img.png"), cv2.IMREAD_UNCHANGED)
    alpha = rgba[..., 3:4].astype(np.float32) / 255
    image = (rgba[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    ort.set_default_logger_severity(3)
    device = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    print(f"RTMPose device: {device}")
    estimator = Wholebody(
        det=str(detector), det_input_size=(640, 640),
        pose=str(pose_model), pose_input_size=(192, 256),
        backend="onnxruntime", device=device,
    )
    keypoints, scores = estimator(image)
    if not len(keypoints):
        raise ValueError("RTMPose did not detect a person")
    person = int(np.argmax(scores[:, :23].mean(axis=1)))
    keypoints, scores = keypoints[person], scores[person]
    cv2.imwrite(
        str(src_dir / "pose_overlay.png"),
        draw_skeleton(image.copy(), keypoints[None], scores[None], kpt_thr=0.25, radius=5, line_width=3),
    )
    (src_dir / "pose_keypoints.json").write_text(json.dumps({
        "keypoints": keypoints.tolist(), "scores": scores.tolist(),
    }, indent=2) + "\n")
    return keypoints, scores


def _distance_to_segment(points, start, end):
    segment = end - start
    scale = np.dot(segment, segment)
    if scale == 0:
        return ((points - start) ** 2).sum(axis=1)
    amount = np.clip(((points - start) @ segment) / scale, 0, 1)[:, None]
    return ((points - (start + amount * segment)) ** 2).sum(axis=1)


def _partition_mask(mask, xyxy, bones, overlap_ratio=0.2):
    if not 0 <= overlap_ratio <= 0.5:
        raise ValueError("joint overlap ratio must be between 0 and 0.5")
    ys, xs = np.nonzero(mask)
    points = np.column_stack((xs + xyxy[0], ys + xyxy[1])).astype(np.float32)
    owner = np.argmin(np.column_stack([
        _distance_to_segment(points, start, end) for _, start, end in bones
    ]), axis=1)
    parts = []
    for index in range(len(bones)):
        part = np.zeros_like(mask, dtype=bool)
        part[ys[owner == index], xs[owner == index]] = True
        parts.append(part)

    if overlap_ratio == 0:
        return parts

    # Keep model-generated pixels on both sides of every joint. The duplicated
    # band stays hidden in the rest pose and prevents a hole when either bone moves.
    for first in range(len(bones)):
        for second in range(first + 1, len(bones)):
            endpoints_a = bones[first][1:]
            endpoints_b = bones[second][1:]
            joint = next((a for a in endpoints_a for b in endpoints_b if np.allclose(a, b)), None)
            if joint is None:
                continue
            length_a = np.linalg.norm(endpoints_a[1] - endpoints_a[0])
            length_b = np.linalg.norm(endpoints_b[1] - endpoints_b[0])
            radius = max(2.0, min(length_a, length_b) * overlap_ratio)
            shared = ((points - joint) ** 2).sum(axis=1) <= radius ** 2
            parts[first][ys[shared], xs[shared]] = True
            parts[second][ys[shared], xs[shared]] = True
    return parts


def _split_part(part, bones, overlap_ratio=0.2):
    source_mask = part["img"][..., 3] > 0
    masks = _partition_mask(source_mask, part["xyxy"], bones, overlap_ratio=overlap_ratio)
    if not all(mask.any() for mask in masks):
        raise ValueError(f"empty ragdoll cut while splitting {part['tag']}")
    result = {}
    for (name, start, end), mask in zip(bones, masks):
        image = part["img"].copy()
        image[..., 3] *= mask
        depth = part["depth"].copy()
        depth[~mask] = 1
        result[name] = {
            "img": image, "depth": depth, "mask": mask, "xyxy": list(part["xyxy"]),
            "tag": name, "preserve_alpha": True, "pivot": np.asarray(start).tolist(),
            "bone": [np.asarray(start).tolist(), np.asarray(end).tolist()],
            "joint_overlap_ratio": overlap_ratio,
        }
        if name in RIG_PARENT:
            result[name]["parent"] = RIG_PARENT[name]
    assert np.array_equal(np.logical_or.reduce(masks), source_mask)
    return result


def split_ragdoll_parts(parts, keypoints, scores, threshold=0.3, overlap_ratio=0.2):
    import warnings
    k = np.asarray(keypoints, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    required = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                95, 99, 103, 107, 111, 116, 120, 124, 128, 132]
    low_joints = [i for i in required if s[i] < threshold]
    if low_joints:
        worst = float(np.min(s[low_joints]))
        warnings.warn(
            f"ragdoll: {len(low_joints)} joint(s) scored below confidence threshold "
            f"{threshold:.2f} (worst={worst:.3f}, indices={low_joints}). "
            "Skipping ragdoll split — limb layers will not be bone-cut. "
            "Lower the threshold with --ragdoll-confidence if the pose is correct."
        )
        return parts

    left_hand = k[[95, 99, 103, 107, 111]].mean(axis=0)
    right_hand = k[[116, 120, 124, 128, 132]].mean(axis=0)
    groups = {
        "handwear": [
            ("upper-arm-l", k[5], k[7]), ("forearm-l", k[7], k[9]), ("hand-l", k[9], left_hand),
            ("upper-arm-r", k[6], k[8]), ("forearm-r", k[8], k[10]), ("hand-r", k[10], right_hand),
        ],
        "legwear": [
            ("thigh-l", k[11], k[13]), ("calf-l", k[13], k[15]),
            ("thigh-r", k[12], k[14]), ("calf-r", k[14], k[16]),
        ],
        "footwear": [
            ("foot-l", k[15], k[[17, 18, 19]].mean(axis=0)),
            ("foot-r", k[16], k[[20, 21, 22]].mean(axis=0)),
        ],
    }
    for tag, bones in groups.items():
        if tag in parts:
            parts.update(_split_part(parts.pop(tag), bones, overlap_ratio=overlap_ratio))
    return parts


if __name__ == "__main__":
    test_mask = np.ones((3, 10), dtype=bool)
    bones = [("a", np.array([0, 1]), np.array([5, 1])),
             ("b", np.array([5, 1]), np.array([9, 1]))]
    cuts = _partition_mask(test_mask, [0, 0, 10, 3], bones, overlap_ratio=0.5)
    assert np.array_equal(np.logical_or.reduce(cuts), test_mask)
    assert np.any(cuts[0] & cuts[1])
    exclusive = _partition_mask(test_mask, [0, 0, 10, 3], bones, overlap_ratio=0)
    assert np.array_equal(np.sum(exclusive, axis=0), test_mask)
    try:
        _partition_mask(test_mask, [0, 0, 10, 3], bones, overlap_ratio=0.6)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid overlap ratio accepted")
    print("ragdoll partition self-check passed")
