"""
recognizer.py — Core face recognition engine.

Loads labeled images from the dataset folder once at startup,
precomputes L2-normalized ArcFace embeddings, and matches incoming
face crops using vectorized cosine similarity.
"""

import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from typing import List, Dict, Tuple, Optional

from appearance_cues import detect_appearance_cues


class FaceRecognizer:
    """
    Real-time face recognizer backed by InsightFace (buffalo_l).

    The dataset folder must contain one image per person, named:
        FirstName_LastName.jpg   →  displayed as "FirstName LastName"

    Embeddings are computed once at construction and stored as a
    (N, 512) float32 matrix so every frame only needs one matrix
    multiply to compare against all registered faces.
    """

    SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        dataset_path: str,
        threshold: float = 0.45,
        det_size: Tuple[int, int] = (640, 640),
    ) -> None:
        """
        Args:
            dataset_path: Path to the folder containing labeled images.
            threshold:    Minimum cosine similarity to accept a match [0, 1].
            det_size:     Internal resolution for the face detector.
        """
        self.dataset_path = dataset_path
        self.threshold = threshold

        # Parallel arrays — index i → name i, embedding row i
        self._names: List[str] = []
        self._embeddings: Optional[np.ndarray] = None  # shape (N, 512)

        self._init_model(det_size)
        self._load_dataset()

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_model(self, det_size: Tuple[int, int]) -> None:
        """
        Prepare the InsightFace pipeline.
        Models are downloaded automatically on first run (~500 MB).
        CUDA is used when its runtime dependencies are available; otherwise use CPU.
        """
        for providers, ctx_id in (
            (["CUDAExecutionProvider", "CPUExecutionProvider"], 0),
            (["CPUExecutionProvider"], -1),
        ):
            try:
                model = FaceAnalysis(name="buffalo_l", providers=providers)
                model.prepare(ctx_id=ctx_id, det_size=det_size)
                model.get(np.zeros((det_size[1], det_size[0], 3), dtype=np.uint8))
                self.model = model
                provider = "CUDA" if ctx_id == 0 else "CPU"
                print(f"[INFO] InsightFace inference provider: {provider}")
                return
            except Exception as exc:
                if ctx_id == 0:
                    print(f"[WARN] CUDA inference unavailable ({exc}); retrying with CPU.")
                else:
                    raise

    def _load_dataset(self) -> None:
        """
        Walk the dataset folder, detect the primary face in each image,
        and store its L2-normalized embedding together with the person's name.
        """
        raw_embeddings: List[np.ndarray] = []

        for filename in sorted(os.listdir(self.dataset_path)):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in self.SUPPORTED_EXT:
                continue

            label = os.path.splitext(filename)[0].replace("_", " ")
            img_path = os.path.join(self.dataset_path, filename)

            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Cannot read image: {img_path}")
                continue

            faces = self.model.get(img)
            if not faces:
                print(f"[WARN] No face detected in: {img_path}")
                continue

            # Use the largest face when the image contains multiple people
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )

            self._names.append(label)
            raw_embeddings.append(self._l2_normalize(face.embedding))
            print(f"[INFO] Loaded  →  {label}")

        if raw_embeddings:
            # Stack into a single matrix for vectorized similarity queries
            self._embeddings = np.stack(raw_embeddings).astype(np.float32)

        print(f"\n[INFO] Dataset ready: {len(self._names)} registered person(s)\n")

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        """Return the L2-normalized version of vector v."""
        norm = np.linalg.norm(v)
        return (v / norm).astype(np.float32) if norm > 0.0 else v.astype(np.float32)

    def _best_match(self, embedding: np.ndarray) -> Tuple[str, float]:
        """
        Find the closest identity in the dataset.

        Returns:
            (name, cosine_similarity)  — name is "Unknown" when below threshold.
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return "Unknown", 0.0

        query = self._l2_normalize(embedding)
        # Dot product of normalized vectors == cosine similarity
        scores = self._embeddings @ query          # shape (N,)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= self.threshold:
            return self._names[best_idx], best_score
        return "Unknown", best_score

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect and identify every face in a BGR frame.

        Returns:
            List of dicts, one per face:
                bbox       (x1, y1, x2, y2) — pixel coordinates
                name       str               — label or "Unknown"
                score      float             — cosine similarity
                recognized bool             — True when above threshold
        """
        faces = self.model.get(frame)
        results: List[Dict] = []

        for face in faces:
            name, score = self._best_match(face.embedding)
            x1, y1, x2, y2 = face.bbox.astype(int)
            results.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "name": name,
                    "score": score,
                    "recognized": name != "Unknown",
                    "appearance_cues": detect_appearance_cues(
                        frame, (x1, y1, x2, y2)
                    ),
                }
            )

        return results

    @property
    def num_registered(self) -> int:
        """Number of persons loaded from the dataset."""
        return len(self._names)

    @property
    def registered_names(self) -> List[str]:
        """Names available to the classroom roster dashboard."""
        return list(self._names)

    def add_person(self, name: str, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Register a new person at runtime from a live camera frame.

        Re-detects faces in `frame` and picks the one closest to `bbox`
        (the face that triggered the "unknown" prompt), stores its
        embedding in memory for immediate recognition, and saves a
        cropped image into the dataset folder so it also survives restarts.

        Returns:
            True on success, False if no face could be found in `frame`.
        """
        faces = self.model.get(frame)
        if not faces:
            return False

        target_center = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        face = min(
            faces,
            key=lambda f: np.linalg.norm(
                np.array([(f.bbox[0] + f.bbox[2]) / 2.0, (f.bbox[1] + f.bbox[3]) / 2.0]) - target_center
            ),
        )

        embedding = self._l2_normalize(face.embedding).reshape(1, -1)
        if self._embeddings is None:
            self._embeddings = embedding
        else:
            self._embeddings = np.vstack([self._embeddings, embedding])
        self._names.append(name)

        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = frame.shape[:2]
        pad_x = int((x2 - x1) * 0.3)
        pad_y = int((y2 - y1) * 0.3)
        cx1, cy1 = max(x1 - pad_x, 0), max(y1 - pad_y, 0)
        cx2, cy2 = min(x2 + pad_x, w), min(y2 + pad_y, h)
        crop = frame[cy1:cy2, cx1:cx2]

        safe_name = name.strip().replace(" ", "_")
        path = os.path.join(self.dataset_path, f"{safe_name}.jpg")
        cv2.imwrite(path, crop)

        print(f"[INFO] Added new person → {name}  ({path})")
        return True
