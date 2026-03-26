"""
coin_loader.py -- Download and parse CoIN-Bench episodes from HuggingFace.

This loader supplies **episode metadata** and a **content-aware observation resolver** for
auxiliary offline integration tests. It does **not** implement online navigation or official SR/SPL.

Image resolution policy (especially for ``offline_static_coin``):
  Only paths that can be linked from scene JSON metadata to the **episode** and/or
  **target object** are returned as candidates. We **never** fall back to "first image in
  scene folder" -- that heuristic is intentionally not implemented here.

Dataset layouts vary across releases; all resolution steps are defensive and heavily commented.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

SPLITS = ["val_seen", "val_seen_synonyms", "val_unseen"]


@dataclass
class CoINEpisode:
    """Single CoIN-Bench evaluation episode."""
    episode_id: str
    scene_id: str
    target_category: str
    target_description: str
    start_position: list[float]
    start_rotation: list[float]
    target_position: list[float]
    target_object_id: int | None = None
    distractors: list[dict] = field(default_factory=list)
    split: str = ""

    @property
    def num_distractors(self) -> int:
        return len(self.distractors)


@dataclass
class CoINScene:
    """Per-scene content data from CoIN-Bench."""
    scene_id: str
    episodes: list[dict] = field(default_factory=list)
    object_instances: list[dict] = field(default_factory=list)
    target_images: dict = field(default_factory=dict)


class CoINBenchLoader:
    """
    Downloads and parses the CoIN-Bench dataset from HuggingFace Hub and resolves
    episode-linked static images from ``content/*.json(.gz)`` when possible.
    """

    def __init__(self, local_dir: str = "./data/CoIN-Bench"):
        self._local_dir = Path(local_dir)
        self._repo_id = "ftaioli/CoIN-Bench"

    # ------------------------------------------------------------------
    # Local presence / download
    # ------------------------------------------------------------------

    def has_local_data(self) -> bool:
        """True if the local directory exists and at least one known split is present."""
        if not self._local_dir.exists():
            return False
        return any((self._local_dir / s).exists() for s in SPLITS)

    def ensure_downloaded(self) -> None:
        """Idempotent: fetch dataset from the Hub if splits are missing."""
        if self.has_local_data():
            return
        self.download()

    def download(self, force: bool = False) -> None:
        """Download CoIN-Bench from HuggingFace Hub."""
        if self._local_dir.exists() and not force:
            if any((self._local_dir / s).exists() for s in SPLITS):
                print(f"[CoINBenchLoader] Dataset already exists at {self._local_dir}")
                return

        from huggingface_hub import snapshot_download

        print(f"[CoINBenchLoader] Downloading {self._repo_id}...")
        snapshot_download(
            repo_id=self._repo_id,
            repo_type="dataset",
            local_dir=str(self._local_dir),
        )
        print(f"[CoINBenchLoader] Downloaded to {self._local_dir}")

    def get_split_dir(self, split: str) -> Path:
        return self._local_dir / split

    def get_content_dir(self, split: str) -> Path:
        """Directory holding per-scene content JSON and/or imagery (layout varies)."""
        return self.get_split_dir(split) / "content"

    # ------------------------------------------------------------------
    # JSON loading
    # ------------------------------------------------------------------

    def _load_gzip_json(self, path: Path) -> dict | list:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)

    def load_scene_content(self, split: str, scene_id: str) -> dict | None:
        """
        Load parsed JSON for one scene.

        Tries, in order:
          - ``content/<scene_id>.json.gz``
          - ``content/<scene_id>.json``
          - ``content/<scene_id>/<scene_id>.json.gz`` (nested layout)
        Returns None if no readable file exists or JSON is not a dict.
        """
        cdir = self.get_content_dir(split)
        if not cdir.exists():
            return None

        candidates = [
            cdir / f"{scene_id}.json.gz",
            cdir / f"{scene_id}.json",
            cdir / scene_id / f"{scene_id}.json.gz",
            cdir / scene_id / f"{scene_id}.json",
        ]
        for p in candidates:
            if not p.exists():
                continue
            try:
                if p.suffix == ".gz":
                    data = self._load_gzip_json(p)
                else:
                    with open(p, encoding="utf-8") as f:
                        data = json.load(f)
                if isinstance(data, dict):
                    return data
            except (OSError, json.JSONDecodeError, gzip.BadGzipFile):
                continue
        return None

    # ------------------------------------------------------------------
    # Episode-linked image candidates (trustworthy only)
    # ------------------------------------------------------------------

    def _resolve_existing_path(
        self,
        split: str,
        scene_id: str,
        raw: str | Path,
    ) -> Path | None:
        """Map a string from JSON to an on-disk file under the split, if it exists."""
        if not raw:
            return None
        p = Path(raw)
        if p.is_absolute() and p.is_file():
            return p
        split_dir = self.get_split_dir(split)
        content_dir = self.get_content_dir(split)
        trials = [
            content_dir / raw,
            content_dir / scene_id / raw,
            split_dir / raw,
            self._local_dir / raw,
        ]
        for t in trials:
            try:
                if t.is_file():
                    return t.resolve()
            except OSError:
                continue
        # basename-only fallback within scene subfolder (still metadata-driven path stem)
        base = os.path.basename(str(raw))
        for parent in (content_dir / scene_id, content_dir):
            cand = parent / base
            if cand.is_file():
                return cand.resolve()
        return None

    @staticmethod
    def _collect_path_like_values(obj: Any, out: list[str]) -> None:
        """Recursively collect strings that look like image file paths."""
        if isinstance(obj, str):
            lower = obj.lower()
            if any(lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
                out.append(obj)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("image", "rgb", "path", "file", "observation", "frame", "uri"):
                    if isinstance(v, str):
                        out.append(v)
                CoINBenchLoader._collect_path_like_values(v, out)
        elif isinstance(obj, list):
            for item in obj:
                CoINBenchLoader._collect_path_like_values(item, out)

    def get_episode_image_candidates(self, split: str, episode: CoINEpisode) -> list[Path]:
        """
        Return image paths that can be **trustworthily** associated with this episode.

        Strategy (all best-effort; returns [] if nothing defensible is found):

        1. Load per-scene ``content`` JSON. If missing -> [].

        2. If ``episodes`` array exists, find the entry whose ``episode_id`` / ``id`` matches
           ``episode.episode_id`` and collect any image-like path fields on **that** object only.

        3. If ``episode.target_object_id`` is set, find ``objects`` / ``instances`` entry with
           matching ``object_id`` / ``id`` and collect image paths from **that** object.

        4. If ``target_images`` maps object_id -> path(s), use the entry for ``target_object_id``.

        5. Deduplicate, keep only files that exist on disk.

        Never scans arbitrary scene directories for "any image" -- only JSON-linked paths.
        """
        content = self.load_scene_content(split, episode.scene_id)
        if not content or not isinstance(content, dict):
            return []

        string_candidates: list[str] = []

        eid = str(episode.episode_id).strip()
        for ep_raw in content.get("episodes") or []:
            if not isinstance(ep_raw, dict):
                continue
            rid = str(ep_raw.get("episode_id", ep_raw.get("id", ""))).strip()
            if rid and rid == eid:
                for key in (
                    "image", "rgb", "observation", "frame_path", "image_path",
                    "file", "path", "rgb_path", "observation_path",
                ):
                    if key in ep_raw:
                        CoINBenchLoader._collect_path_like_values(ep_raw[key], string_candidates)
                CoINBenchLoader._collect_path_like_values(ep_raw, string_candidates)

        tid = episode.target_object_id
        if tid is not None:
            for obj in content.get("objects") or content.get("instances") or []:
                if not isinstance(obj, dict):
                    continue
                oid = obj.get("object_id", obj.get("id"))
                if oid is None:
                    continue
                try:
                    if int(oid) != int(tid):
                        continue
                except (TypeError, ValueError):
                    if str(oid) != str(tid):
                        continue
                for key in (
                    "image", "rgb", "image_path", "observation_path", "path",
                    "frames", "images", "rgb_path",
                ):
                    if key in obj:
                        CoINBenchLoader._collect_path_like_values(obj[key], string_candidates)

        timg = content.get("target_images")
        if isinstance(timg, dict) and tid is not None:
            for key in (str(tid), tid):
                if key in timg:
                    CoINBenchLoader._collect_path_like_values(timg[key], string_candidates)

        # Dedupe while preserving order
        seen: set[str] = set()
        resolved: list[Path] = []
        for s in string_candidates:
            if s in seen:
                continue
            seen.add(s)
            rp = self._resolve_existing_path(split, episode.scene_id, s)
            if rp is not None:
                resolved.append(rp)

        return resolved

    def available_splits(self) -> list[str]:
        return [s for s in SPLITS if (self._local_dir / s).exists()]

    def load_episodes(self, split: str) -> list[CoINEpisode]:
        split_dir = self._local_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split '{split}' not found at {split_dir}. Run download() first."
            )

        index_file = split_dir / f"{split}.json.gz"
        if not index_file.exists():
            index_file = split_dir / f"{split}.json"

        episodes: list[CoINEpisode] = []

        if index_file.exists():
            if index_file.suffix == ".gz":
                raw_episodes = self._load_gzip_json(index_file)
            else:
                with open(index_file, encoding="utf-8") as f:
                    raw_episodes = json.load(f)
            if isinstance(raw_episodes, list):
                for raw in raw_episodes:
                    episodes.append(self._parse_episode(raw, split))
            elif isinstance(raw_episodes, dict):
                for raw in raw_episodes.get("episodes", []):
                    episodes.append(self._parse_episode(raw, split))

        content_dir = split_dir / "content"
        if content_dir.exists():
            scene_data: dict[str, dict] = {}
            for gz_file in sorted(content_dir.glob("*.json.gz")):
                scene_id = gz_file.stem.replace(".json", "")
                data = self._load_gzip_json(gz_file)
                if isinstance(data, dict):
                    scene_data[scene_id] = data

            for gz_file in sorted(content_dir.glob("*.json")):
                if gz_file.name.endswith(".json.gz"):
                    continue
                scene_id = gz_file.stem
                if scene_id not in scene_data:
                    try:
                        with open(gz_file, encoding="utf-8") as f:
                            data = json.load(f)
                        if isinstance(data, dict):
                            scene_data[scene_id] = data
                    except (OSError, json.JSONDecodeError):
                        pass

            if not episodes:
                for sid, data in scene_data.items():
                    if isinstance(data, dict):
                        for ep_raw in data.get("episodes", []):
                            if isinstance(ep_raw, dict):
                                ep_raw = {**ep_raw, "scene_id": ep_raw.get("scene_id", sid)}
                                episodes.append(self._parse_episode(ep_raw, split))

            for ep in episodes:
                if ep.scene_id in scene_data:
                    sd = scene_data[ep.scene_id]
                    if isinstance(sd, dict):
                        objects = sd.get("objects", sd.get("instances", []))
                        if isinstance(objects, list):
                            ep.distractors = [
                                o for o in objects
                                if isinstance(o, dict)
                                and str(o.get("category", "")).lower() == ep.target_category.lower()
                                and o.get("object_id") != ep.target_object_id
                            ]

        print(f"[CoINBenchLoader] Loaded {len(episodes)} episodes from '{split}'")
        return episodes

    def _parse_episode(self, raw: dict, split: str) -> CoINEpisode:
        scene_id = raw.get("scene_id", raw.get("scene", ""))
        target_cat = raw.get(
            "object_category",
            raw.get("target_category", raw.get("category", "")),
        )
        target_desc = raw.get(
            "description",
            raw.get("target_description", raw.get("lang_desc", "")),
        )
        start_pos = raw.get("start_position", [0, 0, 0])
        start_rot = raw.get("start_rotation", [1, 0, 0, 0])
        target_pos = raw.get(
            "target_position",
            raw.get("goals_position", raw.get("goal_position", [0, 0, 0])),
        )
        goals = raw.get("goals", [])
        if goals and isinstance(goals, list) and isinstance(goals[0], dict):
            target_pos = goals[0].get("position", target_pos)

        oid = raw.get("object_id", raw.get("target_object_id"))
        if oid is not None:
            try:
                oid = int(oid)
            except (TypeError, ValueError):
                pass

        return CoINEpisode(
            episode_id=str(raw.get("episode_id", raw.get("id", ""))),
            scene_id=scene_id,
            target_category=target_cat,
            target_description=target_desc,
            start_position=start_pos,
            start_rotation=start_rot,
            target_position=target_pos,
            target_object_id=oid,
            split=split,
        )

    def iter_episodes(self, split: str) -> Iterator[CoINEpisode]:
        yield from self.load_episodes(split)

    def get_all_episodes(self) -> dict[str, list[CoINEpisode]]:
        result = {}
        for split in self.available_splits():
            result[split] = self.load_episodes(split)
        return result

    def summary(self) -> dict:
        stats = {}
        for split in self.available_splits():
            eps = self.load_episodes(split)
            cats = set(e.target_category for e in eps)
            scenes = set(e.scene_id for e in eps)
            stats[split] = {
                "num_episodes": len(eps),
                "num_scenes": len(scenes),
                "num_categories": len(cats),
                "categories": sorted(cats),
                "avg_distractors": (
                    sum(e.num_distractors for e in eps) / len(eps) if eps else 0
                ),
            }
        return stats
