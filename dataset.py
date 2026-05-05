"""Online ViewTransfer dataset.

Workers do disk I/O only (PNG decode + EXR depth load).  All GPU work — equi2pers,
**lift-and-render from static panorama**, VAE encode, plücker compute, mask packing —
happens in the main process inside `gpu_preprocess`, driven by `prefetcher.py`.

Per-location data layout:

    location_dir/
        c2w_PanoCam_00.pt              [T_video, 4, 4]   OpenCV (run_prep cameras)
        c2w_PanoCam_01.pt              [T_video, 4, 4]
        text_emb.pt                    [L, 4096]         (run_prep text)
        Pano_00/rgb/*.png              dynamic 360 (UE-rendered, e.g. 4096x2048)
        Pano_00_static/rgb/*.png       static 360 (no agents)
        Pano_00_static/depth/*.exr     static depth (radial distance, meters)
        Pano_01/rgb/*.png
        Pano_01_static/rgb/*.png
        Pano_01_static/depth/*.exr
        Pano_00/blobs/*.png            user pre-computed gaussian blobs (rendered at Pano_00 cam)
        Pano_01/blobs/*.png            (rendered at Pano_01 cam)

The dataset emits two entries per location with both panos: 00→01 and 01→00.
"""

from __future__ import annotations

import os
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np

from .prepare_data.video_io import load_png_sequence
from .prepare_data.extract_perspectives import sample_perspective_trajectory
from .prepare_data.lift_and_render import load_depth, load_depth_ue


@dataclass
class SampleEntry:
    """One source/target panorama pairing within a location."""

    location_dir: str
    src_idx: str  # "00" or "01"
    tgt_idx: str

    @property
    def rgb_src_dir(self) -> str:
        return os.path.join(self.location_dir, f"Pano_{self.src_idx}", "rgb")

    @property
    def rgb_tgt_dir(self) -> str:
        return os.path.join(self.location_dir, f"Pano_{self.tgt_idx}", "rgb")

    @property
    def static_rgb_dir(self) -> str:
        return os.path.join(self.location_dir, f"Pano_{self.src_idx}_static", "rgb")

    @property
    def static_depth_dir(self) -> str:
        return os.path.join(self.location_dir, f"Pano_{self.src_idx}_static", "depth")

    @property
    def blob_dir(self) -> str:
        # Blob video is rendered at the target pano's camera positions, stored
        # under the target pano's directory: Pano_{tgt}/blobs/.
        return os.path.join(self.location_dir, f"Pano_{self.tgt_idx}", "blobs")

    @property
    def c2w_src_path(self) -> str:
        return os.path.join(self.location_dir, f"c2w_PanoCam_{self.src_idx}.pt")

    @property
    def c2w_tgt_path(self) -> str:
        return os.path.join(self.location_dir, f"c2w_PanoCam_{self.tgt_idx}.pt")

    @property
    def text_emb_path(self) -> str:
        return os.path.join(self.location_dir, "text_emb.pt")

    def is_complete(self, verbose: bool = False) -> bool:
        """Existence + length-consistency check.

        Returns True iff:
          1. Every required directory and file exists.
          2. All sequence sources have an identical length:
             - PNG counts in `rgb_src`, `rgb_tgt`, `static_rgb`, `blob`
             - depth-file count in `static_depth` (.exr / .npy / .pt / .pth)
             - pose-tensor lengths in `c2w_src` and `c2w_tgt`
             — and that common length is > 0.

        The length check catches partially-prepared locations like the one we
        hit in training: `c2w_*.pt` had T poses but `Pano_XX_static/depth/` only
        had 7 EXRs, so a t0 sampled against the c2w length would IndexError on
        depth load. With this check, such locations are skipped at discovery.

        If `verbose=True`, prints a one-line reason for any rejection so the
        operator can see which locations are bad.
        """
        def _fail(reason: str) -> bool:
            if verbose:
                print(
                    f"  [skip] {self.location_dir}  src={self.src_idx} tgt={self.tgt_idx}: {reason}"
                )
            return False

        # 1. Existence
        for p in (
            self.rgb_src_dir, self.rgb_tgt_dir,
            self.static_rgb_dir, self.static_depth_dir, self.blob_dir,
        ):
            if not os.path.isdir(p):
                return _fail(f"missing dir {os.path.relpath(p, self.location_dir)}")
        for p in (self.c2w_src_path, self.c2w_tgt_path, self.text_emb_path):
            if not os.path.exists(p):
                return _fail(f"missing file {os.path.basename(p)}")

        # 2. Length consistency
        def _count(d: str, exts: tuple) -> int:
            try:
                return sum(1 for f in os.listdir(d) if f.lower().endswith(exts))
            except OSError:
                return -1

        counts = {
            "rgb_src":      _count(self.rgb_src_dir,    (".png",)),
            "rgb_tgt":      _count(self.rgb_tgt_dir,    (".png",)),
            "static_rgb":   _count(self.static_rgb_dir, (".png",)),
            "blob":         _count(self.blob_dir,       (".png",)),
            "static_depth": _count(self.static_depth_dir, (".exr", ".npy", ".pt", ".pth")),
        }

        # c2w tensor shapes (cheap; small files, sub-ms each)
        try:
            counts["c2w_src"] = torch.load(
                self.c2w_src_path, map_location="cpu", weights_only=True,
            ).shape[0]
            counts["c2w_tgt"] = torch.load(
                self.c2w_tgt_path, map_location="cpu", weights_only=True,
            ).shape[0]
        except Exception as e:
            return _fail(f"failed to load c2w tensors: {e}")

        if any(c <= 0 for c in counts.values()):
            return _fail(f"empty source: {counts}")
        if len(set(counts.values())) > 1:
            return _fail(f"inconsistent lengths: {counts}")
        return True


def discover_entries(
    data_root: Optional[str] = None,
    require_complete: bool = True,
    locations: Optional[List[str]] = None,
    verbose: bool = False,
) -> List[SampleEntry]:
    """Emit valid (src→tgt) entries.

    Enumerates all 4 pairings (00,00), (00,01), (01,00), (01,01) per location and
    keeps the ones that pass `SampleEntry.is_complete()` (existence + sequence-length
    consistency across rgb / depth / blob / c2w). In the current UE renders only
    `Pano_00_static` exists, so only `src="00"` pairings will pass. (00,00) is a
    same-camera pairing useful for pure-rotation training; (00,01) gives translation.

    Either `data_root` (walk scene/location subdirs) or `locations` (an explicit
    list of location_dir paths) must be provided.

    If `verbose=True`, prints a per-rejection reason from `is_complete()` and a
    final summary count.
    """
    if (data_root is None) == (locations is None):
        raise ValueError("Provide exactly one of `data_root` or `locations`.")

    if data_root is not None:
        loc_dirs: List[str] = []
        for scene in sorted(os.listdir(data_root)):
            scene_dir = os.path.join(data_root, scene)
            if not os.path.isdir(scene_dir):
                continue
            for loc in sorted(os.listdir(scene_dir)):
                loc_dir = os.path.join(scene_dir, loc)
                if os.path.isdir(loc_dir):
                    loc_dirs.append(loc_dir)
    else:
        loc_dirs = list(locations)

    entries: List[SampleEntry] = []
    n_skipped = 0
    for loc_dir in loc_dirs:
        for s in ("00", "01"):
            for t in ("00", "01"):
                e = SampleEntry(location_dir=loc_dir, src_idx=s, tgt_idx=t)
                if (not require_complete) or e.is_complete(verbose=verbose):
                    entries.append(e)
                else:
                    n_skipped += 1
    if verbose:
        print(
            f"discover_entries: kept {len(entries)} / {len(entries) + n_skipped} "
            f"({n_skipped} skipped)"
        )
    return entries


def load_locations_file(path: str) -> List[str]:
    """Read a newline-delimited list of location dirs (skips blanks and `#` comments)."""
    out: List[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


def _list_files(dir_path: str, exts: tuple) -> List[str]:
    files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(exts)])
    return [os.path.join(dir_path, f) for f in files]


class ViewTransferDataset(Dataset):
    """Returns raw 360 windows + ONE static RGB+depth frame at the chosen t0.

    Per sample dict:
      rgb_src_360       [3, T, He, We]  uint8
      rgb_tgt_360       [3, T, He, We]  uint8     (GT)
      blob_360          [3, T, He, We]  uint8
      static_rgb_t0     [3, He, We]     uint8     source-static at frame t0
      static_depth_t0   [He, We]        float32   radial distance, meters
      pano_c2w_src      [T, 4, 4]       float32   windowed
      pano_c2w_tgt      [T, 4, 4]       float32
      src_c2w_at_t0     [4, 4]          float32   for lifting (= pano_c2w_src[0])
      text_emb          [L, 4096]       float32
      src_*_deg         [T] / scalar    perspective trajectory parameters (source view)
      tgt_*_deg         [T] / scalar    perspective trajectory parameters (target view)
      t_offset          int             chosen t0 in the original sequence
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        num_video_frames: int = 81,
        same_orientation: bool = False,
        seed: int = 0,
        locations_file: Optional[str] = None,
    ):
        if locations_file is not None:
            locations = load_locations_file(locations_file)
            self.entries: List[SampleEntry] = discover_entries(
                locations=locations, require_complete=True,
            )
            source_desc = locations_file
        else:
            self.entries = discover_entries(data_root=data_root, require_complete=True)
            source_desc = data_root
        if not self.entries:
            raise RuntimeError(f"No complete sample entries found from {source_desc}")
        self.num_video_frames = num_video_frames
        self.same_orientation = same_orientation
        self.base_seed = seed

    def __len__(self) -> int:
        return len(self.entries)

    def _make_rng(self, idx: int) -> np.random.Generator:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        salt = torch.randint(0, 2**31 - 1, (1,)).item()
        return np.random.default_rng(self.base_seed + idx * 7919 + worker_id * 11 + salt)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.entries[idx]
        rng = self._make_rng(idx)

        pano_c2w_src = torch.load(entry.c2w_src_path, map_location="cpu", weights_only=True).float()
        pano_c2w_tgt = torch.load(entry.c2w_tgt_path, map_location="cpu", weights_only=True).float()
        T_full = pano_c2w_src.shape[0]
        assert pano_c2w_tgt.shape[0] == T_full

        if T_full > self.num_video_frames:
            t0 = int(rng.integers(0, T_full - self.num_video_frames + 1))
        else:
            t0 = 0
        t_slice = slice(t0, t0 + self.num_video_frames)

        # 360 windows (uint8 to keep memory small until GPU)
        def _load_window(d):
            return load_png_sequence(
                d, start=t0, num_frames=self.num_video_frames, return_dtype=torch.uint8,
            ).permute(1, 0, 2, 3).contiguous()  # [3, T, H, W]

        rgb_src = _load_window(entry.rgb_src_dir)
        rgb_tgt = _load_window(entry.rgb_tgt_dir)
        blob = _load_window(entry.blob_dir)

        # Single source-static frame at t0 (RGB + radial depth)
        static_rgb_t0 = load_png_sequence(
            entry.static_rgb_dir, start=t0, num_frames=1, return_dtype=torch.uint8,
        )[0].permute(0, 1, 2)  # [3, He, We]

        depth_files = _list_files(entry.static_depth_dir, exts=(".exr", ".npy", ".pt", ".pth"))
        if t0 >= len(depth_files):
            raise IndexError(
                f"t0={t0} but only {len(depth_files)} depth frames in {entry.static_depth_dir}"
            )
        # UE depth is in cm with float16-max sentinel for sky.  For non-EXR formats
        # (tests) assume already-converted meters.
        if depth_files[t0].endswith(".exr"):
            depth_np = load_depth_ue(depth_files[t0])
        else:
            depth_np = load_depth(depth_files[t0])
        static_depth_t0 = torch.from_numpy(depth_np)

        # Trajectories
        src_traj = sample_perspective_trajectory(self.num_video_frames, rng=rng)
        if self.same_orientation:
            tgt_traj = {
                k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in src_traj.items()
            }
        else:
            tgt_traj = sample_perspective_trajectory(self.num_video_frames, rng=rng)

        text_emb = torch.load(entry.text_emb_path, map_location="cpu", weights_only=True).float()

        pano_c2w_src_win = pano_c2w_src[t_slice].clone()
        pano_c2w_tgt_win = pano_c2w_tgt[t_slice].clone()

        return {
            "rgb_src_360": rgb_src,
            "rgb_tgt_360": rgb_tgt,
            "blob_360": blob,
            "static_rgb_t0": static_rgb_t0,
            "static_depth_t0": static_depth_t0,
            "pano_c2w_src": pano_c2w_src_win,
            "pano_c2w_tgt": pano_c2w_tgt_win,
            "src_c2w_at_t0": pano_c2w_src_win[0].clone(),  # frame at t0
            "text_emb": text_emb,
            "src_fov_h_deg": float(src_traj["fov_h_deg"]),
            "src_yaw_deg": torch.from_numpy(src_traj["yaw_deg"]),
            "src_pitch_deg": torch.from_numpy(src_traj["pitch_deg"]),
            "src_roll_deg": torch.from_numpy(src_traj["roll_deg"]),
            "tgt_fov_h_deg": float(tgt_traj["fov_h_deg"]),
            "tgt_yaw_deg": torch.from_numpy(tgt_traj["yaw_deg"]),
            "tgt_pitch_deg": torch.from_numpy(tgt_traj["pitch_deg"]),
            "tgt_roll_deg": torch.from_numpy(tgt_traj["roll_deg"]),
            "t_offset": t0,
            # Provenance (used by train-time artifact logging)
            "location_dir": entry.location_dir,
            "src_idx": entry.src_idx,
            "tgt_idx": entry.tgt_idx,
        }


def collate_view_transfer(batch: List[Dict]) -> Dict:
    """Collate a list of samples into a batched dict."""
    out: Dict = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            if k == "text_emb":
                max_L = max(v.shape[0] for v in vals)
                padded = []
                for v in vals:
                    if v.shape[0] < max_L:
                        pad = torch.zeros(max_L - v.shape[0], v.shape[1], dtype=v.dtype)
                        v = torch.cat([v, pad], dim=0)
                    padded.append(v)
                out[k] = torch.stack(padded, dim=0)
            else:
                out[k] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], (int, float)):
            out[k] = torch.tensor(vals)
        else:
            out[k] = vals
    return out
