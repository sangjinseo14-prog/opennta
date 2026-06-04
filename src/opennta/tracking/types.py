"""Shared tracking-tab value types."""

from __future__ import annotations


class ProcessingParams:
    def __init__(
        self,
        normalization_percentile: float = 99.5,
        particle_radius_pixels: float = 6,
        fit_fraction: float = 0.85,
        significance_alpha: float = 0.05,
        linking_max_distance: int = 10,
        gap_closing_max_distance: int = 10,
        max_frame_gap: int = 0,
        min_track_frames: int = 0,
        border_margin_pixels: int = 0,
        method: str = "trackmate",
        detection_method: str = "",
        linking_method: str = "",
        quality_model: str = "Cheng-Schwartzman",
    ) -> None:
        self.normalization_percentile: float = float(normalization_percentile)
        self.particle_radius_pixels: float = float(particle_radius_pixels)
        self.fit_fraction: float = float(fit_fraction)
        self.significance_alpha: float = float(significance_alpha)
        self.linking_max_distance: int = int(linking_max_distance)
        self.gap_closing_max_distance: int = int(gap_closing_max_distance)
        self.max_frame_gap: int = int(max_frame_gap)
        self.min_track_frames: int = int(min_track_frames)
        self.border_margin_pixels: int = int(border_margin_pixels)
        self.method: str = str(method or "trackmate").lower()
        self.detection_method: str = str(detection_method or "").lower()
        self.linking_method: str = str(linking_method or "").lower()
        self.quality_model: str = str(quality_model or "Cheng-Schwartzman")


# (sub_path, tiff_paths, nta_folder, sub_name)
FolderItem = tuple[str, list[str], str, str]


__all__ = ["FolderItem", "ProcessingParams"]
