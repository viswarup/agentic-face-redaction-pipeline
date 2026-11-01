"""Central configuration module for Face Anonymization & Verification Pipeline."""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AppConfig(BaseModel):
    # Base and Data Directory Paths
    BASE_DIR: Path = Field(default_factory=lambda: Path(os.getenv("BASE_DIR", str(PROJECT_ROOT))))
    DATA_DIR: Path = Field(default_factory=lambda: Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))))
    IMAGES_100_DIR: Path = Field(
        default_factory=lambda: Path(os.getenv("IMAGES_100_DIR", str(PROJECT_ROOT / "data" / "images_100")))
    )
    VIDEO_SNIPPETS_DIR: Path = Field(
        default_factory=lambda: Path(os.getenv("VIDEO_SNIPPETS_DIR", str(PROJECT_ROOT / "data" / "video_snippets")))
    )
    REDACTED_VIDEOS_DIR: Path = Field(
        default_factory=lambda: Path(
            os.getenv("REDACTED_VIDEOS_DIR", str(PROJECT_ROOT / "data" / "outputs" / "redacted_videos"))
        )
    )
    EVAL_100_REDACTED_DIR: Path = Field(
        default_factory=lambda: Path(
            os.getenv("EVAL_100_REDACTED_DIR", str(PROJECT_ROOT / "data" / "outputs" / "eval_100_redacted"))
        )
    )
    DIAGNOSTICS_DIR: Path = Field(
        default_factory=lambda: Path(
            os.getenv("DIAGNOSTICS_DIR", str(PROJECT_ROOT / "data" / "outputs" / "diagnostics"))
        )
    )
    JOBS_DB_PATH: Path = Field(
        default_factory=lambda: Path(os.getenv("JOBS_DB_PATH", str(PROJECT_ROOT / "data" / "jobs.db")))
    )

    # Primary detector settings
    PRIMARY_MODEL: str = Field(default_factory=lambda: os.getenv("PRIMARY_MODEL", "scrfd_34g"))
    PRIMARY_FALLBACK_MODELS: list[str] = Field(
        default_factory=lambda: ["buffalo_l", "scrfd_10g", "yolov11-face"]
    )
    PRIMARY_CONF_THRESH: float = Field(default_factory=lambda: float(os.getenv("PRIMARY_CONF_THRESH", "0.25")))
    BBOX_DILATION_RATIO: float = Field(default_factory=lambda: float(os.getenv("BBOX_DILATION_RATIO", "0.28")))
    TRACKER_MAX_AGE: int = Field(default_factory=lambda: int(os.getenv("TRACKER_MAX_AGE", "10")))
    INTERPOLATION_WINDOW: int = Field(default_factory=lambda: int(os.getenv("INTERPOLATION_WINDOW", "12")))

    # Verifier settings
    VERIFIER_BACKBONE: str = Field(
        default_factory=lambda: os.getenv("VERIFIER_BACKBONE", "retinaface_r152_or_r50")
    )
    RETINAFACE_R152_PATH: Optional[str] = Field(
        default_factory=lambda: os.getenv("RETINAFACE_R152_PATH", None)
    )
    VERIFIER_CONF_THRESH: float = Field(default_factory=lambda: float(os.getenv("VERIFIER_CONF_THRESH", "0.30")))
    VERIFIER_STRIDE: int = Field(default_factory=lambda: int(os.getenv("VERIFIER_STRIDE", "2")))

    def ensure_directories(self) -> None:
        """Create required directories on disk if they do not exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.IMAGES_100_DIR.mkdir(parents=True, exist_ok=True)
        self.VIDEO_SNIPPETS_DIR.mkdir(parents=True, exist_ok=True)
        self.REDACTED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        self.EVAL_100_REDACTED_DIR.mkdir(parents=True, exist_ok=True)
        self.DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
        self.JOBS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


config = AppConfig()
config.ensure_directories()
