"""
HIT137 Assignment 3 - Spot the Difference Game
A desktop application using Tkinter GUI and OpenCV image processing.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import random
import math
import os

# ─────────────────────────────────────────────
# Class 1: DifferenceRegion
#   Represents a single difference region in the image.
# ─────────────────────────────────────────────
class DifferenceRegion:
    """Encapsulates a single rectangular difference region."""

    def __init__(self, x: int, y: int, w: int, h: int, alteration_type: str):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.alteration_type = alteration_type
        self.found = False
        self.cx = x + w // 2   # centre x
        self.cy = y + h // 2   # centre y

    def contains_point(self, px: int, py: int, tolerance: int = 40) -> bool:
        """Return True if (px, py) is within tolerance pixels of the region centre."""
        dist = math.sqrt((px - self.cx) ** 2 + (py - self.cy) ** 2)
        return dist <= tolerance

    def mark_found(self):
        self.found = True

    def __repr__(self):
        return (f"DifferenceRegion(type={self.alteration_type}, "
                f"pos=({self.x},{self.y}), size=({self.w}x{self.h}), found={self.found})")


# ─────────────────────────────────────────────
# Class 2: ImageProcessor
#   Handles all OpenCV image manipulation.
# ─────────────────────────────────────────────
class ImageProcessor:
    """Handles image loading, cloning, and applying alterations using OpenCV."""

    NUM_DIFFERENCES = 5
    MIN_REGION_SIZE = 40
    MAX_REGION_SIZE = 90
    MARGIN = 20

    ALTERATION_TYPES = [
        "colour_shift",
        "brightness_patch",
        "blur_patch",
        "hue_rotate",
        "contrast_invert",
    ]

    def __init__(self):
        self.original_bgr: np.ndarray | None = None
        self.modified_bgr: np.ndarray | None = None
        self.regions: list[DifferenceRegion] = []
        self.img_h = 0
        self.img_w = 0

    # ── Loading ──────────────────────────────
    def load_image(self, path: str) -> bool:
        """Load image from disk. Returns True on success."""
        img = cv2.imread(path)
        if img is None:
            return False
        self.original_bgr = img
        self.img_h, self.img_w = img.shape[:2]
        return True

    # ── Scale to fit a canvas ────────────────
    def get_display_image(self, bgr: np.ndarray, target_w: int, target_h: int) -> ImageTk.PhotoImage:
        """Scale BGR image to fit (target_w x target_h), preserving aspect ratio."""
        h, w = bgr.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        return ImageTk.PhotoImage(pil_img), new_w, new_h

    # ── Generate Differences ─────────────────
    def generate_differences(self) -> list[DifferenceRegion]:
        """Create a modified clone with 5 non-overlapping random differences."""
        self.modified_bgr = self.original_bgr.copy()
        self.regions = []
        placed_boxes = []

        attempts = 0
        while len(self.regions) < self.NUM_DIFFERENCES and attempts < 500:
            attempts += 1
            w = random.randint(self.MIN_REGION_SIZE, self.MAX_REGION_SIZE)
            h = random.randint(self.MIN_REGION_SIZE, self.MAX_REGION_SIZE)
            x = random.randint(self.MARGIN, self.img_w - w - self.MARGIN)
            y = random.randint(self.MARGIN, self.img_h - h - self.MARGIN)

            # Ensure no overlap with existing regions
            if self._overlaps(x, y, w, h, placed_boxes):
                continue

            alt_type = random.choice(self.ALTERATION_TYPES)
            self._apply_alteration(x, y, w, h, alt_type)
            region = DifferenceRegion(x, y, w, h, alt_type)
            self.regions.append(region)
            placed_boxes.append((x, y, w, h))

        return self.regions

    def _overlaps(self, x, y, w, h, boxes, padding=15) -> bool:
        for bx, by, bw, bh in boxes:
            if (x < bx + bw + padding and x + w + padding > bx and
                    y < by + bh + padding and y + h + padding > by):
                return True
        return False

    # ── Alteration methods ───────────────────
    def _apply_alteration(self, x, y, w, h, alt_type: str):
        roi = self.modified_bgr[y:y+h, x:x+w]

        if alt_type == "colour_shift":
            # Shift hue in HSV space
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.int32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(25, 50)) % 180
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            self.modified_bgr[y:y+h, x:x+w] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        elif alt_type == "brightness_patch":
            # Increase/decrease brightness
            delta = random.choice([-45, -35, 35, 45])
            patch = roi.astype(np.int32) + delta
            self.modified_bgr[y:y+h, x:x+w] = np.clip(patch, 0, 255).astype(np.uint8)

        elif alt_type == "blur_patch":
            # Gaussian blur
            ksize = random.choice([15, 19, 23])
            blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)
            self.modified_bgr[y:y+h, x:x+w] = blurred

        elif alt_type == "hue_rotate":
            # Swap colour channels
            b, g, r = cv2.split(roi)
            swapped = cv2.merge([r, b, g])
            self.modified_bgr[y:y+h, x:x+w] = swapped

        elif alt_type == "contrast_invert":
            # Local contrast enhancement + slight invert
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            l_eq = clahe.apply(l)
            lab_eq = cv2.merge([l_eq, a, b])
            result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
            self.modified_bgr[y:y+h, x:x+w] = result

    # ── Draw Markers ─────────────────────────
    def draw_circle_on_bgr(self, bgr: np.ndarray, region: DifferenceRegion,
                           colour: tuple, thickness: int = 3) -> np.ndarray:
        """Draw a circle around a region on a copy of the image."""
        img = bgr.copy()
        radius = max(region.w, region.h) // 2 + 10
        cv2.circle(img, (region.cx, region.cy), radius, colour, thickness)
        return img

    # ── Polymorphism demo: apply_effect ──────
    def apply_effect(self, effect_name: str, img: np.ndarray) -> np.ndarray:
        """Polymorphic method — apply a named global effect to a full image."""
        if effect_name == "grayscale":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif effect_name == "flip":
            return cv2.flip(img, 1)
        return img


# ─────────────────────────────────────────────
# Class 3: GameState
#   Tracks score, mistakes, and round logic.
# ─────────────────────────────────────────────
class GameState:
    """Manages game progress: score, mistakes, and win/lose conditions."""

    MAX_MISTAKES = 3

    def __init__(self):
        self.total_score = 0
        self.mistakes = 0
        self.locked = False
        self.round_complete = False

    def reset_round(self):
        self.mistakes = 0
        self.locked = False
        self.round_complete = False

    def record_correct(self):
        self.total_score += 1

    def record_mistake(self):
        if not self.locked:
            self.mistakes += 1
            if self.mistakes >= self.MAX_MISTAKES:
                self.locked = True

    def check_complete(self, regions: list[DifferenceRegion]) -> bool:
        if all(r.found for r in regions):
            self.round_complete = True
        return self.round_complete

    @property
    def remaining(self) -> int:
        """Must be called with current regions list — computed externally."""
        return 0  # placeholder; GUI computes this directly


