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


# ─────────────────────────────────────────────
# Class 4: SpotTheDifferenceApp  (inherits tk.Tk)
#   Main application window — GUI + event wiring.
# ─────────────────────────────────────────────
class SpotTheDifferenceApp(tk.Tk):
    """Main Tkinter application. Inherits from tk.Tk (demonstrating inheritance)."""

    CANVAS_W = 520
    CANVAS_H = 420
    ACCENT   = "#E8572A"
    ACCENT2  = "#3ABFB8"
    BG       = "#0F1117"
    SURFACE  = "#1A1D27"
    SURFACE2 = "#252836"
    TEXT     = "#F0F2F5"
    SUBTEXT  = "#8B90A0"
    FOUND_COL_BGR   = (50, 50, 220)   # red in BGR
    UNFOUND_COL_BGR = (220, 150, 30)  # blue in BGR

    def __init__(self):
        super().__init__()
        self.title("Spot the Difference  ·  HIT137")
        self.configure(bg=self.BG)
        self.resizable(True, True)

        self.processor = ImageProcessor()
        self.state     = GameState()

        # Runtime display state
        self._orig_display_bgr: np.ndarray | None = None
        self._mod_display_bgr:  np.ndarray | None = None
        self._display_scale = 1.0
        self._display_offset_x = 0
        self._display_offset_y = 0
        self._canvas_w = self.CANVAS_W
        self._canvas_h = self.CANVAS_H

        self._build_ui()
        self._update_status()

    # ── UI Construction ──────────────────────
    def _build_ui(self):
        # ── Top bar ──
        topbar = tk.Frame(self, bg=self.BG, pady=12)
        topbar.pack(fill=tk.X, padx=24)

        title_lbl = tk.Label(topbar, text="SPOT  THE  DIFFERENCE",
                             font=("Courier New", 16, "bold"),
                             bg=self.BG, fg=self.ACCENT,)
        title_lbl.pack(side=tk.LEFT)

        # Score badge
        self.score_lbl = tk.Label(topbar,
                                  text="SCORE  0",
                                  font=("Courier New", 11, "bold"),
                                  bg=self.ACCENT, fg="white",
                                  padx=12, pady=4)
        self.score_lbl.pack(side=tk.RIGHT, padx=(8, 0))

        # ── Divider ──
        tk.Frame(self, bg=self.ACCENT, height=2).pack(fill=tk.X, padx=0)

        # ── Image area ──
        img_frame = tk.Frame(self, bg=self.BG)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(16, 8))

        # Original canvas
        orig_wrap = tk.Frame(img_frame, bg=self.SURFACE, bd=0)
        orig_wrap.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 8))

        tk.Label(orig_wrap, text="ORIGINAL",
                 font=("Courier New", 9, "bold"),
                 bg=self.SURFACE, fg=self.SUBTEXT, pady=6).pack()

        self.canvas_orig = tk.Canvas(orig_wrap,
                                     width=self.CANVAS_W, height=self.CANVAS_H,
                                     bg=self.SURFACE2, highlightthickness=0, cursor="arrow")
        self.canvas_orig.pack(padx=8, pady=(0, 8))
        self._draw_placeholder(self.canvas_orig, "Load an image to begin")

        # Modified canvas
        mod_wrap = tk.Frame(img_frame, bg=self.SURFACE, bd=0)
        mod_wrap.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(8, 0))

        tk.Label(mod_wrap, text="FIND THE DIFFERENCES  ← click here",
                 font=("Courier New", 9, "bold"),
                 bg=self.SURFACE, fg=self.ACCENT2, pady=6).pack()

        self.canvas_mod = tk.Canvas(mod_wrap,
                                    width=self.CANVAS_W, height=self.CANVAS_H,
                                    bg=self.SURFACE2, highlightthickness=0, cursor="crosshair")
        self.canvas_mod.pack(padx=8, pady=(0, 8))
        self.canvas_mod.bind("<Button-1>", self._on_canvas_click)
        self._draw_placeholder(self.canvas_mod, "Modified image will appear here")

        # ── Status bar ──
        status_frame = tk.Frame(self, bg=self.SURFACE, pady=10)
        status_frame.pack(fill=tk.X, padx=20, pady=(0, 8))

        self.remaining_lbl = tk.Label(status_frame,
                                      text="Remaining: —",
                                      font=("Courier New", 11, "bold"),
                                      bg=self.SURFACE, fg=self.ACCENT2)
        self.remaining_lbl.pack(side=tk.LEFT, padx=16)

        self.mistakes_lbl = tk.Label(status_frame,
                                     text="Mistakes: 0 / 3",
                                     font=("Courier New", 11, "bold"),
                                     bg=self.SURFACE, fg=self.TEXT)
        self.mistakes_lbl.pack(side=tk.LEFT, padx=16)

        self.status_msg = tk.Label(status_frame,
                                   text="Load an image to start!",
                                   font=("Courier New", 10),
                                   bg=self.SURFACE, fg=self.SUBTEXT)
        self.status_msg.pack(side=tk.LEFT, padx=16)

        # ── Button bar ──
        btn_frame = tk.Frame(self, bg=self.BG, pady=12)
        btn_frame.pack(fill=tk.X, padx=20, pady=(0, 16))

        btn_style = dict(font=("Courier New", 10, "bold"),
                         padx=20, pady=8, bd=0, cursor="hand2",
                         activeforeground="white")

        self.load_btn = tk.Button(btn_frame, text="⊕  LOAD IMAGE",
                                  bg=self.ACCENT, fg="black",
                                  activebackground="#C4441F",
                                  command=self._load_image, **btn_style)
        self.load_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.reveal_btn = tk.Button(btn_frame, text="◎  REVEAL ALL",
                                    bg=self.SURFACE2, fg="black",
                                    activebackground=self.SURFACE,
                                    command=self._reveal_all, state=tk.DISABLED,
                                    **btn_style)
        self.reveal_btn.pack(side=tk.LEFT, padx=(0, 10))

        hint_lbl = tk.Label(btn_frame,
                            text="Click on the right image to spot differences",
                            font=("Courier New", 9),
                            bg=self.BG, fg=self.SUBTEXT)
        hint_lbl.pack(side=tk.RIGHT)

    def _draw_placeholder(self, canvas: tk.Canvas, text: str):
        canvas.delete("all")
        w = int(canvas["width"])
        h = int(canvas["height"])
        # Dashed border
        canvas.create_rectangle(10, 10, w-10, h-10,
                                 outline=self.SUBTEXT, dash=(6, 4), width=1)
        canvas.create_text(w//2, h//2,
                           text=text, fill=self.SUBTEXT,
                           font=("Courier New", 10), width=w-40)

    # ── Image Loading ────────────────────────
    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All", "*.*")]
        )
        if not path:
            return

        if not self.processor.load_image(path):
            messagebox.showerror("Error", "Could not load image. Please try another file.")
            return

        # Generate differences
        self.processor.generate_differences()
        self.state.reset_round()
        self._revealed_regions = []

        # Take fresh copies for display (we'll draw circles on top)
        self._orig_display_bgr = self.processor.original_bgr.copy()
        self._mod_display_bgr  = self.processor.modified_bgr.copy()

        # Compute canvas size to match image aspect ratio
        self._fit_canvases()

        self._refresh_canvases()
        self._update_status()
        self.reveal_btn.config(state=tk.NORMAL)
        self.status_msg.config(text="Image loaded! Find the 5 differences.", fg=self.SUBTEXT)

    def _fit_canvases(self):
        """Resize canvases to match loaded image aspect ratio (within limits)."""
        ih, iw = self.processor.original_bgr.shape[:2]
        max_w, max_h = 560, 460
        min_w, min_h = 300, 240
        scale = min(max_w / iw, max_h / ih)
        scale = max(scale, min_w / iw)
        nw = max(min_w, int(iw * scale))
        nh = max(min_h, int(ih * scale))

        self._canvas_w = nw
        self._canvas_h = nh

        self.canvas_orig.config(width=nw, height=nh)
        self.canvas_mod.config(width=nw, height=nh)

    # ── Display helpers ──────────────────────
    def _bgr_to_photoimage(self, bgr: np.ndarray) -> tuple:
        """Scale bgr to canvas size, return (PhotoImage, displayed_w, displayed_h, scale, ox, oy)."""
        cw, ch = self._canvas_w, self._canvas_h
        h, w = bgr.shape[:2]
        scale = min(cw / w, ch / h)
        dw, dh = int(w * scale), int(h * scale)
        resized = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        ox = (cw - dw) // 2
        oy = (ch - dh) // 2
        return photo, dw, dh, scale, ox, oy

    def _refresh_canvases(self):
        """Redraw both canvases from current display BGRs."""
        orig_bgr = self._orig_display_bgr
        mod_bgr  = self._mod_display_bgr

        # Draw circles onto display copies based on found/unfound state
        orig_draw = orig_bgr.copy()
        mod_draw  = mod_bgr.copy()

        for region in self.processor.regions:
            if region.found:
                col = self.FOUND_COL_BGR
                thickness = 3
            else:
                continue  # only draw found/revealed ones
            r = max(region.w, region.h) // 2 + 12
            cv2.circle(orig_draw, (region.cx, region.cy), r, col, thickness)
            cv2.circle(mod_draw,  (region.cx, region.cy), r, col, thickness)

        # Revealed (blue)
        for region in getattr(self, '_revealed_regions', []):
            if not region.found:
                col = self.UNFOUND_COL_BGR
                r = max(region.w, region.h) // 2 + 12
                cv2.circle(orig_draw, (region.cx, region.cy), r, col, 3)
                cv2.circle(mod_draw,  (region.cx, region.cy), r, col, 3)

        p_orig, dw, dh, scale, ox, oy = self._bgr_to_photoimage(orig_draw)
        p_mod,  _,  _,  _,    _,  _  = self._bgr_to_photoimage(mod_draw)

        self._display_scale  = scale
        self._display_offset_x = ox
        self._display_offset_y = oy

        # Keep references to avoid GC
        self._photo_orig = p_orig
        self._photo_mod  = p_mod

        self.canvas_orig.delete("all")
        self.canvas_mod.delete("all")
        self.canvas_orig.create_image(ox, oy, anchor=tk.NW, image=p_orig)
        self.canvas_mod.create_image(ox, oy, anchor=tk.NW, image=p_mod)

    # ── Click Handling ───────────────────────
    def _on_canvas_click(self, event):
        if self.processor.original_bgr is None:
            return
        if self.state.locked or self.state.round_complete:
            return

        # Map canvas coords → original image coords
        ix = int((event.x - self._display_offset_x) / self._display_scale)
        iy = int((event.y - self._display_offset_y) / self._display_scale)

        # Check against regions
        for region in self.processor.regions:
            if region.found:
                continue
            if region.contains_point(ix, iy, tolerance=45):
                region.mark_found()
                self.state.record_correct()
                self._flash_canvas(self.canvas_mod, "#1DB954")
                self._refresh_canvases()
                self._update_status()

                if self.state.check_complete(self.processor.regions):
                    self.status_msg.config(text="🎉 All differences found! Load a new image.", fg="#1DB954")
                    messagebox.showinfo("Round Complete!",
                                        f"You found all 5 differences!\n"
                                        f"Total score: {self.state.total_score}")
                else:
                    remaining = sum(1 for r in self.processor.regions if not r.found)
                    self.status_msg.config(text=f"✓ Correct! {remaining} left.", fg="#1DB954")
                return

        # Miss
        self.state.record_mistake()
        self._flash_canvas(self.canvas_mod, "#E8572A")
        self._update_status()

        if self.state.locked:
            found_count = sum(1 for r in self.processor.regions if r.found)
            self.status_msg.config(
                text=f"✗ 3 mistakes reached. Found {found_count}/5. Load a new image.",
                fg=self.ACCENT)
            messagebox.showwarning("Too Many Mistakes",
                                   f"You've made 3 mistakes.\n"
                                   f"Differences found: {found_count} / 5\n\n"
                                   "Load a new image to try again.")
        else:
            self.status_msg.config(
                text=f"✗ Miss! {self.MAX_MISTAKES - self.state.mistakes} guesses remaining.",
                fg=self.ACCENT)

    @property
    def MAX_MISTAKES(self):
        return GameState.MAX_MISTAKES

    def _flash_canvas(self, canvas: tk.Canvas, colour: str):
        orig_bg = canvas.cget("bg")
        canvas.config(bg=colour)
        self.after(120, lambda: canvas.config(bg=orig_bg))

    # ── Reveal ───────────────────────────────
    def _reveal_all(self):
        if self.processor.original_bgr is None:
            return
        self._revealed_regions = [r for r in self.processor.regions if not r.found]
        self.state.locked = True
        self._refresh_canvases()
        self._update_status()
        self.status_msg.config(text="All differences revealed in blue. Load a new image.", fg=self.ACCENT2)

    # ── Status Bar Update ────────────────────
    def _update_status(self):
        if self.processor.original_bgr is None:
            self.remaining_lbl.config(text="Remaining: —")
            self.mistakes_lbl.config(text="Mistakes: 0 / 3")
            self.score_lbl.config(text="SCORE  0")
            return

        remaining = sum(1 for r in self.processor.regions if not r.found)
        self.remaining_lbl.config(text=f"Remaining: {remaining}")

        mistakes = self.state.mistakes
        col = self.ACCENT if mistakes >= 2 else self.TEXT
        self.mistakes_lbl.config(text=f"Mistakes: {mistakes} / 3", fg=col)
        self.score_lbl.config(text=f"SCORE  {self.state.total_score}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = SpotTheDifferenceApp()
    app.mainloop()
