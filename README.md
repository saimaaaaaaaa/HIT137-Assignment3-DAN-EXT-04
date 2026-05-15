# HIT137 — Assignment 3: Spot the Difference Game

A desktop "Spot the Difference" game built with **Python**, **Tkinter** (GUI) and **OpenCV** (image processing), demonstrating solid Object-Oriented Programming principles.


---
# Student Name and ID

| Name | Student ID |
|------|-----------|
| Saima Akter | S394703 |

---

## 🚀 Setup & Running

### Prerequisites

| Package | Install command |
|---------|----------------|
| Python 3.9+ | https://python.org |
| OpenCV | `pip3 install opencv-python` |
| Pillow | `pip3 install Pillow` |
| NumPy | `pip3 install numpy` |

Install all at once:

```bash
pip3 install -r requirements.txt
```

### Run

```bash
python3 game.py
```

---

## 🎮 How to Play

1. **Click "⊕ LOAD IMAGE"** — choose any JPG, PNG or BMP image from disk.
2. The app displays the **original** image on the left and a **modified copy** (with 5 hidden differences) on the right.
3. **Click on the right image** wherever you think a difference is.
   - ✅ Correct click → a **red circle** appears on both images; score increments.
   - ❌ Wrong click → a mistake is recorded (max **3 mistakes** per image).
4. Find all 5 differences to complete the round and load the next image.
5. Use **"◎ REVEAL ALL"** at any time to reveal unfound differences in **blue**.

---

## 🏗️ OOP Design

The application is split into **four classes**, each with a clear responsibility.

### `DifferenceRegion`
Encapsulates a single rectangular difference region.

| Member | Role |
|--------|------|
| `x, y, w, h` | Bounding box on the original image |
| `alteration_type` | Which of the 5 alteration types was applied |
| `found` | Whether the player has located it |
| `contains_point(px, py, tolerance)` | Hit-test with proximity radius |
| `mark_found()` | Marks region as discovered |

Demonstrates: **encapsulation**, **constructor**, **methods**.

---

### `ImageProcessor`
Handles all OpenCV image loading and manipulation.

| Member | Role |
|--------|------|
| `load_image(path)` | Reads image from disk via `cv2.imread` |
| `generate_differences()` | Places 5 non-overlapping random alterations |
| `_apply_alteration(...)` | Polymorphic dispatch to one of 5 alteration methods |
| `draw_circle_on_bgr(...)` | Draws a coloured circle on an image copy |
| `apply_effect(name, img)` | Polymorphic global-effect API |

**5 alteration types implemented:**

| Type | Description |
|------|-------------|
| `colour_shift` | Shifts the hue of a region in HSV colour space |
| `brightness_patch` | Increases or decreases pixel brightness |
| `blur_patch` | Applies Gaussian blur to a rectangular patch |
| `hue_rotate` | Swaps BGR colour channels |
| `contrast_invert` | Applies CLAHE local contrast enhancement |

Demonstrates: **encapsulation**, **methods**, **class interaction**.

---

### `GameState`
Tracks game progress independently of the UI.

| Member | Role |
|--------|------|
| `total_score` | Cumulative correct finds across all rounds |
| `mistakes` | Mistakes for the current image |
| `locked` | True once 3 mistakes are reached |
| `record_correct()` | Increments score |
| `record_mistake()` | Increments mistakes; locks if limit hit |
| `check_complete(regions)` | Returns True when all 5 found |
| `reset_round()` | Resets per-round state for a new image |

Demonstrates: **encapsulation**, **constructor**, **methods**, **class interaction**.

---

### `SpotTheDifferenceApp(tk.Tk)`
Main Tkinter window — **inherits from `tk.Tk`**.

Wires the three model classes above into a complete interactive UI. Key responsibilities:

- Builds and lays out all widgets
- Handles `<Button-1>` click events on the modified canvas
- Converts canvas coordinates back to original image coordinates using scale + offset
- Redraws canvases after every state change
- Manages image aspect-ratio-aware canvas resizing

Demonstrates: **inheritance** (extends `tk.Tk`), **polymorphism** (overrides `__init__`), **class interaction** (uses all three other classes).

---

## 🖼️ Image Processing Details

When an image is loaded:

1. An exact **clone** of the original is made.
2. The algorithm randomly places **5 non-overlapping regions**, each with a random alteration type and random position/size.
3. A padding check prevents regions from touching each other or the image border.
4. All manipulation uses **OpenCV only** (`cv2`, `numpy`).

---

## 🎨 UI Features

- Dark, high-contrast theme with accent colours
- Canvases **resize to match the loaded image's aspect ratio** — no wasted space
- Canvas flashes **green** on a correct click, **red** on a miss
- Live status bar: remaining differences, mistake counter, total score
- Placeholder text shown before any image is loaded
- All 5 found → popup notification + on-screen message
- 3 mistakes → lockout with clear warning
- Reveal button marks unfound differences in blue on both images


---
---
