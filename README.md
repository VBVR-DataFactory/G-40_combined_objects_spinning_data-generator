# Combined Objects Spinning Data Generator

Generates tasks where a connected cluster of shapes rotates in place and then moves right into dashed targets.

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-org/your-task-generator.git
cd your-task-generator

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 4. Generate tasks
python examples/generate.py --num-samples 50
```

---

## 📁 Structure

```
template-data-generator/
├── core/                    # ✅ KEEP: Standard utilities
│   ├── base_generator.py   # Abstract base class
│   ├── schemas.py          # Pydantic models
│   ├── image_utils.py      # Image helpers
│   ├── video_utils.py      # Video generation
│   └── output_writer.py    # File output
├── src/                     # ⚠️ CUSTOMIZE: Your task logic
│   ├── generator.py        # Your task generator
│   ├── prompts.py          # Your prompt templates
│   └── config.py           # Your configuration
├── examples/
│   └── generate.py         # Entry point
└── data/questions/         # Generated output
```

---

## 📦 Output Format

Every generator produces:

```
data/questions/{domain}_task/{task_id}/
├── first_frame.png          # Initial state (REQUIRED)
├── final_frame.png          # Goal state (or goal.txt)
├── prompt.txt               # Instructions (REQUIRED)
└── ground_truth.mp4         # Solution video (OPTIONAL)
```

---

## Task Description

- Each sample contains 2-5 geometric objects with random shape, color, and size.
- The initial layout is a connected cluster: every object touches at least one other, with no overlaps.
- Dashed target outlines are shown to the right; each object has a target outline at the same y-position.
- Objects first rotate in place to match the target orientation, then slide horizontally to the right.
- Target positions are separated (no touching) and the final frame shows only the solid objects (no dashed outlines).
- The background is a plain white square.
- Prompts are in English and describe the rotation-then-translation sequence.

## Configuration

Edit `src/config.py` to change object counts, sizes, rotation ranges, spacing, and animation timing.

**Single entry point:** `python examples/generate.py --num-samples 50`
