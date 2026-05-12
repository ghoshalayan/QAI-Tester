"""Phase A — Set-of-Mark (SoM) screenshot annotator.

Draws colored bounding boxes + numbered labels onto a PNG so a
vision LLM can refer to "box 5" instead of inventing pixel
coordinates. Published benchmarks (Microsoft's SoM paper, GPT-4V
evaluations) consistently show ~10-15% targeting-accuracy
improvement on agent tasks.

Two distinct use sites, same drawing primitives:

1. **Live VL calls** — system annotates before sending to the
   model: red boxes = the agent tried this and failed, green =
   tried and succeeded, blue = VL-recommended candidates, yellow
   = sub-goal target zone. Numbered (1, 2, 3...) so the model
   responds with "box 3".

2. **HITL overlay (Layer 1)** — the same image, plus the user
   draws on Layer 2 (rect / pen / text) in the test-browser
   overlay. On submit, Layers 1+2 are flattened and shipped to
   the model with the user's typed guidance.

Color legend (locked v1)
------------------------
- ``red``     — agent tried; failed
- ``green``   — agent tried; succeeded
- ``blue``    — VL-recommended candidate (not tried)
- ``yellow``  — sub-goal target zone (e.g., "this section is
  incomplete")
- ``cyan``    — reserved for user marks (Layer 2 — drawn in the
  browser, not by this module)

Why Pillow and not opencv: Pillow is already in the wheel for
screenshot processing, has zero new deps, draws clean text + AA
shapes, and ships everywhere our backend runs (Windows + Linux).
opencv would add ~50 MB to the image and brings nothing we need.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


MarkColor = Literal["red", "green", "blue", "yellow", "cyan"]


@dataclass
class Mark:
    """One bounding box to draw on the screenshot.

    Coordinates are in image-pixel space (the same space the VL
    sees). ``x``/``y`` is the top-left corner; ``w``/``h`` are
    width/height. Label is drawn as a small tag pinned to the
    top-left of the box.
    """

    x: int
    y: int
    w: int
    h: int
    label: str
    color: MarkColor = "blue"


# RGB tuples — bright, high-contrast against typical web UIs.
# Alpha applied separately in the drawing routine so the box
# outline stays opaque while the fill is translucent.
_COLOR_RGB: dict[MarkColor, tuple[int, int, int]] = {
    "red":    (220,  53,  69),
    "green":  ( 40, 167,  69),
    "blue":   ( 13, 110, 253),
    "yellow": (255, 193,   7),
    "cyan":   ( 13, 202, 240),
}

# Outline thickness (px). 3 px reads cleanly on the 1280px-wide
# screenshots we typically capture without obscuring the
# underlying button.
_LINE_WIDTH = 3
# Translucent fill alpha (0-255). Light tint helps the box pop
# without making the underlying text unreadable.
_FILL_ALPHA = 40
# Label tag padding inside the corner badge.
_LABEL_PADX = 6
_LABEL_PADY = 3


def _load_font(size: int = 16) -> ImageFont.ImageFont:
    """Best-effort font load. Falls back to PIL's default bitmap
    when no TrueType is available (some CI containers ship without
    DejaVu)."""
    for candidate in (
        "DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf",
        "arial.ttf", "Helvetica.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    """Width/height of rendered text. Wraps the Pillow version
    skew (textbbox new, textsize deprecated, default font has no
    .getbbox in older builds)."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        try:
            return draw.textsize(text, font=font)  # type: ignore[attr-defined]
        except Exception:
            return 8 * len(text), 14


def annotate_screenshot(
    png_bytes: bytes,
    marks: list[Mark],
    *,
    number_marks: bool = True,
) -> bytes:
    """Draw the given marks onto a PNG and return the new PNG.

    ``number_marks=True`` (default) prepends a 1-based index to
    every label so the VL can refer to "box 3". Set False when the
    label is already a unique identifier you want the model to
    quote verbatim.

    Failures here are NEVER fatal — if we can't decode the image,
    we return the original bytes unchanged. The agent loses the
    SoM hint for that call but the run continues.
    """
    if not marks:
        return png_bytes
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception as e:
        logger.warning("SoM: cannot decode screenshot (%s); skipping", e)
        return png_bytes

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(16)

    width, height = img.size

    for idx, m in enumerate(marks, start=1):
        x = max(0, min(m.x, width - 1))
        y = max(0, min(m.y, height - 1))
        x2 = max(x + 1, min(m.x + m.w, width))
        y2 = max(y + 1, min(m.y + m.h, height))

        rgb = _COLOR_RGB.get(m.color, _COLOR_RGB["blue"])
        outline = (*rgb, 255)
        fill = (*rgb, _FILL_ALPHA)

        # Box: translucent fill + opaque outline.
        draw.rectangle((x, y, x2, y2), fill=fill, outline=outline, width=_LINE_WIDTH)

        # Label tag in the top-left corner of the box.
        label_text = (
            f"{idx} · {m.label}" if number_marks else m.label
        )
        tw, th = _text_size(draw, label_text, font)
        # Pin the tag ABOVE the box when there's room (more readable);
        # fall back to inside-top-left when the box is at y=0.
        tag_x = x
        tag_y = max(0, y - (th + 2 * _LABEL_PADY) - 2)
        if tag_y == 0 and y <= th + 2 * _LABEL_PADY + 2:
            tag_y = y + 2  # inside the box

        tag_x2 = tag_x + tw + 2 * _LABEL_PADX
        tag_y2 = tag_y + th + 2 * _LABEL_PADY
        draw.rectangle(
            (tag_x, tag_y, tag_x2, tag_y2),
            fill=outline,
        )
        draw.text(
            (tag_x + _LABEL_PADX, tag_y + _LABEL_PADY),
            label_text,
            fill=(255, 255, 255, 255),
            font=font,
        )

    composited = Image.alpha_composite(img, overlay).convert("RGB")
    out = io.BytesIO()
    composited.save(out, format="PNG", optimize=True)
    return out.getvalue()


def annotate_from_turn_history(
    png_bytes: bytes,
    *,
    tried_targets: list[dict],
    candidates: list[dict] | None = None,
    sub_goal_zone: tuple[int, int, int, int] | None = None,
) -> bytes:
    """Convenience wrapper: build Mark list from agent state.

    ``tried_targets`` — list of ``{x, y, w, h, label, status}``
    where status is "ok" (→ green) or "failed" (→ red).
    ``candidates`` — list of ``{x, y, w, h, label}`` for
    VL-recommended alternatives (→ blue).
    ``sub_goal_zone`` — optional (x, y, w, h) of the area the
    sub-goal cares about (→ yellow).

    The order of marks (and therefore numbering) is:
        sub_goal_zone (if any) → tried failed → tried ok → candidates
    so the most-relevant items get the lowest numbers in the VL prompt.
    """
    marks: list[Mark] = []
    if sub_goal_zone is not None:
        x, y, w, h = sub_goal_zone
        marks.append(Mark(x=x, y=y, w=w, h=h, label="sub-goal zone", color="yellow"))

    failed = [t for t in tried_targets if t.get("status") == "failed"]
    ok = [t for t in tried_targets if t.get("status") == "ok"]
    for t in failed:
        marks.append(Mark(
            x=int(t.get("x", 0)), y=int(t.get("y", 0)),
            w=int(t.get("w", 0)), h=int(t.get("h", 0)),
            label=str(t.get("label", "tried"))[:40],
            color="red",
        ))
    for t in ok:
        marks.append(Mark(
            x=int(t.get("x", 0)), y=int(t.get("y", 0)),
            w=int(t.get("w", 0)), h=int(t.get("h", 0)),
            label=str(t.get("label", "ok"))[:40],
            color="green",
        ))
    for c in (candidates or []):
        marks.append(Mark(
            x=int(c.get("x", 0)), y=int(c.get("y", 0)),
            w=int(c.get("w", 0)), h=int(c.get("h", 0)),
            label=str(c.get("label", "candidate"))[:40],
            color="blue",
        ))
    return annotate_screenshot(png_bytes, marks)
