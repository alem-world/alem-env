"""LLM renderer: presentation-first observation cards plus compact agent traces.

Designed for website/paper presentation rather than raw debugging:
- Large, pixel-preserving observation panels
- Compact action / message / reasoning sections with fixed rhythm
- Clean widescreen composition that still exposes the LLM trace

Outputs:
- render_llm_frame()  -> single PIL Image per timestep
- render_llm_video()  -> MP4 video (pauseable)
- render_llm_html()   -> interactive HTML viewer with step slider
- render_llm_gif()    -> GIF fallback
"""

import base64
import io
import json
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================================
# Theme — presentation-first debug layout with stronger hierarchy
# ============================================================================

BG_TOP = (14, 17, 25)
BG_BOTTOM = (8, 10, 15)
BG_COLOR = BG_BOTTOM

CARD_BG = (25, 28, 38)
CARD_BORDER = (70, 75, 95)
CARD_SHADOW = (5, 7, 10)

OBS_BG = (17, 19, 27)
OBS_FRAME_BG = (10, 12, 18)
OBS_BORDER = (58, 63, 82)

SECTION_BG = (32, 35, 47)
SECTION_BORDER = (67, 72, 91)
SECTION_INNER_BG = (40, 44, 58)

TEXT_COLOR = (228, 232, 241)
TEXT_DIM = (133, 141, 159)
TEXT_SOFT = (184, 189, 201)
TEXT_WHITE = (248, 249, 252)
ACTION_COLOR = (255, 223, 126)
STEP_COLOR = (176, 182, 198)
STEP_TRACK = (45, 49, 64)
STEP_FILL = (99, 116, 164)

ROLE_COLORS = {
    "warrior": {"accent": (161, 80, 80), "label": (227, 176, 176)},
    "forager": {"accent": (82, 145, 92), "label": (180, 223, 186)},
    "miner": {"accent": (86, 118, 182), "label": (181, 200, 237)},
}
DEFAULT_ROLE = {"accent": (110, 114, 132), "label": (196, 200, 214)}

SECTION_ACCENTS = {
    "action": (208, 167, 60),
    "sent": (86, 182, 163),
    "inbox": (101, 145, 220),
    "scratchpad": (138, 187, 106),
    "reasoning": (217, 148, 86),
}

# Layout
COLUMN_WIDTH = 560
AGENT_GAP = 22
OUTER_PAD = 24
TOP_BAR_H = 58
CARD_RADIUS = 22
CARD_PAD = 16
CARD_HEADER_H = 42
SECTION_PAD = 12
SECTION_GAP = 12
SECTION_RADIUS = 14
LINE_H = 16
FONT_SIZE = 13
FONT_SIZE_SMALL = 10
FONT_SIZE_TITLE = 16
FONT_SIZE_ACTION = 14
OBS_PAD = 14
OBS_LABEL_H = 32
OBS_MIN_H = 170
OBS_MAX_H = 280

# Max display lengths (chars) — keeps frames consistent
MAX_COMM_DISPLAY = 400
MAX_SCRATCHPAD_DISPLAY = 600
MAX_REASONING_DISPLAY = 400
MAX_ACTION_LINES = 2
MAX_SENT_LINES = 5
MAX_INBOX_LINES = 6
MAX_SCRATCHPAD_LINES = 6
MAX_REASONING_LINES = 5


def _get_font(size=FONT_SIZE, preferred=None):
    search = []
    if preferred:
        search.extend(preferred)
    search.extend(
        [
            "DejaVuSans.ttf",
            "LiberationSans-Regular.ttf",
            "Ubuntu-R.ttf",
            "Arial.ttf",
            "DejaVuSansMono.ttf",
            "LiberationMono-Regular.ttf",
            "UbuntuMono-R.ttf",
            "Consolas.ttf",
        ]
    )
    for name in search:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT = _get_font(FONT_SIZE, preferred=["DejaVuSans.ttf", "LiberationSans-Regular.ttf"])
FONT_SMALL = _get_font(FONT_SIZE_SMALL, preferred=["DejaVuSans.ttf", "LiberationSans-Regular.ttf"])
FONT_TITLE = _get_font(
    FONT_SIZE_TITLE, preferred=["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"]
)
FONT_ACTION = _get_font(
    FONT_SIZE_ACTION, preferred=["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"]
)
FONT_MONO = _get_font(
    FONT_SIZE_SMALL + 1, preferred=["DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"]
)

_MEASURE_DRAW = ImageDraw.Draw(Image.new("RGB", (1, 1)))


def _mix(color_a, color_b, alpha):
    return tuple(int(round(color_a[i] * (1.0 - alpha) + color_b[i] * alpha)) for i in range(3))


def _truncate(text, max_len):
    if not text:
        return text
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _wrap(text, width=80):
    if not text:
        return []
    lines = []
    for para in str(text).split("\n"):
        lines.extend(textwrap.wrap(para, width=width) or [""])
    return lines


def _clamp_lines(lines, max_lines, width):
    if max_lines is None or len(lines) <= max_lines:
        return lines
    clipped = list(lines[:max_lines])
    if clipped:
        clipped[-1] = textwrap.shorten(clipped[-1], width=max(8, width), placeholder="...")
    return clipped


def _wrap_clamped(text, width, max_lines=None):
    return _clamp_lines(_wrap(text, width=width), max_lines, width)


def _bubble_h(lines):
    return 2 * SECTION_PAD + max(len(lines), 1) * LINE_H


def _text_size(text, font=FONT):
    bbox = _MEASURE_DRAW.textbbox((0, 0), str(text), font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_vertical_gradient(draw, width, height, top_color, bottom_color):
    if height <= 1:
        draw.rectangle((0, 0, width, height), fill=top_color)
        return
    for y in range(height):
        t = y / float(height - 1)
        fill = tuple(int(round(top_color[i] * (1.0 - t) + bottom_color[i] * t)) for i in range(3))
        draw.line((0, y, width, y), fill=fill)


def _draw_card_shell(draw, box, radius=CARD_RADIUS, fill=CARD_BG, outline=CARD_BORDER):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(
        (x0 + 3, y0 + 5, x1 + 3, y1 + 5),
        radius=radius,
        fill=CARD_SHADOW,
    )
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=1)
    draw.rounded_rectangle(
        (x0 + 1, y0 + 1, x1 - 1, y0 + 26),
        radius=radius,
        fill=_mix(fill, TEXT_WHITE, 0.03),
    )


def _chip_h(font=FONT_SMALL, pad_y=4):
    return _text_size("Ag", font=font)[1] + 2 * pad_y


def _draw_chip(
    draw,
    x,
    y,
    text,
    fill,
    text_fill=TEXT_WHITE,
    border=None,
    font=FONT_SMALL,
    pad_x=8,
    pad_y=4,
    radius=9,
):
    text_w, text_h = _text_size(text, font=font)
    w = text_w + 2 * pad_x
    h = text_h + 2 * pad_y
    draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline=border)
    draw.text((x + pad_x, y + pad_y - 1), text, fill=text_fill, font=font)
    return w, h


def _measure_section_h(lines):
    return _chip_h() + 6 + _bubble_h(lines) + SECTION_GAP


def _draw_section(
    draw,
    x,
    y,
    w,
    label,
    lines,
    accent,
    *,
    text_fill=TEXT_COLOR,
    body_fill=SECTION_BG,
    empty_text="--",
    font=FONT,
):
    chip_fill = _mix(body_fill, accent, 0.34)
    _, chip_h = _draw_chip(draw, x, y, label, fill=chip_fill, text_fill=TEXT_WHITE)
    panel_y = y + chip_h + 6
    panel_h = _bubble_h(lines)
    draw.rounded_rectangle(
        (x, panel_y, x + w, panel_y + panel_h),
        radius=SECTION_RADIUS,
        fill=body_fill,
        outline=SECTION_BORDER,
        width=1,
    )
    draw.rounded_rectangle(
        (x + 1, panel_y + 2, x + 6, panel_y + panel_h - 2),
        radius=3,
        fill=accent,
    )
    tx = x + SECTION_PAD + 8
    ty = panel_y + SECTION_PAD
    if not lines:
        draw.text((tx, ty), empty_text, fill=TEXT_DIM, font=FONT)
    else:
        for line in lines:
            draw.text((tx, ty), line, fill=text_fill, font=font)
            ty += LINE_H
    return panel_y + panel_h + SECTION_GAP


def _received_lines(received, chars):
    if not received:
        return []

    lines = []
    for aid, msg in received.items():
        prefix = str(aid)
        if not prefix.startswith("A"):
            prefix = f"A{prefix}"
        wrapped = textwrap.wrap(
            f"{prefix}: {_truncate(msg, MAX_COMM_DISPLAY)}",
            width=max(16, chars),
            break_long_words=True,
            break_on_hyphens=False,
        ) or [f"{prefix}: --"]
        lines.extend(wrapped)
    return _clamp_lines(lines, MAX_INBOX_LINES, max(16, chars))


def _measure_received_h(received, chars):
    return _measure_section_h(_received_lines(received, chars))


def _draw_received(draw, x, y, w, received, chars):
    return _draw_section(
        draw,
        x,
        y,
        w,
        "INBOX",
        _received_lines(received, chars),
        SECTION_ACCENTS["inbox"],
        body_fill=_mix(SECTION_BG, SECTION_ACCENTS["inbox"], 0.08),
    )


def _draw_top_bar(draw, total_w, step, max_steps):
    bar_x0 = OUTER_PAD
    bar_y0 = OUTER_PAD
    bar_x1 = total_w - OUTER_PAD
    bar_y1 = bar_y0 + TOP_BAR_H
    draw.rounded_rectangle(
        (bar_x0, bar_y0, bar_x1, bar_y1),
        radius=16,
        fill=(18, 20, 29),
        outline=(46, 50, 66),
        width=1,
    )
    _draw_chip(
        draw,
        bar_x0 + 12,
        bar_y0 + 11,
        "ALEM • LLM AGENTS",
        fill=(41, 47, 64),
        text_fill=TEXT_WHITE,
        font=FONT_SMALL,
        pad_x=10,
    )
    title = "Reasoning + Coordination Trace"
    title_w, title_h = _text_size(title, font=FONT_TITLE)
    draw.text(
        ((bar_x0 + bar_x1 - title_w) / 2, bar_y0 + 10),
        title,
        fill=TEXT_SOFT,
        font=FONT_TITLE,
    )
    step_text = f"STEP {step} / {max_steps}"
    step_w, _ = _text_size(step_text, font=FONT_SMALL)
    step_chip_w = step_w + 20
    _draw_chip(
        draw,
        bar_x1 - step_chip_w - 12,
        bar_y0 + 11,
        step_text,
        fill=(34, 39, 54),
        text_fill=STEP_COLOR,
        font=FONT_SMALL,
        pad_x=10,
    )
    progress = max(0.0, min(float(step) / max(float(max_steps), 1.0), 1.0))
    track_x0 = bar_x0 + 14
    track_x1 = bar_x1 - 14
    track_y0 = bar_y1 - 10
    track_y1 = track_y0 + 4
    draw.rounded_rectangle((track_x0, track_y0, track_x1, track_y1), radius=3, fill=STEP_TRACK)
    fill_x1 = track_x0 + int((track_x1 - track_x0) * progress)
    if fill_x1 > track_x0:
        draw.rounded_rectangle((track_x0, track_y0, fill_x1, track_y1), radius=3, fill=STEP_FILL)


def _draw_agent_header(draw, x, y, w, data, role_theme):
    accent = role_theme["accent"]
    draw.rounded_rectangle((x, y, x + w, y + 8), radius=4, fill=accent)
    title = f"Agent {data.get('id', 0)}"
    draw.text((x, y + 14), title, fill=TEXT_WHITE, font=FONT_TITLE)

    title_w, _ = _text_size(title, font=FONT_TITLE)
    role_x = x + title_w + 14
    role_fill = _mix(CARD_BG, accent, 0.40)
    role_text = str(data.get("role", "unknown")).upper()
    _draw_chip(
        draw,
        role_x,
        y + 12,
        role_text,
        fill=role_fill,
        text_fill=TEXT_WHITE,
        font=FONT_SMALL,
    )

    inbox_count = len(data.get("comm_received") or {})
    sent_count = 1 if data.get("comm_sent") else 0
    chips = [
        (f"OUT {sent_count}", _mix(CARD_BG, SECTION_ACCENTS["sent"], 0.28)),
        (f"IN {inbox_count}", _mix(CARD_BG, SECTION_ACCENTS["inbox"], 0.28)),
    ]
    right_x = x + w
    for text, fill in reversed(chips):
        chip_w, _ = _text_size(text, font=FONT_SMALL)
        chip_w += 16
        right_x -= chip_w
        _draw_chip(draw, right_x, y + 12, text, fill=fill, text_fill=TEXT_SOFT, font=FONT_SMALL)
        right_x -= 6


def _draw_obs_panel(canvas, draw, x, y, w, h, img, accent):
    draw.rounded_rectangle(
        (x, y, x + w, y + h),
        radius=SECTION_RADIUS,
        fill=OBS_BG,
        outline=OBS_BORDER,
        width=1,
    )
    _draw_chip(
        draw,
        x + 10,
        y + 8,
        "OBSERVATION",
        fill=_mix(OBS_BG, accent, 0.26),
        text_fill=TEXT_WHITE,
    )
    inner_x0 = x + 10
    inner_y0 = y + OBS_LABEL_H
    inner_x1 = x + w - 10
    inner_y1 = y + h - 10
    draw.rounded_rectangle(
        (inner_x0, inner_y0, inner_x1, inner_y1),
        radius=10,
        fill=OBS_FRAME_BG,
        outline=_mix(OBS_BORDER, accent, 0.18),
        width=1,
    )
    draw.rounded_rectangle(
        (inner_x0 + 1, inner_y0 + 1, inner_x1 - 1, inner_y0 + 24),
        radius=10,
        fill=_mix(OBS_FRAME_BG, accent, 0.10),
    )
    if img is None:
        text = "No image"
        text_w, text_h = _text_size(text, font=FONT)
        draw.text(
            (
                inner_x0 + (inner_x1 - inner_x0 - text_w) / 2,
                inner_y0 + (inner_y1 - inner_y0 - text_h) / 2,
            ),
            text,
            fill=TEXT_DIM,
            font=FONT,
        )
        return

    slot_w = inner_x1 - inner_x0
    slot_h = inner_y1 - inner_y0
    scale = min(slot_w / max(img.width, 1), slot_h / max(img.height, 1))
    scaled_w = max(1, int(round(img.width * scale)))
    scaled_h = max(1, int(round(img.height * scale)))
    scaled = img.resize((scaled_w, scaled_h), resample=Image.NEAREST)
    ix = inner_x0 + (slot_w - scaled_w) // 2
    iy = inner_y0 + (slot_h - scaled_h) // 2
    canvas.paste(scaled, (ix, iy))


def _measure_column(data, chars):
    side_chars = max(18, chars // 2)
    action_lines = _wrap_clamped(
        str(data.get("action", "Noop")), width=chars, max_lines=MAX_ACTION_LINES
    )
    sent_lines = _wrap_clamped(
        _truncate(data.get("comm_sent"), MAX_COMM_DISPLAY),
        width=side_chars,
        max_lines=MAX_SENT_LINES,
    )
    scratch_lines = _wrap_clamped(
        _truncate(data.get("scratchpad"), MAX_SCRATCHPAD_DISPLAY),
        width=side_chars,
        max_lines=MAX_SCRATCHPAD_LINES,
    )
    reasoning_lines = _wrap_clamped(
        _truncate(data.get("reasoning"), MAX_REASONING_DISPLAY),
        width=side_chars,
        max_lines=MAX_REASONING_LINES,
    )
    action_h = _measure_section_h(action_lines)
    row1_h = max(
        _measure_section_h(sent_lines),
        _measure_received_h(data.get("comm_received"), side_chars),
    )
    row2_h = max(
        _measure_section_h(scratch_lines),
        _measure_section_h(reasoning_lines),
    )
    return action_h + row1_h + row2_h


def render_llm_frame(
    agent_images: list,
    agent_data: list,
    step: int = 0,
    max_steps: int = 10000,
    panel_width: int | None = None,
    fixed_size: tuple | None = None,
) -> Image.Image:
    """Render observation cards and structured agent traces into one frame.

    Args:
        agent_images: Per-agent observation images, or ``None`` placeholders.
        agent_data: Per-agent text, action, communication, and role fields.
        step: Current episode step displayed in the top bar.
        max_steps: Episode length displayed in the top bar.
        panel_width: Optional width of each agent column.
        fixed_size: Optional fixed output width and height.

    Returns:
        Composite RGB PIL image.
    """
    col_w = panel_width or COLUMN_WIDTH
    num_agents = len(agent_data)
    content_w = col_w - 2 * CARD_PAD
    chars = max(30, (content_w - 2 * SECTION_PAD - 18) // 8)

    obs_height = 0
    for img in agent_images:
        if img is not None:
            obs_height = max(obs_height, img.height)

    obs_content_h = max(int(round(obs_height * 0.75)) if obs_height else 0, OBS_MIN_H)
    obs_content_h = min(obs_content_h, OBS_MAX_H)
    obs_panel_h = obs_content_h + 2 * OBS_PAD + OBS_LABEL_H
    text_heights = [_measure_column(d, chars) for d in agent_data]
    card_heights = [
        2 * CARD_PAD + CARD_HEADER_H + obs_panel_h + SECTION_GAP + text_h for text_h in text_heights
    ]
    max_card_h = max(card_heights) if card_heights else 0

    natural_w = 2 * OUTER_PAD + num_agents * col_w + max(num_agents - 1, 0) * AGENT_GAP
    natural_h = 2 * OUTER_PAD + TOP_BAR_H + 16 + max_card_h + 8

    if fixed_size:
        total_w, total_h = fixed_size
    else:
        total_w, total_h = natural_w, natural_h

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    _draw_vertical_gradient(draw, total_w, total_h, BG_TOP, BG_BOTTOM)

    _draw_top_bar(draw, total_w, step, max_steps)

    for i, data in enumerate(agent_data):
        x = OUTER_PAD + i * (col_w + AGENT_GAP)
        card_y = OUTER_PAD + TOP_BAR_H + 12
        role = data.get("role", "unknown")
        role_theme = ROLE_COLORS.get(role, DEFAULT_ROLE)
        _draw_card_shell(draw, (x, card_y, x + col_w, card_y + card_heights[i]))

        inner_x = x + CARD_PAD
        inner_y = card_y + CARD_PAD
        inner_w = col_w - 2 * CARD_PAD

        _draw_agent_header(draw, inner_x, inner_y, inner_w, data, role_theme)

        obs_y = inner_y + CARD_HEADER_H
        img = agent_images[i] if i < len(agent_images) else None
        _draw_obs_panel(
            canvas, draw, inner_x, obs_y, inner_w, obs_panel_h, img, role_theme["accent"]
        )

        grid_gap = 12
        half_w = (inner_w - grid_gap) // 2

        ty = obs_y + obs_panel_h + SECTION_GAP
        ty = _draw_section(
            draw,
            inner_x,
            ty,
            inner_w,
            "ACTION",
            _wrap_clamped(str(data.get("action", "Noop")), width=chars, max_lines=MAX_ACTION_LINES),
            SECTION_ACCENTS["action"],
            text_fill=ACTION_COLOR,
            body_fill=_mix(SECTION_BG, SECTION_ACCENTS["action"], 0.14),
            font=FONT_ACTION,
        )

        sent_lines = _wrap_clamped(
            _truncate(data.get("comm_sent"), MAX_COMM_DISPLAY),
            width=max(16, chars // 2),
            max_lines=MAX_SENT_LINES,
        )
        inbox_chars = max(20, chars // 2)
        scratch_lines = _wrap_clamped(
            _truncate(data.get("scratchpad"), MAX_SCRATCHPAD_DISPLAY),
            width=max(18, chars // 2),
            max_lines=MAX_SCRATCHPAD_LINES,
        )
        reasoning_lines = _wrap_clamped(
            _truncate(data.get("reasoning"), MAX_REASONING_DISPLAY),
            width=max(18, chars // 2),
            max_lines=MAX_REASONING_LINES,
        )

        row1_h = max(
            _measure_section_h(sent_lines),
            _measure_received_h(data.get("comm_received"), inbox_chars),
        )
        _row2_h = max(
            _measure_section_h(scratch_lines),
            _measure_section_h(reasoning_lines),
        )

        _draw_section(
            draw,
            inner_x,
            ty,
            half_w,
            "SENT",
            sent_lines,
            SECTION_ACCENTS["sent"],
        )
        _draw_received(
            draw,
            inner_x + half_w + grid_gap,
            ty,
            half_w,
            data.get("comm_received"),
            inbox_chars,
        )
        ty += row1_h

        _draw_section(
            draw,
            inner_x,
            ty,
            half_w,
            "SCRATCHPAD",
            scratch_lines,
            SECTION_ACCENTS["scratchpad"],
            body_fill=_mix(SECTION_BG, SECTION_ACCENTS["scratchpad"], 0.08),
            font=FONT_MONO,
        )
        _draw_section(
            draw,
            inner_x + half_w + grid_gap,
            ty,
            half_w,
            "REASONING",
            reasoning_lines,
            SECTION_ACCENTS["reasoning"],
            text_fill=TEXT_SOFT,
            body_fill=_mix(SECTION_BG, SECTION_ACCENTS["reasoning"], 0.08),
        )

    return canvas


# ============================================================================
# Output formats
# ============================================================================


def _normalize_frames(frames):
    """Pad all frames to the same size (multiple of 16 for codec compat)."""
    if not frames:
        return frames
    max_w = max(f.width for f in frames)
    max_h = max(f.height for f in frames)
    max_w = (max_w + 15) // 16 * 16
    max_h = (max_h + 15) // 16 * 16
    out = []
    for f in frames:
        if f.width == max_w and f.height == max_h:
            out.append(f)
        else:
            c = Image.new("RGB", (max_w, max_h), BG_COLOR)
            c.paste(f, (0, 0))
            out.append(c)
    return out


def render_llm_gif(frames: list, output_path: str, fps: int = 3):
    """Write rendered LLM frames as an animated GIF.

    Args:
        frames: Ordered PIL images to encode.
        output_path: Destination GIF path.
        fps: Playback frames per second.
    """
    if not frames:
        return
    frames = _normalize_frames(frames)
    frames[0].save(
        output_path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0
    )


def render_llm_video(frames: list, output_path: str, fps: int = 1):
    """Write rendered LLM frames as an H.264 video.

    Args:
        frames: Ordered PIL images to encode.
        output_path: Destination video path.
        fps: Playback frames per second.
    """
    if not frames:
        return
    import imageio

    frames = _normalize_frames(frames)
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()


def render_llm_html(frames: list, agent_data_per_step: list, output_path: str, fps: int = 3):
    """Write frames and agent metadata as an interactive HTML viewer.

    Args:
        frames: Ordered PIL images embedded in the page.
        agent_data_per_step: Agent details aligned with each frame.
        output_path: Destination HTML path.
        fps: Initial viewer playback rate.
    """
    if not frames:
        return
    frames = _normalize_frames(frames)

    frame_data = []
    for frame in frames:
        buf = io.BytesIO()
        frame.save(buf, format="PNG", optimize=True)
        frame_data.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LLM Agent Viewer</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#12141c;color:#e8ebf2;font-family:'Consolas','DejaVu Sans Mono',monospace}}
.bar{{position:sticky;top:0;z-index:20;background:#0d1017;padding:10px 16px;display:flex;align-items:center;gap:12px;border-bottom:1px solid #2b3244;flex-wrap:wrap}}
.bar button{{background:#273044;color:#dfe4ef;border:1px solid #394560;padding:6px 14px;border-radius:6px;cursor:pointer;font:13px inherit;transition:background .15s}}
.bar button:hover{{background:#34405a}}
.bar button.on{{background:#4d5d84;border-color:#7b92c9}}
input[type=range]{{flex:1;min-width:220px;accent-color:#7d92cc}}
.lbl{{font-size:13px;min-width:120px;color:#aeb8cf}}
.spd{{font-size:11px;color:#7b86a1}}
.filters{{display:flex;gap:14px;flex-wrap:wrap;padding:10px 16px;border-bottom:1px solid #252c3a;background:#151923}}
.filters label{{font-size:12px;color:#afb7cb;display:flex;align-items:center;gap:6px;cursor:pointer}}
.layout{{display:grid;grid-template-columns:minmax(420px,1.05fr) minmax(420px,0.95fr);gap:18px;padding:14px 16px 18px}}
.panel{{background:#171b25;border:1px solid #2b3447;border-radius:12px;overflow:hidden}}
.panel-head{{padding:10px 14px;border-bottom:1px solid #252d3d;background:#11151d;color:#cfd6e6;font-size:12px;text-transform:uppercase;letter-spacing:.6px}}
.frame-wrap{{padding:10px;display:flex;justify-content:center;align-items:flex-start;background:#0d1017}}
.frame-wrap img{{max-width:100%;height:auto;border-radius:8px}}
.details{{padding:12px;display:grid;gap:12px;align-content:start}}
.agent-card{{background:#1d2330;border:1px solid #313b50;border-radius:12px;padding:12px}}
.agent-top{{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:10px}}
.agent-name{{font-size:14px;font-weight:700;color:#f0f3fa}}
.agent-role{{font-size:11px;padding:3px 8px;border-radius:999px;background:#303950;color:#dce2ef;text-transform:uppercase}}
.field{{margin-bottom:10px}}
.field:last-child{{margin-bottom:0}}
.field-label{{font-size:10px;text-transform:uppercase;letter-spacing:.6px;color:#8e9ab3;margin-bottom:4px}}
.text-block{{background:#0f131b;border:1px solid #283041;border-radius:8px;padding:10px 12px;white-space:pre-wrap;word-break:break-word;line-height:1.45;font-size:12px;color:#eef2f8}}
.text-block.soft{{color:#cad2e3}}
.action-block{{background:#3d372c;border-left:4px solid #e0bb56;color:#f7dd94}}
.message-block{{border-left:4px solid #74c7b8}}
.prompt-block{{background:#0f131b;border:1px solid #283041;border-radius:8px;overflow:hidden}}
.prompt-msg{{padding:8px 10px;border-bottom:1px solid #232a38}}
.prompt-msg:last-child{{border-bottom:none}}
.prompt-role{{font-size:10px;text-transform:uppercase;letter-spacing:.6px;margin-bottom:4px;font-weight:700}}
.prompt-role.system{{color:#e0c06f}}
.prompt-role.user{{color:#85a8f7}}
.prompt-role.assistant{{color:#82cb9b}}
.inbox-list{{display:grid;gap:6px}}
.hidden{{display:none}}
@media (max-width: 1300px){{.layout{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="bar">
<button id="pb" onclick="tp()">Play</button>
<button onclick="sb()">&larr;</button>
<button onclick="sf()">&rarr;</button>
<input type="range" id="sl" min="0" max="{len(frames) - 1}" value="0" oninput="go(+this.value)">
<span class="lbl" id="lb">Step 0/{len(frames) - 1}</span>
<button onclick="cs(-1)">-</button>
<span class="spd" id="sp">{fps} fps</span>
<button onclick="cs(1)">+</button>
</div>
<div class="filters">
<label><input type="checkbox" id="showObs" checked> Current obs</label>
<label><input type="checkbox" id="showStatus"> Short-term status</label>
<label><input type="checkbox" id="showMessages" checked> Messages</label>
<label><input type="checkbox" id="showScratchpad" checked> Scratchpad</label>
<label><input type="checkbox" id="showReasoning" checked> Reasoning</label>
<label><input type="checkbox" id="showPrompt"> Full prompt + history</label>
</div>
<div class="layout">
  <div class="panel">
    <div class="panel-head">Rendered Frame</div>
    <div class="frame-wrap"><img id="fi" src="" alt="frame"></div>
  </div>
  <div class="panel">
    <div class="panel-head">LLM Trace</div>
    <div class="details" id="details"></div>
  </div>
</div>
<script>
const F={json.dumps(frame_data)};
const D={json.dumps(agent_data_per_step)};
let c=0,p=false,iv=null,fps={fps};
function esc(s){{if(s===null||s===undefined||s==='')return '<span style="color:#7f8aa3">--</span>';const d=document.createElement('div');d.textContent=String(s);return d.innerHTML;}}
function promptHtml(msgs){{if(!msgs||!msgs.length)return '';return msgs.map(m=>'<div class="prompt-msg"><div class="prompt-role '+(m.role||'user')+'">'+esc(m.role||'user')+'</div><div>'+esc(m.content)+'</div></div>').join('');}}
function inboxHtml(received){{if(!received)return '<div class="text-block soft">--</div>';const entries=Object.entries(received);if(!entries.length)return '<div class="text-block soft">--</div>';return '<div class="inbox-list">'+entries.map(([k,v])=>'<div class="text-block message-block"><strong>'+esc(k)+':</strong> '+esc(v)+'</div>').join('')+'</div>';}}
function fieldBlock(label, cls, value){{return '<div class="field '+cls+'"><div class="field-label">'+label+'</div><div class="text-block '+(cls.includes('action')?'action-block':'')+'">'+esc(value)+'</div></div>';}}
function renderAgentCard(agent, idx){{const role=(agent.role||'agent').toUpperCase();let html='<div class="agent-card">';html+='<div class="agent-top"><div class="agent-name">Agent '+(agent.id??idx)+'</div><div class="agent-role">'+esc(role)+'</div></div>';html+=fieldBlock('Action','field-action',agent.action||'Noop');html+='<div class="field field-obs"><div class="field-label">Current observation</div><div class="text-block">'+esc(agent.obs_long_term)+'</div></div>';html+='<div class="field field-status"><div class="field-label">Short-term status</div><div class="text-block soft">'+esc(agent.obs_short_term)+'</div></div>';html+='<div class="field field-messages"><div class="field-label">Sent message</div><div class="text-block message-block">'+esc(agent.comm_sent)+'</div></div>';html+='<div class="field field-messages"><div class="field-label">Inbox</div>'+inboxHtml(agent.comm_received)+'</div>';html+='<div class="field field-scratchpad"><div class="field-label">Scratchpad</div><div class="text-block">'+esc(agent.scratchpad)+'</div></div>';html+='<div class="field field-reasoning"><div class="field-label">Reasoning</div><div class="text-block soft">'+esc(agent.reasoning)+'</div></div>';if(agent.prompt_messages&&agent.prompt_messages.length){{html+='<div class="field field-prompt"><div class="field-label">Full prompt + history ('+agent.prompt_messages.length+' messages)</div><div class="prompt-block">'+promptHtml(agent.prompt_messages)+'</div></div>';}}html+='</div>';return html;}}
function applyFilters(){{const showObs=document.getElementById('showObs').checked;const showStatus=document.getElementById('showStatus').checked;const showMessages=document.getElementById('showMessages').checked;const showScratchpad=document.getElementById('showScratchpad').checked;const showReasoning=document.getElementById('showReasoning').checked;const showPrompt=document.getElementById('showPrompt').checked;document.querySelectorAll('.field-obs').forEach(el=>el.classList.toggle('hidden',!showObs));document.querySelectorAll('.field-status').forEach(el=>el.classList.toggle('hidden',!showStatus));document.querySelectorAll('.field-messages').forEach(el=>el.classList.toggle('hidden',!showMessages));document.querySelectorAll('.field-scratchpad').forEach(el=>el.classList.toggle('hidden',!showScratchpad));document.querySelectorAll('.field-reasoning').forEach(el=>el.classList.toggle('hidden',!showReasoning));document.querySelectorAll('.field-prompt').forEach(el=>el.classList.toggle('hidden',!showPrompt));}}
function renderDetails(){{const stepData=D[c]||[];document.getElementById('details').innerHTML=stepData.map((agent,idx)=>renderAgentCard(agent,idx)).join('');applyFilters();}}
function sh(i){{c=Math.max(0,Math.min(i,F.length-1));document.getElementById('fi').src='data:image/png;base64,'+F[c];document.getElementById('sl').value=c;document.getElementById('lb').textContent='Step '+c+'/'+(F.length-1);renderDetails();}}
function go(i){{sh(i)}}function sf(){{sh(c+1)}}function sb(){{sh(c-1)}}
function tp(){{p=!p;const b=document.getElementById('pb');b.textContent=p?'Pause':'Play';b.classList.toggle('on',p);if(p)iv=setInterval(()=>{{if(c>=F.length-1){{tp();return}}sf()}},1000/fps);else clearInterval(iv)}}
function cs(d){{fps=Math.max(1,Math.min(30,fps+d));document.getElementById('sp').textContent=fps+' fps';if(p){{clearInterval(iv);p=false;tp()}}}}
['showObs','showStatus','showMessages','showScratchpad','showReasoning','showPrompt'].forEach(id=>document.getElementById(id).addEventListener('change',applyFilters));
document.addEventListener('keydown',e=>{{if(e.key===' '){{e.preventDefault();tp()}}if(e.key==='ArrowLeft')sb();if(e.key==='ArrowRight')sf()}});
sh(0);
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
