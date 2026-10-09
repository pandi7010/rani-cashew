"""Look and widgets of the grader UI (design only; no grading, tracking or relay logic).

Ported from the 5-track CASHEW X AI project (home_ui.py, external drive, read only) so that this
machine has the same look. Changes for this machine: Roboto instead of Segoe UI (Linux), two more
menu icons, a narrower right column for the 1920 px screen.

All round shapes (start/stop button, EXIT / SHUTDOWN, lane LEDs, status dots) are drawn
here with Pillow at 4x size and scaled down, so they are smooth. No external images.
"""
import os
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageTk

# ---------------------------------------------------------------- theme
BG, PANEL, BORDER, TEXT, MUTED = "#0E1420", "#161E2E", "#263044", "#E6EAF2", "#8A94A8"
ACCENT, GOOD, WARN, BAD, IDLE = "#22B8CF", "#2FBF71", "#F2A93B", "#E5484D", "#3A4558"
FIELD = "#0B111B"      # input fields
FONT = "Roboto"
# Four text sizes, in pixels (negative Tk sizes are pixels).
F_SMALL, F_BODY, F_HEAD, F_KPI = -13, -16, -22, -34
GRID = 8
NAV_W, RIGHT_W, TOP_H, LANE_BAR_H, KPI_H = 176, 640, 64, 40, 104
SS = 4      # supersampling factor for the drawn shapes
_FONT_DIRS = ("/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF", os.path.expanduser("~/.local/share/fonts/roboto"),
              os.path.expanduser("~/.fonts"))

# The same colours for drawing on the camera image (OpenCV uses B, G, R).
BGR_GOOD, BGR_BAD, BGR_MUTED, BGR_ACCENT = (113, 191, 47), (77, 72, 229), (168, 148, 138), (207, 184, 34)
BGR_TEXT, BGR_PANEL = (242, 234, 230), (46, 30, 22)


def rgb(colour):
    return tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))


def mix(a, b, t):
    a, b = rgb(a), rgb(b)
    return "#%02X%02X%02X" % tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _pil_font(size, bold=True):
    name = "Roboto-Bold.ttf" if bold else "Roboto-Regular.ttf"
    for folder in _FONT_DIRS:
        try:
            return ImageFont.truetype(os.path.join(folder, name), size)
        except Exception:
            continue
    try:
        return ImageFont.truetype(name, size)           # found through fontconfig, if at all
    except Exception:
        return ImageFont.load_default()


def _photo(image, bg):
    """RGBA drawing -> opaque Tk image on the given background colour."""
    base = Image.new("RGBA", image.size, rgb(bg) + (255,))
    return ImageTk.PhotoImage(Image.alpha_composite(base, image).convert("RGB"))


# ------------------------------------------------------- drawn shapes
def _disc(size_px, centre, painter):
    yy, xx = np.mgrid[0:size_px, 0:size_px].astype(np.float32)
    dx, dy = xx - centre, yy - centre
    return painter(np.hypot(dx, dy), np.arctan2(dy, dx), dy)


def _chrome(theta, t):
    """Polished metal: light and dark bands around the ring, darker towards both edges."""
    band = 0.5 + 0.5 * np.cos(2.0 * (theta + np.pi * 0.30))
    band = 0.25 + 0.75 * band ** 1.4
    edge = 1.0 - 0.45 * np.abs(t - 0.45) * 2.0
    return np.clip(band * edge, 0.0, 1.0)


def engine_button_image(running, hover=False, pressed=False, d=176):
    """Car-style push button: chrome bezel, glowing LED ring (red = stopped, green = running),
    matte charcoal centre with MACHINE / START / STOP. Returns an RGBA image."""
    colour = np.array(rgb(GOOD if running else BAD), dtype=np.float32)
    pad = 26
    size = d + 2 * pad
    S = size * SS
    c = S / 2.0
    R = d * SS / 2.0 * (0.97 if pressed else 1.0)
    r_bezel_in = R * 0.845
    r_led_out, r_led_in = R * 0.815, R * 0.745
    r_centre = R * 0.715

    def paint(r, theta, dy):
        out = np.zeros(r.shape + (4,), dtype=np.float32)
        m = (r <= R) & (r > r_bezel_in)
        shade = _chrome(theta, (r - r_bezel_in) / (R - r_bezel_in))
        for i, (lo, hi) in enumerate(((58, 248), (64, 250), (74, 252))):
            out[..., i] = np.where(m, lo + (hi - lo) * shade, out[..., i])
        out[..., 3] = np.where(m, 255, out[..., 3])
        m = (r <= r_bezel_in) & (r > r_centre)
        out[m] = (9, 11, 15, 255)
        m = (r <= r_led_out) & (r > r_led_in)
        core = 1.0 - np.abs((r - r_led_in) / (r_led_out - r_led_in) - 0.5) * 2.0
        boost = 1.18 if hover else 1.0
        for i in range(3):
            out[..., i] = np.where(m, np.clip((colour[i] * (0.80 + 0.20 * core) + 70.0 * core ** 3) * boost, 0, 255), out[..., i])
        m = r <= r_centre
        g = 66.0 - 30.0 * (r / r_centre) ** 1.6 - 12.0 * (dy / r_centre)
        if pressed:
            g = g * 0.82
        for i, off in enumerate((0, 2, 5)):
            out[..., i] = np.where(m, np.clip(g + off, 0, 255), out[..., i])
        out[..., 3] = np.where(m, 255, out[..., 3])
        return out

    face = Image.fromarray(_disc(S, c, paint).astype(np.uint8), "RGBA")

    def glow(radius, width, blur, alpha):
        layer = Image.new("L", (S, S), 0)
        ImageDraw.Draw(layer).ellipse([c - radius, c - radius, c + radius, c + radius], outline=255, width=int(width))
        layer = layer.filter(ImageFilter.GaussianBlur(blur * SS))
        tinted = Image.new("RGBA", (S, S), tuple(int(v) for v in colour) + (0,))
        tinted.putalpha(layer.point(lambda v: int(min(255, v * alpha))))
        return tinted

    strength = 1.35 if hover else 1.0
    canvas = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    shadow = Image.new("L", (S, S), 0)
    ImageDraw.Draw(shadow).ellipse([c - R, c - R + 7 * SS, c + R, c + R + 7 * SS], fill=170)
    canvas.putalpha(shadow.filter(ImageFilter.GaussianBlur(6 * SS)))
    canvas = Image.alpha_composite(canvas, glow(R + 4 * SS, 10 * SS, 9, 1.5 * strength))
    canvas = Image.alpha_composite(canvas, face)
    canvas = Image.alpha_composite(canvas, glow((r_led_in + r_led_out) / 2, r_led_out - r_led_in, 3, 2.2 * strength))
    high = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    rb = (R + r_bezel_in) / 2
    ImageDraw.Draw(high).arc([c - rb, c - rb, c + rb, c + rb], start=205, end=255, fill=(255, 255, 255, 150), width=int(2.2 * SS))
    canvas = Image.alpha_composite(canvas, high.filter(ImageFilter.GaussianBlur(1.2 * SS)))
    out = canvas.resize((size, size), Image.LANCZOS)
    p = ImageDraw.Draw(out)
    k = 0.97 if pressed else 1.0
    white = (235, 238, 244, 255) if pressed else (255, 255, 255, 255)
    p.text((size / 2, size / 2 - 30 * k), "MACHINE", font=_pil_font(int(13 * k)), fill=(200, 206, 216, 255), anchor="mm")
    p.text((size / 2, size / 2 - 5 * k), "START", font=_pil_font(int(21 * k)), fill=white, anchor="mm")
    p.text((size / 2, size / 2 + 22 * k), "STOP", font=_pil_font(int(21 * k)), fill=white, anchor="mm")
    return out


def icon_button_image(kind, hover=False, pressed=False, d=44):
    """Small round button in the same style: chrome bezel, dark centre, line icon (exit / power)."""
    S = d * SS
    c = S / 2.0
    R = (S / 2.0 - 1 * SS) * (0.95 if pressed else 1.0)
    r_in = R * 0.80

    def paint(r, theta, dy):
        out = np.zeros(r.shape + (4,), dtype=np.float32)
        m = (r <= R) & (r > r_in)
        shade = _chrome(theta, (r - r_in) / (R - r_in)) * (1.12 if hover else 1.0)
        for i, (lo, hi) in enumerate(((58, 248), (64, 250), (74, 252))):
            out[..., i] = np.where(m, np.clip(lo + (hi - lo) * shade, 0, 255), out[..., i])
        out[..., 3] = np.where(m, 255, out[..., 3])
        m = r <= r_in
        g = (60.0 - 28.0 * (r / r_in) ** 1.6 - 10.0 * (dy / r_in)) * (0.8 if pressed else 1.0)
        for i, off in enumerate((0, 2, 5)):
            out[..., i] = np.where(m, np.clip(g + off, 0, 255), out[..., i])
        out[..., 3] = np.where(m, 255, out[..., 3])
        return out

    big = Image.fromarray(_disc(S, c, paint).astype(np.uint8), "RGBA")
    p = ImageDraw.Draw(big)
    s, cx = SS, int(c)
    line = rgb(BAD if kind == "power" else TEXT) + (255,)
    if kind == "power":
        r = 8 * s
        p.arc([cx - r, cx - r + 1 * s, cx + r, cx + r + 1 * s], start=-60, end=240, fill=line, width=int(2.2 * s))
        p.line([cx, cx - 10 * s, cx, cx - 1 * s], fill=line, width=int(2.2 * s))
    else:
        p.line([cx - 2 * s, cx - 8 * s, cx - 8 * s, cx - 8 * s, cx - 8 * s, cx + 8 * s, cx - 2 * s, cx + 8 * s], fill=line, width=2 * s)
        p.line([cx - 3 * s, cx, cx + 8 * s, cx], fill=line, width=2 * s)
        p.line([cx + 4 * s, cx - 4 * s, cx + 8 * s, cx, cx + 4 * s, cx + 4 * s], fill=line, width=2 * s)
    return big.resize((d, d), Image.LANCZOS)


def led_image(state, d=18):
    """Round LED of diameter d: 'idle' (dark grey with a thin ring), 'good' (green), 'bad' (red) or
    'firing' (cyan), the lit ones with a soft glow. The image is a little larger than d for the glow."""
    margin = max(3, d // 4)
    size = d + 2 * margin
    big = Image.new("RGBA", (size * SS, size * SS), (0, 0, 0, 0))
    c, r = size * SS / 2.0, d * SS / 2.0
    if state not in ("good", "bad", "firing"):
        p = ImageDraw.Draw(big)
        p.ellipse([c - r, c - r, c + r, c + r], fill=rgb("#232C3D") + (255,), outline=rgb("#4A566C") + (255,), width=max(1, SS))
        return big.resize((size, size), Image.LANCZOS)
    colour = rgb({"good": GOOD, "bad": BAD, "firing": ACCENT}[state])
    glow = Image.new("L", (size * SS, size * SS), 0)
    ImageDraw.Draw(glow).ellipse([c - r - margin * SS * 0.5, c - r - margin * SS * 0.5, c + r + margin * SS * 0.5, c + r + margin * SS * 0.5], fill=150)
    halo = Image.new("RGBA", big.size, colour + (0,))
    halo.putalpha(glow.filter(ImageFilter.GaussianBlur(margin * SS * 0.45)))
    big = Image.alpha_composite(big, halo)
    p = ImageDraw.Draw(big)
    p.ellipse([c - r, c - r, c + r, c + r], fill=colour + (255,))
    return big.resize((size, size), Image.LANCZOS)


def dot_image(colour, d=12):
    big = Image.new("RGBA", (d * SS, d * SS), (0, 0, 0, 0))
    ImageDraw.Draw(big).ellipse([0, 0, d * SS - 1, d * SS - 1], fill=rgb(colour) + (255,))
    return big.resize((d, d), Image.LANCZOS)


def nav_icon_image(name, colour, d=24):
    """Line icons, one per page, all with the same line weight (drawn on a 24 x 24 grid)."""
    big = Image.new("RGBA", (d * SS, d * SS), (0, 0, 0, 0))
    p = ImageDraw.Draw(big)
    k = d * SS / 24.0
    c, w = rgb(colour) + (255,), max(1, int(round(1.7 * k)))

    def line(*pts):
        p.line([v * k for v in pts], fill=c, width=w, joint="curve")
        for i in range(0, len(pts), 2):                      # round ends and corners
            dot(pts[i], pts[i + 1], 0.85)

    def box(x1, y1, x2, y2):
        line(x1, y1, x2, y1, x2, y2, x1, y2, x1, y1)

    def ring(x, y, r):
        p.ellipse([(x - r) * k, (y - r) * k, (x + r) * k, (y + r) * k], outline=c, width=w)

    def dot(x, y, r):
        p.ellipse([(x - r) * k, (y - r) * k, (x + r) * k, (y + r) * k], fill=c)

    def corners(x1, y1, x2, y2, n):
        line(x1, y1 + n, x1, y1, x1 + n, y1)
        line(x2 - n, y1, x2, y1, x2, y1 + n)
        line(x2, y2 - n, x2, y2, x2 - n, y2)
        line(x1 + n, y2, x1, y2, x1, y2 - n)

    if name == "Home":                         # house
        line(3, 12, 12, 4, 21, 12)
        line(5.5, 10.5, 5.5, 20, 18.5, 20, 18.5, 10.5)
        line(10.5, 20, 10.5, 14.5, 13.5, 14.5, 13.5, 20)
    elif name == "Pneumatic":                  # air nozzle blowing to the right
        line(2.5, 9, 9, 9, 13.5, 11, 13.5, 13, 9, 15, 2.5, 15, 2.5, 9)
        line(16.5, 12, 21.5, 12)
        line(16, 9.2, 20.5, 6.8)
        line(16, 14.8, 20.5, 17.2)
    elif name == "Colour Size":                # colour palette
        p.ellipse([3 * k, 4 * k, 21 * k, 20 * k], outline=c, width=w)
        dot(8, 9.5, 1.4)
        dot(12.5, 7.8, 1.4)
        dot(16.8, 10.5, 1.4)
        ring(14.5, 15.3, 1.9)
    elif name == "Vision ROI":                 # camera inside a frame
        corners(2.5, 3.5, 21.5, 20.5, 3.5)
        box(7, 9.5, 17, 16.5)
        line(10, 9.5, 11, 7.8, 13, 7.8, 14, 9.5)
        ring(12, 13, 1.9)
    elif name == "Teaching CNN":               # neural network
        nodes_in, nodes_mid, node_out = ((5, 7), (5, 17)), ((12, 4.5), (12, 12), (12, 19.5)), (19.5, 12)
        thin = max(1, int(round(1.1 * k)))
        for x1, y1 in nodes_in:
            for x2, y2 in nodes_mid:
                p.line([x1 * k, y1 * k, x2 * k, y2 * k], fill=c, width=thin)
        for x1, y1 in nodes_mid:
            p.line([x1 * k, y1 * k, node_out[0] * k, node_out[1] * k], fill=c, width=thin)
        for x, y in nodes_in + nodes_mid + (node_out,):
            dot(x, y, 2.1)
    elif name == "Dataset Capture":            # camera on a stack of images
        line(7, 4, 21, 4, 21, 15)
        box(3, 7.5, 17.5, 20)
        ring(10.2, 13.8, 2.9)
        dot(14.8, 10.2, 0.9)
    elif name == "Teaching YOLO":              # target crosshair
        ring(12, 12, 6.2)
        line(12, 2, 12, 8)
        line(12, 16, 12, 22)
        line(2, 12, 8, 12)
        line(16, 12, 22, 12)
        dot(12, 12, 1.3)
    elif name == "AI Model Select":            # processor chip
        box(7, 7, 17, 17)
        box(10.3, 10.3, 13.7, 13.7)
        for v in (9, 12, 15):
            line(v, 3.5, v, 7)
            line(v, 17, v, 20.5)
            line(3.5, v, 7, v)
            line(17, v, 20.5, v)
    elif name == "Network":                    # globe
        ring(12, 12, 9)
        line(3, 12, 21, 12)
        p.ellipse([8 * k, 3 * k, 16 * k, 21 * k], outline=c, width=w)
    elif name == "Info":                       # i in a circle
        ring(12, 12, 9)
        line(12, 11, 12, 17)
        dot(12, 7.3, 1.2)
    elif name == "Sorting Logic":              # three sliders
        for y, x in ((6.5, 15), (12, 8.5), (17.5, 13)):
            line(3, y, 21, y)
            p.ellipse([(x - 2.3) * k, (y - 2.3) * k, (x + 2.3) * k, (y + 2.3) * k], fill=c)
    elif name == "Test Console":               # terminal window with a prompt
        box(3, 5, 21, 19)
        line(7, 9.5, 10, 12, 7, 14.5)
        line(12.5, 15, 17, 15)
    elif name == "Speed":                      # gauge
        p.arc([3 * k, 5 * k, 21 * k, 23 * k], start=180, end=360, fill=c, width=w)
        line(12, 14, 16.5, 9)
        dot(12, 14, 1.4)
    else:
        box(4, 4, 20, 20)
        line(8, 10, 16, 10)
        line(8, 15, 16, 15)
    return big.resize((d, d), Image.LANCZOS)


# ----------------------------------------------------------- wording
def short_result(decision, reason):
    """The same short words everywhere: (text, is_good). GOOD, or the defect class / cause."""
    reason = str(reason or "")
    if decision == "ACCEPT":
        return "GOOD", True
    if reason.startswith("DEFECT:") or reason.startswith("DEFECT_LOW_CONF:"):
        body = reason.split(":", 1)[1]
        name = body.split("(")[0].strip()
        parts = name.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].replace(".", "", 1).isdigit():
            return f"{parts[0].replace('_', ' ')}  {parts[1]}", False
        return name.replace("_", " "), False
    # A reject added by an inspection engine names the engine: "OPENCV: DARK SPOT", "SHAPE: BROKEN", "HYBRID: LOW CONF".
    if reason.startswith("COLOR_"):             # COLOR ONLY mode: the colour reason itself, e.g. "DARK SPOT  3.1 mm2"
        code, _, rest = reason.partition(" ")
        return code[6:].replace("_", " ") + (f"  {rest}" if rest else ""), False
    for prefix, engine in (("OPENCV_", "OPENCV"), ("SHAPE_", "SHAPE"), ("SIZE_", "SIZE"), ("HYBRID_", "HYBRID")):
        if reason.startswith(prefix):
            code, _, rest = reason.partition(" ")
            return f"{engine}: {code[len(prefix):].replace('_', ' ')}" + (f"  {rest}" if rest else ""), False
    table = (("DOUBLE_OR_TOUCHING", "DOUBLE OR TOUCHING"), ("STUCK", "STUCK"), ("APPEARED_AFTER_LINE", "SEEN AFTER LINE"),
             ("SEEN_ONLY_ONCE", "SEEN ONLY ONCE"), ("LEFT_ROI_UNDECIDED", "LEFT UNDECIDED"), ("LOST_BEFORE_LINE", "LOST BEFORE LINE"))
    for key, text in table:
        if reason.startswith(key):
            return text, False
    if reason.startswith("NOT_ENOUGH_VIEWS"):
        return "NOT ENOUGH VIEWS  " + reason.split(":", 1)[-1], False
    if reason.startswith("LOW_CONFIDENCE"):
        return "LOW CONFIDENCE  " + reason.split(" ")[-1], False
    if reason.startswith("UNKNOWN_CLASS"):
        return "UNKNOWN  " + reason.split(":", 1)[-1], False
    if reason.startswith("LOST_"):
        return "LOST " + reason[5:].replace("MS", " ms"), False
    return (reason.replace("_", " ") or decision), False


def short_id(track_id):
    """L3-000154 -> 3-154"""
    try:
        lane, number = str(track_id)[1:].split("-")
        return f"{lane}-{int(number):03d}"
    except Exception:
        return str(track_id)


# ------------------------------------------------------------- widgets
class Sink:
    """Stands in for a label of the old layout that code still writes to."""

    def config(self, *args, **kwargs):
        pass

    configure = config

    def cget(self, key):
        return ""


class Pill(tk.Frame):
    """Status pill: coloured dot, muted name, bold value."""

    def __init__(self, parent, name, value="", colour=MUTED, short=None):
        super().__init__(parent, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        self._dots = {}
        self._dot = tk.Label(self, bg=PANEL, bd=0)
        self._dot.pack(side=tk.LEFT, padx=(10, 6), pady=9)
        self._name = tk.Label(self, text=name, font=(FONT, F_SMALL), fg=MUTED, bg=PANEL)
        self._name.pack(side=tk.LEFT)
        self._value = tk.Label(self, text=value, font=(FONT, F_SMALL, "bold"), fg=TEXT, bg=PANEL)
        self._value.pack(side=tk.LEFT, padx=(6, 12))
        self._state = None
        self._full_name, self._short_name, self._level = name, short or name, 0
        self.set(value, colour)

    def set_level(self, level):
        """Space saving when the bar is too narrow: 0 normal, 1 tighter padding, 2 short name too, 3 smaller font too."""
        if level == self._level:
            return
        self._level = level
        self._dot.pack_configure(padx=(6, 4) if level >= 1 else (10, 6))
        self._value.pack_configure(padx=(4, 8) if level >= 1 else (6, 12))
        size = F_SMALL + 1 if level >= 3 else F_SMALL       # sizes are negative pixels: -12 is smaller than -13
        self._name.config(text=self._short_name if level >= 2 else self._full_name, font=(FONT, size))
        self._value.config(font=(FONT, size, "bold"))

    def set(self, value, colour, name=None):
        if (value, colour, name) == self._state:
            return
        self._state = (value, colour, name)
        if name is not None:
            self._full_name = self._short_name = name
            self._name.config(text=name)
        if colour not in self._dots:
            self._dots[colour] = _photo(dot_image(colour), PANEL)
        self._dot.config(image=self._dots[colour])
        self._value.config(text=value)


class ImageButton(tk.Label):
    """Round button made of drawn images: normal, hover and pressed (150 ms)."""

    def __init__(self, parent, bg, command=None):
        super().__init__(parent, bg=bg, bd=0, cursor="hand2")
        self._bg = bg
        self.command = command
        self._enabled = True
        self._hover = False
        self._pressed = False
        self._images = {}
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self.bind("<Button-1>", self._click)

    def _make(self, hover, pressed):
        raise NotImplementedError

    def _key(self):
        return (self._hover and self._enabled, self._pressed)

    def _show(self):
        key = self._variant() + self._key()
        if key not in self._images:
            self._images[key] = _photo(self._make(*self._key()), self._bg)
        self.config_image(self._images[key])

    def config_image(self, image):
        tk.Label.config(self, image=image)

    def _variant(self):
        return ()

    def _set_hover(self, value):
        self._hover = value
        self._show()

    def _click(self, event=None):
        if not self._enabled:
            return
        self._pressed = True
        self._show()
        self.after(150, self._release)

    def _release(self):
        self._pressed = False
        self._show()
        if self.command:
            self.command()


class EngineButton(ImageButton):
    """MACHINE START / STOP push button. Green ring = running, red ring = stopped.

    Accepts the config(text=..., state=..., command=...) calls of the previous start button.
    """

    def __init__(self, parent, bg, command=None):
        super().__init__(parent, bg, command)
        self._text = "START RUNNING"
        self._running = False
        self._show()

    def _variant(self):
        return (self._running,)

    def _make(self, hover, pressed):
        return engine_button_image(self._running, hover, pressed)

    def config(self, **kwargs):
        state = kwargs.pop("state", None)
        if state is not None:
            self._enabled = state != tk.DISABLED
            tk.Label.config(self, cursor="hand2" if self._enabled else "arrow")
        if "text" in kwargs:
            self._text = kwargs.pop("text")
            running = "STOP" in self._text.upper()
            if running != self._running:
                self._running = running
                self._show()
        kwargs.pop("style", None)
        if "command" in kwargs:
            self.command = kwargs.pop("command")
        if kwargs:
            tk.Label.config(self, **kwargs)

    configure = config

    def cget(self, key):
        return self._text if key == "text" else tk.Label.cget(self, key)


class IconButton(ImageButton):
    def __init__(self, parent, kind, bg, command=None):
        self._kind = kind
        super().__init__(parent, bg, command)
        self._show()

    def _make(self, hover, pressed):
        return icon_button_image(self._kind, hover, pressed)


class LedLabel(tk.Label):
    """Lane operating LED. set('good' | 'bad' | 'idle')."""

    def __init__(self, parent, bg, d=18, extra=()):
        super().__init__(parent, bg=bg, bd=0)
        self._images = {state: _photo(led_image(state, d), bg) for state in ("idle", "good", "bad") + tuple(extra)}
        self._state = None
        self.set("idle")

    def set(self, state):
        if state != self._state:
            self._state = state
            tk.Label.config(self, image=self._images[state])


class NavItem(tk.Frame):
    def __init__(self, parent, name, command):
        super().__init__(parent, bg=PANEL, height=48, cursor="hand2")
        self.pack_propagate(False)
        self.command = command
        self._active = False
        self._bar = tk.Frame(self, bg=PANEL, width=4)
        self._bar.pack(side=tk.LEFT, fill=tk.Y)
        self._icons = {}
        self._name = name
        self._icon = tk.Label(self, bg=PANEL, bd=0)
        self._icon.pack(side=tk.LEFT, padx=(14, 10))
        self._label = tk.Label(self, text=name, font=(FONT, F_BODY), fg=MUTED, bg=PANEL, anchor="w")
        self._label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        for widget in (self, self._bar, self._icon, self._label):
            widget.bind("<Button-1>", lambda e: self.command())
            widget.bind("<Enter>", lambda e: self._paint(hover=True))
            widget.bind("<Leave>", lambda e: self._paint(hover=False))
        self._paint()

    def _paint(self, hover=False):
        bg = mix(PANEL, ACCENT, 0.14) if self._active else (mix(PANEL, TEXT, 0.05) if hover else PANEL)
        fg = TEXT if (self._active or hover) else MUTED
        icon_colour = ACCENT if self._active else fg
        key = (icon_colour, bg)
        if key not in self._icons:
            self._icons[key] = _photo(nav_icon_image(self._name, icon_colour), bg)
        self.config(bg=bg)
        self._bar.config(bg=ACCENT if self._active else bg)
        self._icon.config(image=self._icons[key], bg=bg)
        self._label.config(bg=bg, fg=fg, font=(FONT, F_BODY, "bold") if self._active else (FONT, F_BODY))

    def set_active(self, active):
        if active != self._active:
            self._active = active
            self._paint()


class KpiTile(tk.Frame):
    def __init__(self, parent, title, colour=TEXT):
        super().__init__(parent, bg=PANEL, highlightthickness=1, highlightbackground=BORDER, height=KPI_H)
        self.pack_propagate(False)
        tk.Label(self, text=title, font=(FONT, F_SMALL), fg=MUTED, bg=PANEL, anchor="w").pack(fill=tk.X, padx=16, pady=(10, 0))
        self._value = tk.Label(self, text="0", font=(FONT, F_KPI, "bold"), fg=colour, bg=PANEL, anchor="w")
        self._value.pack(fill=tk.X, padx=16)
        self._sub = tk.Label(self, text="", font=(FONT, F_SMALL), fg=MUTED, bg=PANEL, anchor="w")
        self._sub.pack(fill=tk.X, padx=16)
        self._last = None

    def set(self, value, sub=""):
        if (value, sub) != self._last:
            self._last = (value, sub)
            self._value.config(text=value)
            self._sub.config(text=sub)


def tag_image(label, colour, text_colour):
    """Small rounded tag (PASS / FIRE)."""
    font = _pil_font(13 * SS)
    w = int(font.getlength(label) / SS) + 22
    big = Image.new("RGBA", (w * SS, 26 * SS), (0, 0, 0, 0))
    p = ImageDraw.Draw(big)
    p.rounded_rectangle([0, 0, w * SS - 1, 26 * SS - 1], radius=13 * SS, fill=rgb(colour) + (255,))
    p.text((w * SS / 2, 13 * SS), label, font=font, fill=rgb(text_colour) + (255,), anchor="mm")
    return big.resize((w, 26), Image.LANCZOS)


def switch_image(on):
    w, h = 64, 32
    big = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    p = ImageDraw.Draw(big)
    p.rounded_rectangle([0, 0, w * SS - 1, h * SS - 1], radius=h * SS // 2, fill=rgb(GOOD if on else IDLE) + (255,))
    x = (w - h + 3) if on else 3
    p.ellipse([x * SS, 3 * SS, (x + h - 6) * SS, (h - 3) * SS], fill=(255, 255, 255, 255))
    return big.resize((w, h), Image.LANCZOS)


class FlatButton(tk.Frame):
    """Flat button: 'outline' (accent), 'danger' (red outline) or 'primary' (filled). Greyed when disabled."""

    def __init__(self, parent, text, command=None, kind="outline", height=32, width=None, pad=16, font=None):
        super().__init__(parent, height=height, highlightthickness=1, cursor="hand2")
        self.command, self._kind, self._enabled, self._hover = command, kind, True, False
        self._label = tk.Label(self, text=text, font=font or (FONT, F_SMALL, "bold"), cursor="hand2")
        self._label.place(relx=0.5, rely=0.5, anchor="center")
        tk.Frame.config(self, width=width or self._label.winfo_reqwidth() + 2 * pad)
        self.pack_propagate(False)
        for widget in (self, self._label):
            widget.bind("<Button-1>", self._click)
            widget.bind("<Enter>", lambda e: self._set_hover(True))
            widget.bind("<Leave>", lambda e: self._set_hover(False))
        self._paint()

    def _paint(self):
        parent_bg = self.master.cget("bg")
        if self._kind == "primary":
            fill = (mix(ACCENT, "#FFFFFF", 0.15) if self._hover else ACCENT) if self._enabled else IDLE
            border, fg = fill, "#06222A" if self._enabled else "#8A94A8"
        else:
            colour = {"danger": BAD, "muted": MUTED}.get(self._kind, ACCENT) if self._enabled else IDLE
            fill = mix(parent_bg, colour, 0.16) if (self._hover and self._enabled) else parent_bg
            border, fg = colour, colour if self._enabled else "#566178"
        tk.Frame.config(self, bg=fill, highlightbackground=border, highlightcolor=border, cursor="hand2" if self._enabled else "arrow")
        self._label.config(bg=fill, fg=fg, cursor="hand2" if self._enabled else "arrow")

    def _set_hover(self, value):
        self._hover = value
        self._paint()

    def _click(self, event=None):
        if self._enabled and self.command:
            self.command()

    def set_enabled(self, enabled):
        if enabled != self._enabled:
            self._enabled = enabled
            self._paint()

    def set_text(self, text):
        if self._label.cget("text") != text:
            self._label.config(text=text)

    def set_kind(self, kind):
        if kind != self._kind:
            self._kind = kind
            self._paint()

    def config(self, **kwargs):
        """Accepts the text= / state= / command= calls older code makes on its buttons; colours are owned by the theme."""
        if "text" in kwargs:
            self.set_text(kwargs.pop("text"))
        if "state" in kwargs:
            self.set_enabled(kwargs.pop("state") != tk.DISABLED)
        if "command" in kwargs:
            self.command = kwargs.pop("command")
        for key in ("bg", "fg", "background", "foreground", "style", "font"):
            kwargs.pop(key, None)
        if kwargs:
            tk.Frame.config(self, **kwargs)

    configure = config


class ToggleSwitch(tk.Label):
    """ON / OFF switch. command(state) is called after a click."""

    def __init__(self, parent, bg, on=True, command=None):
        super().__init__(parent, bg=bg, bd=0, cursor="hand2")
        self._images = {state: _photo(switch_image(state), bg) for state in (True, False)}
        self.command, self.on = command, bool(on)
        tk.Label.config(self, image=self._images[self.on])
        self.bind("<Button-1>", self._click)

    def _click(self, event=None):
        self.on = not self.on
        tk.Label.config(self, image=self._images[self.on])
        if self.command:
            self.command(self.on)

    def set(self, on):
        """Show a state without calling the command (for a change that was refused or made elsewhere)."""
        self.on = bool(on)
        tk.Label.config(self, image=self._images[self.on])


class Chip(tk.Frame):
    """Class chip: green PASS or red FIRE, click to switch. A fixed chip cannot be switched."""

    def __init__(self, parent, name, fire, command=None, fixed=False):
        super().__init__(parent, height=48, highlightthickness=1, cursor="arrow" if fixed else "hand2")
        self.pack_propagate(False)
        self.command, self._fire = command, None
        self._name = tk.Label(self, text=name, font=(FONT, F_BODY, "bold"), fg=TEXT, anchor="w")
        self._name.pack(side=tk.LEFT, padx=(14, 0))
        self._tag = tk.Label(self, bd=0)
        self._tag.pack(side=tk.RIGHT, padx=(0, 10))
        self._fixed = tk.Label(self, text="FIXED RULE" if fixed else "", font=(FONT, -9, "bold"), fg=MUTED)
        self._fixed.pack(side=tk.RIGHT, padx=(0, 8))
        self._tags = {}
        if not fixed:
            for widget in (self, self._name, self._tag, self._fixed):
                widget.config(cursor="hand2")
                widget.bind("<Button-1>", lambda e: self.command and self.command())
        self.set(fire)

    def set(self, fire):
        if fire == self._fire:
            return
        self._fire = fire
        colour = BAD if fire else GOOD
        bg = mix(PANEL, colour, 0.10)
        if fire not in self._tags:
            self._tags[fire] = _photo(tag_image("FIRE" if fire else "PASS", colour, "#FFFFFF" if fire else "#06222A"), bg)
        self.config(bg=bg, highlightbackground=mix(BORDER, colour, 0.45))
        self._name.config(bg=bg)
        self._fixed.config(bg=bg)
        self._tag.config(image=self._tags[fire], bg=bg)


class Segmented(tk.Frame):
    """Small selector, e.g. OFF / SHADOW / ACTIVE. command(value) is called after a click."""

    def __init__(self, parent, options, value, command=None, colours=None, font=None, pad=9):
        super().__init__(parent, bg=BORDER, highlightthickness=1, highlightbackground=BORDER)
        self.command, self.value, self._colours, self._labels = command, value, colours or {}, {}
        for option in options:
            label = tk.Label(self, text=option, font=font or (FONT, -11, "bold"), padx=pad, pady=5, cursor="hand2")
            label.pack(side=tk.LEFT, padx=(0, 1) if option != options[-1] else 0)
            label.bind("<Button-1>", lambda e, o=option: self._click(o))
            self._labels[option] = label
        self.set(value)

    def set(self, value):
        if value == self.value and getattr(self, "_painted", False):
            return
        self.value, self._painted = value, True
        for option, label in self._labels.items():
            if option == value:
                label.config(bg=self._colours.get(option, ACCENT), fg="#06222A")
            else:
                label.config(bg=PANEL, fg=MUTED)

    def _click(self, option):
        if option != self.value and self.command:
            self.command(option)


class Slider(tk.Canvas):
    """Dark slider bound to a StringVar (shared with an entry field). Drag or click to set."""

    def __init__(self, parent, var, lo, hi, step, bg, width=220):
        super().__init__(parent, bg=bg, height=24, width=width, highlightthickness=0, bd=0, cursor="hand2")
        self.var, self.lo, self.hi, self.step = var, float(lo), float(hi), float(step)
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Button-1>", self._drag)
        self.bind("<B1-Motion>", self._drag)
        var.trace_add("write", lambda *_: self._draw())

    def _value(self):
        try:
            return min(self.hi, max(self.lo, float(self.var.get().strip().replace(",", "."))))
        except ValueError:
            return self.lo

    def _draw(self):
        self.delete("all")
        w = max(40, self.winfo_width())
        x0, x1, y = 9, w - 9, 12
        x = x0 + (x1 - x0) * (self._value() - self.lo) / (self.hi - self.lo)
        self.create_line(x0, y, x1, y, fill=BORDER, width=4, capstyle="round")
        self.create_line(x0, y, x, y, fill=ACCENT, width=4, capstyle="round")
        self.create_oval(x - 8, y - 8, x + 8, y + 8, fill="#FFFFFF", outline=ACCENT, width=2)

    def _drag(self, event):
        w = max(40, self.winfo_width())
        fraction = min(1.0, max(0.0, (event.x - 9) / (w - 18)))
        value = round((self.lo + fraction * (self.hi - self.lo)) / self.step) * self.step
        decimals = max(0, len(f"{self.step:g}".partition(".")[2]))
        text = f"{value:.{decimals}f}"
        if text != self.var.get():
            self.var.set(text)


class Histogram(tk.Canvas):
    """Small histogram of taught values with the mean -/+ k x std lines."""

    def __init__(self, parent, title, unit, height=132):
        super().__init__(parent, bg=FIELD, height=height, highlightthickness=1, highlightbackground=BORDER, bd=0)
        self.title, self.unit, self._data, self._k = title, unit, [], 3.0
        self.bind("<Configure>", lambda e: self._draw())

    def show(self, values, k):
        self._data, self._k = list(values), k
        self._draw()

    def _draw(self):
        self.delete("all")
        w, h = max(80, self.winfo_width()), max(60, self.winfo_height())
        self.create_text(10, 12, text=self.title, anchor="w", fill=MUTED, font=(FONT, -11, "bold"))
        if len(self._data) < 2:
            self.create_text(w / 2, h / 2, text="no samples yet", fill=MUTED, font=(FONT, F_SMALL))
            return
        n = len(self._data)
        mean = sum(self._data) / n
        std = (sum((v - mean) ** 2 for v in self._data) / n) ** 0.5
        self.create_text(w - 10, 12, text=f"mean {mean:.1f}  std {std:.2f} {self.unit}", anchor="e", fill=MUTED, font=(FONT, -11))
        spread = max(std * (self._k + 1.2), 1e-6, (max(self._data) - min(self._data)) / 2 * 1.1)
        lo, hi, bins = mean - spread, mean + spread, 16
        counts = [0] * bins
        for v in self._data:
            counts[min(bins - 1, max(0, int((v - lo) / (hi - lo) * bins)))] += 1
        x0, x1, base, top = 10, w - 10, h - 22, 28
        bw = (x1 - x0) / bins
        for i, count in enumerate(counts):
            if count:
                self.create_rectangle(x0 + i * bw + 1, base - (base - top) * count / max(counts), x0 + (i + 1) * bw - 1, base, fill=ACCENT, outline="")
        self.create_line(x0, base, x1, base, fill=BORDER)
        for limit in (mean - self._k * std, mean + self._k * std):
            x = x0 + (limit - lo) / (hi - lo) * (x1 - x0)
            self.create_line(x, top - 4, x, base + 2, fill=WARN)
            self.create_text(x, base + 11, text=f"{limit:.1f}", fill=WARN, font=(FONT, -11, "bold"))


class DualHistogram(tk.Canvas):
    """GSN_V21: two groups of values (e.g. WW and SW cashews) in one histogram, with a threshold line."""

    def __init__(self, parent, height=150):
        super().__init__(parent, bg=FIELD, height=height, highlightthickness=1, highlightbackground=BORDER, bd=0)
        self._groups, self._threshold, self._title = [], None, ""
        self.bind("<Configure>", lambda e: self._draw())

    def show(self, groups, threshold=None, title=""):
        """groups: [(name, colour, values)]."""
        self._groups, self._threshold, self._title = [(n, c, list(v)) for n, c, v in groups], threshold, title
        self._draw()

    def _draw(self):
        self.delete("all")
        w, h = max(120, self.winfo_width()), max(60, self.winfo_height())
        self.create_text(10, 12, text=self._title, anchor="w", fill=MUTED, font=(FONT, -11, "bold"))
        x = w - 10
        for name, colour, values in reversed(self._groups):
            label = f"{name} {len(values)}"
            self.create_text(x, 12, text=label, anchor="e", fill=colour, font=(FONT, -11, "bold"))
            x -= 9 * len(label) + 14
        everything = [v for _, _, values in self._groups for v in values]
        if self._threshold is not None:
            everything = everything + [self._threshold]
        if len(everything) < 2:
            self.create_text(w / 2, h / 2, text="no samples yet", fill=MUTED, font=(FONT, F_SMALL))
            return
        lo, hi = min(everything), max(everything)
        pad = max((hi - lo) * 0.08, 0.5)
        lo, hi, bins = lo - pad, hi + pad, 32
        x0, x1, base, top = 10, w - 10, h - 22, 28
        bw = (x1 - x0) / bins
        counts = []
        for _, _, values in self._groups:
            row = [0] * bins
            for v in values:
                row[min(bins - 1, max(0, int((v - lo) / (hi - lo) * bins)))] += 1
            counts.append(row)
        peak = max([max(row) for row in counts] + [1])
        for (_, colour, _), row in zip(self._groups, counts):
            for i, count in enumerate(row):
                if count:
                    self.create_rectangle(x0 + i * bw + 1, base - (base - top) * count / peak, x0 + (i + 1) * bw - 1, base,
                                          fill=colour, outline="", stipple="gray50")
        self.create_line(x0, base, x1, base, fill=BORDER)
        for value in (lo + pad, hi - pad):
            self.create_text(x0 + (value - lo) / (hi - lo) * (x1 - x0), base + 11, text=f"{value:.1f}", fill=MUTED, font=(FONT, -11))
        if self._threshold is not None:
            tx = x0 + (self._threshold - lo) / (hi - lo) * (x1 - x0)
            self.create_line(tx, top - 6, tx, base + 2, fill="#FFFFFF", width=2)
            self.create_text(tx, base + 11, text=f"{self._threshold:.1f}", fill="#FFFFFF", font=(FONT, -11, "bold"))


class ProgressBar(tk.Canvas):
    """Thin progress bar, 0.0 - 1.0."""

    def __init__(self, parent, bg, colour=None, height=8):
        super().__init__(parent, bg=bg, height=height, highlightthickness=0, bd=0)
        self._fraction, self._colour = 0.0, colour or ACCENT
        self.bind("<Configure>", lambda e: self._draw())

    def set(self, fraction, colour=None):
        self._fraction = min(1.0, max(0.0, float(fraction)))
        if colour:
            self._colour = colour
        self._draw()

    def _draw(self):
        self.delete("all")
        w, h = max(10, self.winfo_width()), max(4, self.winfo_height())
        self.create_line(h / 2, h / 2, w - h / 2, h / 2, fill=BORDER, width=h, capstyle="round")
        if self._fraction > 0:
            self.create_line(h / 2, h / 2, max(h / 2 + 1, (w - h / 2) * self._fraction), h / 2, fill=self._colour, width=h, capstyle="round")


class LineChart(tk.Canvas):
    """Small line chart of one series (training loss, accuracy)."""

    def __init__(self, parent, title, colour, height=200):
        super().__init__(parent, bg=FIELD, height=height, highlightthickness=1, highlightbackground=BORDER, bd=0)
        self.title, self.colour, self._values = title, colour, []
        self.bind("<Configure>", lambda e: self._draw())

    def show(self, values):
        self._values = list(values)
        self._draw()

    def _draw(self):
        self.delete("all")
        w, h = max(80, self.winfo_width()), max(60, self.winfo_height())
        self.create_text(10, 12, text=self.title, anchor="w", fill=MUTED, font=(FONT, -11, "bold"))
        if not self._values:
            self.create_text(w / 2, h / 2, text="no data yet", fill=MUTED, font=(FONT, F_SMALL))
            return
        lo, hi = min(self._values), max(self._values)
        if hi - lo < 1e-9:
            lo, hi = lo - 0.5, hi + 0.5
        self.create_text(w - 10, 12, text=f"{self._values[-1]:.3f}", anchor="e", fill=self.colour, font=(FONT, -11, "bold"))
        x0, x1, top, base = 12, w - 12, 30, h - 14
        n = len(self._values)
        points = []
        for i, value in enumerate(self._values):
            points += [x0 + (x1 - x0) * (i / (n - 1) if n > 1 else 0.5), base - (value - lo) / (hi - lo) * (base - top)]
        if n > 1:
            self.create_line(*points, fill=self.colour, width=2, smooth=False)
        self.create_oval(points[-2] - 3, points[-1] - 3, points[-2] + 3, points[-1] + 3, fill=self.colour, outline="")


def thousands(n):
    return f"{int(n):,}".replace(",", " ")
