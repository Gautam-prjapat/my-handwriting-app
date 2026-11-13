"""
Diffusion Handwriting Generation — full-featured Streamlit app.

Exercises the full project surface:
  * nn.py         → DiffusionWriter, StyleExtractor
  * utils.py      → Tokenizer, get_beta_set, new/standard_diffusion_step, pad_img
  * preprocessing → remove_whitespace, photo cleanup
Adds: filmstrip of denoising steps, drawing-GIF animation, multi-step attention,
      stroke smoothing, PDF/SVG/NPZ export, debug panel (β-schedule, tokens, params).
"""
import io, math, os, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
import gdown
from tensorflow.keras.applications import MobileNetV2
from PIL import Image, ImageFilter, ImageOps

import nn, preprocessing, utils

# ---------------------------------------------------------------- config
st.set_page_config(page_title="Diffusion Handwriting Generation",
                   page_icon="✍️", layout="wide")

STYLE_H, STYLE_W   = 96, 1400
NUM_STYLE_TOKENS   = 14
STYLE_DIM          = 1280
WEIGHTS_ID         = "1U1h_sfTx2F51vmBGnELfZvKenospyITo"

FILMSTRIP_STEPS    = [0, 2, 4, 8, 12, 16]      # i indices to capture x_t
ATTENTION_STEPS    = [0, 5, 10, 20, 40]        # i indices to capture attention

SAMPLE_TEXTS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "To be or not to be, that is the question.",
    "Sphinx of black quartz, judge my vow.",
]

# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner=False)
def load_bundle():
    wdir, wpath = "./weights", "./weights/model_weights.h5"
    if not os.path.exists(wpath):
        os.makedirs(wdir, exist_ok=True)
        gdown.download(f"https://drive.google.com/uc?id={WEIGHTS_ID}", wpath, quiet=False)

    c1 = 128
    c2, c3 = c1 * 3 // 2, c1 * 2

    extractor = nn.StyleExtractor()
    extractor.mobilenet = MobileNetV2(include_top=False, pooling=None,
                                      weights="imagenet",
                                      input_shape=(STYLE_H, STYLE_W, 3))
    for l in extractor.mobilenet.layers:
        l.trainable = False

    model = nn.DiffusionWriter(num_layers=2, c1=c1, c2=c2, c3=c3)
    model(tf.random.normal([1, 400, 2]),
          tf.random.uniform([1, 40], dtype=tf.int32, maxval=50),
          tf.random.uniform([1, 1]),
          tf.random.normal([1, NUM_STYLE_TOKENS, STYLE_DIM]))
    model.load_weights(wpath)

    @tf.function(reduce_retracing=True)
    def denoise(x, text, sigma, style):
        return model(x, text, sigma, style)

    return dict(denoise=denoise, model=model, extractor=extractor,
                tokenizer=utils.Tokenizer(), beta_set=utils.get_beta_set())

# ---------------------------------------------------------------- style img
def load_gray(data):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", img.size, (255,)*4), img)
    return np.array(img.convert("L"), dtype=np.uint8)

def clean_photo(gray):
    img = Image.fromarray(gray)
    s = 400 / max(img.size)
    small = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s)))) if s < 1 else img
    bg = (small.filter(ImageFilter.MaxFilter(9))
              .filter(ImageFilter.GaussianBlur(5))
              .resize(img.size, Image.BILINEAR))
    flat = np.clip(gray.astype(np.float32) /
                   np.maximum(np.asarray(bg, np.float32), 1), 0, 1) * 255
    lo = np.percentile(flat, 1)
    if lo < 200:
        flat = np.clip((flat - lo) / (230 - lo) * 255, 0, 255)
    return flat.astype(np.uint8)

def split_into_lines(gray):
    prof = (gray < 127).sum(axis=1).astype(np.float32)
    if prof.max() == 0: return [gray]
    k = max(3, gray.shape[0] // 150)
    prof = np.convolve(prof, np.ones(k) / k, mode="same")
    active = (prof > 0.04 * prof.max()).astype(int)
    edges = np.diff(np.concatenate([[0], active, [0]]))
    runs = list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))
    if not runs: return [gray]
    med = np.median([e - s for s, e in runs])
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        if s - merged[-1][1] < 0.3 * med: merged[-1][1] = e
        else: merged.append([s, e])
    pad = int(0.15 * med)
    out = [gray[max(0, s - pad): min(gray.shape[0], e + pad)]
           for s, e in merged if e - s >= 0.35 * med]
    return out or [gray]

def prepare_line(gray):
    arr = gray[..., None]
    if not (arr < 127).any(): raise ValueError("no ink found")
    arr = preprocessing.remove_whitespace(arr, thresh=127)
    h, w, _ = arr.shape
    if h < 4 or w < 4: raise ValueError("ink region too small")
    arr = tf.image.resize(arr, (STYLE_H, max(8, STYLE_H * w // h)))
    return arr.numpy().astype("uint8")

@st.cache_data(show_spinner=False)
def extract_style(_extractor, data, clean, split_lines):
    gray = load_gray(data)
    if clean: gray = clean_photo(gray)
    pieces = split_into_lines(gray) if split_lines else [gray]

    strips = []
    for p in pieces:
        try: strips.append(prepare_line(p))
        except ValueError: continue
    if not strips: raise ValueError("No handwriting detected.")

    windows = []
    for s in strips:
        for start in range(0, s.shape[1], STYLE_W):
            w = s[:, start:start + STYLE_W]
            if start == 0 or w.shape[1] >= 300:
                windows.append(utils.pad_img(w, STYLE_W, STYLE_H))
    batch = np.stack(windows).astype("float32")
    vecs = [_extractor(batch[i:i + 8]).numpy() for i in range(0, len(batch), 8)]
    return np.concatenate(vecs).mean(axis=0).astype("float32"), strips

# ---------------------------------------------------------------- sampling
def sample(denoise, beta_set, tokenizer, text, style, n_samples,
           steps_per_char, mode, seed, temperature=1.0,
           capture_steps=None, track_att_steps=None, on_step=None):
    tokens = tokenizer.encode(text) + [1]
    text_t = tf.constant([tokens] * n_samples, dtype=tf.int32)
    style_t = tf.repeat(style, n_samples, axis=0)

    T = len(text) * steps_per_char
    T = T - (T % 8) + 8

    tf.random.set_seed(seed)
    L = len(beta_set)
    alpha_set = tf.math.cumprod(1 - beta_set)
    x = tf.random.normal([n_samples, T, 2])

    frames, att_history = [], []

    for i in range(L - 1, -1, -1):
        alpha  = alpha_set[i] * tf.ones([n_samples, 1, 1])
        beta   = beta_set[i] * tf.ones([n_samples, 1, 1])
        a_next = alpha_set[i - 1] if i > 1 else 1.
        eps, pen, att = denoise(x, text_t, tf.sqrt(alpha), style_t)

        if track_att_steps is not None and i in track_att_steps:
            att_history.append((i, att[0].numpy().astype("float16")))
        if capture_steps is not None and i in capture_steps:
            frames.append((i, tf.concat([x, pen], axis=-1).numpy()))

        if mode == "standard":
            x = utils.standard_diffusion_step(x, eps, beta, alpha, add_sigma=bool(i))
        elif temperature == 1.0:
            x = utils.new_diffusion_step(x, eps, beta, alpha, a_next)
        else:
            x_t = (x - tf.sqrt(1 - alpha) * eps) / tf.sqrt(1 - beta)
            x_t += tf.random.normal(x.shape) * (temperature * tf.sqrt(1 - a_next))
            x = x_t

        if on_step is not None: on_step(i)

    final = tf.concat([x, pen], axis=-1).numpy()
    return final, att.numpy(), frames, att_history

# ---------------------------------------------------------------- rendering
def pen_segments(pen):
    cuts = np.where(pen > 0.5)[0]
    bounds = np.concatenate([[0], cuts, [len(pen)]])
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a > 1]

def smooth_strokes(s, window):
    if window < 2: return s
    pos = np.cumsum(s[:, :2], axis=0).astype(np.float32)
    out = pos.copy()
    kernel = np.ones(window) / window
    for a, b in pen_segments(s[:, 2]):
        if b - a < window: continue
        out[a:b, 0] = np.convolve(pos[a:b, 0], kernel, mode="same")
        out[a:b, 1] = np.convolve(pos[a:b, 1], kernel, mode="same")
        out[a:a + window // 2] = pos[a:a + window // 2]
        out[b - window // 2:b] = pos[b - window // 2:b]
    new = s.copy()
    new[0, :2] = out[0]
    new[1:, :2] = out[1:] - out[:-1]
    return new

def _layout(lines, straighten, spacing, smooth):
    placed = []
    for s in lines:
        s = np.asarray(s, dtype=np.float32)
        if smooth > 0: s = smooth_strokes(s, smooth)
        pos = np.cumsum(s[:, :2], axis=0)
        if straighten and len(pos) > 2:
            slope = np.polyfit(pos[:, 0], pos[:, 1], 1)[0]
            pos[:, 1] -= slope * pos[:, 0]
        pos[:, 0] -= pos[:, 0].min()
        pos[:, 1] -= np.median(pos[:, 1])
        placed.append((pos, s[:, 2]))
    if not placed: raise ValueError("Nothing to render.")
    gaps = [max(1e-3, np.ptp(p[:, 1])) for p, _ in placed]
    gap = spacing * np.mean(gaps)
    xmax = max(p[:, 0].max() for p, _ in placed)
    ymin = -gap * (len(placed) - 1) + min(p[:, 1].min() for p, _ in placed)
    ymax = max(p[:, 1].max() for p, _ in placed)
    return placed, gap, xmax, ymin, ymax

def render_lines(lines, ink="#111111", lw=1.5, straighten=True, spacing=1.3,
                 transparent=False, fmt="png", dpi=220, pad=0.5, smooth=0):
    placed, gap, xmax, ymin, ymax = _layout(lines, straighten, spacing, smooth)
    k = 0.12
    fig = plt.figure(figsize=(max(1, (xmax + 2 * pad) * k),
                              max(0.5, (ymax - ymin + 2 * pad) * k)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.axis("off")
    for i, (pos, pen) in enumerate(placed):
        for a, b in pen_segments(pen):
            ax.plot(pos[a:b, 0], pos[a:b, 1] - i * gap, color=ink, linewidth=lw,
                    solid_capstyle="round", solid_joinstyle="round")
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, transparent=transparent)
    plt.close(fig)
    return buf.getvalue()

def render_filmstrip(frames, ink="#111111", lw=1.5, cols=3):
    n = len(frames)
    if n == 0: return None
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 1.3))
    axes = np.array(axes).flatten() if rows * cols > 1 else np.array([axes])
    for idx, (step, strokes) in enumerate(frames):
        ax = axes[idx]
        pos = np.cumsum(strokes[0, :, :2], axis=0)
        for a, b in pen_segments(strokes[0, :, 2]):
            ax.plot(pos[a:b, 0], pos[a:b, 1], color=ink, linewidth=lw,
                    solid_capstyle="round")
        ax.set_title(f"step {step}", fontsize=8); ax.axis("off")
    for j in range(n, len(axes)): axes[j].axis("off")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def render_drawing_gif(lines, ink, lw, straighten, spacing, smooth,
                       n_frames=30, duration_ms=80):
    placed, gap, xmax, ymin, ymax = _layout(lines, straighten, spacing, smooth)
    pad, k = 0.5, 0.12
    figsize = (max(1, (xmax + 2 * pad) * k), max(0.5, (ymax - ymin + 2 * pad) * k))

    segs = [[(pos[a:b, 0], pos[a:b, 1] - i * gap)
             for a, b in pen_segments(pen)]
            for i, (pos, pen) in enumerate(placed)]
    total = sum(len(xs) for line in segs for xs, _ in line)
    if total == 0: return None

    images = []
    for f in range(n_frames + 1):
        budget = (f / n_frames) * total
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(-pad, xmax + pad); ax.set_ylim(ymin - pad, ymax + pad)
        ax.axis("off")
        for line in segs:
            for xs, ys in line:
                if budget <= 0: break
                if budget >= len(xs):
                    ax.plot(xs, ys, color=ink, lw=lw, solid_capstyle="round")
                    budget -= len(xs)
                else:
                    n = int(budget)
                    if n > 1: ax.plot(xs[:n], ys[:n], color=ink, lw=lw,
                                      solid_capstyle="round")
                    budget = 0
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor="white")
        plt.close(fig)
        images.append(Image.open(buf).convert("RGB"))
    gbuf = io.BytesIO()
    images[0].save(gbuf, format="GIF", save_all=True, append_images=images[1:],
                   duration=duration_ms, loop=0, disposal=2, optimize=True)
    return gbuf.getvalue()

def attention_figure(att, text, head="mean"):
    a = att.mean(axis=0) if head == "mean" else att[int(head)]
    fig, ax = plt.subplots(figsize=(max(4, len(text) * 0.25), 3))
    ax.imshow(a.astype("float32"), aspect="auto", cmap="magma")
    ax.set_xticks(range(len(text) + 2))
    ax.set_xticklabels(list(text) + ["⏎", "⏎"], fontsize=7)
    ax.set_ylabel("stroke position (1/8)"); ax.set_xlabel("text token")
    fig.tight_layout()
    return fig

def beta_schedule_figure(beta_set):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    betas = beta_set.numpy(); alphas = np.cumprod(1 - betas)
    axes[0].plot(betas);  axes[0].set_title(r"$\beta_t$");  axes[0].set_xlabel("t")
    axes[1].plot(alphas); axes[1].set_title(r"$\bar\alpha_t$"); axes[1].set_xlabel("t")
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------- page
st.title("✍️ Diffusion Handwriting Generation")
st.caption("Upload a handwriting sample, type text, and synthesise it in that style "
           "with a diffusion model.")

with st.spinner("Loading model (first run downloads weights)…"):
    try:
        bundle = load_bundle()
    except Exception as e:
        st.error(f"Failed to load model: {e}"); st.stop()

denoise        = bundle["denoise"]
model          = bundle["model"]
style_extractor= bundle["extractor"]
tokenizer      = bundle["tokenizer"]
beta_set       = bundle["beta_set"]

# ---- sidebar
with st.sidebar:
    st.header("Sampling")
    mode            = st.selectbox("Diffusion mode", ["new", "standard"],
                                   help="'new' = Eq.12; 'standard' = Eq.9.")
    steps_per_char  = st.slider("Stroke points per character", 12, 24, 16)
    wrap            = st.slider("Max chars per line", 15, 45, 30)
    n_var           = st.slider("Variations", 1, 4, 1)
    randomize       = st.checkbox("Random seed each run", value=False)
    seed            = st.number_input("Seed", 0, 2**31 - 1, 42, disabled=randomize)
    with st.expander("Advanced"):
        temperature = st.slider("Noise temperature", 0.5, 1.5, 1.0, 0.05,
                                help="1.0 = original. Higher = more variation.")

    st.header("Style extraction")
    clean       = st.checkbox("Clean up photo (lighting/contrast)", value=True)
    split_lines = st.checkbox("Auto-split page into lines", value=False)

    st.header("Appearance")
    ink         = st.color_picker("Ink colour", "#111111")
    lw          = st.slider("Pen thickness", 0.5, 4.0, 1.5, 0.1)
    straighten  = st.checkbox("Straighten baseline drift", value=True)
    spacing     = st.slider("Line spacing", 0.8, 2.5, 1.3, 0.1)
    smooth      = st.slider("Stroke smoothing (window)", 0, 7, 0, 1,
                            help="0 = raw. 3–5 noticeably smoother.")
    transparent = st.checkbox("Transparent background", value=False)

    st.header("Visualisations")
    do_filmstrip = st.checkbox("Denoising filmstrip", value=False)
    do_animation = st.checkbox("Drawing animation (GIF)", value=False)
    do_attention = st.checkbox("Attention map", value=True)
    track_att_t  = st.checkbox("Attention over denoising steps", value=False,
                               help="Captures attention at several steps (slower).")

# ---- 1. style
st.subheader("1 · Reference handwriting")
files = st.file_uploader("Upload one or more handwriting images",
                         type=["png","jpg","jpeg","tif","tiff","bmp","webp"],
                         accept_multiple_files=True)
if not files:
    st.info("Upload at least one image of handwriting.")
    st.stop()

styles, weights = [], []
cols = st.columns(min(len(files), 4))
for idx, f in enumerate(files):
    try:
        with st.spinner(f"Extracting style from {f.name}…"):
            vec, strips = extract_style(style_extractor, f.getvalue(), clean, split_lines)
    except Exception as e:
        st.error(f"{f.name}: {e}"); st.stop()
    with cols[idx % len(cols)]:
        st.image(strips[0][..., 0], caption=f"{f.name} ({len(strips)} line(s))", clamp=True)
        w = (st.slider(f"Weight · {f.name}", 0.0, 1.0, 1.0, 0.05, key=f"w_{idx}")
             if len(files) > 1 else 1.0)
    styles.append(vec); weights.append(w)

if sum(weights) == 0:
    st.warning("All style weights are zero."); st.stop()
style = (sum(w * v for w, v in zip(weights, styles)) / sum(weights))[None].astype("float32")

# ---- 2. text
st.subheader("2 · Text")
example = st.selectbox("Load an example (or write below)", [""] + SAMPLE_TEXTS)
text_in = st.text_area("Text to write", example or "Hello Streamlit!", height=110)

wrapped = []
for para in text_in.split("\n"):
    if para.strip(): wrapped += textwrap.wrap(para.strip(), width=wrap)

bad = sorted({c for c in text_in if c != "\n" and c not in tokenizer.text})
if bad: st.warning(f"Unsupported characters → drawn as '_': {' '.join(bad)}")

with st.expander(f"Line breakdown ({len(wrapped)} line(s))"):
    for i, line in enumerate(wrapped):
        ntok = len(tokenizer.encode(line)) + 1
        st.text(f"Line {i+1} ({len(line)} chars, ~{ntok} tokens): {line}")

# ---- 3. generate
st.subheader("3 · Generate")
if st.button("Generate handwriting", type="primary"):
    if not wrapped:
        st.warning("Enter some text first."); st.stop()

    run_seed = int(np.random.randint(0, 2**31 - 1)) if randomize else int(seed)
    total    = len(wrapped) * len(beta_set)
    done     = [0]
    bar      = st.progress(0.0, text="Denoising…")

    def on_step(_):
        done[0] += 1
        if done[0] % 3 == 0 or done[0] == total:
            bar.progress(min(1.0, done[0] / total),
                         text=f"Denoising… {done[0]}/{total} steps")

    strokes_all  = [[None] * len(wrapped) for _ in range(n_var)]
    atts_per_line, att_over_time = [], []
    frames_var   = [[] for _ in range(n_var)]

    try:
        for li, line in enumerate(wrapped):
            final, att, frames, att_hist = sample(
                denoise, beta_set, tokenizer, line, style, n_var,
                steps_per_char, mode, run_seed + li,
                temperature=temperature,
                capture_steps=FILMSTRIP_STEPS if do_filmstrip else None,
                track_att_steps=ATTENTION_STEPS if track_att_t else None,
                on_step=on_step)
            for v in range(n_var):
                strokes_all[v][li] = final[v]
                if do_filmstrip: frames_var[v].append(frames)
            atts_per_line.append(att[0].astype("float16"))
            if track_att_t: att_over_time.append((li, att_hist))
    except Exception as e:
        bar.empty(); st.error(f"Generation failed: {e}"); st.stop()
    bar.empty()
    st.session_state["result"] = dict(lines=wrapped, strokes=strokes_all,
                                      atts=atts_per_line, att_over_time=att_over_time,
                                      frames=frames_var, seed=run_seed)

# ---- 4. results
res = st.session_state.get("result")
if res:
    st.subheader("4 · Result")
    st.caption(f"Seed: {res['seed']}")

    for i, tab in enumerate(st.tabs([f"Sample {i+1}" for i in range(len(res["strokes"]))])):
        with tab:
            png = render_lines(res["strokes"][i], ink, lw, straighten, spacing,
                               transparent, "png", smooth=smooth)
            st.image(png)

            c1, c2, c3, c4 = st.columns(4)
            c1.download_button("PNG", png, f"hw_{i+1}.png", "image/png", key=f"p{i}")
            svg = render_lines(res["strokes"][i], ink, lw, straighten, spacing,
                               transparent, "svg", smooth=smooth)
            c2.download_button("SVG", svg, f"hw_{i+1}.svg", "image/svg+xml", key=f"s{i}")
            pdf = render_lines(res["strokes"][i], ink, lw, straighten, spacing,
                               False, "pdf", smooth=smooth)
            c3.download_button("PDF", pdf, f"hw_{i+1}.pdf", "application/pdf", key=f"d{i}")
            npz_buf = io.BytesIO()
            np.savez(npz_buf, **{f"line_{j}": s for j, s in enumerate(res["strokes"][i])})
            c4.download_button("Raw NPZ", npz_buf.getvalue(), f"hw_{i+1}.npz", key=f"z{i}")

            if do_filmstrip and res["frames"][i]:
                with st.expander("Denoising filmstrip (first line)"):
                    strip = render_filmstrip(res["frames"][i][0], ink, lw)
                    if strip: st.image(strip)

            if do_animation:
                with st.expander("Drawing animation"):
                    gif = render_drawing_gif(res["strokes"][i], ink, lw,
                                             straighten, spacing, smooth)
                    if gif:
                        st.image(gif)
                        st.download_button("Download GIF", gif,
                                           f"hw_{i+1}.gif", "image/gif", key=f"g{i}")

    if do_attention and res["atts"]:
        with st.expander("Text ↔ stroke attention"):
            li   = st.selectbox("Line", range(len(res["lines"])),
                                format_func=lambda j: res["lines"][j], key="al")
            att  = res["atts"][li]
            head = st.selectbox("Head", ["mean"] + list(range(att.shape[0])), key="ah")
            fig  = attention_figure(att, res["lines"][li], head)
            st.pyplot(fig); plt.close(fig)

    if track_att_t and res["att_over_time"]:
        with st.expander("Attention over denoising steps"):
            li    = st.selectbox("Line", range(len(res["lines"])),
                                 format_func=lambda j: res["lines"][j], key="atl")
            entry = next((h for i, h in res["att_over_time"] if i == li), None)
            if entry:
                fig, axes = plt.subplots(1, len(entry), figsize=(3 * len(entry), 2.5))
                axes = [axes] if len(entry) == 1 else axes
                for ax, (step, a) in zip(axes, entry):
                    ax.imshow(a.mean(axis=0).astype("float32"),
                              aspect="auto", cmap="magma")
                    ax.set_title(f"step {step}", fontsize=8); ax.axis("off")
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with st.expander("Debug panel"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("**β / ᾱ schedule**")
            fig = beta_schedule_figure(beta_set); st.pyplot(fig); plt.close(fig)
        with c2:
            st.write("**Model**")
            n = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
            st.metric("Trainable parameters", f"{n:,}")
            st.caption(f"{len(model.trainable_weights)} tensors")
        st.write("**Tokenisation preview**")
        for line in res["lines"]:
            st.code(f"{line!r}\n→ {tokenizer.encode(line)}", language="text")
