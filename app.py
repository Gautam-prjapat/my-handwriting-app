import io
import os
import textwrap

import matplotlib
matplotlib.use("Agg")  # must happen before nn/utils/preprocessing import pyplot
import matplotlib.pyplot as plt
import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageFilter, ImageOps

import nn
import preprocessing
import utils

st.set_page_config(page_title="Diffusion Handwriting Generation", page_icon="✍️", layout="wide")

STYLE_H, STYLE_W = 96, 1400          # size the style extractor was trained on (14 style tokens)
WEIGHTS_ID = "1U1h_sfTx2F51vmBGnELfZvKenospyITo"


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
@st.cache_resource
def load_diffusion_model():
    weights_dir = "./weights"
    weights_path = os.path.join(weights_dir, "model_weights.h5")
    if not os.path.exists(weights_path):
        os.makedirs(weights_dir, exist_ok=True)
        gdown.download(f"https://drive.google.com/uc?id={WEIGHTS_ID}", weights_path, quiet=False)

    C1 = 128
    C2, C3 = C1 * 3 // 2, C1 * 2
    style_extractor = nn.StyleExtractor()
    model = nn.DiffusionWriter(num_layers=2, c1=C1, c2=C2, c3=C3)

    # warm-up pass builds the variables so the weights can be loaded
    model(tf.random.normal([1, 400, 2]), tf.random.uniform([1, 40], dtype=tf.int32, maxval=50),
          tf.random.uniform([1, 1]), tf.random.normal([1, 14, 1280]))
    model.load_weights(weights_path)

    @tf.function(reduce_retracing=True)
    def denoise(x, text, sigma, style):
        return model(x, text, sigma, style)

    return denoise, style_extractor, utils.Tokenizer(), utils.get_beta_set()


# ----------------------------------------------------------------------------
# Style-image processing (uses preprocessing.remove_whitespace / utils.pad_img)
# ----------------------------------------------------------------------------
def load_gray(data: bytes) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
    return np.array(img.convert("L"), dtype=np.uint8)


def clean_photo(gray: np.ndarray) -> np.ndarray:
    """Flatten uneven lighting (phone photos) and push paper to white."""
    img = Image.fromarray(gray)
    s = 400 / max(img.size)
    small = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s)))) if s < 1 else img
    bg = small.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(5)).resize(img.size, Image.BILINEAR)
    flat = np.clip(gray.astype(np.float32) / np.maximum(np.asarray(bg, np.float32), 1), 0, 1) * 255
    lo = np.percentile(flat, 1)
    if lo < 200:
        flat = np.clip((flat - lo) / (230 - lo) * 255, 0, 255)
    return flat.astype(np.uint8)


def split_into_lines(gray: np.ndarray):
    """Horizontal-projection line segmentation for page photos."""
    prof = (gray < 127).sum(axis=1).astype(np.float32)
    if prof.max() == 0:
        return [gray]
    k = max(3, gray.shape[0] // 150)
    prof = np.convolve(prof, np.ones(k) / k, mode="same")
    active = (prof > 0.04 * prof.max()).astype(int)
    edges = np.diff(np.concatenate([[0], active, [0]]))
    runs = list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))
    if not runs:
        return [gray]
    med = np.median([e - s for s, e in runs])
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        if s - merged[-1][1] < 0.3 * med:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    pad = int(0.15 * med)
    out = [gray[max(0, s - pad): min(gray.shape[0], e + pad)] for s, e in merged if e - s >= 0.35 * med]
    return out or [gray]


def prepare_line(gray: np.ndarray) -> np.ndarray:
    """Same recipe as preprocessing.read_img: crop whitespace, resize to height 96."""
    arr = gray[..., None]
    if not (arr < 127).any():
        raise ValueError("no ink found")
    arr = preprocessing.remove_whitespace(arr, thresh=127)
    h, w, _ = arr.shape
    if h < 4 or w < 4:
        raise ValueError("ink region too small")
    arr = tf.image.resize(arr, (STYLE_H, max(8, STYLE_H * w // h)))
    return arr.numpy().astype("uint8")


@st.cache_data(show_spinner=False)
def extract_file_style(_extractor, data: bytes, clean: bool, split_lines: bool):
    gray = load_gray(data)
    if clean:
        gray = clean_photo(gray)
    pieces = split_into_lines(gray) if split_lines else [gray]

    strips = []
    for p in pieces:
        try:
            strips.append(prepare_line(p))
        except ValueError:
            continue
    if not strips:
        raise ValueError("No handwriting detected in this image.")

    # lines wider than 1400px are cut into windows instead of being squashed
    windows = []
    for s in strips:
        for start in range(0, s.shape[1], STYLE_W):
            w = s[:, start:start + STYLE_W]
            if start == 0 or w.shape[1] >= 300:
                windows.append(utils.pad_img(w, STYLE_W, STYLE_H))
    batch = np.stack(windows).astype("float32")

    vecs = [_extractor(batch[i:i + 8]).numpy() for i in range(0, len(batch), 8)]
    vec = np.concatenate(vecs, axis=0).mean(axis=0)     # (14, 1280)
    return vec.astype("float32"), strips


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------
def sample_line(denoise, beta_set, tokenizer, text, style, n_samples, steps_per_char,
                mode, seed, on_step):
    tokens = tokenizer.encode(text) + [1]       # identical to utils.run_batch_inference
    text_t = tf.constant([tokens] * n_samples, dtype=tf.int32)
    style_t = tf.repeat(style, n_samples, axis=0)

    T = len(text) * steps_per_char
    T = T - (T % 8) + 8                          # must be divisible by 8 (three 2x poolings)

    tf.random.set_seed(seed)
    L = len(beta_set)
    alpha_set = tf.math.cumprod(1 - beta_set)
    x = tf.random.normal([n_samples, T, 2])

    for i in range(L - 1, -1, -1):
        alpha = alpha_set[i] * tf.ones([n_samples, 1, 1])
        beta = beta_set[i] * tf.ones([n_samples, 1, 1])
        a_next = alpha_set[i - 1] if i > 1 else 1.   # same quirk as the reference sampler
        eps, pen, att = denoise(x, text_t, tf.sqrt(alpha), style_t)
        if mode == "standard":
            x = utils.standard_diffusion_step(x, eps, beta, alpha, add_sigma=bool(i))
        else:
            x = utils.new_diffusion_step(x, eps, beta, alpha, a_next)
        on_step()

    return tf.concat([x, pen], axis=-1).numpy(), att.numpy()


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def pen_segments(pen):
    cuts = np.where(pen > 0.5)[0]
    bounds = np.concatenate([[0], cuts, [len(pen)]])
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a > 1]


def render(lines, color, lw, transparent, straighten, spacing, fmt):
    placed = []
    for s in lines:
        pos = np.cumsum(s[:, :2], axis=0)
        if straighten and len(pos) > 2:
            pos[:, 1] -= np.polyfit(pos[:, 0], pos[:, 1], 1)[0] * pos[:, 0]
        pos[:, 0] -= pos[:, 0].min()
        pos[:, 1] -= np.median(pos[:, 1])
        placed.append((pos, s[:, 2]))

    gap = spacing * np.mean([np.ptp(p[:, 1]) for p, _ in placed])
    xmax = max(p[:, 0].max() for p, _ in placed)
    ymin = -gap * (len(placed) - 1) + min(p[:, 1].min() for p, _ in placed)
    ymax = max(p[:, 1].max() for p, _ in placed)
    m = 0.5
    k = 0.12  # inches per stroke unit

    fig = plt.figure(figsize=(max(1, (xmax + 2 * m) * k), max(0.5, (ymax - ymin + 2 * m) * k)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-m, xmax + m)
    ax.set_ylim(ymin - m, ymax + m)
    ax.axis("off")
    for i, (pos, pen) in enumerate(placed):
        for a, b in pen_segments(pen):
            ax.plot(pos[a:b, 0], pos[a:b, 1] - i * gap, color=color, linewidth=lw,
                    solid_capstyle="round", solid_joinstyle="round")
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=220, transparent=transparent)
    plt.close(fig)
    return buf.getvalue()


def attention_figure(att, text, head):
    a = att.mean(axis=0) if head == "mean" else att[int(head)]
    fig, ax = plt.subplots(figsize=(max(4, len(text) * 0.25), 3))
    ax.imshow(a.astype("float32"), aspect="auto", cmap="magma")
    ax.set_xticks(range(len(text) + 2))
    ax.set_xticklabels(list(text) + ["⏎", "⏎"], fontsize=7)
    ax.set_ylabel("stroke position (downsampled)")
    ax.set_xlabel("text token")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("✍️ Diffusion Handwriting Generation")
st.caption("Upload samples of someone's handwriting, type text, and generate it in that style.")

with st.spinner("Loading model (first boot downloads weights)..."):
    try:
        denoise, style_extractor, tokenizer, beta_set = load_diffusion_model()
    except Exception as e:
        st.error(f"Failed to load model or download weights: {e}")
        st.stop()

with st.sidebar:
    st.header("Sampling")
    mode = st.selectbox("Diffusion mode", ["new", "standard"],
                        help="'new' = Eq. 12 of the paper, 'standard' = Eq. 9.")
    steps_per_char = st.slider("Stroke points per character", 12, 24, 16,
                               help="16 is what the model was trained around.")
    wrap = st.slider("Max characters per line", 15, 45, 30,
                     help="The model was trained on short lines (<50 chars); long text is wrapped.")
    n_var = st.slider("Variations", 1, 4, 1, help="Different random samples of the same text/style.")
    randomize = st.checkbox("Random seed every run", value=False)
    seed = st.number_input("Seed", 0, 2**31 - 1, 42, disabled=randomize)

    st.header("Reference image handling")
    clean = st.checkbox("Clean up photo (even out lighting, boost contrast)", value=True)
    split_lines = st.checkbox("Page with several lines → auto-split", value=False,
                              help="Each detected line yields a style sample; they are averaged.")

    st.header("Appearance")
    ink = st.color_picker("Ink colour", "#111111")
    lw = st.slider("Pen thickness", 0.5, 4.0, 1.5, 0.1)
    straighten = st.checkbox("Straighten baseline drift", value=True)
    spacing = st.slider("Line spacing", 0.8, 2.5, 1.3, 0.1)
    transparent = st.checkbox("Transparent background", value=False)

# ---- 1. style images
st.subheader("1 · Reference handwriting")
files = st.file_uploader("Upload one or more handwriting images", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
                         accept_multiple_files=True)
if not files:
    st.info("Upload at least one image of handwriting (dark ink on light paper works best).")
    st.stop()

styles, weights = [], []
cols = st.columns(min(len(files), 4))
for idx, f in enumerate(files):
    try:
        vec, strips = extract_file_style(style_extractor, f.getvalue(), clean, split_lines)
    except Exception as e:
        st.error(f"{f.name}: {e}")
        st.stop()
    with cols[idx % len(cols)]:
        st.image(strips[0][..., 0], caption=f"{f.name} (model's view, {len(strips)} line(s))", clamp=True)
        w = st.slider(f"Weight · {f.name}", 0.0, 1.0, 1.0, 0.05, key=f"w_{idx}") if len(files) > 1 else 1.0
    styles.append(vec)
    weights.append(w)

if sum(weights) == 0:
    st.warning("All style weights are zero.")
    st.stop()
style = sum(w * v for w, v in zip(weights, styles)) / sum(weights)
style = style[None].astype("float32")      # (1, 14, 1280)

# ---- 2. text
st.subheader("2 · Text")
text_in = st.text_area("Text to write (new lines are respected, long lines are wrapped)", "Hello Streamlit!", height=110)
lines = []
for para in text_in.split("\n"):
    if para.strip():
        lines += textwrap.wrap(para.strip(), width=wrap)
bad = sorted({c for c in text_in if c != "\n" and c not in tokenizer.text})
if bad:
    st.warning(f"Unsupported characters will be drawn as '_': {' '.join(bad)}")

# ---- 3. generate
if st.button("Generate Handwriting", type="primary"):
    if not lines:
        st.warning("Please enter some text first.")
    else:
        run_seed = int(np.random.randint(0, 2**31 - 1)) if randomize else int(seed)
        total = len(lines) * len(beta_set)
        bar = st.progress(0.0, text="Denoising...")
        done = [0]

        def tick():
            done[0] += 1
            bar.progress(done[0] / total, text=f"Denoising... {done[0]}/{total} steps")

        strokes = [[None] * len(lines) for _ in range(n_var)]
        atts = []
        try:
            for li, line in enumerate(lines):
                out, att = sample_line(denoise, beta_set, tokenizer, line, style, n_var,
                                       steps_per_char, mode, run_seed + li, tick)
                for v in range(n_var):
                    strokes[v][li] = out[v]
                atts.append(att[0].astype("float16"))
        except Exception as e:
            bar.empty()
            st.error(f"Error during generation: {e}")
            st.stop()
        bar.empty()
        st.session_state["result"] = dict(lines=lines, strokes=strokes, atts=atts, seed=run_seed)

# ---- 4. results (kept in session_state so download clicks don't wipe them)
res = st.session_state.get("result")
if res:
    st.subheader("3 · Result")
    st.caption(f"Seed used: {res['seed']}")
    tabs = st.tabs([f"Sample {i + 1}" for i in range(len(res["strokes"]))])
    for i, tab in enumerate(tabs):
        with tab:
            png = render(res["strokes"][i], ink, lw, transparent, straighten, spacing, "png")
            svg = render(res["strokes"][i], ink, lw, transparent, straighten, spacing, "svg")
            st.image(png)
            c1, c2, c3 = st.columns(3)
            c1.download_button("Download PNG", png, f"handwriting_{i + 1}.png", "image/png", key=f"png{i}")
            c2.download_button("Download SVG", svg, f"handwriting_{i + 1}.svg", "image/svg+xml", key=f"svg{i}")
            buf = io.BytesIO()
            np.savez(buf, **{f"line_{j}": s for j, s in enumerate(res["strokes"][i])})
            c3.download_button("Raw strokes (.npz)", buf.getvalue(), f"strokes_{i + 1}.npz", key=f"npz{i}")

    with st.expander("Text ↔ stroke attention (debug)"):
        li = st.selectbox("Line", range(len(res["lines"])), format_func=lambda j: res["lines"][j])
        n_heads = res["atts"][li].shape[0]
        head = st.selectbox("Head", ["mean"] + list(range(n_heads)))
        fig = attention_figure(res["atts"][li], res["lines"][li], head)
        st.pyplot(fig)
        plt.close(fig)
