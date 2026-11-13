import io, os, textwrap
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

st.set_page_config(page_title="Diffusion Handwriting", page_icon="✍️", layout="wide")

STYLE_H = 96
STYLE_W = 1400                                # 1400/32 -> 43 -> pool 3x3 -> 14 tokens
WEIGHTS_ID = "1U1h_sfTx2F51vmBGnELfZvKenospyITo"
EXAMPLES = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "To be, or not to be, that is the question.",
]

# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner=False)
def load_bundle():
    wpath = "./weights/model_weights.h5"
    os.makedirs("./weights", exist_ok=True)
    if not os.path.exists(wpath):
        gdown.download(f"https://drive.google.com/uc?id={WEIGHTS_ID}", wpath, quiet=False)

    c1 = 128
    c2, c3 = c1 * 3 // 2, c1 * 2

    ext = nn.StyleExtractor()
    ext.mobilenet = MobileNetV2(include_top=False, pooling=None, weights="imagenet",
                                input_shape=(STYLE_H, STYLE_W, 3))
    for l in ext.mobilenet.layers:
        l.trainable = False

    m = nn.DiffusionWriter(num_layers=2, c1=c1, c2=c2, c3=c3)
    m(tf.random.normal([1, 400, 2]),
      tf.random.uniform([1, 40], dtype=tf.int32, maxval=50),
      tf.random.uniform([1, 1]),
      tf.random.normal([1, 14, 1280]))
    m.load_weights(wpath)

    @tf.function(reduce_retracing=True)
    def denoise(x, text, sigma, style):
        return m(x, text, sigma, style)

    return dict(denoise=denoise, model=m, ext=ext,
                tok=utils.Tokenizer(), beta=utils.get_beta_set())


# ---------------------------------------------------------------- style
def to_gray(data):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", img.size, (255,) * 4), img)
    return np.array(img.convert("L"), dtype=np.uint8)


def clean_photo(gray):
    img = Image.fromarray(gray)
    s = 400 / max(img.size)
    small = img.resize((max(1, int(img.width * s)),
                        max(1, int(img.height * s)))) if s < 1 else img
    bg = (small.filter(ImageFilter.MaxFilter(9))
              .filter(ImageFilter.GaussianBlur(5))
              .resize(img.size, Image.BILINEAR))
    flat = np.clip(gray.astype(np.float32) /
                   np.maximum(np.asarray(bg, np.float32), 1), 0, 1) * 255
    lo = np.percentile(flat, 1)
    if lo < 200:
        flat = np.clip((flat - lo) / (230 - lo) * 255, 0, 255)
    return flat.astype(np.uint8)


def split_lines(gray):
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
    return [gray[max(0, s - pad): min(gray.shape[0], e + pad)]
            for s, e in merged if e - s >= 0.35 * med] or [gray]


def prepare_line(gray):
    arr = gray[..., None]
    if not (arr < 127).any():
        raise ValueError("no ink")
    arr = preprocessing.remove_whitespace(arr, thresh=127)
    h, w, _ = arr.shape
    if h < 4 or w < 4:
        raise ValueError("ink region too small")
    arr = tf.image.resize(arr, (STYLE_H, max(8, STYLE_H * w // h)))
    return arr.numpy().astype("uint8")


def strip_to_windows(strip):
    W = strip.shape[1]
    if W <= STYLE_W:
        return [utils.pad_img(strip, STYLE_W, STYLE_H)]
    out = []
    for start in range(0, W, STYLE_W):
        w = strip[:, start:start + STYLE_W]
        if start > 0 and w.shape[1] < 300:
            continue
        out.append(utils.pad_img(w, STYLE_W, STYLE_H))
    return out


@st.cache_data(show_spinner=False)
def extract_style(_ext, data, clean, do_split, agg):
    gray = to_gray(data)
    if clean: gray = clean_photo(gray)
    pieces = split_lines(gray) if do_split else [gray]

    strips = []
    for p in pieces:
        try: strips.append(prepare_line(p))
        except ValueError: continue
    if not strips:
        raise ValueError("No handwriting detected")

    windows = []
    for s in strips: windows += strip_to_windows(s)
    batch = np.stack(windows).astype("float32")
    vecs = [_ext(batch[i:i + 8]).numpy() for i in range(0, len(batch), 8)]
    vecs = np.concatenate(vecs)
    vec = {"mean":   vecs.mean(0),
           "median": np.median(vecs, 0),
           "first":  vecs[0]}[agg]
    return vec.astype("float32"), strips


# ---------------------------------------------------------------- sampling
def sample(denoise, beta_set, tok, text, style, n, spc, mode, seed, on_step=None):
    tokens = tok.encode(text) + [1]                       # same as run_batch_inference
    text_t = tf.constant([tokens] * n, dtype=tf.int32)
    style_t = tf.repeat(style, n, axis=0)

    T = len(text) * spc
    T = T - (T % 8) + 8                                   # original inference formula

    tf.random.set_seed(seed)
    L = len(beta_set)
    alpha_set = tf.math.cumprod(1 - beta_set)
    x = tf.random.normal([n, T, 2])

    for i in range(L - 1, -1, -1):
        alpha = alpha_set[i] * tf.ones([n, 1, 1])
        beta = beta_set[i] * tf.ones([n, 1, 1])
        # match run_batch_inference exactly: at i==1 use 1.0 (deterministic x_0)
        a_next = alpha_set[i - 1] if i > 1 else 1.0
        eps, pen, att = denoise(x, text_t, tf.sqrt(alpha), style_t)

        if mode == "standard":
            x = utils.standard_diffusion_step(x, eps, beta, alpha, add_sigma=bool(i))
        else:
            x = utils.new_diffusion_step(x, eps, beta, alpha, a_next)

        if on_step: on_step()

    return tf.concat([x, pen], axis=-1).numpy(), att.numpy()


# ---------------------------------------------------------------- rendering
def pen_segments(pen):
    cuts = np.where(pen > 0.5)[0]
    b = np.concatenate([[0], cuts, [len(pen)]])
    return [(a, c) for a, c in zip(b[:-1], b[1:]) if c - a > 1]


def render(lines, ink="#111111", lw=1.5, straighten=True, spacing=1.3,
           transparent=False, fmt="png", dpi=220, pad=0.5):
    placed = []
    for s in lines:
        pos = np.cumsum(s[:, :2], axis=0)
        if straighten and len(pos) > 2:
            slope = np.polyfit(pos[:, 0], pos[:, 1], 1)[0]
            pos[:, 1] -= slope * pos[:, 0]
        pos[:, 0] -= pos[:, 0].min()
        pos[:, 1] -= np.median(pos[:, 1])
        placed.append((pos, s[:, 2]))
    if not placed:
        raise ValueError("Nothing to render")

    gap = spacing * np.mean([np.ptp(p[:, 1]) for p, _ in placed])
    xmax = max(p[:, 0].max() for p, _ in placed)
    ymin = -gap * (len(placed) - 1) + min(p[:, 1].min() for p, _ in placed)
    ymax = max(p[:, 1].max() for p, _ in placed)

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


def attn_fig(att, text, head="mean"):
    a = att.mean(0) if head == "mean" else att[int(head)]
    fig, ax = plt.subplots(figsize=(max(4, len(text) * 0.25), 3))
    ax.imshow(a.astype("float32"), aspect="auto", cmap="magma")
    ax.set_xticks(range(len(text) + 2))
    ax.set_xticklabels(list(text) + ["⏎", "⏎"], fontsize=7)
    ax.set_ylabel("stroke pos (1/8)")
    ax.set_xlabel("text token")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- UI
st.title("✍️ Diffusion Handwriting")
st.caption("Type text, upload a handwriting sample, generate that style.")

with st.spinner("Loading model…"):
    try:
        B = load_bundle()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

with st.sidebar:
    st.header("Sampling")
    mode = st.selectbox("Diffusion mode", ["new", "standard"],
                        help="'new' = Eq. 12 of the paper (usually better)")
    spc = st.slider("Stroke points per character", 12, 24, 16)
    wrap = st.slider("Max chars per line", 15, 45, 30)
    n_var = st.slider("Variations", 1, 4, 1)
    randomize = st.checkbox("Random seed each run", value=False)
    seed = st.number_input("Seed", 0, 2**31 - 1, 42, disabled=randomize)

    st.header("Style extraction")
    clean = st.checkbox("Clean photo (lighting)", value=True)
    do_split = st.checkbox("Auto-split multi-line image", value=False)
    agg = st.radio("Aggregation", ["mean", "median", "first"], index=1,
                   help="How to combine style vectors. 'first' matches the "
                        "reference inference (one line).")

    st.header("Appearance")
    ink = st.color_picker("Ink colour", "#111111")
    lw = st.slider("Pen thickness", 0.5, 4.0, 1.5, 0.1)
    straighten = st.checkbox("Straighten baseline", value=True)
    spacing = st.slider("Line spacing", 0.8, 2.5, 1.3, 0.1)
    transparent = st.checkbox("Transparent background", value=False)

    st.header("Visualisations")
    do_att = st.checkbox("Show attention map", value=True)

# ---- 1
st.subheader("1 · Reference handwriting")
files = st.file_uploader("Upload handwriting image(s)",
                         type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
                         accept_multiple_files=True)
if not files:
    st.info("Upload a handwriting image to continue.")
    st.stop()

styles, weights = [], []
cols = st.columns(min(len(files), 4))
for idx, f in enumerate(files):
    try:
        vec, strips = extract_style(B["ext"], f.getvalue(), clean, do_split, agg)
    except Exception as e:
        st.error(f"{f.name}: {e}")
        st.stop()
    with cols[idx % len(cols)]:
        st.image(strips[0][..., 0],
                 caption=f"{f.name} — {len(strips)} line(s)",
                 clamp=True)
        w = (st.slider(f"Weight · {f.name}", 0.0, 1.0, 1.0, 0.05, key=f"w{idx}")
             if len(files) > 1 else 1.0)
    styles.append(vec)
    weights.append(w)

if sum(weights) == 0:
    st.warning("All style weights are zero.")
    st.stop()
style = (sum(w * v for w, v in zip(weights, styles)) / sum(weights))[None].astype("float32")

# ---- 2
st.subheader("2 · Text")
example = st.selectbox("Load an example (optional)", [""] + EXAMPLES)
text_in = st.text_area("Text to write", example or "Hello, world!", height=100)

lines = []
for para in text_in.split("\n"):
    if para.strip():
        lines += textwrap.wrap(para.strip(), width=wrap)

bad = sorted({c for c in text_in if c != "\n" and c not in B["tok"].text})
if bad:
    st.warning(f"Unsupported characters → '_': {' '.join(bad)}")

# ---- 3
st.subheader("3 · Generate")
if st.button("Generate handwriting", type="primary"):
    if not lines:
        st.warning("Enter some text.")
        st.stop()

    run_seed = int(np.random.randint(0, 2**31 - 1)) if randomize else int(seed)
    total = len(lines) * len(B["beta"])
    done = [0]
    bar = st.progress(0.0, text="Denoising…")

    def tick():
        done[0] += 1
        if done[0] % 3 == 0 or done[0] == total:
            bar.progress(min(1.0, done[0] / total),
                         text=f"Denoising… {done[0]}/{total}")

    strokes = [[None] * len(lines) for _ in range(n_var)]
    atts = []
    try:
        for li, line in enumerate(lines):
            out, att = sample(B["denoise"], B["beta"], B["tok"], line, style, n_var,
                              spc, mode, run_seed + li, tick)
            for v in range(n_var):
                strokes[v][li] = out[v]
            atts.append(att[0].astype("float16"))
    except Exception as e:
        bar.empty()
        st.error(f"Error: {e}")
        st.stop()
    bar.empty()
    st.session_state["result"] = dict(lines=lines, strokes=strokes, atts=atts, seed=run_seed)

# ---- 4
res = st.session_state.get("result")
if res:
    st.subheader("4 · Result")
    st.caption(f"Seed: {res['seed']}")
    tabs = st.tabs([f"Sample {i + 1}" for i in range(len(res["strokes"]))])
    for i, tab in enumerate(tabs):
        with tab:
            png = render(res["strokes"][i], ink, lw, straighten, spacing, transparent, "png")
            svg = render(res["strokes"][i], ink, lw, straighten, spacing, transparent, "svg")
            st.image(png)
            c1, c2, c3 = st.columns(3)
            c1.download_button("PNG", png, f"hw_{i+1}.png", "image/png", key=f"p{i}")
            c2.download_button("SVG", svg, f"hw_{i+1}.svg", "image/svg+xml", key=f"s{i}")
            buf = io.BytesIO()
            np.savez(buf, **{f"line_{j}": s for j, s in enumerate(res["strokes"][i])})
            c3.download_button("Raw NPZ", buf.getvalue(), f"hw_{i+1}.npz", key=f"z{i}")

    if do_att and res["atts"]:
        with st.expander("Attention (text ↔ stroke)"):
            li = st.selectbox("Line", range(len(res["lines"])),
                              format_func=lambda j: res["lines"][j], key="al")
            att = res["atts"][li]
            head = st.selectbox("Head", ["mean"] + list(range(att.shape[0])), key="ah")
            fig = attn_fig(att, res["lines"][li], head)
            st.pyplot(fig)
            plt.close(fig)
