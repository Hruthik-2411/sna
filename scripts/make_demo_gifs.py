"""
Backup demo: export animated GIFs of the live replay and the race.
Use them if Streamlit or the laptop misbehaves during the presentation.

    python scripts/make_demo_gifs.py        # -> outputs/demo/*.gif
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DEMO_DIR  # noqa: E402
from src.demo_viz import frame, race_frame, radial_layout, replay_times  # noqa: E402

INF = float("inf")


def fig_to_img(fig):
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return img


def save_gif(frames, path, ms=110, hold_last=25):
    frames = frames + [frames[-1]] * hold_last
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
    print(f"  saved {path}")


def main(n_frames=50):
    data = json.loads((DEMO_DIR / "demo_cascades.json").read_text())["cascades"]
    fals = [c for c in data if c["kind"] == "false"]
    tru = [c for c in data if c["kind"] == "true"]

    for name, c in (("replay_false", fals[0] if fals else None), ("replay_true", tru[0] if tru else None)):
        if c is None:
            continue
        pos = radial_layout(c["nodes"])
        tmax = min(max(n["t"] for n in c["nodes"]), 180)
        imgs = [fig_to_img(frame(c, t, pos, figsize=(9, 4.8))[0]) for t in replay_times(tmax, n_frames)]
        imgs.append(fig_to_img(frame(c, INF, pos, reveal=True, figsize=(9, 4.8))[0]))
        save_gif(imgs, DEMO_DIR / f"{name}.gif")

    if fals and tru:
        a, b = fals[1 % len(fals)], tru[1 % len(tru)]
        pa, pb = radial_layout(a["nodes"]), radial_layout(b["nodes"])
        tmax = min(max(max(n["t"] for n in a["nodes"]), max(n["t"] for n in b["nodes"])), 180)
        imgs = [fig_to_img(race_frame(a, b, t, pa, pb, figsize=(10, 4.6)))
                for t in replay_times(tmax, n_frames)]
        imgs.append(fig_to_img(race_frame(a, b, INF, pa, pb, reveal=True, figsize=(10, 4.6))))
        save_gif(imgs, DEMO_DIR / "race.gif")


if __name__ == "__main__":
    main()
