"""Reference clips: what each game in the study actually looks like.

One short GIF per Procgen game used anywhere in README (coinrun, starpilot, dodgeball,
bossfight, jumper, plunder, maze, heist), driven by a UNIFORM RANDOM policy.

Why random and not a trained agent: these clips exist to show the games, and a trained-agent
clip would currently be dishonest for the exploration suite -- README 3.6.2 measures that the
SB3 maze/heist policies emit the same action for five different games, so their "gameplay" is
an artifact of that defect, not a demonstration of a policy. The label is burned into every
frame for the same reason: a clip that only says "coinrun" in the filename is a claim nobody
can check by watching it.

Each frame carries: game, distribution mode, the level seed actually rendered, the policy
label, the step index and the reward collected so far.

Usage:
    py -3.10 record_game_clips.py                      # all eight games
    py -3.10 record_game_clips.py --games coinrun      # one, for a look
    py -3.10 record_game_clips.py --frames 40 --preview-dir /tmp/clip_preview
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3.common.monitor import Monitor

from procgen_wrapper import make_procgen_env

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DEFAULT = os.path.join(BASE, "results", "games")
# game -> (distribution_mode, steps, fps). The modes and levels are the ones the study runs.
GAMES = {
    "coinrun":   ("easy", 90, 15),
    "starpilot": ("easy", 90, 15),
    "dodgeball": ("easy", 90, 15),
    "bossfight": ("easy", 90, 15),
    "jumper":    ("easy", 90, 15),
    "plunder":   ("easy", 90, 15),
    "maze":      ("easy", 90, 12),
    "heist":     ("easy", 90, 12),
}
SCALE = 4                    # 64x64 -> 256x256
LINE_H = 11                  # px per label line, for the default PIL bitmap font


def wrap(draw, text, width):
    """Split a label so nothing is silently cut off at the canvas edge."""
    words, out, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial) <= width or not cur:
            cur = trial
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


def render(frame_chw, lines):
    """Upscale one CHW uint8 observation and stamp the label band under it."""
    img = Image.fromarray(np.transpose(frame_chw, (1, 2, 0))).resize(
        (64 * SCALE, 64 * SCALE), Image.NEAREST).convert("RGB")
    probe = ImageDraw.Draw(img)
    wrapped = []
    for line in lines:
        wrapped.extend(wrap(probe, line, img.width - 12))
    band = 6 + LINE_H * len(wrapped)
    canvas = Image.new("RGB", (img.width, img.height + band), (12, 15, 24))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    for i, text in enumerate(wrapped):
        d.text((6, img.height + 3 + i * LINE_H), text, fill=(235, 235, 235))
    return canvas


def record(game, mode, n_frames, fps, seed, out_dir, preview_dir=None, num_levels=200):
    """Fill n_frames across episode boundaries: a level the random policy dies on instantly
    must not turn into a one-frame clip, and respawns are part of what the game looks like."""
    env = Monitor(make_procgen_env(game, num_levels=num_levels, distribution_mode=mode,
                                   seed=seed, vector=False))
    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    rng = np.random.default_rng(seed)
    frames, level, episodes, ep_reward = [], 0, 1, 0.0
    nonzero = 0
    for step in range(n_frames):
        lines = [f"{game}  ·  {mode}, {num_levels} train levels  ·  level seed {level}",
                 f"policy: uniform random (reference clip, NOT a trained agent)  ·  "
                 f"frame {step + 1}/{n_frames}  ·  episode {episodes}",
                 f"reward this episode: {ep_reward:+.2f}"]
        frames.append(render(obs, lines))
        obs, r, term, trunc, info = env.step(int(rng.integers(0, 15)))
        obs = obs[0] if isinstance(obs, tuple) else obs
        ep_reward += float(r)
        nonzero += 1 if r else 0
        if isinstance(info, dict) and info.get("level_seed") is not None:
            level = int(info["level_seed"])
        if term or trunc:
            episodes += 1
            ep_reward = 0.0
            obs = env.reset()
            obs = obs[0] if isinstance(obs, tuple) else obs

    distinct = len({f.tobytes() for f in [np.asarray(x) for x in frames]}) if frames else 0
    distinct = len({f.tobytes() for f in frames})
    if distinct < 5:
        raise RuntimeError(f"{game}: only {distinct} distinct frames rendered — this clip "
                           f"would not show the game, refusing to write it")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{game}.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps),
                   loop=0, optimize=True)
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)
        frames[0].save(os.path.join(preview_dir, f"{game}_first.png"))
        frames[len(frames) // 2].save(os.path.join(preview_dir, f"{game}_middle.png"))
        frames[-1].save(os.path.join(preview_dir, f"{game}_last.png"))
    return {"game": game, "mode": mode, "num_levels": num_levels, "frames": len(frames),
            "distinct_frames": distinct, "fps": fps, "seconds": round(len(frames) / fps, 2),
            "episodes_shown": episodes, "bytes": os.path.getsize(path),
            "reward_events": nonzero,
            "path": os.path.relpath(path, BASE).replace(os.sep, "/")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=list(GAMES))
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--preview-dir", default=None)
    args = ap.parse_args()

    rows = []
    for g in args.games:
        mode, n, fps = GAMES[g]
        rows.append(record(g, mode, args.frames or n, fps, args.seed, args.out, args.preview_dir))
        print(f"  {g:10s} {rows[-1]['frames']:3d} frames ({rows[-1]['distinct_frames']:3d} distintas)  "
              f"{rows[-1]['seconds']:4.1f}s  {rows[-1]['bytes'] / 1024:6.0f} KiB  "
              f"{rows[-1]['episodes_shown']} episódios  {rows[-1]['reward_events']} eventos de recompensa")
    with open(os.path.join(args.out, "clips_provenance.json"), "w", encoding="utf-8") as f:
        json.dump({"_provenance": {
            "produced_by": "record_game_clips.py",
            "policy": "uniform random over the 15 Procgen actions",
            "purpose": "show what each game looks like; not a result",
            "num_levels": 200, "seed": args.seed,
            "why_not_a_trained_agent": "README 3.6.2: the SB3 maze/heist policies emit the same "
                                       "action across five different games, so their gameplay "
                                       "would demonstrate the defect rather than the game",
        }, "clips": rows}, f, indent=2)
    print(f"total: {sum(r['bytes'] for r in rows) / 1e6:.2f} MB em {len(rows)} clipes")


if __name__ == "__main__":
    main()
