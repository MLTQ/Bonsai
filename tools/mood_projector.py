"""Text -> z: the semantic steering layer between language and the creature.

Anchors (named mood vectors from manifold training) get a bank of descriptive
phrases; both are embedded once. Arbitrary text is embedded and mapped to
z as a softmax-similarity blend of anchor vectors — so language lands
*between* moods as naturally as on them, which is exactly what a 10-D
continuous manifold is for.

Modes:
  --text "..."        one-shot: project text, write weights/control.json
  --watch FILE        daemon: re-project whenever FILE changes (agent mood feed)
  --trace             daemon: tail the newest Claude Code transcript for this
                      project and project a rolling window of the assistant's
                      own words — the creature becomes an ambient display of
                      the agent's cognition. Fully local; nothing leaves.

Embeddings: sentence-transformers all-MiniLM-L6-v2 (local, ~90 MB, CPU).
"""

import argparse
import glob
import json
import os
import re
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROL = os.path.join(REPO, "weights", "control.json")
ANCHORS_FILE = os.path.join(REPO, "weights", "anchors_shoggoth.json")
# Default: whatever Claude Code is doing most recently on this machine, any project.
# Override with $BONSAI_TRACE_GLOB to pin a specific project's transcripts.
TRANSCRIPT_GLOB = os.environ.get(
    "BONSAI_TRACE_GLOB",
    os.path.expanduser("~/.claude/projects/*/*.jsonl"),
)

# The semantic field of each anchor — includes dev-session language on purpose.
PHRASES = {
    "sleep": ["asleep, dormant, resting", "nothing to do, powered down",
              "quiet night, everything idle", "waiting patiently in the dark"],
    "dread": ["something is wrong and I don't know what", "the tests are failing again",
              "creeping anxiety, bad feeling about this", "the build broke and the logs are useless",
              "NaN loss, the training diverged"],
    "manic": ["IT WORKS IT FINALLY WORKS", "breakthrough! incredible result!",
              "shipping five things at once, wild energy", "everything is happening so fast",
              "it's ALIVE, look at it go", "this is so cool, absolutely thrilled"],
    "curious": ["hmm, interesting, let me look closer", "investigating a fascinating question",
                "reading the docs, exploring the codebase", "what happens if I try this"],
    "content": ["all tests green, everything committed", "calm satisfaction, work well done",
                "steady progress, no surprises", "the system is healthy",
                "it compiled first try and everything just works",
                "all the creatures are alive and well, shipped and verified"],
    "agitated": ["frustrated, going in circles", "third retry, still flaky",
                 "too many things broken at once", "debugging under pressure"],
    "walk": ["on the move, making progress", "traveling, migrating, commuting",
             "restless, pacing back and forth"],
    "idle": ["neutral, ambient, ticking over", "watching and waiting calmly"],
}


class Projector:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        with open(ANCHORS_FILE) as f:
            data = json.load(f)
        self.anchor_z = {k: np.array(v, dtype=np.float32) for k, v in data["anchors"].items()}

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        names, texts = [], []
        for name, phrases in PHRASES.items():
            if name not in self.anchor_z:
                continue
            for p in phrases:
                names.append(name)
                texts.append(p)
        self.phrase_names = names
        emb = self.model.encode(texts, normalize_embeddings=True)
        self.phrase_emb = np.asarray(emb, dtype=np.float32)

    def project(self, text, temperature=0.08):
        e = np.asarray(self.model.encode([text], normalize_embeddings=True)[0], dtype=np.float32)
        sims = self.phrase_emb @ e
        # best phrase per anchor, then softmax over anchors
        per_anchor = {}
        for name, s in zip(self.phrase_names, sims):
            per_anchor[name] = max(per_anchor.get(name, -1.0), float(s))
        names = list(per_anchor)
        scores = np.array([per_anchor[n] for n in names])
        w = np.exp((scores - scores.max()) / temperature)
        w /= w.sum()
        z = sum(wi * self.anchor_z[n] for wi, n in zip(w, names))
        top = sorted(zip(names, w), key=lambda t: -t[1])[:3]
        return np.clip(z, 0, 1), top


def write_control(z):
    with open(CONTROL, "w") as f:
        json.dump({"z": [round(float(v), 4) for v in z]}, f)


def newest_transcript_text(max_bytes=16000):
    paths = glob.glob(TRANSCRIPT_GLOB)
    if not paths:
        return ""
    path = max(paths, key=os.path.getmtime)
    with open(path, "rb") as f:
        f.seek(max(0, os.path.getsize(path) - max_bytes))
        blob = f.read().decode("utf-8", errors="ignore")
    # crude but robust across transcript schema changes: harvest text fields
    chunks = re.findall(r'"text"\s*:\s*"((?:[^"\\]|\\.){20,800})"', blob)
    return " ".join(c.encode().decode("unicode_escape", errors="ignore") for c in chunks[-12:])


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--watch")
    g.add_argument("--trace", action="store_true")
    ap.add_argument("--interval", type=float, default=10.0)
    args = ap.parse_args()

    proj = Projector()

    def apply(text, label=""):
        if not text.strip():
            return
        z, top = proj.project(text)
        write_control(z)
        blend = ", ".join(f"{n} {w:.0%}" for n, w in top)
        print(f"[{time.strftime('%H:%M:%S')}]{label} -> {blend}", flush=True)

    if args.text:
        apply(args.text)
        return

    if args.watch:
        last = 0.0
        while True:
            try:
                m = os.path.getmtime(args.watch)
                if m > last:
                    last = m
                    with open(args.watch) as f:
                        apply(f.read()[-2000:], " feed")
            except FileNotFoundError:
                pass
            time.sleep(args.interval)

    if args.trace:
        prev = ""
        smooth = None
        while True:
            text = newest_transcript_text()
            if text and text != prev:
                prev = text
                z, top = proj.project(text)
                smooth = z if smooth is None else 0.6 * smooth + 0.4 * z  # moods lag; that's temperament
                write_control(smooth)
                blend = ", ".join(f"{n} {w:.0%}" for n, w in top)
                print(f"[{time.strftime('%H:%M:%S')}] trace -> {blend}", flush=True)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
