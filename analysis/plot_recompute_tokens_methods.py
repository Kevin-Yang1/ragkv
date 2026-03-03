#!/usr/bin/env python3
"""Plot per-method recomputed token distributions for one dataset item.

Example:
python analysis/plot_recompute_tokens_methods.py \
  --model Llama-3-8B-Instruct \
  --dataset 2wikimqa \
  --item_id 1
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


ITEM_ID_RE = re.compile(r"^item_id:\s*(\d+)\s*$")
CHUNK_OBJ_RE = re.compile(r"^\{.*\}\s*,?\s*$")

COLOR_RECOMPUTE = np.array([139, 26, 26], dtype=np.uint8)  # dark red
COLOR_NORMAL = np.array([253, 236, 234], dtype=np.uint8)  # light pink


def parse_recomputed_chunks(path: Path) -> Dict[int, Optional[List[dict]]]:
    """Parse txt file into {item_id: chunks_or_none}."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    result: Dict[int, Optional[List[dict]]] = {}
    current_item_id: Optional[int] = None
    current_chunks: List[dict] = []
    saw_null = False

    def flush() -> None:
        nonlocal current_item_id, current_chunks, saw_null
        if current_item_id is None:
            return
        if saw_null and not current_chunks:
            result[current_item_id] = None
        else:
            result[current_item_id] = current_chunks
        current_item_id = None
        current_chunks = []
        saw_null = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        m = ITEM_ID_RE.match(line)
        if m:
            flush()
            current_item_id = int(m.group(1))
            continue

        if current_item_id is None:
            continue

        if line == "null":
            saw_null = True
            continue

        if CHUNK_OBJ_RE.match(line):
            try:
                obj = json.loads(line.rstrip(","))
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            chunk_len = int(obj.get("len", 0))
            indices = [int(x) for x in obj.get("indices", [])]
            current_chunks.append({"len": chunk_len, "indices": indices})

    flush()
    return result


def resolve_recomputed_file(method_dir: Path, dataset: str) -> Optional[Path]:
    direct = method_dir / dataset / "recomputed_chunks.txt"
    if direct.is_file():
        return direct

    matches = sorted(method_dir.glob(f"**/{dataset}/recomputed_chunks.txt"))
    if matches:
        return matches[0]
    return None


def build_token_bar(chunks: List[dict]) -> tuple[np.ndarray, List[int], int, int]:
    total_len = sum(int(c["len"]) for c in chunks)
    img = np.zeros((1, total_len, 3), dtype=np.uint8)
    img[0, :] = COLOR_NORMAL

    chunk_starts: List[int] = []
    recompute_count = 0
    offset = 0
    for c in chunks:
        c_len = int(c["len"])
        chunk_starts.append(offset)
        for idx in c["indices"]:
            if 0 <= idx < c_len:
                img[0, offset + idx] = COLOR_RECOMPUTE
                recompute_count += 1
        offset += c_len

    return img, chunk_starts, total_len, recompute_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare recomputed token positions across methods for one item."
    )
    parser.add_argument("--model", "--model_name", dest="model", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--item_id", required=True, type=int)
    parser.add_argument("--outputs_root", default="outputs", type=str)
    parser.add_argument("--methods", nargs="*", default=None, help="optional method dir names")
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--dpi", default=220, type=int)
    parser.add_argument("--hide_chunk_boundaries", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outputs_root = Path(args.outputs_root)
    model_dir = outputs_root / args.model
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")

    if args.methods:
        method_dirs = []
        for method in args.methods:
            p = model_dir / method
            if p.is_dir():
                method_dirs.append(p)
            else:
                print(f"[skip] method dir not found: {p}")
    else:
        method_dirs = sorted([p for p in model_dir.iterdir() if p.is_dir()], key=lambda x: x.name)

    if not method_dirs:
        raise RuntimeError("no method directories to inspect")

    traces: List[dict] = []
    for method_dir in method_dirs:
        recompute_path = resolve_recomputed_file(method_dir, args.dataset)
        if recompute_path is None:
            print(f"[skip] {method_dir.name}: no recomputed_chunks.txt for dataset '{args.dataset}'")
            continue

        items = parse_recomputed_chunks(recompute_path)
        if args.item_id not in items:
            print(f"[skip] {method_dir.name}: item_id {args.item_id} not found")
            continue

        chunks = items[args.item_id]
        traces.append(
            {
                "method": method_dir.name,
                "path": recompute_path,
                "chunks": chunks,
                "imputed_from_null": False,
            }
        )

    if not traces:
        raise RuntimeError("no valid method data found for the given model/dataset/item_id")

    reference_chunks = next((t["chunks"] for t in traces if t["chunks"] is not None), None)
    if reference_chunks is None:
        raise RuntimeError("all selected methods have null chunks; cannot infer token length")

    for t in traces:
        if t["chunks"] is None:
            t["chunks"] = [{"len": int(c["len"]), "indices": []} for c in reference_chunks]
            t["imputed_from_null"] = True

    for t in traces:
        img, chunk_starts, total_len, recompute_count = build_token_bar(t["chunks"])
        t["img"] = img
        t["chunk_starts"] = chunk_starts
        t["total_len"] = total_len
        t["recompute_count"] = recompute_count

    total_lens = {t["total_len"] for t in traces}
    share_x = len(total_lens) == 1

    fig_w = 14
    fig_h = max(2.8, 0.92 * len(traces) + 1.2)
    fig, axes = plt.subplots(len(traces), 1, figsize=(fig_w, fig_h), sharex=share_x)
    if len(traces) == 1:
        axes = [axes]

    for i, (ax, t) in enumerate(zip(axes, traces)):
        total_len = t["total_len"]
        recompute_count = t["recompute_count"]
        ratio = recompute_count / total_len if total_len > 0 else 0.0

        ax.imshow(
            t["img"],
            aspect="auto",
            interpolation="nearest",
            extent=[0, total_len, 0, 1],
        )

        if not args.hide_chunk_boundaries:
            for cs in t["chunk_starts"][1:]:
                ax.plot(
                    [cs, cs],
                    [1.0, 1.07],
                    color="#666666",
                    linewidth=0.6,
                    clip_on=False,
                    transform=ax.get_xaxis_transform(),
                )

        suffix = " | null -> empty" if t["imputed_from_null"] else ""
        ax.set_title(
            f"{t['method']}  recomputed={recompute_count}/{total_len} ({ratio:.2%}){suffix}",
            loc="left",
            fontsize=9,
        )
        ax.set_ylim(0, 1)
        ax.set_xlim(0, total_len)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=8)
        if i < len(traces) - 1:
            ax.tick_params(axis="x", labelbottom=False)

    token_label = (
        f"Token Position (total {next(iter(total_lens))} tokens)"
        if share_x
        else "Token Position"
    )
    axes[-1].set_xlabel(token_label, fontsize=9)

    fig.suptitle(
        f"Recomputed Token Distribution | model={args.model} dataset={args.dataset} item_id={args.item_id}",
        fontsize=11,
        y=0.995,
    )

    patch_r = mpatches.Patch(color=tuple(COLOR_RECOMPUTE / 255), label="Recomputed")
    patch_n = mpatches.Patch(
        facecolor=tuple(COLOR_NORMAL / 255),
        label="Normal",
        edgecolor="#cccccc",
        linewidth=0.6,
    )
    fig.legend(handles=[patch_r, patch_n], loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if args.save_path:
        out_path = Path(args.save_path)
    else:
        out_path = model_dir / f"recompute_tokens_{args.dataset}_item{args.item_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {out_path}")
    print("[methods]")
    for t in traces:
        mark = " (null->empty)" if t["imputed_from_null"] else ""
        print(f"- {t['method']}{mark}: {t['path']}")


if __name__ == "__main__":
    main()
