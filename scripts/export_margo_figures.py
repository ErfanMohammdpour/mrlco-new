#!/usr/bin/env python3
"""
Export slide figures from MARGO presentation PDF (or PPTX) into ../figures/

Default mapping (1-based slide/page numbers, as in the MARGO deck):
  8  -> system_model.png
  4  -> margo_architecture.png
  5  -> meta_reptile_training.png
  6  -> graph2seq_encoder.png
  7  -> joint_objective.png
  9  -> triple_readout.png
  10 -> dataset_generation.png
  11 -> performance_metrics.png

Requires: pip install pymupdf
Optional PPTX path: pip install pywin32 (Windows) for COM export if PDF is missing.

Usage:
  python scripts/export_margo_figures.py
  python scripts/export_margo_figures.py --pdf "C:/path/to/MARGO-Meta-learning-for-Task-Offloading_main.pdf"
  python scripts/export_margo_figures.py --pptx "C:/path/to/deck.pptx"  # Windows + PowerPoint
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repo root = parent of scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"

DEFAULT_PDF = Path(
    r"C:\Users\thisi\Downloads\Telegram Desktop\MARGO-Meta-learning-for-Task-Offloading_main.pdf"
)
DEFAULT_PPTX = Path(
    r"C:\Users\thisi\Downloads\Telegram Desktop\MARGO-Meta-learning-for-Task-Offloading_main (4).pptx"
)

# (1-based page index, output filename)
DEFAULT_PAGE_MAP: list[tuple[int, str]] = [
    (8, "system_model.png"),
    (4, "margo_architecture.png"),
    (5, "meta_reptile_training.png"),
    (6, "graph2seq_encoder.png"),
    (7, "joint_objective.png"),
    (9, "triple_readout.png"),
    (10, "dataset_generation.png"),
    (11, "performance_metrics.png"),
]


def export_pdf_pages(pdf_path: Path, page_map: list[tuple[int, str]], dpi: float) -> None:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("Install PyMuPDF: pip install pymupdf", file=sys.stderr)
        sys.exit(1)

    doc = fitz.open(pdf_path)
    n = doc.page_count
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for page_one_based, filename in page_map:
        idx = page_one_based - 1
        if idx < 0 or idx >= n:
            print(f"Skip {filename}: page {page_one_based} out of range (PDF has {n} pages)")
            continue
        pix = doc.load_page(idx).get_pixmap(matrix=mat, alpha=False)
        out_path = FIGURES_DIR / filename
        pix.save(str(out_path))
        print(f"Wrote {out_path} (page {page_one_based} @ {dpi} dpi)")


def export_pptx_windows(pptx_path: Path, slide_map: list[tuple[int, str]]) -> None:
    try:
        import win32com.client  # type: ignore
    except ImportError:
        print("PPTX export needs pywin32: pip install pywin32", file=sys.stderr)
        sys.exit(1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ppt = win32com.client.Dispatch("Powerpoint.Application")
    ppt.Visible = 1
    pres = ppt.Presentations.Open(str(pptx_path), WithWindow=False)
    try:
        for slide_num, filename in slide_map:
            try:
                slide = pres.Slides(slide_num)
            except Exception as e:
                print(f"Skip {filename}: slide {slide_num}: {e}")
                continue
            out_path = FIGURES_DIR / filename
            slide.Export(str(out_path), "PNG")
            print(f"Wrote {out_path} (slide {slide_num})")
    finally:
        pres.Close()
        ppt.Quit()


def main() -> None:
    p = argparse.ArgumentParser(description="Export MARGO deck figures to figures/")
    p.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="Path to presentation PDF")
    p.add_argument("--pptx", type=Path, default=DEFAULT_PPTX, help="Path to presentation PPTX (Windows)")
    p.add_argument("--dpi", type=float, default=300.0, help="Raster DPI for PDF export (default 300)")
    p.add_argument("--use-pptx", action="store_true", help="Force PPTX+PowerPoint export (Windows)")
    args = p.parse_args()

    if args.use_pptx or not args.pdf.is_file():
        if args.pptx.is_file():
            print(f"Exporting from PPTX: {args.pptx}")
            export_pptx_windows(args.pptx, DEFAULT_PAGE_MAP)
            return
        if args.pdf.is_file():
            print(f"PDF found, exporting: {args.pdf}")
            export_pdf_pages(args.pdf, DEFAULT_PAGE_MAP, args.dpi)
            return
        print("Neither PDF nor PPTX found at default paths.", file=sys.stderr)
        print(f"  PDF:  {args.pdf}", file=sys.stderr)
        print(f"  PPTX: {args.pptx}", file=sys.stderr)
        sys.exit(2)

    print(f"Exporting from PDF: {args.pdf}")
    export_pdf_pages(args.pdf, DEFAULT_PAGE_MAP, args.dpi)


if __name__ == "__main__":
    main()
