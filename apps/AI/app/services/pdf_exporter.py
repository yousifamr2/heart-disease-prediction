"""
report/pdf_exporter.py
──────────────────────────────────────────────────────────────────────────
PDF Conversion Layer — converts rendered HTML string to PDF bytes.

Strategy (in order):
  1. WeasyPrint  — best quality (needs libcairo2 + libgobject via apt)
  2. Playwright  — Chromium headless (playwright install chromium in build)
  3. fpdf2       — Pure Python fallback, ZERO system dependencies ✅
  4. pdfkit      — needs wkhtmltopdf binary installed

Single responsibility: HTML string → PDF bytes.
"""

import io
import os
import re


def html_to_pdf(html: str) -> bytes:
    """
    Convert a rendered HTML string to a PDF byte stream.

    Tries backends in order:
      1. WeasyPrint (Linux: libcairo2 + libgobject via apt)
      2. Playwright (Chromium headless — playwright install chromium)

    Parameters
    ----------
    html : str  Fully rendered HTML string.
    Returns     bytes : Raw PDF content.
    Raises      RuntimeError : If all backends fail.
    """
    errors = {}

    # ── 1. WeasyPrint ─────────────────────────────────────────────────
    try:
        from weasyprint import HTML
        return HTML(string=html).write_pdf()
    except Exception as e:
        errors["weasyprint"] = str(e)
        print(f"[pdf_exporter] WeasyPrint unavailable: {e}")

    # ── 2. Playwright ─────────────────────────────────────────────────
    try:
        from playwright.sync_api import sync_playwright
        timeout_ms = int(os.getenv("PDF_PLAYWRIGHT_TIMEOUT_MS", "120000"))
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content(html, wait_until="networkidle", timeout=timeout_ms)
            pdf_bytes = page.pdf(
                format="A4",
                margin={"top": "20px", "right": "20px", "bottom": "20px", "left": "20px"},
            )
            browser.close()
        return pdf_bytes
    except Exception as e:
        errors["playwright"] = str(e)
        print(f"[pdf_exporter] Playwright unavailable: {e}")

    # ── 3. xhtml2pdf ──────────────────────────────────────────────────
    try:
        from xhtml2pdf import pisa
        print("[pdf_exporter] Trying xhtml2pdf fallback...")
        pdf_io = io.BytesIO()
        pisa_status = pisa.CreatePDF(html, dest=pdf_io)
        if not pisa_status.err:
            return pdf_io.getvalue()
        else:
            errors["xhtml2pdf"] = f"xhtml2pdf error code: {pisa_status.err}"
            print(f"[pdf_exporter] xhtml2pdf failed: {pisa_status.err}")
    except Exception as e:
        errors["xhtml2pdf"] = str(e)
        print(f"[pdf_exporter] xhtml2pdf unavailable: {e}")

    raise RuntimeError(
        "All PDF backends failed (WeasyPrint, Playwright, and xhtml2pdf).\n" +
        "\n".join(f"  {k}: {v}" for k, v in errors.items())
    )
