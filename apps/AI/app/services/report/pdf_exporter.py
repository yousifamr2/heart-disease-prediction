"""
report/pdf_exporter.py
──────────────────────────────────────────────────────────────────────────
PDF Conversion Layer — converts rendered HTML string to PDF bytes.

Strategy (in order):
  1. WeasyPrint  — best quality (needs GTK3 on Windows)
  2. xhtml2pdf   — pure Python, no binaries, works on Windows
  3. pdfkit      — needs wkhtmltopdf binary installed

Single responsibility: HTML string → PDF bytes.
"""

import io


def html_to_pdf(html: str) -> bytes:
    """
    Convert a rendered HTML string to a PDF byte stream.

    Tries backends in order:
      1. WeasyPrint  (preferred — best CSS support)
      2. xhtml2pdf   (pure Python fallback — works everywhere)
      3. pdfkit      (last resort — needs wkhtmltopdf binary)

    Parameters
    ----------
    html : str
        Fully rendered HTML string from renderer.render_report().

    Returns
    -------
    bytes : Raw PDF content.

    Raises
    ------
    RuntimeError : If all backends fail.
    """
    errors = {}

    # ── 1. WeasyPrint ─────────────────────────────────────────────────
    try:
        from weasyprint import HTML
        return HTML(string=html).write_pdf()
    except Exception as e:
        errors["weasyprint"] = str(e)
        print(f"[pdf_exporter] WeasyPrint unavailable: {e}")

    # ── 2. xhtml2pdf (pure Python — works on Windows without binaries) ─
    try:
        from xhtml2pdf import pisa
        from pathlib import Path as _Path
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        # Register Amiri Arabic font if available
        _font_dir = _Path(__file__).resolve().parent.parent.parent.parent / "app" / "assets" / "fonts"
        _regular = _font_dir / "Amiri-Regular.ttf"
        _bold    = _font_dir / "Amiri-Bold.ttf"
        if _regular.exists():
            pdfmetrics.registerFont(TTFont("Amiri", str(_regular)))
        if _bold.exists():
            pdfmetrics.registerFont(TTFont("Amiri-Bold", str(_bold)))

        buf = io.BytesIO()
        result = pisa.CreatePDF(
            src=html.encode("utf-8"),
            dest=buf,
            encoding="utf-8",
        )
        if not result.err:
            buf.seek(0)
            return buf.read()
        else:
            errors["xhtml2pdf"] = f"pisa reported errors: {result.err}"
            print(f"[pdf_exporter] xhtml2pdf errors: {result.err}")
    except Exception as e:
        errors["xhtml2pdf"] = str(e)
        print(f"[pdf_exporter] xhtml2pdf failed: {e}")


    # ── 3. pdfkit (needs wkhtmltopdf binary) ──────────────────────────
    try:
        import pdfkit
        options = {
            "page-size":    "A4",
            "encoding":     "UTF-8",
            "margin-top":   "15mm",
            "margin-right": "15mm",
            "margin-bottom":"15mm",
            "margin-left":  "15mm",
        }
        return pdfkit.from_string(html, False, options=options)
    except Exception as e:
        errors["pdfkit"] = str(e)
        print(f"[pdf_exporter] pdfkit failed: {e}")

    raise RuntimeError(
        "All PDF backends failed.\n" +
        "\n".join(f"  {k}: {v}" for k, v in errors.items()) +
        "\n\nInstall one of:\n"
        "  - xhtml2pdf  : pip install xhtml2pdf          (pure Python)\n"
        "  - WeasyPrint : pip install weasyprint + GTK3  (Windows: needs runtime)\n"
        "  - pdfkit     : pip install pdfkit + wkhtmltopdf binary"
    )
