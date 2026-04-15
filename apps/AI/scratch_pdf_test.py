from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display

class PDF(FPDF):
    pass

pdf = PDF()
pdf.add_page()
pdf.add_font("Arial", "", r"C:\Windows\Fonts\arial.ttf")
pdf.set_font("Arial", size=14)

text = "مرحبا بكم في التطبيق الخاص بالتنبؤ بأمراض القلب"
reshaped_text = arabic_reshaper.reshape(text)
bidi_text = get_display(reshaped_text)

pdf.cell(200, 10, txt=bidi_text, align='C')
pdf.output("test_arabic.pdf")
print("PDF created successfully!")
