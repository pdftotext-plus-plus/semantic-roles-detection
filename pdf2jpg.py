from PyPDF2
from pdf2image
import io

def pdf_area_to_jpg(pdf_file, jpg_file, page_num, min_x, min_y, max_x, max_y):
    """
    Crops the given page of the given PDF file to the area defined by min_x, min_y, max_x and max_y
    and writes the result to the given JPG file.
    """
    # Read the PDF file.
    reader = PyPDF2.PdfFileReader(pdf_file, "r")
    # Get the respective page.
    page = reader.getPage(page_num)
    # Crop the page to the given area.
    page.cropBox.setLowerLeft((min_x, min_y))
    page.cropBox.setUpperRight((max_x, max_y))

    # Write the result to an in-memory pdf.
    writer = PdfFileWriter()
    writer.addPage(page)

    tmp_pdf = io.BytesIO()
    writer.write(tmp_pdf)
    tmp_pdf.seek(0)

    pdf2image.convert_from_bytes(tmp.read(), use_cropbox=True)
    pages[0].save(output_file, "JPEG")


if __name__ == "__main__":
    pdf_area_to_jpg("/home/korzen/Downloads/2003.02320.pdf", 1, 145, 345, 200, 600, "/home/korzen/Downloads/2003.02320.jpg")
