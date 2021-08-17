import fitz

def to_cropped_png(pdf_file, png_file, page_num, crop_box, highlight_box=None):
  """
  Crops the given page of the given PDF file to the given crop_box (defined by a list or tuple of 4
  numbers minX, minY, maxX, maxY), generates a PNG image from the resulting page and writes the
  image to the given png file path. Set highlight_box to a list or tuple of four numbers minX,
  minY, maxX, maxY (relative to the original page size, within the specified crop_box) to draw a
  respective rectangle with a semi-transparent filling color into the page (this can be used to
  highlight a specific part within the page)
  """

  # Open the PDF file.
  pdf = fitz.open(pdf_file)

  # Shrink the PDF to the given page and select the page.
  pdf.select([page_num])
  page = pdf.load_page(0)

  if highlight_box is not None:
      # Draw the highlight box into the PDF.
      shape = page.new_shape()
      # Fit the given highlight box to the page boundaries. Translate the y values so that they
      # are relative to the page's *upper* left.
      shape.draw_rect(fitz.Rect(
          max(0, highlight_box[0]),
          max(0, page.rect.height - highlight_box[3]),
          min(page.rect.width, highlight_box[2]),
          min(page.rect.height, page.rect.height - highlight_box[1])
      ))
      shape.finish(fill=(1, 0, 0), fill_opacity=0.5)
      shape.commit()

  # Crop the page to the given crop box. Fit the crop box to the page boundaries. Translate the y
  # values so that they are relative to the page's *upper* left.
  page.set_cropbox(fitz.Rect(
       max(0, crop_box[0]),
       max(0, page.rect.height - crop_box[3]),
       min(page.rect.width, crop_box[2]),
       min(page.rect.height, page.rect.height - crop_box[1])
  ))

  # Generate a PNG from the page.
  pic = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha = False)
  pic.writePNG(png_file)


if __name__ == "__main__":
    to_cropped_png(
        pdf_file="/local/data/korzen/datasets/semantic-roles-detection/svjour3.one-column.cStxFTFd.pdf",
        png_file="x.png",
        page_num=0,
        crop_box=[22.0,657.7,133.0,766.8],
        highlight_box=[72.0,707.7,83.0,716.8]
    )