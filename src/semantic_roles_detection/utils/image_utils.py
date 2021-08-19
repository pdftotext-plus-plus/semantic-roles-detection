"""
This file is part of the "semantic-roles-detection" module of PdfActML. It contains code to create
images from specific areas (e.g., text blocks) in PDF documents.

Copyright 2021, University of Freiburg.

Claudius Korzen <korzen@cs.uni-freiburg.de>
"""

from typing import Tuple

import fitz

# The border color for the rectangles that visualize the highlight areas.
HIGHLIGHT_AREA_STROKE_COLOR = (.43, .43, .07)
# The filling color for the rectangles that visualize the highlight areas.
HIGHLIGHT_AREA_FILLING_COLOR = (.93, .93, .67)
# The filling opacity for the rectangles that visualize the highlight areas.
HIGHLIGHT_AREA_FILL_OPACITY: float = 0.3

# ==================================================================================================


def create_png(pdf_file_path: str, png_file_path: str, page_num: int,
               area: Tuple[float, float, float, float],
               highlight_area: Tuple[float, float, float, float] = None):
    """
    This method creates a PNG image from the given area on the given page of the given PDF file
    and writes it to the given target path. If `highlight_area` is set to a tuple
    [minX, minY, maxX, maxY], this area will be highlighted in the PNG image with a rectangle that
    has a semi-transparent filling color.

    Args:
        pdf_file_path: str
            The path to the PDF document from which to create the PNG image.
        png_file_path: str
            The target path of the PNG image.
        page_num: int
            The page number of the area in the PDF document from which to create a PNG image.
        area: Tuple[float, float, float, float]
            A tuple [minX, minY, maxX, maxY] defining the area from which to create a PNG image.
        highlight_area: Tuple[float, float, float, float]
            An area within `area` to highlight with a rectangle of a semi-transparent filling
            color. If set to None, no such rectangle will be drawn to the PNG image.
    """
    # Open the PDF file.
    pdf = fitz.open(pdf_file_path)

    # Shrink the PDF to the given page and select the page.
    pdf.select([page_num])
    page = pdf.load_page(0)

    page_height = page.rect.height
    page_width = page.rect.width

    # Draw the highlight area into the PDF (if any).
    if highlight_area is not None:
        min_x, min_y, max_x, max_y = highlight_area
        shape = page.new_shape()
        # Fit the given highlight area to the page boundaries. Translate the y-coordinates so that
        # they are relative to the page's *upper* left (as required by fitz).
        shape.draw_rect(fitz.Rect(
            max(0, min_x),
            max(0, page_height - max_y),
            min(page_width, max_x),
            min(page_height, page_height - min_y)
        ))
        shape.finish(
            color=HIGHLIGHT_AREA_STROKE_COLOR,
            fill=HIGHLIGHT_AREA_STROKE_COLOR,
            fill_opacity=HIGHLIGHT_AREA_FILL_OPACITY
        )
        shape.commit()

    # Crop the page to the given crop box. Fit the crop box to the page boundaries. Translate the y
    # values so that they are relative to the page's *upper* left (as required by fitz).
    min_x, min_y, max_x, max_y = area
    page.set_cropbox(fitz.Rect(
        max(0, min_x),
        max(0, page_height - max_y),
        min(page_width, max_x),
        min(page_height, page_height - min_y)
    ))

    # Generate a PNG from the page. Zoom into the page by factor 2 to increase the resolution.
    pic = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
    pic.writePNG(png_file_path)
