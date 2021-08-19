"""
This file is part of the "semantic-roles-detection" module of PdfActML. It contains all classes
required to model the information required for predicting the semantic roles of text blocks
extracted from PDF files.

Copyright 2021, University of Freiburg.

Claudius Korzen <korzen@cs.uni-freiburg.de>
"""

from typing import List

# ==================================================================================================


class Document:
    """
    This class represents the content of a single document, with information about the contained
    text blocks and pages.
    """

    def __init__(self, blocks: List["TextBlock"] = None, pages: List["Page"] = None):
        """
        This constructor creates and initializes a new `Document`.

        Args:
            blocks: List[TextBlock]
                The information about the text blocks contained in this document.
            pages: List[Page]
                The information about the pages contained in this document.
        """
        self.blocks = blocks if blocks is not None else []
        self.pages = pages if pages is not None else []

# ==================================================================================================


class TextBlock:
    """
    This class represents a single text block of a document.
    """

    def __init__(self, id: str = None, text: str = None, page_num: int = None,
                 lower_left_x: float = None, lower_left_y: float = None,
                 upper_right_x: float = None, upper_right_y: float = None,
                 font_name: str = None, font_size: float = None, is_bold: bool = None,
                 is_italic: bool = None, role: str = None, ground_truth_file_path: str = None):
        """
        This constructor creates and initializes a new `TextBlock`.

        Args:
            id: str
                A unique id of this text block.
            text: str
                The text of this text block.
            page_num: int
                The page number of this text block.
            lower_left_x: float
                The minimum x-coordinate of this text block.
            lower_left_y: float
                The minimum y-coordinate of this text block.
            upper_right_x: float
                The maximum x-coordinate of this text block.
            upper_right_y: float
                The maximum y-coordinate of this text block.
            font_name: str
                The name of the most common font in this text block.
            font_size: float
                The average font size among the characters in this text block.
            is_bold: bool
                A boolean flag indicating whether or not more than half of the characters in this
                text block are printed in bold.
            is_italic: bool
                A boolean flag indicating whether or not more than half of the characters in this
                text block are printed in italics.
            role: str
                The (expected) semantic role of this text block.
            ground_truth_file_path: str
                The path to the ground truth file from which this text block originates. Set it to
                None if this text block does not originate from a ground truth file.
        """
        self.id = id
        self.text = text
        self.page_num = page_num
        self.lower_left_x = lower_left_x
        self.lower_left_y = lower_left_y
        self.upper_right_x = upper_right_x
        self.upper_right_y = upper_right_y
        self.font_name = font_name
        self.font_size = font_size
        self.is_bold = is_bold
        self.is_italic = is_italic
        self.role = role
        self.ground_truth_file_path = ground_truth_file_path

    def __str__(self):
        return f"TextBlock({self.id}; {self.role}; {self.page_num}; {self.lower_left_x}; " \
                  f"{self.lower_left_y}; {self.upper_right_x}; {self.upper_right_y}; " \
                  f"{self.font_name}; {self.font_size}; {self.is_bold}; {self.is_italic}; " \
                  f"{self.text}\")"

    def __repr__(self):
        return self.__str__()

# ==================================================================================================


class Page:
    """
    This class represents a single page of a document.
    """

    def __init__(self, page_num: int = None, width: float = None, height: float = None):
        """
        This constructor creates and initializes a new `Page`.

        Args:
            page_num: int
                The page number of this page.
            width: float
                The width of this page.
            height: float
                The height of this page.
        """
        self.page_num = page_num
        self.width = width
        self.height = height

    def __str__(self):
        return f"Page({self.page_num}; {self.width}; {self.height})"

    def __repr__(self):
        return self.__str__()
