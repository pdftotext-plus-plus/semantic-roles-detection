"""
This file is part of the "semantic-roles-detection" module of PdfActML. It contains code to read
text blocks (together with their expected semantic roles) from ground truth files.

Copyright 2021, University of Freiburg.

Claudius Korzen <korzen@cs.uni-freiburg.de>
"""

import fnmatch
import os
import random
import string
from typing import List

from semantic_roles_detection.utils import log_utils
from semantic_roles_detection.utils.models import Document, Page, TextBlock

# ==================================================================================================
# Parameters.

# The logger.
LOG = log_utils.get_logger(__name__)

# ==================================================================================================


class GroundTruthReader():
    """
    This class parses a given directory for ground truth files, reads the ground truth files and
    stores the content of each ground truth file in form of a `Document`.
    """

    def read(self, directory: str, file_name_pattern: str = "*", num_files: int = -1,
             shuffle_files: bool = False) -> List[Document]:
        """
        This method recursively parses the given directory for ground truth files that match the
        given file name pattern and reads the ground truth files. It returns the content of each
        ground truth file in form of a `Document`.

        Args:
            directory: str
                The path to the directory to parse for ground truth files.
            file_name_pattern: str
                The file name pattern of the ground truth files to read from the directory, for
                example: "*" or "*.tsv" or "vldb*"
            num_files: int
                The maximum number of ground truth files to read from the directory. If set to -1,
                this method reads *all* ground truth files.
            shuffle_files: bool
                A boolean flag indicating whether or not to shuffle the ground truth files before
                selecting `num_files`-many files.
        Returns:
            List[Document]
                The contents of the ground truth files, each in form of a `Document`.
        """
        # Parse the directory for ground truth files.
        files = self.parse_dir(directory, file_name_pattern, num_files, shuffle_files)

        # Read each ground truth file into a `Document`.
        docs = []
        for file in files:
            doc = self.read_ground_truth_file(file)
            docs.append(doc)

        return docs

    # ==============================================================================================

    def parse_dir(self, directory: str, file_name_pattern: str = "*", max_num_files: int = -1,
                  shuffle_files: bool = False) -> List[str]:
        """
        This method recursively parses the given directory for ground truth files that match the
        given file name pattern. If `shuffle_files` is set to True, this method returns
        `max_num_files`-many *random* ground truth files. Otherwise (if `shuffle_files` is set to
        False), this method returns the first `max_num_files`-many ground truth files (ordered by
        filename).

        Args:
            directory: str
                The path to the directory to parse for ground truth files.
            file_name_pattern: str
                The file name pattern of the files to read, for example: "*" or "*.tsv" or "vldb*"
            max_num_files: int
                The maximum number of ground truth files to read. If set to -1, this method reads
                *all* ground truth files.
            shuffle_files: bool
                A boolean flag indicating whether or not to shuffle the ground truth files before
                selecting `max_num_files`-many files.
        Returns:
            List[str]:
                The paths to the ground truth files.

        TODO
        >>> base_dir = os.path.dirname(os.path.realpath(__file__))
        >>> directory = os.path.join(base_dir, "../examples")

        >>> files = parse_dir(directory)
        >>> sorted([os.path.basename(f) for f in files]) #doctest: +NORMALIZE_WHITESPACE
        ['article.4.tsv', 'country-names.example.txt', 'human-names.example.txt', \
        'sig-alternate.1.tsv', 'vldb.2.tsv', 'vldb.3.txt', 'vocab-roles.example.txt', \
        'vocab-words.example.txt']

        >>> files = parse_dir(directory, pattern="*.tsv")
        >>> sorted([os.path.basename(f) for f in files])
        ['article.4.tsv', 'sig-alternate.1.tsv', 'vldb.2.tsv']

        >>> files = parse_dir(directory, pattern="vldb*")
        >>> sorted([os.path.basename(f) for f in files])
        ['vldb.2.tsv', 'vldb.3.txt']

        >>> files = parse_dir(directory, max_num_files=-1)
        >>> len(files)
        8

        >>> files = parse_dir(directory, max_num_files=0)
        >>> len(files)
        0

        >>> files = parse_dir(directory, max_num_files=2)
        >>> len(files)
        2

        >>> files = parse_dir(directory, pattern="vldb*", max_num_files=1)
        >>> len(files)
        1
        """
        LOG.debug(f"Parsing '{directory}' for files matching pattern '{file_name_pattern}' ...")

        files = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, file_name_pattern):
                files.append(os.path.join(root, filename))

        LOG.debug(f"Found {len(files)} ground truth files.")

        if shuffle_files:
            LOG.debug("Shuffling the ground truth files.")
            random.shuffle(files)
        else:
            LOG.debug("Sorting the ground truth files.")
            files.sort()

        if max_num_files is not None and max_num_files >= 0:
            LOG.debug(f"Limiting the number of ground truth files to {max_num_files}.")
            files = files[:max_num_files]

        LOG.debug(f"Selected {len(files)}{' random' if shuffle_files else ''} files.")

        return files

    # ==============================================================================================

    def read_ground_truth_file(self, file_path: str) -> Document:
        """
        This method reads the given ground truth file and returns its content in form of a
        `Document`.

        Args:
            file_path: str
                The path to the ground truth file to read.
        Returns:
            Document
                The content of the ground truth file.

        TODO
        >>> base_dir = os.path.dirname(os.path.realpath(__file__))
        >>> file_2 = os.path.join(base_dir, "../examples/vldb.2.tsv")
        >>> file_4 = os.path.join(base_dir, "../examples/article.4.tsv")
        >>> blocks, pages, min_fs, max_fs = read_groundtruth_files([file_4])
        >>> blocks
        ... #doctest: +NORMALIZE_WHITESPACE
        [BB(AUTHOR_NAME; 1; 200.0;540.0;280.0;550.0; lmroman; 12.0; 0; 0; "Ben Müller"),
        BB(PUBLICATION-DATE; 1; 0.0;0.0;0.0;0.0; lmroman; 12.0; 0; 0; "September 2017"),
        BB(TITLE; 1; 141.9;571.4;469.4;627.3; arial; 17.2; 0; 1; "A catchy title!"),
        BB(HEADING_HEADING; 2; 158.0;332.3;219.7;342.4; lmroman; 14.3; 1; 0; "Abstract"),
        BB(PARAGRAPHS; 3; 210.0;460.0;230.0;500.0; lmroman; 10.0; 0; 0; "Bullshit-Bingo.")]
        >>> pages
        [[Page(1; 1000.0; 480.0), Page(2; 841.89; 595.276), Page(3; 480.0; 220.0)]]
        >>> min_fs
        [10.0]
        >>> max_fs
        [17.2]

        >>> blocks, pages, min_fs, max_fs = read_groundtruth_files([file_4, file_2])
        >>> blocks
        ... #doctest: +NORMALIZE_WHITESPACE
        [BB(AUTHOR_NAME; 1; 200.0;540.0;280.0;550.0; lmroman; 12.0; 0; 0; "Ben Müller"), \
        BB(PUBLICATION-DATE; 1; 0.0;0.0;0.0;0.0; lmroman; 12.0; 0; 0; "September 2017"), \
        BB(TITLE; 1; 141.9;571.4;469.4;627.3; arial; 17.2; 0; 1; "A catchy title!"), \
        BB(HEADING_HEADING; 2; 158.0;332.3;219.7;342.4; lmroman; 14.3; 1; 0; "Abstract"), \
        BB(PARAGRAPHS; 3; 210.0;460.0;230.0;500.0; lmroman; 10.0; 0; 0; "Bullshit-Bingo."), \
        BB(TITLE; 1; 62.0;682.8;547.7;720.0; nimbussanl; 17.9; 1; 0; "A comprehensive survey."), \
        BB(AUTHOR_NAME; 1; 114.8;642.9;202.4;654.2; arial; 12.0; 0; 0; "Anne Müller"), \
        BB(AUTHOR_COUNTRY; 1; 143.0;613.9;174.2;621.4; nimbussanl; 10.0; 0; 0; "Mexico"), \
        BB(PARAGRAPHS; 2; 62.8;176.1;292.9;184.2; lmroman; 9.0; 0; 0; "1 Introduction.")]
        >>> pages #doctest: +NORMALIZE_WHITESPACE
        [[Page(1; 1000.0; 480.0), Page(2; 841.89; 595.276), Page(3; 480.0; 220.0)], \
        [Page(1; 850.0; 575.2), Page(2; 750.0; 675.7)]]
        >>> min_fs
        [10.0, 9.0]
        >>> max_fs
        [17.2, 17.9]
        """
        LOG.debug(f"Reading ground truth file '{file_path}' ...")

        doc = Document()
        with open(file_path, "r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()

                # Ignore empty lines.
                if len(line) == 0:
                    continue

                # Ignore comment lines.
                if line.startswith("#") or line.startswith("%"):
                    continue

                fields = line.split("\t")

                if fields[0] == "BLOCK":
                    # Parse a BLOCK line.
                    block = TextBlock()

                    # Create a random id of length 5.
                    block.id = ''.join(random.choice(string.ascii_letters) for i in range(5))
                    # Keep track of the path of the ground truth file.
                    block.ground_truth_file_path = file_path

                    # Get the semantic role.
                    block.role = fields[1].strip()

                    # Get the page number.
                    block.page_num = int(fields[2].strip())

                    # Get the x-coordinate of the lower left.
                    block.lower_left_x = float(fields[3].strip())

                    # Get the y-coordinate of the lower left.
                    block.lower_left_y = float(fields[4].strip())

                    # Get the x-coordinate of the upper right.
                    block.upper_right_x = float(fields[5].strip())

                    # Get the y-coordinate of the upper right.
                    block.upper_right_y = float(fields[6].strip())

                    # Get the font base name.
                    block.font_name = fields[9].strip()

                    # Get the font size.
                    block.font_size = float(fields[10].strip())

                    # Get the isBold and isItalic flags.
                    block.is_bold = int(fields[11].strip())
                    block.is_italic = int(fields[12].strip())

                    # Get the text.
                    block.text = fields[14].strip()

                    doc.blocks.append(block)

                if fields[0] == "PAGE":
                    # Parse a PAGE line.
                    page = Page()
                    # Get the page number.
                    page.page_num = int(fields[1].strip())
                    # Get the page width.
                    page.width = float(fields[2].strip())
                    # Get the page height.
                    page.height = float(fields[3].strip())

                    doc.pages.append(page)

        return doc
