import fnmatch
import glob
import os.path
import random
import string

import utils.log

# =================================================================================================

# The logger.
LOG = utils.log.get_logger(__name__)

# =================================================================================================


def parse_dir(directory, pattern="*", shuffle_files=False, max_num_files=-1):
    """
    Parses the given directory for files matching the given file name pattern recursively. If
    shuffle_files is set to True, returns <max_num_files>-many *random* of these files.
    Otherwise, if shuffle_files is set to False, returns the first <max_num_files>-many input
    files (in the order as returned by Python's directory reader).

    Args:
        directory (str):
            The path to the directory to read.
        pattern (str):
            The file name pattern of the files to read, for example: "*" or "*.tsv" or "vldb*"
        shuffle_files (bool):
            Whether or not to shuffle the files before selecting <max_num_files>-many files.
        max_num_files (int):
            The maximum number of files to read. Reads *all* files if set to -1.

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

    LOG.debug("Parsing '{0}' for files matching pattern '{1}' ...".format(directory, pattern))
    #files = glob.glob(os.path.join(directory, pattern))
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    LOG.debug("Found {0} files.".format(len(files)))

    if shuffle_files:
        LOG.debug("Shuffling the files.")
        random.shuffle(files)
    else:
        LOG.debug("Sorting the files.")
        files.sort()

    if max_num_files is not None and max_num_files >= 0:
        LOG.debug("Limiting the number of files to {0}.".format(max_num_files))
        files = files[:max_num_files]

    LOG.debug("Selected {0}{1} files.".format(len(files), " random" if shuffle_files else ""))

    return files

# =================================================================================================


def read_groundtruth_files(files):
    """
    Reads the given input files and returns the following four lists:

    (1) building_blocks: The building blocks stored in the input files, each with the provided
                         attributes, like: the semantic role, the text, the position in the
                         visual text, etc. Each block also contains the index of the document
                         in which it occurs, for example: building_blocks[i].doc_index == j iff
                         the building block at position i occurs in document j.
    (2) pages:           The provided metadata about the pages, that is: the numbers, the heights
                         and the widths of the pages. pages[i][j] is the metadata of page j in
                         document i.
    (3) min_font_sizes:  The minimum font sizes in the documents; min_font_sizes[i] is the
                         minimum font size in document i. Needed later for normalizing the font
                         sizes in the input sequence.
    (4) max_font_sizes:  The maximum font sizes in the documents; max_font_sizes[i] is the
                         maximum font size in document i. Needed later for normalizing the font
                         sizes in the input sequence.

    Args:
        files (list of str):
            The paths to the files to read.

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
    building_blocks = []
    pages = []
    min_font_sizes = []
    max_font_sizes = []

    LOG.debug("Reading the building blocks from {0} files ...".format(len(files)))

    for doc_index, input_file in enumerate(files):
        with open(input_file, "r", encoding="utf-8") as f:
            doc_pages = []
            doc_min_font_size = float('inf')
            doc_max_font_size = 0

            for line in f:
                # Ignore comment lines.
                if line.startswith("#"):
                    continue

                fields = line.split("\t")

                if fields[0] == "BLOCK":
                    # Parse a BLOCK line.
                    block = BuildingBlock()
                    # Create a random block id of length 4.
                    block.id = ''.join(random.choice(string.ascii_letters) for i in range(4))
                    block.doc_index = doc_index
                    # Compute the doc slug, that is: the file name without the extension.
                    block.doc_slug = os.path.splitext(os.path.basename(input_file))[0]
                    block.doc_path = input_file

                    # (1) Read the role.
                    block.role = fields[1].strip()

                    # (2) Read the page number.
                    block.page_num = int(fields[2].strip())

                    # (3) Read the lower left x.
                    block.lower_left_x = float(fields[3].strip())

                    # (4) Read the lower left y.
                    block.lower_left_y = float(fields[4].strip())

                    # (5) Read the upper right x.
                    block.upper_right_x = float(fields[5].strip())

                    # (6) Read the upper right y.
                    block.upper_right_y = float(fields[6].strip())

                    # (7) Read the font base name.
                    block.font_name = fields[9].strip()

                    # (8) Read the font size.
                    block.font_size = float(fields[10])
                    doc_min_font_size = min(doc_min_font_size, block.font_size)
                    doc_max_font_size = max(doc_max_font_size, block.font_size)

                    # (9) Read the isBold, isItalic flags.
                    block.is_bold = int(fields[11].strip())
                    block.is_italic = int(fields[12].strip())

                    # (10) Read the text.
                    block.text = fields[14].strip()

                    building_blocks.append(block)

                if fields[0] == "PAGE":
                    # Parse a PAGE line.
                    page = Page()
                    page.page_num = int(fields[1])
                    page.width = float(fields[2])
                    page.height = float(fields[3])
                    doc_pages.append(page)

            pages.append(doc_pages)
            min_font_sizes.append(doc_min_font_size)
            max_font_sizes.append(doc_max_font_size)

    return building_blocks, pages, min_font_sizes, max_font_sizes

# =================================================================================================


def write_vocabulary_file(vocab, path):
    """
    Writes the given vocabulary to the given file.

    Args:
        vocab (dict str:int):
            The vocabulary to write to file.
        path (str):
            The path to the target file for the vocabulary.

    >>> vocab = {"x": 0, "y": 1, "xy": 2}
    >>> base_dir = os.path.dirname(os.path.realpath(__file__))
    >>> path = os.path.join(base_dir, "../examples/example-vocab.tmp.txt")
    >>> write_vocabulary_file(vocab, path)
    >>> open(path).read()
    'x\\t0\\ny\\t1\\nxy\\t2\\n'
    >>> os.remove(path)
    """
    with open(path, "w") as fout:
        for word, id in vocab.items():
            fout.write("{}\t{}\n".format(word, id))


def read_vocabulary_file(path):
    """
    Reads the given vocabulary file.

    Args:
        path (str):
            The path to the file from which the vocabulary should be read.

    >>> base_dir = os.path.dirname(os.path.realpath(__file__))
    >>> vocab = read_vocabulary_file(os.path.join(base_dir, "../examples/vocab-words.example.txt"))
    >>> vocab_list = sorted(vocab.items(), key=lambda x: x[1])
    >>> len(vocab_list)
    281
    >>> vocab_list[:25] #doctest: +NORMALIZE_WHITESPACE
    [('!', 0), ('"', 1), ('#', 2), ('$', 3), ('%', 4), ('&', 5), ("'", 6), ('(', 7), (')', 8), \
     ('*', 9), ('+', 10), (',', 11), ('-', 12), ('.', 13), ('/', 14), ('0', 15), ('1', 16), \
     ('2', 17), ('3', 18), ('4', 19), ('5', 20), ('6', 21), ('7', 22), ('8', 23), ('9', 24)]
    >>> vocab_list[-25:] #doctest: +NORMALIZE_WHITESPACE
    [('ll', 256), ('er', 257), ('er✂', 258), ('ller✂', 259), ('e✂', 260), ('en', 261), \
     ('Mü', 262), ('Müller✂', 263), ('A✂', 264), ('y✂', 265), ('ti', 266), ('tr', 267), \
     ('t✂', 268), ('co', 269), ('com', 270), ('Ben', 271), ('Ben✂', 272), ('Se', 273), \
     ('Sep', 274), ('Sept', 275), ('Septe', 276), ('Septem', 277), ('Septemb', 278), \
     ('September✂', 279), ('20', 280)]
    """
    vocab = {}
    with open(path, "r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                key, value = line.split("\t")
                vocab[key] = int(value)
    return vocab

# =================================================================================================


def read_lines(file_path):
    """
    Reads the given file line by line. Returns a list of the lines read.
    """
    lines = []
    with open(file_path, "r") as fin:
        for line in fin:
            lines.append(line.strip())
    return lines


# =================================================================================================

class BuildingBlock:
    """
    A single building block of a visual text.
    """

    def __init__(self):
        self.id = None
        self.text = None
        self.page_num = -1
        self.lower_left_x = 0.0
        self.lower_left_y = 0.0
        self.upper_right_x = 0.0
        self.upper_right_y = 0.0
        self.font_name = None
        self.font_size = -1
        self.is_bold = 0
        self.is_italic = 0
        self.role = None
        self.doc_index = -1
        # The doc slug, that is: the filename of the belonging ground truth file (resp. PDF file)
        # without the file extension. For example, if the path to the ground truth file is
        # "/data/semantic-roles-detection/svjour3.twocolumn.14qnqsHi.tsv" the doc slug is
        # "svjour3.twocolumn.14qnqsHi". With the doc slug, the path to the respective PDF file can
        # be deduced. This is particularly useful for the evaluation app, where we want to add a
        # PDF link to each building block with a wrong prediction result).
        self.doc_slug = None
        # The path to the document (i.e., JSON file) from which this building block was read.
        self.doc_path = None

    def __str__(self):
        return "BB({0}; {1}; {2};{3};{4};{5}; {6}; {7}; {8}; {9}; \"{10}\")".format(
            self.role, self.page_num, self.lower_left_x, self.lower_left_y,
            self.upper_right_x, self.upper_right_y, self.font_name, self.font_size, self.is_bold,
            self.is_italic, self.text)

    def __repr__(self):
        return self.__str__()


class Page:
    """
    A single page of a visual text.
    """

    def __init__(self):
        self.page_num = None
        self.height = 0.0
        self.width = 0.0

    def __str__(self):
        return "Page({0}; {1}; {2})".format(self.page_num, self.height, self.width)

    def __repr__(self):
        return self.__str__()
