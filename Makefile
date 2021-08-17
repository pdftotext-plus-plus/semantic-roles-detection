DOCTEST_CMD = python3 -m doctest
UNITTEST_CMD = python3 -m unittest
COMPILE_CMD = python3 -m py_compile
CHECKSTYLE_CMD = flake8 --max-line-length=99
DOCKER_CMD = $(shell which wharfer || which docker)

PROJECT_NAME = pdfact-ml.semantic-roles-detection
TEX_INPUT_DIR = /local/data/korzen/datasets/arxiv-subset-pdf-extraction-benchmark
GROUND_TRUTH_DIR = $(shell pwd)/data/ground-truth/
VOCABS_DIR = $(shell pwd)/data/vocabs/
MODEL_DIR = $(shell pwd)/data/models/$(shell date +'%y-%m-%d_%H-%M-%S')/
EVALUATION_RESULT_DIR = $(shell pwd)/data/evaluation-results/$(shell date +'%y-%m-%d_%H-%M-%S')
NUM_DOCS = -1
SPLIT_RATIO = 0.8
CREATE_BLOCKS_VISUALIZATION = true
CREATE_WORDS_VISUALIZATION = true
NUM_EPOCHS=25
LOG = info

# ==================================================================================================

help:
	@echo "This is the 'semantic-roles-detection' module of PdfActML, responsible for predicting the"
	@echo "semantic roles of text blocks extracted from PDF files in a previous step. There are the"
	@echo "following targets:"
	@echo
	@echo "\033[1mmake ground-truth\033[0m"
	@echo "Creates a new ground truth dataset, splitted into a training dataset, ready to be used"
	@echo "for training a new learning model for semantic roles detection, and an evaluation"
	@echo "dataset, ready to be used for evaluating a model."
	@echo
	@echo "\033[1mmake vocabularies\033[0m"
	@echo "Creates the vocabularies from the ground truth."
	@echo
	@echo "\033[1mmake train\033[0m"
	@echo "Trains a new learning model for semantic roles detection."
	@echo
	@echo "\033[1mmake evaluate\033[0m"
	@echo "Evaluates an existing learning model and creates the evaluation files as required by our"
	@echo "evaluation app."
	@echo
	@echo "For more information about each target, type \"make help-<target>\"."

# ==================================================================================================

help-ground-truth:
	@echo "\033[1mmake ground-truth\033[0m"
	@echo "Creates a new ground truth, splitted into a training dataset and an evaluation dataset,"
	@echo "ready to be used for training and evaluating a learning model for semantic roles"
	@echo "detection."
	@echo
	@echo "Args:"
	@echo "  - TEX_INPUT_DIR:               The path to a directory of TeX files from which to"
	@echo "                                 create the ground truth."
	@echo "                                 Default: \"$(TEX_INPUT_DIR)\""
	@echo "  - GROUND_TRUTH_DIR:            The path to the directory to which the ground truth"
	@echo "                                 should be written."
	@echo "                                 Default: \"$(GROUND_TRUTH_DIR)\""
	@echo "  - NUM_DOCS:                    The maximum number of ground truth documents to create."
	@echo "                                 Setting this value to -1 is equivalent to setting it to"
	@echo "                                 the number of TeX files in <TEX_INPUT_DIR>."
	@echo "                                 Default: $(NUM_DOCS)"
	@echo "  - SPLIT_RATIO:                 The split ratio (a float value in range [0,1]) to use on"
	@echo "                                 splitting the ground truth into a training dataset and an"
	@echo "                                 evaluation dataset. The created training dataset will"
	@echo "                                 consists of <SPLIT_RATIO>*<NUM_DOCS>-many documents. The"
	@echo "                                 created evaluation dataset will consists of"
	@echo "                                 (1 - <SPLIT_RATIO>)*<NUM_DOCS>-many documents."
	@echo "                                 Default: $(SPLIT_RATIO)"
	@echo "  - CREATE_WORDS_VISUALIZATION:  Whether or not to create visualizations of the extracted"
	@echo "                                 words (possible values are "true" or "false")."
	@echo "                                 Default: $(CREATE_WORDS_VISUALIZATION)"
	@echo "  - CREATE_BLOCKS_VISUALIZATION: Whether or not to create visualizations of the extracted"
	@echo "                                 blocks (possible values are "true" or "false")."
	@echo "                                 Default: $(CREATE_BLOCKS_VISUALIZATION)"

.PHONY: ground-truth
ground-truth:
	cd ../ground-truth-generator; $(DOCKER_CMD) build -t pdfact-ml.ground-truth-generator .
	$(DOCKER_CMD) run --rm -it -v $(shell realpath $(TEX_INPUT_DIR)):/input -v $(shell realpath $(GROUND_TRUTH_DIR)):/output --name pdfact-ml.ground-truth-generator pdfact-ml.ground-truth-generator --max-num-docs $(NUM_DOCS) --serializer srd --split-train-evaluate $(SPLIT_RATIO) --create-words-visualization $(CREATE_WORDS_VISUALIZATION) --create-blocks-visualization $(CREATE_BLOCKS_VISUALIZATION) --log $(LOG)

# TODO: This should be moved to a project related to page segmentation.
ground-truth-elias:
	$(DOCKER_CMD) run --rm -it -v $(shell realpath $(TEX_INPUT_DIR)):/input -v /local/data/korzen/pdfact-ml/page-segmentation/ground-truth-new-format:/output --name pdfact-ml.ground-truth-generator-ps pdfact-ml.ground-truth-generator --max-num-docs $(NUM_DOCS) --serializer ps --create-words-visualization --create-blocks-visualization --log $(LOG)

# ==================================================================================================

help-vocabularies:
	@echo "\033[1mmake vocabularies\033[0m"
	@echo "Creates the vocabularies from the ground truth."
	@echo
	@echo "Args:"
	@echo "  - TEX_INPUT_DIR:               The path to a directory of TeX files from which to"
	@echo "                                 create the ground truth."
	@echo "                                 Default: \"$(TEX_INPUT_DIR)\""
	@echo "  - GROUND_TRUTH_DIR:            The path to the directory to which the ground truth"
	@echo "                                 should be written."
	@echo "                                 Default: \"$(GROUND_TRUTH_DIR)\""
	@echo "  - NUM_DOCS:                    The maximum number of ground truth documents to create."
	@echo "                                 Setting this value to -1 is equivalent to setting it to"
	@echo "                                 the number of TeX files in <TEX_INPUT_DIR>."
	@echo "                                 Default: $(NUM_DOCS)"
	@echo "  - SPLIT_RATIO:                 The split ratio (a float value in range [0,1]) to use on"
	@echo "                                 splitting the ground truth into a training dataset and an"
	@echo "                                 evaluation dataset. The created training dataset will"
	@echo "                                 consists of <SPLIT_RATIO>*<NUM_DOCS>-many documents. The"
	@echo "                                 created evaluation dataset will consists of"
	@echo "                                 (1 - <SPLIT_RATIO>)*<NUM_DOCS>-many documents."
	@echo "                                 Default: $(SPLIT_RATIO)"
	@echo "  - CREATE_WORDS_VISUALIZATION:  Whether or not to create visualizations of the extracted"
	@echo "                                 words (possible values are "true" or "false")."
	@echo "                                 Default: $(CREATE_WORDS_VISUALIZATION)"
	@echo "  - CREATE_BLOCKS_VISUALIZATION: Whether or not to create visualizations of the extracted"
	@echo "                                 blocks (possible values are "true" or "false")."
	@echo "                                 Default: $(CREATE_BLOCKS_VISUALIZATION)"

.PHONY: vocabularies
vocabularies:
	$(DOCKER_CMD) build -t $(PROJECT_NAME) .
	$(DOCKER_CMD) run --rm -it -v $(shell realpath $(GROUND_TRUTH_DIR)):/input -v $(shell realpath $(VOCABS_DIR)):/output --name $(PROJECT_NAME)-vocabularies $(PROJECT_NAME) scripts/create_bpe_words_vocab_file.py --input_dir /input --output_file /output/vocab-words.tsv
	$(DOCKER_CMD) run --rm -it -v $(shell realpath $(GROUND_TRUTH_DIR)):/input -v $(shell realpath $(VOCABS_DIR)):/output --name $(PROJECT_NAME)-vocabularies $(PROJECT_NAME) scripts/create_roles_vocab_file.py --input_dir /input --output_file /output/vocab-roles.tsv

# ==================================================================================================

help-train:
	@echo "\033[1mmake train\033[0m"
	@echo "Trains a new learning model for semantic roles detection."
	@echo
	@echo "Args:"
	@echo "  - TRAIN_GROUND_TRUTH_DIR: The path to the training ground truth."
	@echo "                            Default: \"$(TRAIN_GROUND_TRUTH_DIR)\""
	@echo "  - MODEL_DIR:              The path to the directory to which the trained learning"
	@echo "                            model (and the vocab files) should be written."
	@echo "                            Default: \"$(MODEL_DIR)\""
	@echo "  - NUM_EPOCHS:             The number of training epochs."
	@echo "                            Default: $(NUM_EPOCHS)"

train:
	$(DOCKER_CMD) build -t $(PROJECT_NAME) .
	$(DOCKER_CMD) run --rm -it -v $(shell realpath $(GROUND_TRUTH_DIR)/train):/input -v $(shell realpath $(MODEL_DIR)):/output -v $(shell realpath $(VOCABS_DIR)):/vocabs --name $(PROJECT_NAME)-train $(PROJECT_NAME) train.py --epochs $(NUM_EPOCHS) --words_vocab_file /vocabs/vocab-words.tsv --roles_vocab_file /vocabs/vocab-roles.tsv

# ==================================================================================================

help-evaluate:
	@echo "\033[1mmake evaluate\033[0m"
	@echo "Evaluates an existing learning model and creates the evaluation data as required by our"
	@echo "evaluation app."
	@echo
	@echo "Args:"
	@echo "  - GROUND_TRUTH_DIR:      The path to the evaluation ground truth."
	@echo "                           Default: \"$(GROUND_TRUTH_DIR)\""
	@echo "  - MODEL_DIR:             The path to the directory of the learning model to"
	@echo "                           evaluate. Default: \"$(MODEL_DIR)\""
	@echo "  - EVALUATION_RESULT_DIR: The path to the directory to which the evaluation files"
	@echo "                           should be written. Default: $(EVALUATION_RESULT_DIR)"

evaluate:
	$(DOCKER_CMD) build -t $(PROJECT_NAME) .
	$(DOCKER_CMD) run --rm -it -v $(shell realpath $(GROUND_TRUTH_DIR)/evaluate):/input -v $(shell realpath $(MODEL_DIR)):/model -v $(shell realpath $(EVALUATION_RESULT_DIR)):/output --name $(PROJECT_NAME)-evaluate $(PROJECT_NAME) evaluate.py --create-images

# ==================================================================================================

checkstyle:
	@find * -name "*.py" | xargs $(CHECKSTYLE_CMD)

compile:
	@find * -name "*.py" | xargs $(COMPILE_CMD)

test: doctest unittest

doctest:
	@find * -name "*.py" | xargs $(DOCTEST_CMD)

unittest:
	@find * -name "test_*.py" | xargs $(UNITTEST_CMD)

clean:
	@find . -name *.pyc | xargs rm -rf
	@find . -name __pycache__ | xargs rm -rf