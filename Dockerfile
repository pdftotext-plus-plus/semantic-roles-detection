FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

# ==================================================================================================

# Install some dependencies.
RUN apt-get update && apt-get install -y python3-pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

# ==================================================================================================

COPY src src
COPY models models
COPY scripts scripts
COPY vocabs vocabs

# CMD ["bash"]
ENTRYPOINT ["python3"]

# ==================================================================================================

# Building an image:
# docker build -t pdfact-ml.semantic-roles-detection .
# wharfer build -t pdfact-ml.semantic-roles-detection .

# Running a container for training:
# docker run --rm -v <input-dir>:/input -v <output-dir>:/output -it pdfact-ml.semantic-roles-detection train.py --words_vocab_file <words_vocab_file> --roles_vocab_file <roles_vocab_file> --epochs <num>
# wharfer run --rm -v <input-dir>:/input -v <output-dir>:/output -it pdfact-ml.semantic-roles-detection train.py --words_vocab_file <words_vocab_file> --roles_vocab_file <roles_vocab_file> --epochs <num>
