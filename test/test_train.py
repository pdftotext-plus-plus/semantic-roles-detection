"""
End-to-End tests for the train method.
"""

import argparse
import copy
import os
import unittest
from os.path import dirname, exists, join, realpath

import numpy as np

import train

np.set_printoptions(formatter={'float': lambda x: ("%.2f" % x)})

# =================================================================================================


class TrainTest(unittest.TestCase):

    def __init__(self, methodName='runTest'):  # NOQA
        super(TrainTest, self).__init__(methodName)

        # Use the root directory of this file as base directory.
        base_dir = dirname(realpath(__file__))

        # The base arguments, that can be adapted in the below test cases to individual needs.
        self.args = argparse.Namespace()
        setattr(self.args, "input_dir", join(base_dir, "examples"))
        setattr(self.args, "output_dir", base_dir)
        setattr(self.args, "prefix", "")
        setattr(self.args, "suffix", ".tsv")
        setattr(self.args, "max_num_files", -1)
        setattr(self.args, "shuffle", False)
        setattr(self.args, "encoding", "bpe")
        setattr(self.args, "words_vocab_file", join(base_dir, "examples/vocab-words.example.txt"))
        setattr(self.args, "roles_vocab_file", join(base_dir, "examples/vocab-roles.example.txt"))
        setattr(self.args, "word_delimiters", "!#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n ")
        setattr(self.args, "lowercase_words", False)
        setattr(self.args, "num_words_per_seq", 8)
        setattr(self.args, "include_positions", True)
        setattr(self.args, "include_font_sizes", True)
        setattr(self.args, "include_font_styles", True)
        setattr(self.args, "include_char_features", True)
        setattr(self.args, "include_semantic_features", False)
        setattr(self.args, "human_names_db_file", None)
        setattr(self.args, "countries_db_file", None)
        setattr(self.args, "dropout", 0.2)
        setattr(self.args, "activation", "softmax")
        setattr(self.args, "loss", "categorical_crossentropy")
        setattr(self.args, "optimizer", "adam")
        setattr(self.args, "log_learning_rate", -3)
        setattr(self.args, "validation_split", 0.1)
        setattr(self.args, "epochs", 3)
        setattr(self.args, "use_tensorboard", False)
        setattr(self.args, "progress_bar_update_frequency", 0.1)
        setattr(self.args, "log_level", "error")

        self.files_to_delete = []

    def tearDown(self):
        # Delete all files marked for deletion once all tests were finished.
        for f in self.files_to_delete:
            os.remove(f)

    # =============================================================================================

    def test_base(self):
        """
        Run a test with the base arguments.
        """
        args = copy.copy(self.args)

        expected_seqs = np.array([
            [272, 263, 281, 281, 281, 281, 281, 281],
            [279, 280, 16, 22, 283, 281, 281, 281],
            [264, 66, 64, 83, 66, 71, 265, 266],
            [32, 65, 82, 267, 64, 66, 268, 281],
            [33, 84, 256, 82, 71, 72, 268, 33],
            [76, 72, 259, 70, 76, 64, 72, 75],
            [264, 270, 79, 81, 68, 71, 261, 82],
            [32, 77, 77, 260, 263, 281, 281, 281],
            [44, 68, 87, 72, 269, 283, 281, 281],
            [16, 283, 40, 77, 267, 78, 67, 84]
        ])

        expected_features = np.array([
            [0.00, 0.50, 0.55, 0.28, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 1.00, 0.22],
            [0.00, 0.00, 0.00, 0.28, 0.00, 0.00, 0.00, 0.00, 0.29, 0.00, 0.00, 0.50, 0.08],
            [0.00, 0.64, 0.60, 1.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.07, 0.33, 0.08],
            [0.50, 0.32, 0.40, 0.60, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.12],
            [1.00, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.13, 1.00, 0.13],
            [0.00, 0.53, 0.84, 0.00, 1.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.18, 0.00, 0.00],
            [0.00, 0.53, 0.83, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.04, 0.33, 0.05],
            [0.00, 0.28, 0.76, 0.34, 0.00, 0.00, 0.00, 0.00, 0.00, 0.09, 0.00, 1.00, 0.20],
            [0.00, 0.28, 0.73, 0.11, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.17],
            [1.00, 0.26, 0.24, 0.00, 0.00, 0.00, 0.00, 1.00, 0.07, 0.00, 0.07, 0.50, 0.07]
        ])

        expected_roles = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0]
        ])

        expected_roles_distr = [('AUTHOR_COUNTRY', '0.10'), ('AUTHOR_MAIL', '0.10'),
                                ('AUTHOR_NAME', '0.20'), ('HEADING_HEADING', '0.10'),
                                ('PARAGRAPHS', '0.20'), ('PUBLICATION-DATE', '0.10'),
                                ('TITLE', '0.20')]

        self.run_test(args, expected_seqs, expected_features, expected_roles, expected_roles_distr)

    # =============================================================================================

    def test_lowercase(self):
        """
        Run a test with lowercased word sequences.
        """
        args = copy.copy(self.args)
        setattr(args, "lowercase_words", True)

        expected_seqs = np.array([
            [65, 261, 283,  76, 184, 259, 281, 281],
            [82,  68,  79,  83,  68,  76,  65, 258],
            [64, 283,  66,  64,  83,  66,  71, 265],
            [64,  65,  82, 267,  64,  66, 268, 281],
            [65,  84, 256,  82,  71,  72, 268,  65],
            [76,  72, 259,  70,  76,  64,  72,  75],
            [64, 283, 270,  79,  81,  68,  71, 261],
            [64,  77,  77, 260,  76, 184, 259, 281],
            [76,  68,  87,  72, 269, 283, 281, 281],
            [16, 283,  72,  77, 267,  78,  67,  84]
        ])

        expected_features = np.array([
            [0.00, 0.50, 0.55, 0.28, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 1.00, 0.22],
            [0.00, 0.00, 0.00, 0.28, 0.00, 0.00, 0.00, 0.00, 0.29, 0.00, 0.00, 0.50, 0.08],
            [0.00, 0.64, 0.60, 1.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.07, 0.33, 0.08],
            [0.50, 0.32, 0.40, 0.60, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.12],
            [1.00, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.13, 1.00, 0.13],
            [0.00, 0.53, 0.84, 0.00, 1.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.18, 0.00, 0.00],
            [0.00, 0.53, 0.83, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.04, 0.33, 0.05],
            [0.00, 0.28, 0.76, 0.34, 0.00, 0.00, 0.00, 0.00, 0.00, 0.09, 0.00, 1.00, 0.20],
            [0.00, 0.28, 0.73, 0.11, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.17],
            [1.00, 0.26, 0.24, 0.00, 0.00, 0.00, 0.00, 1.00, 0.07, 0.00, 0.07, 0.50, 0.07]
        ])

        expected_roles = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0]
        ])

        expected_roles_distr = [('AUTHOR_COUNTRY', '0.10'), ('AUTHOR_MAIL', '0.10'),
                                ('AUTHOR_NAME', '0.20'), ('HEADING_HEADING', '0.10'),
                                ('PARAGRAPHS', '0.20'), ('PUBLICATION-DATE', '0.10'),
                                ('TITLE', '0.20')]

        self.run_test(args, expected_seqs, expected_features, expected_roles, expected_roles_distr)

    # =============================================================================================

    def test_with_only_position_features(self):
        """
        Run a test with the base arguments.
        """
        args = copy.copy(self.args)
        setattr(args, "include_positions", True)
        setattr(args, "include_font_sizes", False)
        setattr(args, "include_font_styles", False)
        setattr(args, "include_char_features", False)
        setattr(args, "include_semantic_features", False)

        expected_seqs = np.array([
            [272, 263, 281, 281, 281, 281, 281, 281],
            [279, 280, 16, 22, 283, 281, 281, 281],
            [264, 66, 64, 83, 66, 71, 265, 266],
            [32, 65, 82, 267, 64, 66, 268, 281],
            [33, 84, 256, 82, 71, 72, 268, 33],
            [76, 72, 259, 70, 76, 64, 72, 75],
            [264, 270, 79, 81, 68, 71, 261, 82],
            [32, 77, 77, 260, 263, 281, 281, 281],
            [44, 68, 87, 72, 269, 283, 281, 281],
            [16, 283, 40, 77, 267, 78, 67, 84]
        ])

        expected_features = np.array([
            [0.000, 0.500, 0.545],
            [0.000, 0.000, 0.000],
            [0.000, 0.637, 0.599],
            [0.500, 0.317, 0.401],
            [1.000, 1.000, 1.000],
            [0.000, 0.530, 0.837],
            [0.000, 0.530, 0.825],
            [0.000, 0.276, 0.763],
            [0.000, 0.276, 0.727],
            [1.000, 0.263, 0.240],
        ])

        expected_roles = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0]
        ])

        expected_roles_distr = [('AUTHOR_COUNTRY', '0.10'), ('AUTHOR_MAIL', '0.10'),
                                ('AUTHOR_NAME', '0.20'), ('HEADING_HEADING', '0.10'),
                                ('PARAGRAPHS', '0.20'), ('PUBLICATION-DATE', '0.10'),
                                ('TITLE', '0.20')]

        self.run_test(args, expected_seqs, expected_features, expected_roles, expected_roles_distr)

    # =============================================================================================

    def run_test(self, args, expected_seqs, expected_features, expected_roles, expected_distr):
        """
        Runs a test case with the given arguments.
        """

        word_seqs, features, roles, roles_distr, model_file_path, args_file_path = train.main(args)

        # print(features)

        # Delete the files once running the tests finished.
        self.files_to_delete.append(model_file_path)
        self.files_to_delete.append(args_file_path)

        # Test the word sequences.
        np.testing.assert_array_equal(word_seqs, expected_seqs)

        # Test the features.
        np.testing.assert_allclose(features, expected_features, atol=1e-2)

        # Test the roles.
        np.testing.assert_array_equal(roles, expected_roles)

        # Test roles distribution.
        roles_distr = sorted([(k, "%.2f" % v) for k, v in roles_distr])
        self.assertEqual(roles_distr, expected_distr)
        # Test whether the model file exists and is not empty.
        self.assertTrue(exists(model_file_path))
        self.assertTrue(os.stat(model_file_path).st_size > 0)

        # Test whether the args file exists and is not empty.
        self.assertTrue(exists(args_file_path))
        self.assertTrue(os.stat(args_file_path).st_size > 0)


if __name__ == "__main__":
    unittest.main()
