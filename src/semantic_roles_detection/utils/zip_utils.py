"""
This file is part of the "semantic-roles-detection" module of PdfActML. It contains a special
method to merge two lists.

Copyright 2021, University of Freiburg.

Claudius Korzen <korzen@cs.uni-freiburg.de>
"""

from typing import List


def zip_lists(list_1: List, list_2: List) -> List:
    """
    This method returns a list of length max(len(list_1), len(list_2)), where the i-th element is
    the concatenation of the i-th element in list_1 and the i-th element in list_2. On
    concatenating, the i-th elements in list_1 and list_2 are considered as lists, even if they are
    not. If an element is not a list (but a single element like an int or a string or any other
    object) it is considered as a single-element list. The i-th element in the returned list is
    then list_1[i] + list_2[i].
    If one element in one list does not have a matching partner in the other list (i.e., if
    i >= len(list_1) or i >= len(list_2)) the element will be concatenated with the empty list.
    Here are some examples, showing the resulting entry in the result list, depending on the types
    of list_1[i] and list_2[i]:

    list_1[i]       list_2[i]       result_list[i]
    ------------------------------------------------
    [1, 2, 3]       [A, B, C]       [1, 2, 3, A, B, C]
    [1, 2, 3]       A               [1, 2, 3, A]
    1               [A, B, C]       [1, A, B, C]
    1               A               [1, A]
    [1, 2, 3]       not existent    [1, 2, 3]
    1               not existent    [1]
    not existent    [A, B, C]       [A, B, C]
    not existent    A               [A]

    Args:
        list_1 (list):
            The first list.
        list_2 (list):
            The second list.
    Returns:
        The list resulted from merging the two lists as described above.

    >>> zip_lists([], [])
    []

    >>> zip_lists([1, 2, 3], [])
    [[1], [2], [3]]
    >>> zip_lists([], [1, 2, 3])
    [[1], [2], [3]]

    >>> zip_lists([[1], [2], [3]], [])
    [[1], [2], [3]]
    >>> zip_lists([], [[1], [2], [3]])
    [[1], [2], [3]]

    >>> zip_lists([1, 2, 3], [[], [], []])
    [[1], [2], [3]]
    >>> zip_lists([[], [], []], [1, 2, 3])
    [[1], [2], [3]]

    >>> zip_lists([[1], [2], [3]], [[], [], []])
    [[1], [2], [3]]
    >>> zip_lists([[], [], []], [[1], [2], [3]])
    [[1], [2], [3]]

    >>> zip_lists([[1, 2, 3], [4, 5, 6], [7, 8]], [['A', 'B'], ['C'], ['D', 'E']])
    [[1, 2, 3, 'A', 'B'], [4, 5, 6, 'C'], [7, 8, 'D', 'E']]

    >>> zip_lists([[1, 2, 3], [4, 5, 6], [7, 8]], ['A', 'B', 'C'])
    [[1, 2, 3, 'A'], [4, 5, 6, 'B'], [7, 8, 'C']]
    >>> zip_lists([[1, 2, 3], [4, 5, 6], [7, 8]], ['A', 'B', ['C', 'D', 'E']])
    [[1, 2, 3, 'A'], [4, 5, 6, 'B'], [7, 8, 'C', 'D', 'E']]
    >>> zip_lists([[1, 2, 3], [4, 5, 6], [7, 8]], ['A', 'B'])
    [[1, 2, 3, 'A'], [4, 5, 6, 'B'], [7, 8]]
    >>> zip_lists([[1, 2, 3], [4, 5, 6], [7, 8]], ['A', 'B', 'C', 'D'])
    [[1, 2, 3, 'A'], [4, 5, 6, 'B'], [7, 8, 'C'], ['D']]

    >>> zip_lists([1, 2, 3], [['A', 'B'], ['C', 'D'], ['E']])
    [[1, 'A', 'B'], [2, 'C', 'D'], [3, 'E']]
    >>> zip_lists([1, 2, [3, 4, 5]], [['A', 'B'], ['C', 'D'], ['E']])
    [[1, 'A', 'B'], [2, 'C', 'D'], [3, 4, 5, 'E']]
    >>> zip_lists([1, 2], [['A', 'B'], ['C', 'D'], ['E']])
    [[1, 'A', 'B'], [2, 'C', 'D'], ['E']]
    >>> zip_lists([[1, 2, 3], [4, 5, 6], [7, 8], [9, 10]], [['A', 'B'], ['C', 'D'], ['E']])
    [[1, 2, 3, 'A', 'B'], [4, 5, 6, 'C', 'D'], [7, 8, 'E'], [9, 10]]
    """

    result_list = []

    for i in range(max(len(list_1), len(list_2))):
        merged_list = []
        if i < len(list_1):
            if type(list_1[i]) == list:
                merged_list.extend(list_1[i])
            else:
                merged_list.append(list_1[i])
        if i < len(list_2):
            if type(list_2[i]) == list:
                merged_list.extend(list_2[i])
            else:
                merged_list.append(list_2[i])
        result_list.append(merged_list)

    return result_list
