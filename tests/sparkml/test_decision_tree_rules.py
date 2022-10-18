# SPDX-License-Identifier: Apache-2.0

import unittest
from onnxmltools.convert.sparkml.operator_converters.tree_helper import Node


class TestSparkmDecisionTreeClassifier(unittest.TestCase):

    def test_rule_in_set(self):
        attrs = {
            'class_ids': [0, 0, 0, 0, 0, 0],
            'class_nodeids': [2, 4, 5, 8, 9, 10],
            'class_treeids': [0, 0, 0, 0, 0, 0],
            'class_weights': [0.8462194428652643,
                           0.2781875658587987,
                           0.5437174290677474,
                           0.6656197654941374,
                           0.4343004513217279,
                           0.2975769813225644],
            'classlabels_int64s': [0, 1],
            'nodes_falsenodeids': [6, 3, 0, 5, 0, 0, 10, 9, 0, 0, 0],
            'nodes_featureids': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'nodes_hitrates': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'nodes_missing_value_tracks_true': [False,
                                             False,
                                             False,
                                             False,
                                             False,
                                             False,
                                             False,
                                             False,
                                             False,
                                             False,
                                             False],
            'nodes_modes': ['BRANCH_LEQ',
                         'BRANCH_LEQ',
                         'LEAF',
                         'BRANCH_LEQ',
                         'LEAF',
                         'LEAF',
                         'BRANCH_LEQ',
                         'BRANCH_LEQ',
                         'LEAF',
                         'LEAF',
                         'LEAF'],
            'nodes_nodeids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'nodes_treeids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'nodes_truenodeids': [1, 2, 0, 4, 0, 0, 7, 8, 0, 0, 0],
            'nodes_values': [21.65000057220459,
                          100.95000076293945,
                          0.0,
                          -22.84999942779541,
                          0.0,
                          0.0,
                          98.0999984741211,
                          37.14999961853027,
                          0.0,
                          0.0,
                          0.0],
            'post_transform': 'NONE'}
        root, _ = Node.create(attrs)
        root.unfold_rule_or()
        new_attrs = root.to_attrs(
                post_transform=attrs['post_transform'],
                classlabels_int64s=attrs["classlabels_int64s"])
        assert len(attrs['nodes_nodeids']) <= len(new_attrs['nodes_nodeids'])
        for i in range(len(new_attrs["nodes_truenodeids"])):
            if new_attrs["nodes_modes"][i] == 'LEAF':
                continue
            assert new_attrs["nodes_truenodeids"][i] > i
        for i in range(len(new_attrs["nodes_falsenodeids"])):
            if new_attrs["nodes_modes"][i] == 'LEAF':
                continue
            assert new_attrs["nodes_falsenodeids"][i] > i


if __name__ == "__main__":
    unittest.main()
