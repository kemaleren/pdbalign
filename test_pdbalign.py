import random
import unittest

import numpy as np

from pdbalign import compute_distance_matrix

class TestPdbalign(unittest.TestCase):


    def test_align_to_pdb(self):
        pass

    def test_get_pdb_coords(self):
        pass

    def test_compute_distance_matrix(self):
        c1 = np.array([[0, 0],
                       [np.nan, np.nan],
                       [1, 1],
                       [1, 0]])
        c2 = c1.copy()
        c1[:, 0] += 1.5
        c1[:, 1] += 1
        coords = np.hstack([c1, c2]).reshape((4, 2, 2))
        expected = np.array([[0, 5, 0.5, 1],
                             [5, 0, 5, -1],
                             [0.5, 5, 0, 1],
                             [1, -1, 1, 0]])
        result = compute_distance_matrix(coords, radius=1.1, default_dist=5)
        self.assertTrue(np.all(expected == result))
    

if __name__ == '__main__':
    unittest.main()
