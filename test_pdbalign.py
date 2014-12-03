import random
import unittest

import numpy as np

from Bio.Seq import Seq
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.Alphabet import Gapped
from Bio.Alphabet import IUPAC

from pdbalign import align_to_pdb
from pdbalign import get_pdb_indices
from pdbalign import compute_distance_matrix

from BioExt.align import Aligner
from BioExt.scorematrices import BLOSUM62

class TestPdbalign(unittest.TestCase):
    # Need to reduce gap penalty to make test alignments work
    aligner = Aligner(BLOSUM62.load(), do_codon=False, open_insertion=-1, open_deletion=-1)

    def test_align_to_pdb(self):
        problems = (
            (Seq("AHSVH"), Seq("AHVH"), [0, 1, -1, 2, 3]),
            (Seq("AHVH"), Seq("AHSVH"), [0, 1, 3, 4]),
            (Seq("AHSVH"), Seq("AHSVH"), [0, 1, 2, 3, 4]),
            (Seq("-HSVH"), Seq("AHSVH"), [-1, 1, 2, 3, 4]),
            (Seq("A-SVH"), Seq("AHSVH"), [0, -1, 2, 3, 4]),
            (Seq("AH-VH"), Seq("AHSVH"), [0, 1, -1, 3, 4]),
            (Seq("AHS-H"), Seq("AHSVH"), [0, 1, 2, -1, 4]),
            (Seq("AHSV-"), Seq("AHSVH"), [0, 1, 2, 3, -1]),
        )

        for s, p, e in problems:
            result = align_to_pdb(s, p, missing=-1, aligner=self.aligner)
            self.assertEqual(e, result)

    def test_get_pdb_indices(self):
        sequences = [Seq("AHSVH"),
                     Seq("AH-VH"),
                     Seq("A-SVH")]
        chain = Chain("A")
        residues = [
            Residue(0, resname="Ala", segid=0),
            Residue(0, resname="His", segid=1),
            Residue(0, resname="Ser", segid=2),
            Residue(0, resname="Val", segid=3),
            Residue(0, resname="His", segid=4),]

        for r in residues:
            chain.add(r)
        indices = get_pdb_indices(sequences, [chain], aligner=self.aligner)
        expected = np.array([[0, 1, 2, 3, 4]])
        self.assertTrue(np.all(indices == expected))

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
