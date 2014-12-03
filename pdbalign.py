"""Align FASTA sequences to PDB structures.

Outputs residue mapping and 3D coordinates.

Usage:
  pdb_align.py [options] <top> <fasta> <pdb>

Options:
  -t --threshold=<FLOAT>  Threshold for aligning to a structure [default: 0.8]
  -h --help  Display this screen.

"""

import os
import sys

from docopt import docopt

import numpy as np

from Bio.Seq import Seq
import Bio.PDB as biopdb
import Bio.SeqIO as seqio
from Bio.Alphabet import IUPAC
from Bio.SeqUtils import seq1

from BioExt.scorematrices import BLOSUM62
from BioExt.align import Aligner


def residue_center(r):
    """the mean of the residue's atom coordinates"""
    return np.vstack(list(a.coord for a in r)).mean(axis=0)


def make_chain_seq(chain):
    residues = list(chain.get_residues())
    seq_3letter = "".join((r.resname for r in residues))
    seq = seq1(seq_3letter)
    coords = list(residue_center(r) for r in residues)
    return seq, coords


if True:
    # args = docopt(__doc__)
    # fasta_file = args["<fasta>"]
    # pdb_file = args["<pdb>"]

    fasta_file = "./env_translated.fa"
    pdb_file = "4NCO.pdb"
    top = 3

    # read FASTA file
    sequences = list(seqio.parse(fasta_file, "fasta", alphabet=IUPAC.protein))

    # read PDB structures; use filename as id
    structure_id = os.path.basename(pdb_file).split('.')[0]
    parser = biopdb.PDBParser()
    structure = parser.get_structure(structure_id, pdb_file)

    chain_seqs = {}
    chain_coords = {}
    for chain in structure.get_chains():
        seq, coords = make_chain_seq(chain)
        chain_seqs[chain.id] = seq
        chain_coords[chain.id] = coords

    # align fasta sequences to PDB sequences
    aligner = Aligner(BLOSUM62.load(), do_codon=False)
    results = []
    for seq in sequences:
        alignments = []
        for chain_id, pdb_seq in chain_seqs.items():
            score, pdb_seq_aligned, seq_aligned = aligner(pdb_seq, seq)
            alignments.append((score, chain_id, pdb_seq_aligned, seq_aligned))
        alignments = sorted(alignments)[::-1][:top]
        results.append(alignments)

    # transfer coordinates to alignment
    all_coords = []
    for seq, alignments in zip(sequences, results):
        seq_coords = []
        for alignment in alignments:
            _, chain_id, pdb_seq_aligned, seq_aligned = alignment
            coords = chain_coords[chain_id]
            pdb_idx = 0
            result = []        
            for idx in range(len(pdb_seq_aligned)):
                if pdb_seq_aligned[idx] != "-" and seq_aligned[idx] != "-":
                    result.append(coords[pdb_idx])
                elif pdb_seq_aligned[idx] == "-" and seq_aligned[idx] != "-":
                    result.append(None)
                if pdb_seq_aligned[idx] != "-":
                    pdb_idx += 1
            seq_coords.append(result)
        all_coords.append(list(zip(*seq_coords)))

    # write output
