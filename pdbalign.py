"""Compute coordinates for MSA from a PDB file.

Usage:
  pdb_align.py [options] <fasta> <pdb> <chains>

Options:
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
from Bio.Alphabet import Gapped
from Bio.SeqUtils import seq1

from BioExt.scorematrices import BLOSUM62
from BioExt.align import Aligner
from BioExt.misc import translate


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

    fasta_file = "rhodopsin.fasta"
    pdb_file = "1U19.pdb"
    chains = ['A', 'B']

    # read FASTA file
    sequences = list(seqio.parse(fasta_file, "fasta",
                                 alphabet=Gapped(IUPAC.unambiguous_dna)))
    sequences = list(translate(s) for s in sequences)

    # read PDB structures; use filename as id
    structure_id = os.path.basename(pdb_file).split('.')[0]
    parser = biopdb.PDBParser()
    structure = parser.get_structure(structure_id, pdb_file)

    # convert each chain into a Bio.Seq
    chain_seqs = {}
    chain_coords = {}
    for c in chains:
        chain = structure[0][c]
        seq, coords = make_chain_seq(chain)
        chain_seqs[c] = seq
        chain_coords[c] = coords

    # align fasta sequences to PDB sequences
    aligner = Aligner(BLOSUM62.load(), do_codon=False)
    results = []
    for seq in sequences:
        alignments = []
        for c in chains:
            pdb_seq = chain_seqs[c]
            _, seq_aligned, pdb_seq_aligned = aligner(seq, pdb_seq)
            alignments.append((c, seq_aligned, pdb_seq_aligned))
        results.append(alignments)

    # transfer coordinates to alignment
    all_coords = []
    for seq, alignments in zip(sequences, results):
        seq_coords = []
        for alignment in alignments:
            chain_id, seq_aligned, pdb_seq_aligned = alignment
            coords = chain_coords[chain_id]
            pdb_idx = 0
            msa_idx = 0  # for inserting original gaps
            result = []
            for idx in range(len(seq_aligned)):
                try:
                    if seq[msa_idx] == "-":
                        result.append(np.nan * np.arange(3))
                        msa_idx += 1
                        continue
                except IndexError:
                    continue
                if pdb_seq_aligned[idx] != "-" and seq_aligned[idx] != "-":
                    result.append(coords[pdb_idx])
                elif pdb_seq_aligned[idx] == "-" and seq_aligned[idx] != "-":
                    result.append(np.nan * np.arange(3))
                if pdb_seq_aligned[idx] != "-":
                    pdb_idx += 1
                if seq_aligned[idx] != "-":
                    msa_idx += 1
            for i in range(msa_idx, len(seq)):
                result.append(np.nan * np.arange(3))
            seq_coords.append(result)
        all_coords.append(seq_coords)
    all_coords = np.array(all_coords)

    # take consensus coordinates

    # write output
