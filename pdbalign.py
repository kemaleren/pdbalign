#!/usr/bin/env python

"""Compute coordinates for positions in MSA from a PDB file.

Each sequence is aligned to PDB chains. The residue coordinates are
mapped back to the MSA. Each position gets the most common
coordinates.

Output has one line per position in the MSA. Each line contains the
coordinates for the given position for all requested chains. If
coordinates are not available, each coordinate gets "nan".

Chains should be given as comma-separated letters.

Example:
  pdb_align.py A,E,I <fasta> <pdb> <outfile>

Usage:
  pdb_align.py <chains> <fasta> <pdb> <outfile>

Options:
  -h --help  Display this screen.

"""

import os
import sys

from docopt import docopt

import numpy as np
from scipy.stats import mode

from Bio.Seq import Seq
import Bio.PDB as biopdb
import Bio.SeqIO as seqio
from Bio.Alphabet import IUPAC
from Bio.Alphabet import Gapped
from Bio.SeqUtils import seq1

from BioExt.scorematrices import BLOSUM62
from BioExt.align import Aligner
from BioExt.misc import translate


def get_chain_seq(chain):
    """Combine residues into a single Bio.Seq."""
    seq = "".join((r.resname for r in chain.get_residues()))
    return seq1(seq)


def residue_center(r):
    """the mean of the residue's atom coordinates"""
    return np.vstack(list(a.coord for a in r)).mean(axis=0)


def get_chain_coords(chain):
    """Residue coordinates."""
    result = (list(residue_center(r) for r in chain.get_residues()))
    # append dummy coordinates
    result.append([np.nan] * 3)
    return np.vstack(result)


def transfer_pdb_indices(seq_aligned, pdb_seq_aligned, missing=-1):
    """Returns a pdb index for each index of the original MSA.

    Unaligned indices (either in original MSA or in PDB alignment) get
    `missing`.

    """
    pdb_idx = 0  # pointer to position in pdb chain sequence
    msa_idx = 0  # pointer to position in original MSA coordinates
    result = []
    # handle leading gaps
    while seq[msa_idx] == "-":
        result.append(missing)
        msa_idx += 1
    # handle alignment
    for idx in range(len(seq_aligned)):
        try:
            if seq[msa_idx] == "-":
                result.append(missing)
                msa_idx += 1
                continue
        except IndexError:
            continue
        if pdb_seq_aligned[idx] != "-" and seq_aligned[idx] != "-":
            result.append(pdb_idx)
        elif pdb_seq_aligned[idx] == "-" and seq_aligned[idx] != "-":
            result.append(missing)
        if pdb_seq_aligned[idx] != "-":
            pdb_idx += 1
        if seq_aligned[idx] != "-":
            msa_idx += 1
    # handle trailing gaps
    for i in range(msa_idx, len(seq)):
        result.append(missing)
    return result


if __name__ == "__main__":
    args = docopt(__doc__)
    chains = args["<chains>"].split(",")
    fasta_file = args["<fasta>"]
    pdb_file = args["<pdb>"]
    outfile = args["<outfile>"]

    # read FASTA file
    sequences = list(seqio.parse(fasta_file, "fasta",
                                 alphabet=Gapped(IUPAC.unambiguous_dna)))
    sequences = list(translate(s) for s in sequences)

    # read PDB structures; use filename as id
    structure_id = os.path.basename(pdb_file).split('.')[0]
    parser = biopdb.PDBParser()
    model = parser.get_structure(structure_id, pdb_file)[0]

    for c in chains:
        if c not in model:
            raise Exception("Chain '{}' not found. Candidates: {}".format(
                c, sorted(model.child_dict.keys())))

    chain_dict = {c: model[c] for c in chains}
    chain_seqs = {c: get_chain_seq(chain) for c, chain in chain_dict.items()}
    chain_coords = {c: get_chain_coords(chain) for c, chain in chain_dict.items()}

    # align and transfer coordinates to alignment
    aligner = Aligner(BLOSUM62.load(), do_codon=False)
    pdb_index_array = []
    for seq in sequences:
        indices = []
        for c in chains:
            _, seq_aligned, pdb_seq_aligned = aligner(seq, chain_seqs[c])
            indices.append(transfer_pdb_indices(seq_aligned, pdb_seq_aligned, missing=-1))
        pdb_index_array.append(indices)

    # collapse to consensus coordinates
    pdb_index_array = np.array(pdb_index_array, dtype=np.int)
    pdb_index_array = pdb_index_array.transpose(1, 2, 0)
    modes, _ = mode(pdb_index_array, axis=2)
    modes = modes.squeeze().astype(np.int)

    # get coordinates from pdb indices
    coord_array = np.array(list(chain_coords[c][modes[i]]
                                for i, c in enumerate(chains)))

    # write output
    n_posns = coord_array.shape[1]
    coord_array = coord_array.transpose(1, 0, 2).reshape((n_posns, -1))
    with open(outfile, 'w') as f:
        header = "\t".join("{}_{}".format(chain, coord)
                           for chain in chains for coord in "xyz")
        f.write(header)
        f.write("\n")
        for line in coord_array:
            f.write("\t".join(map(str, line)))
            f.write("\n")
