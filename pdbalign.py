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
    """Missing indices get `missing`"""
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


if True:
    # args = docopt(__doc__)
    # fasta_file = args["<fasta>"]
    # pdb_file = args["<pdb>"]

    fasta_file = "env.fa"
    pdb_file = "4NCO.pdb"
    chains = ['A', 'E', 'I']

    # read FASTA file
    sequences = list(seqio.parse(fasta_file, "fasta",
                                 alphabet=Gapped(IUPAC.unambiguous_dna)))
    sequences = list(translate(s) for s in sequences)

    # read PDB structures; use filename as id
    structure_id = os.path.basename(pdb_file).split('.')[0]
    parser = biopdb.PDBParser()
    structure = parser.get_structure(structure_id, pdb_file)

    chain_dict = {c: structure[0][c] for c in chains}
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

    # TODO: write output
