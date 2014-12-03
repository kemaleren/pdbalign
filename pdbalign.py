#!/usr/bin/env python

"""Compute coordinates for positions in MSA from a PDB file.

Each sequence is aligned to PDB chains. The residue coordinates are
mapped back to the MSA. Each position gets the most common
coordinates.

Output has one line per position in the MSA. Each line contains the
coordinates for the given position for all requested chains. If
coordinates are not available, each coordinate gets "nan".

Chains should be given as comma-separated letters. Example: A,E,I

Usage:
  pdbalign.py <fasta> <pdb> <chains> <outfile>

"""

import os
import sys

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
    """mean of the residue's atom coordinates"""
    return np.vstack(list(a.coord for a in r)).mean(axis=0)


def get_chain_coords(chain):
    """mean coordinates for each residue in the chain, with dummy at end"""
    result = (list(residue_center(r) for r in chain.get_residues()))
    # append dummy coordinates
    result.append([np.nan] * 3)
    return np.vstack(result)


def align_to_pdb(seq, pdb_seq, missing=-1):
    """Align sequence to PDB chain.

    Returns a PDB index for each position of the original MSA.

    Gaps (either in original MSA or in PDB alignment) get `missing` value.

    """
    aligner = Aligner(BLOSUM62.load(), do_codon=False)
    _, seq_aligned, pdb_seq_aligned = aligner(seq, pdb_seq)
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


def get_pdb_coords(sequences, model, chains):
    """Align each sequence of MSA against chains in the model. Returns
    coordinates.

    Result is a np.ndarray with shape (n_indices, n_chains, 3).

    """
    chain_dict = {c: model[c] for c in chains}
    chain_seqs = {c: get_chain_seq(chain)
                  for c, chain in chain_dict.items()}
    chain_coords = {c: get_chain_coords(chain)
                    for c, chain in chain_dict.items()}

    # align to PDB; get pdb indices for MSA coordinates
    aligner = Aligner(BLOSUM62.load(), do_codon=False)
    pdb_index_array = []
    for seq in sequences:
        indices = []
        for c in chains:
            indices.append(align_to_pdb(seq, chain_seqs[c], missing=-1))
        pdb_index_array.append(indices)

    # collapse to consensus
    pdb_index_array = np.array(pdb_index_array, dtype=np.int)
    pdb_index_array = pdb_index_array.transpose(1, 2, 0)
    modes, _ = mode(pdb_index_array, axis=2)
    modes = modes.squeeze().astype(np.int)

    # get coordinates from pdb indices
    coord_array = np.array(list(chain_coords[c][modes[i]]
                                for i, c in enumerate(chains)))
    return coord_array


def write_coord_array(outfile, coord_array, chains):
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


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 4:
        sys.stderr.write("Usage: pdbalign.py <fasta> <pdb> <chains> <outfile>\n")
        sys.exit(1)
    fasta_file = args[0]
    pdb_file = args[1]
    chains = args[2].split(",")
    outfile = args[3]

    # read FASTA file
    sequences = list(seqio.parse(fasta_file, "fasta",
                                 alphabet=Gapped(IUPAC.unambiguous_dna)))
    sequences = list(translate(s) for s in sequences)

    # read PDB structures; use filename as id
    structure_id = os.path.basename(pdb_file).split('.')[0]
    parser = biopdb.PDBParser()
    model = parser.get_structure(structure_id, pdb_file)[0]

    # check that all chains are present
    for c in chains:
        if c not in model:
            raise Exception("Chain '{}' not found. Candidates: {}".format(
                c, sorted(model.child_dict.keys())))

    # do alignment and get coordinates
    coord_array = get_pdb_coords(sequences, model, chains)
    write_coord_array(outfile, coord_array, chains)
