#!/usr/bin/env python

"""Compute distance matrix for positions in a multiple sequence
alignment.

Each sequence is aligned to multiple PDB chains. The residue
coordinates are mapped back to the MSA. Each position gets the most
common coordinates.

Positions are then connected to neighbors within a certain
radius. Positions without coordinates are connected to their linear
neighbors with a default distance.

Output is a human-readable text file of the distance matrix.

Chains should be given as comma-separated letters. Example: A,E,I

Usage:
  pdbalign.py [options] <fasta> <pdb> <chains> <outfile>

Options:
  -r --radius=<FLOAT>  Radius for connecting neighbors [default: 20]
  --default-distance=<FLOAT>  Distance to assign to linear
                              neighbors [default: 5]
  --delimiter=<STRING>  Delimiter for output [default: \t]
  -h --help  Print this screen

"""

import os
import sys

from docopt import docopt

import numpy as np
from numpy.linalg import norm
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
    coordinates of chain sequence.

    Result is a np.ndarray with shape (n_indices, n_chains, 3).

    """
    chain_seqs = list(get_chain_seq(c) for c in chains)
    # align to PDB; get pdb indices for MSA coordinates
    aligner = Aligner(BLOSUM62.load(), do_codon=False)
    pdb_index_array = []
    for seq in sequences:
        indices = []
        for chain_seq in chain_seqs:
            indices.append(align_to_pdb(seq, chain_seq, missing=-1))
        pdb_index_array.append(indices)

    # collapse to consensus
    pdb_index_array = np.array(pdb_index_array, dtype=np.int)
    pdb_index_array = pdb_index_array.transpose(1, 2, 0)
    modes, _ = mode(pdb_index_array, axis=2)
    modes = modes.squeeze().astype(np.int)
    return modes


def write_coord_array(outfile, coord_array, chains):
    coord_array = coord_array.reshape((n_posns, -1))
    n_posns = coord_array.shape[0]
    with open(outfile, 'w') as f:
        header = "\t".join("{}_{}".format(chain, coord)
                           for chain in chains for coord in "xyz")
        f.write(header)
        f.write("\n")
        for line in coord_array:
            f.write("\t".join(map(str, line)))
            f.write("\n")


def compute_distance_matrix(coord_array, radius, default_dist):
    """make neighbor distance matrix"""
    n_posns = coord_array.shape[0]
    n_chains = coord_array.shape[1]
    dists = np.empty((n_posns, n_posns))
    dists[:] = np.inf
    np.fill_diagonal(dists, 0)
    # find neighbors in 3D space
    for i in range(n_posns):
        for j in range(i + 1, n_posns):
            d = np.inf
            for chain1 in range(n_chains):
                for chain2 in range(n_chains):
                    coord1 = coord_array[i, chain1]
                    coord2 = coord_array[j, chain2]
                    new_d = norm(coord1 - coord2)
                    if np.isnan(new_d):
                        continue
                    d = min(new_d, d)
            if d < radius:
                dists[i, j] = dists[j, i] = d
    # connect linear neighbors
    for i in range(n_posns - 1):
        if np.isinf(dists[i, i + 1]):
            dists[i, i + 1] = dists[i + 1, i] = default_dist
    dists[np.isinf(dists)] = -1
    return dists


if __name__ == "__main__":
    args = docopt(__doc__)
    fasta_file = args["<fasta>"]
    pdb_file = args["<pdb>"]
    chain_ids = args["<chains>"].split(",")
    outfile = args["<outfile>"]
    radius = float(args["--radius"])
    default_dist = float(args["--default-distance"])
    delimiter = args["--delimiter"]

    # read FASTA file
    sequences = list(seqio.parse(fasta_file, "fasta",
                                 alphabet=Gapped(IUPAC.unambiguous_dna)))
    sequences = list(translate(s) for s in sequences)

    # read PDB structures; use filename as id
    structure_id = os.path.basename(pdb_file).split('.')[0]
    parser = biopdb.PDBParser()
    model = parser.get_structure(structure_id, pdb_file)[0]

    # check that all chains are present
    for c in chain_ids:
        if c not in model:
            raise Exception("Chain '{}' not found. Candidates: {}".format(
                c, sorted(model.child_dict.keys())))

    chains = list(model[c] for c in chain_ids)

    # do alignment and get coordinates
    pdb_idx_array = get_pdb_coords(sequences, model, chains)
    # get coordinates from pdb indices
    chain_coords = list(get_chain_coords(c) for c in chains)
    coord_array = np.array(list(c[pdb_idx_array[i]]
                                for i, c in enumerate(chain_coords)))
    coord_array = coord_array.transpose(1, 0, 2)

    dist_matrix = compute_distance_matrix(coord_array, radius, default_dist)
    np.savetxt(outfile, dist_matrix, delimiter=delimiter)
