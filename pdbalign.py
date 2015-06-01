#!/usr/bin/env python

"""Given a multiple sequence alignment and a PDB structure, assign 3D
coordinates to columns in the MSA and compute a distance matrix.

The sequences in the MSA are all assumed to be in-frame. They are
directly translated to a protein sequence before alignment.

Each chain in the PDB is aligned to each sequence in the translated
MSA, using the BLOSUM62 scoring matrix. The residue coordinates are
mapped back to the MSA. For each chain, each column the MSA is
assigned the coordinates of the consensus residue.

The pairwise distance matrix, in angstroms, is then computed for all
columns. Positions without coordinates get a default distance from
their linear neighbors, and are disconnected from any other
positions. If there are multiple chains, the minimum distance is used.

Output:
  - coordinates for each position and each chain: <outname>.coords
  - distance matrix: <outname>.dist
  - translated alignment with chain sequences: <outname>.translated.chains.fasta

Chains should be seperated by commas. Example: A,E,I

Usage:
  pdbalign.py [options] <fasta> <pdb> <chains> <outname>

Options:
  --default-dist=<FLOAT>   Distance, in angstroms, to assign to linear
                           neighbors. [default: 5]
  --disconnected=<FLOAT>   Distance, in angstroms, to assign to disconnected nodes.
                           May be a float or 'inf' [default: inf]
  --delimiter=<STRING>     Delimiter for output [default:  ]
  -h --help                Print this screen

"""

import os
import sys
from collections import Counter

from docopt import docopt

import numpy as np
from numpy.linalg import norm

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import Bio.PDB as biopdb
import Bio.SeqIO as seqio
from Bio.Alphabet import IUPAC
from Bio.Alphabet import Gapped
from Bio.SeqUtils import seq1

from BioExt.scorematrices import BLOSUM62
from BioExt.align import Aligner
from BioExt.misc import translate


def chain_to_seq(chain):
    """Combine residues into a single Bio.Seq."""
    seq = "".join((r.resname for r in chain.get_residues()))
    return seq1(seq)


def residue_center(r):
    """mean of the residue's atom coordinates"""
    return np.vstack(list(a.coord for a in r)).mean(axis=0)


def get_chain_coords(chain):
    """Mean coordinates for each residue in the chain.

    The last entry is a dummy because `-1` is used as the index for
    positions that are missing coordinates. So `result[-1]` gives the
    coordinate `[np.nan, np.nan, np.nan]`.

    """
    result = (list(residue_center(r) for r in chain.get_residues()))
    result.append([np.nan] * 3)
    return np.vstack(result)


def align_and_index(seq, target, missing=-1, aligner=None):
    """Align sequences `seq` and `target` and return indices into `target`.

    Gaps get `missing` value.

    >>> from Bio.Seq import Seq; align_and_index(Seq("AHSVH"), Seq("AHVH"))
    [0, 1, -1, 2, 3]

    """
    if aligner is None:
        # TODO: support different scoring matrices and gap penalties
        # on command line
        aligner = Aligner(BLOSUM62.load(), do_codon=False)
    _, seq_aligned, target_aligned = aligner(seq, target)
    seq_idx = 0
    target_idx = 0
    result = []
    # handle leading gaps
    while seq[seq_idx] == "-":
        result.append(missing)
        seq_idx += 1
    # handle alignment
    for idx in range(len(seq_aligned)):
        try:
            if seq[seq_idx] == "-":
                result.append(missing)
                seq_idx += 1
                if target_aligned[idx] != "-":
                    target_idx += 1
                continue
        except IndexError:
            continue
        if target_aligned[idx] != "-" and seq_aligned[idx] != "-":
            result.append(target_idx)
        elif target_aligned[idx] == "-" and seq_aligned[idx] != "-":
            result.append(missing)
        if target_aligned[idx] != "-":
            target_idx += 1
        if seq_aligned[idx] != "-":
            seq_idx += 1
    # handle trailing gaps
    for i in range(seq_idx, len(seq)):
        result.append(missing)
    return result


def consensus(iterable, flag):
    """Returns `flag` if there is no consensus."""
    c = Counter(iterable)
    if len(c) == 0:
        return flag
    if len(c) == 1:
        return c.most_common()[0][0]
    first, second = c.most_common(2)
    if first[1] > second[1]:
        return first[0]
    return flag


def align_chains_msa(sequences, chains, missing=-1, aligner=None):
    """For each MSA column, assign a consensus index into each chain.

    The result is a np.ndarray with shape (n_chains, n_columns).

    """
    chain_seqs = list(chain_to_seq(c) for c in chains)
    # align to PDB; get pdb indices for MSA coordinates
    pdb_index_array = []
    for seq in sequences:
        indices = []
        for chain_seq in chain_seqs:
            i = align_and_index(seq, chain_seq, missing=missing, aligner=aligner)
            indices.append(i)
        pdb_index_array.append(indices)

    # collapse to consensus
    pdb_index_array = np.array(pdb_index_array, dtype=np.int)
    pdb_index_array = pdb_index_array.transpose(1, 2, 0)
    f = lambda it: consensus((i for i in it if i != missing), flag=missing)
    modes = np.apply_along_axis(f, axis=2, arr=pdb_index_array)
    modes = modes.astype(np.int)
    return modes


def make_coords(idx_array, chains):
    """Get coordinates from pdb indices.

    result shape: [n_columns, n_chains, 3].

    """
    chain_coords = list(get_chain_coords(c) for c in chains)
    coords = np.array(list(c[idx_array[i]]
                           for i, c in enumerate(chain_coords)))
    coords = coords.transpose(1, 0, 2)
    return coords


def write_coords(outfile, coords, chains):
    """Write the coordinates to a file.

    Each line contains all the coordinates for each position.

    """
    n_posns = coords.shape[0]
    coords = coords.reshape((n_posns, -1))
    with open(outfile, 'w') as f:
        header = "\t".join("{}_{}".format(chain.id, coord)
                           for chain in chains for coord in "xyz")
        f.write(header)
        f.write("\n")
        for line in coords:
            f.write("\t".join(map(lambda x: "%.2f" % x, line)))
            f.write("\n")


def compute_distance_matrix(coords, default_dist, disconnected_dist=np.inf):
    n_posns = coords.shape[0]
    n_chains = coords.shape[1]
    dists = np.empty((n_posns, n_posns))
    dists[:] = np.inf
    np.fill_diagonal(dists, 0)
    # find neighbors in 3D space
    for i in range(n_posns):
        for j in range(i + 1, n_posns):
            d = np.inf
            for chain1 in range(n_chains):
                for chain2 in range(n_chains):
                    coord1 = coords[i, chain1]
                    coord2 = coords[j, chain2]
                    new_d = norm(coord1 - coord2)
                    if np.isnan(new_d):
                        continue
                    d = min(new_d, d)
            dists[i, j] = dists[j, i] = d
    # connect linear neighbors
    for i in range(n_posns - 1):
        if np.isinf(dists[i, i + 1]):
            dists[i, i + 1] = dists[i + 1, i] = default_dist
    dists[np.isinf(dists)] = disconnected_dist
    return dists


def aligned_chains(idx_array, chains, chain_ids):
    result = []
    for i, c in enumerate(chain_ids):
        chain_seq = chain_to_seq(chains[i])
        indices = idx_array[i]
        final_seq = ''.join(list(chain_seq[idx] if idx != -1 else "-" for idx in indices))
        result.append(SeqRecord(Seq(final_seq), id="chain {}".format(c)))
    return result


def main(fasta_file, pdb_file, chain_ids, outname, default_dist,
         disconnected_dist, delimiter):
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
    idx_array = align_chains_msa(sequences, chains)
    coords = make_coords(idx_array, chains)
    dist_matrix = compute_distance_matrix(coords, default_dist, disconnected_dist)
    full_alignment = sequences + aligned_chains(idx_array, chains, chain_ids)
    write_coords(outname + ".coords", coords, chains)
    np.savetxt(outname + ".dist", dist_matrix, fmt="%.2f", delimiter=delimiter)
    seqio.write(full_alignment, outname + ".translated.chains.fasta", 'fasta')


if __name__ == "__main__":
    args = docopt(__doc__)
    fasta_file = args["<fasta>"]
    pdb_file = args["<pdb>"]
    chain_ids = args["<chains>"].split(",")
    outname = args["<outname>"]
    default_dist = float(args["--default-dist"])
    disconnected_dist = args["--disconnected"]
    delimiter = args["--delimiter"]

    if disconnected_dist == "inf":
        disconnected_dist = np.inf
    else:
        disconnected_dist = float(disconnected_dist)

    main(fasta_file, pdb_file, chain_ids, outname, default_dist,
         disconnected_dist, delimiter)
