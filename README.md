About
-----

Given a multiple sequence alignment and a PDB structure, assign 3D
coordinates to columns in the MSA and compute a distance matrix for
all column pairs.

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


Dependencies
------------
- [NumPy](<http://www.numpy.org/>)
- [BioPython](<http://biopython.org/wiki/Biopython>)
- [BioExt](<https://github.com/nlhepler/bioext>)


Usage
-----

`pdbalign.py <fasta> <pdb> <chains> <outfile>`

Chains should be comma-seperated and exactly match the PDB
file. Example: `A,E,I`.


Example
-------

Some example data are included in the `example` directory, which
contains the following files:

- `ebola.fasta`: multiple sequence alignment of Ebola GP sequences.
- `3CSY.pdb`: crystal structure of the trimeric prefusion Ebola virus
  glycoprotein (taken from the
  [PDB database](<http://www.rcsb.org/pdb/explore/explore.do?structureId=3csy>))

We want to include chains I, K, M, which represent the three GP1
monomers, and chains J, L, N, which represent the GP2 monomers. The
script is run using the following command:

    pdbalign.py ebola.fasta 3CSY.pdb I,K,M,J,L,N ebola

The run creates two files:

- `ebola.coords`: one line per column in the MSA, with xyz coordinates for each chain.
- `ebola.dist`: minimum distance between each pair of columns.

