About
-----

Calculate residue coordinates for each position in a multiple sequence
alignment.

Supports multiple coordinates for n-mers.


Dependencies
------------
- [docopt](<https://github.com/docopt/docopt>)
- [NumPy](<http://www.numpy.org/>)
- [SciPy](<http://www.scipy.org/>)
- [BioPython](<http://biopython.org/wiki/Biopython>)
- [BioExt](<https://github.com/nlhepler/bioext>)

Usage
-----

`pdbalign.py <fasta> <pdb> <chains> <outfile>`

Coordinates should be comma-seperated and exactly match the PDB
file. Example: `A,E,I`.
