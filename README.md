About
-----

Calculate residue coordinates for each position in a multiple sequence
alignment.

Supports multiple coordinates for n-mers.


Dependencies
------------
- `docopt <https://github.com/docopt/docopt>`_
- `NumPy <http://www.numpy.org/>`_
- `SciPy <http://www.scipy.org/>`_
- `BioPython <http://biopython.org/wiki/Biopython>`_
- `BioExt <https://github.com/nlhepler/bioext>`_

Usage
-----

`pdbalign.py <fasta> <pdb> <chains> <outfile>`

Coordinates should be comma-seperated and exactly match the PDB
file. Example: `A,E,I`.
