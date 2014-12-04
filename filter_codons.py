"""Filter codons from MSA if they are not in some fraction of
sequences.

Usage:
  filter_codons.py [options] <infile> <outfile>

Options:
  -t --threshold=<FLOAT>  Fraction for removing codon [default: 0.5]
  -h --help               Show this screen

"""

from functools import reduce
from operator import add

from docopt import docopt

from Bio.Align import MultipleSeqAlignment
from Bio import AlignIO

from BioExt.misc import translate

if __name__ == "__main__":
    args = docopt(__doc__)
    infile = args["<infile>"]
    outfile = args["<outfile>"]
    thresh = float(args["--threshold"])

    if thresh < 0 or thresh > 1:
        raise Exception("threshold must be between 0 and 1,"
                        " but got {}".format(thresh))

    aln = AlignIO.read(infile, "fasta")
    taln = MultipleSeqAlignment(list(translate(r) for r in aln))
    n_seqs = len(taln)
    percents = list(1 - taln[:, i].count('-') / n_seqs for i in range(len(taln[0])))
    keep = list(i for i, p in enumerate(percents) if p > thresh)

    trunc_aln = reduce(add, (aln[:, i * 3 : i * 3 + 3] for i in keep), aln[:, 0:0])

    AlignIO.write(trunc_aln, outfile, 'fasta')
