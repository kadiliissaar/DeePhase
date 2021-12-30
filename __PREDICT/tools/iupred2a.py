#!/usr/bin/python3

import sys
import os
import iupred2a_lib

PATH = os.path.dirname(os.path.realpath(__file__))

help_msg = """Usage: {} (options) (seqfile) (iupred type)
\tAvailable types: \"long\", \"short\", \"glob\"

Options
\t-d str   -   Location of data directory (default='./')
\t-a       -   Enable ANCHOR2 predition\n""".format(sys.argv[0])
if len(sys.argv) < 2:
    sys.exit(help_msg)
if not os.path.isfile(sys.argv[-2]):
    sys.exit('Input sequence file not found at {}!\n{}'.format(sys.argv[-2], help_msg))
if not os.path.isdir(PATH):
    sys.exit('Data directory not found at {}!\n{}'.format(PATH, help_msg))
if '-d' in sys.argv:
    PATH = sys.argv[sys.argv.index('-d') + 1]
    if not os.path.isdir(os.path.join(PATH, 'data')):
        sys.exit('Data directory not found at {}!\n{}'.format(PATH, help_msg))

if sys.argv[-1] not in ['short', 'long', 'glob']:
    sys.exit('Wrong iupred2 option {}!\n{}'.format(sys.argv[-1], help_msg))

sequence = iupred2a_lib.read_seq(sys.argv[-2])
iupred2_result = iupred2a_lib.iupred(sequence, sys.argv[-1])
if '-a' in sys.argv:
    if sys.argv[-1] == 'long':
        anchor2_res = iupred2a_lib.anchor2(sequence)
    else:
        anchor2_res = iupred2a_lib.anchor2(sequence)
print("""# IUPred2A: context-dependent prediction of protein disorder as a function of redox state and protein binding
# Balint Meszaros, Gabor Erdos, Zsuzsanna Dosztanyi
# Nucleic Acids Research 2018;46(W1):W329-W337.
#
# Prediction type: {}
# Prediction output""".format(sys.argv[-1]))
if sys.argv[-1] == 'glob':
    print(iupred2_result[1])
    iupred2_result = iupred2_result[0]
if '-a' in sys.argv:
    print("# POS\tRES\tIUPRED2\tANCHOR2")
else:
    print("# POS\tRES\tIUPRED2")

for pos, residue in enumerate(sequence):
    print('{}\t{}\t{:.4f}'.format(pos + 1, residue, iupred2_result[pos]), end="")
    if '-a' in sys.argv:
        print("\t{:.4f}".format(anchor2_res[pos]), end="")
    print()
