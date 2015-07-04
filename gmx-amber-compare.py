#!/usr/bin/env python

import os, sys, re
import numpy as np
import copy
from molecule import Molecule
from nifty import _exec, printcool_dictionary
import parmed
from collections import OrderedDict
from atest import Calculate_GMX, Calculate_AMBER, interpret_mdp
import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as u
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ff', type=str, default='fb15', choices=['fb15', 'fb15ni', 'ildn'], help='Select a force field to compare between Gromacs and AMBER.')
args, sys.argv = parser.parse_known_args(sys.argv)

# Disable Gromacs backup files
os.environ["GMX_MAXBACKUP"] = "-1"

# Sections in the .rtp file not corresponding to a residue
rtpknown = ['atoms', 'bonds', 'bondedtypes', 'dihedrals', 'impropers']

# Mapping of amino acid atom names to new atom classes.  Mostly new
# atom classes for beta carbons but a few new gamma carbons are
# defined.  They are named using the number "6" (for carbon) and the
# one-letter amino acid code.  A few exceptions in the case of alternate
# protonation states.
NewAC = {"SER":{"CB":"6S"}, "THR":{"CB":"6T", "CG2":"6t"}, "LEU":{"CB":"6L"},
         "VAL":{"CB":"6V"}, "ILE":{"CB":"6I", "CG2":"6i"}, "ASN":{"CB":"6N"},
         "GLN":{"CB":"6Q", "CG":"6q"}, "ARG":{"CB":"6R"}, "HID":{"CB":"6H"},
         "HIE":{"CB":"6h"}, "HIP":{"CB":"6+"}, "TRP":{"CB":"6W"},
         "TYR":{"CB":"6Y"}, "PHE":{"CB":"6F"}, "GLU":{"CB":"6E", "CG":"6e"},
         "ASP":{"CB":"6D"}, "LYS":{"CB":"6K"}, "LYN":{"CB":"6k"},
         "PRO":{"CB":"6P"}, "CYS":{"CB":"6C"}, "CYM":{"CB":"6c"},
         "MET":{"CB":"6M"}, "ASH":{"CB":"6d"}, "GLH":{"CB":"6J", "CG":"6j"}}
for k, v in NewAC.items():
    NewAC["C"+k] = v
    NewAC["N"+k] = v

def RTPAtomNames(rtpfnm):
    answer = OrderedDict()
    for line in open(rtpfnm).readlines():
        s = line.split()
        # Do nothing for an empty line or a comment line.
        if len(s) == 0 or re.match('^ *;',line): 
            section = 'None'
            continue
        line = line.split(';')[0]
        s = line.split()
        if re.match('^ *\[.*\]',line):
            # Makes a word like "atoms", "bonds" etc.
            section = re.sub('[\[\] \n]','',line.strip())
            if section not in rtpknown:
                resname = section
        elif section == 'atoms':
            answer.setdefault(resname, []).append((s[0], s[1], float(s[2])))
    return answer
# Gromacs .rtp file with residue definitions
gmxrtp = "/home/leeping/opt/gromacs/share/gromacs/top/amberfb15.ff/aminoacids.rtp"
# Gromacs atom names in residues
gmx_resatoms = RTPAtomNames(gmxrtp)

# Load AMBER OFF libraries for amino acids
amb_off = ['all_aminofb15.lib', 'all_aminoctfb15.lib', 'all_aminontfb15.lib']
amb_resatoms = OrderedDict()
for fin in amb_off:
    AOff = parmed.modeller.offlib.AmberOFFLibrary.parse(os.path.join(os.environ['AMBERHOME'], 'dat', 'leap', 'lib', fin))
    for k, v in AOff.items():
        for a in v.atoms:
            amb_resatoms.setdefault(k, []).append((a.name, a.type, a.charge))

gmx_resnames = set(gmx_resatoms.keys())
amb_resnames = set(amb_resatoms.keys())
print "GMX resnames not in AMBER:", ' '.join(sorted(list(gmx_resnames.difference(amb_resnames))))
print "AMBER resnames not in GMX:", ' '.join(sorted(list(amb_resnames.difference(gmx_resnames))))

gmx_amb_amap = OrderedDict()
amb_gmx_amap = OrderedDict()
for resname in sorted(list(gmx_resnames.intersection(amb_resnames))):
    for gmx_atom, amb_atom in zip(gmx_resatoms[resname], amb_resatoms[resname]):
        gmx_aname, gmx_atype, gmx_acharge = gmx_atom
        # Gromacs doesn't use new AMBER-FB15 atom types, so figure it out from the NewAC dictionary
        gmx_atype = NewAC.get(resname, {}).get(gmx_aname, gmx_atype)
        amb_aname, amb_atype, amb_acharge = amb_atom
        # The following code for mapping GMX -> AMBER atom names
        # assumes that the atoms in the GMX .rtp and AMBER .off
        # dictionaries come in the same order.  This is a sanity check
        # that checks to see if the atom type and atomic charge are
        # indeed the same.
        if gmx_atype != amb_atype:
            print "Atom types don't match for", resname, gmx_atom, amb_atom
            raise RuntimeError
        if gmx_acharge != amb_acharge:
            print "Atomic charges don't match for", resname, gmx_atom, amb_atom
            raise RuntimeError
        # Print the atoms that are renamed
        # if gmx_aname != amb_aname:
        #     print "%s (AMBER) %s <-> %s (GMX)" % (resname, amb_aname, gmx_aname)
        gmx_amb_amap.setdefault(resname, OrderedDict())[gmx_aname] = amb_aname
        amb_gmx_amap.setdefault(resname, OrderedDict())[amb_aname] = gmx_aname

amber_atomnames = OrderedDict([(k, set(v.keys())) for k, v in amb_gmx_amap.items()])
max_reslen = max([len(v) for v in amber_atomnames.values()])

# Begin with an AMBER-compatible PDB file
# Please ensure by hand :)
pdb = Molecule(sys.argv[1], build_topology=False)
gmx_pdb = copy.deepcopy(pdb)
del gmx_pdb.Data['elem']

# Convert to a GROMACS-compatible PDB file
# This mainly involves renaming atoms and residues,
# notably hydrogen names.

# List of atoms in the current residue
anameInResidue = []
anumInResidue = []
for i in range(gmx_pdb.na):
    # Rename the ions residue names to be compatible with Gromacs
    if gmx_pdb.resname[i] == 'Na+':
        gmx_pdb.resname[i] = 'NA'
    elif gmx_pdb.resname[i] == 'Cl-':
        gmx_pdb.resname[i] = 'CL'
    elif gmx_pdb.resname[i] in ['HOH','WAT']:
        gmx_pdb.resname[i] = 'SOL'
        if gmx_pdb.atomname[i] == 'O':
            gmx_pdb.atomname[i] = 'OW'
        elif gmx_pdb.atomname[i] == 'H1':
            gmx_pdb.atomname[i] = 'HW1'
        elif gmx_pdb.atomname[i] == 'H2':
            gmx_pdb.atomname[i] = 'HW2'
        else:
            print "Atom number %i unexpected atom name in water molecule: %s" % gmx_pdb.atomname[i]
            raise RuntimeError
    # Rename the ion atom names
    if gmx_pdb.atomname[i] == 'Na+':
        gmx_pdb.atomname[i] = 'Na'
    elif gmx_pdb.atomname[i] == 'Cl-':
        gmx_pdb.atomname[i] = 'Cl'
    # A peculiar thing sometimes happens with a hydrogen being named "3HG1" where in fact it shold be "HG13"
    if gmx_pdb.atomname[i][0] in ['1','2','3'] and gmx_pdb.atomname[i][1] == 'H':
        gmx_pdb.atomname[i] = gmx_pdb.atomname[i][1:] + gmx_pdb.atomname[i][0]
    anumInResidue.append(i)
    anameInResidue.append(gmx_pdb.atomname[i])
    if (i == (gmx_pdb.na - 1)) or ((gmx_pdb.chain[i+1], gmx_pdb.resid[i+1]) != (gmx_pdb.chain[i], gmx_pdb.resid[i])):
        # Try to look up the residue in the amber_atomnames template
        mapped = False
        for k, v in amber_atomnames.items():
            if set(anameInResidue) == v:
                mapped = True
                break
        if mapped:
            for anum, aname in zip(anumInResidue, anameInResidue):
                gmx_pdb.atomname[anum] = amb_gmx_amap[k][aname]
        anameInResidue = []
        anumInResidue = []
    
# Write the Gromacs-compatible PDB file
gmx_pdbfnm = os.path.splitext(sys.argv[1])[0]+"-gmx.pdb"
gmx_pdb.write(gmx_pdbfnm)

gmx_ffnames = {'fb15':'amberfb15', 'fb15ni':'amberfb15ni', 'ildn':'amber99sb-ildn'}
amber_ffnames = {'fb15' : 'fb15', 'fb15ni':'fb15ni', 'ildn':'ff99SBildn'}
# Set up the system in Gromacs
_exec("pdb2gmx -ff %s -f %s" % (gmx_ffnames[args.ff], gmx_pdbfnm), stdin="1\n")

# Set up the system in AMBER
amb_outpdbfnm = os.path.splitext(sys.argv[1])[0]+"-amb.pdb"

with open("stage.leap", 'w') as f:
    print >> f, """source leaprc.{choice}
pdb = loadpdb {pdbin}
savepdb pdb {pdbout}
saveamberparm pdb prmtop inpcrd
quit
""".format(choice = amber_ffnames[args.ff], 
           pdbin = sys.argv[1], pdbout = amb_outpdbfnm)

_exec("tleap -f stage.leap")

# Load the GROMACS output coordinate file
gmx_gro = Molecule("conf.gro", build_topology=False)

# Build a mapping of (atom numbers in original PDB -> atom numbers in Gromacs .gro)
anumInResidue = []
anameInResidue_pdb = []
anameInResidue_gro = []
# For an atom number in the original .pdb file, this is the corresponding atom number in the .gro file
anumMap_gro = []
for i in range(gmx_pdb.na):
    anumInResidue.append(i)
    anameInResidue_pdb.append(gmx_pdb.atomname[i])
    anameInResidue_gro.append(gmx_gro.atomname[i])
    if (i == (gmx_pdb.na - 1)) or ((gmx_pdb.chain[i+1], gmx_pdb.resid[i+1]) != (gmx_pdb.chain[i], gmx_pdb.resid[i])):
        if set(anameInResidue_pdb) != set(anameInResidue_gro):
            print set(anameInResidue_pdb).symmetric_difference(set(anameInResidue_gro))
            print "GROMACS PDB: Atom names do not match for residue %i (%s)" % (gmx_pdb.resid[i], gmx_pdb.resname[i])
            raise RuntimeError
        for anum, aname in zip(anumInResidue, anameInResidue_pdb):
            anumMap_gro.append(anumInResidue[anameInResidue_gro.index(aname)])
        anumInResidue = []
        anameInResidue_pdb = []
        anameInResidue_gro = []

amb_pdb = Molecule(amb_outpdbfnm, build_topology=False)
anumInResidue = []
anameInResidue_pdb = []
anameInResidue_amb = []
# For an atom number in the original .pdb file, this is the corresponding atom number in the .gro file
anumMap_amb = []
for i in range(pdb.na):
    # A peculiar thing sometimes happens with a hydrogen being named "3HG1" where in fact it shold be "HG13"
    if pdb.atomname[i][0] in ['1','2','3'] and pdb.atomname[i][1] == 'H':
        pdb.atomname[i] = pdb.atomname[i][1:] + pdb.atomname[i][0]
    anumInResidue.append(i)
    anameInResidue_pdb.append(pdb.atomname[i])
    anameInResidue_amb.append(amb_pdb.atomname[i])
    if (i == (pdb.na - 1)) or ((pdb.chain[i+1], pdb.resid[i+1]) != (pdb.chain[i], pdb.resid[i])):
        if set(anameInResidue_pdb) != set(anameInResidue_amb):
            print set(anameInResidue_pdb).symmetric_difference(set(anameInResidue_amb))
            print "AMBER PDB: Atom names do not match for residue %i (%s)" % (pdb.resid[i], pdb.resname[i])
            raise RuntimeError
        for anum, aname in zip(anumInResidue, anameInResidue_pdb):
            anumMap_amb.append(anumInResidue[anameInResidue_amb.index(aname)])
        anumInResidue = []
        anameInResidue_pdb = []
        anameInResidue_amb = []

anumMap_amb_gro = [None for i in range(pdb.na)]
for i, (anum_amber, anum_gro) in enumerate(zip(anumMap_amb, anumMap_gro)):
    # if i != anum_amber:
    #     print i, anum_amber, anum_gro
    if anum_amber != anum_gro:
        print "Gromacs and AMBER atoms don't have the same order:", "\x1b[91m", i, anum_amber, anum_gro, "\x1b[0m"
        # raise RuntimeError
    # anumMap_amb_gro[anum_amber] = anum_gro

# Run a quick MD simulation in Gromacs 
_exec("grompp_d -f eq.mdp -o eq.tpr")
# You can set print_to_screen=True to see how fast it's going
_exec("mdrun_d -nt 1 -v -stepout 10 -deffnm eq", print_to_screen=False)
_exec("trjconv_d -s eq.tpr -f eq.trr -o eq.gro -ndec 9 -pbc mol -dump 1", stdin="0\n")

# Confirm that constraints are satisfied
OMM_eqgro = app.GromacsGroFile('eq.gro')
OMM_prmtop = app.AmberPrmtopFile('prmtop')
system = OMM_prmtop.createSystem(nonbondedMethod=app.NoCutoff)
integ = mm.VerletIntegrator(1.0*u.femtosecond)
plat = mm.Platform.getPlatformByName('Reference')
simul = app.Simulation(OMM_prmtop.topology, system, integ)
simul.context.setPositions(OMM_eqgro.positions)
simul.context.applyConstraints(1e-12)
state = simul.context.getState(getPositions=True)
pos = np.array(state.getPositions().value_in_unit(u.angstrom)).reshape(-1,3)
M = Molecule('eq.gro')
M.xyzs[0] = pos
M.write('constrained.gro')

# Gromacs calculation
GMX_Energy, GMX_Force, Ecomps_GMX = Calculate_GMX('constrained.gro', 'topol.top', 'shot.mdp')
GMX_Force = GMX_Force.reshape(-1,3)

# Print Gromacs energy components
printcool_dictionary(Ecomps_GMX, title="GROMACS energy components")

# Parse the .mdp file to inform ParmEd
defines, sysargs, mdp_opts = interpret_mdp('shot.mdp')

parm = parmed.amber.AmberParm('prmtop', 'inpcrd')
GmxGro = parmed.gromacs.GromacsGroFile.parse('constrained.gro')
parm.box = GmxGro.box
parm.positions = GmxGro.positions

# AMBER calculation (optional)
AMBER_Energy, AMBER_Force, Ecomps_AMBER = Calculate_AMBER(parm, mdp_opts)
AMBER_Force = AMBER_Force.reshape(-1,3)
# Print AMBER energy components
printcool_dictionary(Ecomps_AMBER, title="AMBER energy components")

def compare_ecomps(in1, in2):
    groups = [([['Bond'], ['BOND']]),
              ([['Angle'], ['ANGLE']]),
              ([['Proper-Dih.', 'Improper-Dih.'], ['DIHED']]),
              ([['LJ-14'], ['1-4 NB']]),
              ([['Coulomb-14'], ['1-4 EEL']]),
              ([['LJ-(SR)', 'Disper.-corr.'], ['VDWAALS']]),
              ([['Coulomb-(SR)', 'Coul.-recip.'], ['EELEC']]),
              ([['Potential'], ['EPTOT']])]
    t1 = []
    t2 = []
    v1 = []
    v2 = []
    for g1, g2 in groups:
        t1.append('+'.join(g1))
        v1.append(sum([v for k, v in in1.items() if k in g1]))
        t2.append('+'.join(g2))
        v2.append(sum([v for k, v in in2.items() if k in g2]))

    for i in range(len(t1)):
        print "%%%is" % (max([len(t) for t in t1]) + 3) % t1[i], "% 18.6f" % v1[i], "  --vs--",
        print "%%%is" % (max([len(t) for t in t2]) + 3) % t2[i], "% 18.6f" % v2[i],
        print "Diff: % 12.6f" % (v2[i]-v1[i])
        
compare_ecomps(Ecomps_GMX, Ecomps_AMBER)

D_Force = GMX_Force - AMBER_Force
D_FrcRMS = np.sqrt(np.mean([sum(k**2) for k in D_Force]))
D_FrcMax = np.sqrt(np.max(np.array([sum(k**2) for k in D_Force])))

print "RMS / Max Force Difference (kJ/mol/nm): % .6e % .6e" % (D_FrcRMS, D_FrcMax)
