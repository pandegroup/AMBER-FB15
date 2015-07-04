#!/usr/bin/env python

import os, sys
from collections import OrderedDict
import numpy as np
import parmed as pmd
from parmed.amber import parameters
import xml.etree.ElementTree as ET
from nifty import printcool_dictionary

# Atom types from parm99.dat
Type99 = ["C", "CA", "CB", "CC", "CD", "CK", "CM", "CN", "CQ", "CR", "CT", "CV", 
          "CW", "C*", "CY", "CZ", "C0", "H", "HC", "H1", "H2", "H3", "HA", "H4", 
          "H5", "HO", "HS", "HW", "HP", "HZ", "F", "Cl", "Br", "I", "IM", "IB", 
          "MG", "N", "NA", "NB", "NC", "N2", "N3", "NT", "N*", "NY", "O", "O2", 
          "OW", "OH", "OS", "P", "S", "SH", "CU", "FE", "Li", "IP", "Na", "K", 
          "Rb", "Cs", "Zn", "LP"]

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

# Parse the original AMBER99SB XML file.
A99SB = ET.parse('/home/leeping/src/OpenMM/wrappers/python/simtk/openmm/app/data/amber99sb.xml')
root = A99SB.getroot()
A99SB_AtAc = {}
A99SB_AnAc = {}
for force in root:
    if force.tag == 'AtomTypes':
        for atype in force:
            A99SB_AtAc[atype.attrib["name"]] = atype.attrib["class"]

# Build a mapping of Amino Acid / Atom Names -> Atom Types
for force in root:
    if force.tag == 'Residues':
        for res in force:
            A99SB_AnAc[res.attrib["name"]] = {}
            for atom in res:
                if atom.tag == 'Atom':
                    A99SB_AnAc[res.attrib["name"]][atom.attrib["name"]] = A99SB_AtAc[atom.attrib["type"]]

# For each amino acid, create the mapping of the new atom type (in FB15) back to the old atom type (in AMBER99SB).
RevMap = {}
for k1, v1 in NewAC.items():
    for k2, v2 in v1.items():
        if v2 in RevMap.keys():
            print "Atom type already in reverse map"
            raise RuntimeError
        RevMap[v2] = A99SB_AnAc[k1][k2]

# Mappings of AMBER99(SB) atom types to elements and hybridization
A99_Hyb = OrderedDict([("H",  ("H", "sp3")), ("HO", ("H", "sp3")), ("HS", ("H", "sp3")), 
                       ("H1", ("H", "sp3")), ("H2", ("H", "sp3")), ("H3", ("H", "sp3")),
                       ("H4", ("H", "sp3")), ("H5", ("H", "sp3")), ("HW", ("H", "sp3")), 
                       ("HC", ("H", "sp3")), ("HA", ("H", "sp3")), ("HP", ("H", "sp3")),
                       ("OH", ("O", "sp3")), ("OS", ("O", "sp3")), ("O",  ("O", "sp2")), 
                       ("O2", ("O", "sp2")), ("OW", ("O", "sp3")), ("CT", ("C", "sp3")),
                       ("CH", ("C", "sp3")), ("C2", ("C", "sp3")), ("C3", ("C", "sp3")),
                       ("C",  ("C", "sp2")), ("C*", ("C", "sp2")), ("CA", ("C", "sp2")),
                       ("CB", ("C", "sp2")), ("CC", ("C", "sp2")), ("CN", ("C", "sp2")), 
                       ("CM", ("C", "sp2")), ("CK", ("C", "sp2")), ("CQ", ("C", "sp2")),
                       ("CD", ("C", "sp2")), ("CE", ("C", "sp2")), ("CF", ("C", "sp2")), 
                       ("CP", ("C", "sp2")), ("CI", ("C", "sp2")), ("CJ", ("C", "sp2")),
                       ("CW", ("C", "sp2")), ("CV", ("C", "sp2")), ("CR", ("C", "sp2")), 
                       ("CA", ("C", "sp2")), ("CY", ("C", "sp2")), ("C0", ("Ca", "sp3")),
                       ("MG", ("Mg", "sp3")), ("N",  ("N", "sp2")), ("NA", ("N", "sp2")), 
                       ("N2", ("N", "sp2")), ("N*", ("N", "sp2")), ("NP", ("N", "sp2")),
                       ("NQ", ("N", "sp2")), ("NB", ("N", "sp2")), ("NC", ("N", "sp2")), 
                       ("NT", ("N", "sp3")), ("N3", ("N", "sp3")), ("S",  ("S", "sp3")),
                       ("SH", ("S", "sp3")), ("P",  ("P", "sp3")), ("LP", ("",  "sp3")), 
                       ("F",  ("F", "sp3")), ("CL", ("Cl", "sp3")), ("BR", ("Br", "sp3")),
                       ("I",  ("I",  "sp3")), ("FE", ("Fe", "sp3")), ("EP", ("",  "sp3")), 
                       ("OG", ("O", "sp3")), ("OL", ("O", "sp3")), ("AC", ("C", "sp3")),
                       ("EC", ("C", "sp3"))])

fout = open('leaprc.fb15','w')
print >> fout, """logFile leap.log
#
# ----- leaprc for loading the AMBER-FB15 force field
#       with added atom types for amino acid side chains.
#
#	load atom type hybridizations
#
addAtomTypes {"""

AtList = []
for k, v in A99_Hyb.items():
    AtList.append(k)
    print >> fout, "        { %-4s  %3s %5s }" % ("\""+k+"\"", "\""+v[0]+"\"", "\""+v[1]+"\"")

print >> fout, "# FB15 atom types"

# Print out hybridizations of the new FB15 atom types
for k, k2 in RevMap.items():
    AtList.append(k)
    v = A99_Hyb[k2]
    print >> fout, "        { %-4s  %3s %5s }" % ("\""+k+"\"", "\""+v[0]+"\"", "\""+v[1]+"\"")
print >> fout, "}"

print >> fout, """
#
#	Load the main parameter set.
#       The TIP3P-FB water model is used.
#
parm99 = loadamberparams parm99.dat
mods = loadamberparams frcmod.fb15
mods2 = loadamberparams frcmod.tip3pfb
#
#	Load DNA/RNA libraries
#
loadOff all_nucleic94.lib
#
#	Load main chain and terminating 
#	amino acid libraries (modified from ff94)
#
loadOff all_aminofb15.lib
loadOff all_aminoctfb15.lib
loadOff all_aminontfb15.lib
#
#       Load water and ions
# 
loadOff ions94.lib
loadOff solvents.lib
HOH = FB3
WAT = FB3

#
#	Define the PDB name map for the amino acids and DNA.
#
addPdbResMap {
  { 0 "ALA" "NALA" } { 1 "ALA" "CALA" }
  { 0 "ARG" "NARG" } { 1 "ARG" "CARG" }
  { 0 "ASN" "NASN" } { 1 "ASN" "CASN" }
  { 0 "ASP" "NASP" } { 1 "ASP" "CASP" }
  { 0 "CYS" "NCYS" } { 1 "CYS" "CCYS" }
  { 0 "CYX" "NCYX" } { 1 "CYX" "CCYX" }
  { 0 "GLN" "NGLN" } { 1 "GLN" "CGLN" }
  { 0 "GLU" "NGLU" } { 1 "GLU" "CGLU" }
  { 0 "GLY" "NGLY" } { 1 "GLY" "CGLY" }
  { 0 "HID" "NHID" } { 1 "HID" "CHID" }
  { 0 "HIE" "NHIE" } { 1 "HIE" "CHIE" }
  { 0 "HIP" "NHIP" } { 1 "HIP" "CHIP" }
  { 0 "ILE" "NILE" } { 1 "ILE" "CILE" }
  { 0 "LEU" "NLEU" } { 1 "LEU" "CLEU" }
  { 0 "LYS" "NLYS" } { 1 "LYS" "CLYS" }
  { 0 "MET" "NMET" } { 1 "MET" "CMET" }
  { 0 "PHE" "NPHE" } { 1 "PHE" "CPHE" }
  { 0 "PRO" "NPRO" } { 1 "PRO" "CPRO" }
  { 0 "SER" "NSER" } { 1 "SER" "CSER" }
  { 0 "THR" "NTHR" } { 1 "THR" "CTHR" }
  { 0 "TRP" "NTRP" } { 1 "TRP" "CTRP" }
  { 0 "TYR" "NTYR" } { 1 "TYR" "CTYR" }
  { 0 "VAL" "NVAL" } { 1 "VAL" "CVAL" }
  { 0 "HIS" "NHIS" } { 1 "HIS" "CHIS" }
  { 0 "GUA" "DG5"  } { 1 "GUA" "DG3"  } { "GUA" "DG" }
  { 0 "ADE" "DA5"  } { 1 "ADE" "DA3"  } { "ADE" "DA" }
  { 0 "CYT" "DC5"  } { 1 "CYT" "DC3"  } { "CYT" "DC" }
  { 0 "THY" "DT5"  } { 1 "THY" "DT3"  } { "THY" "DT" }
  { 0 "G" "RG5"  } { 1 "G" "RG3"  } { "G" "RG" } { "GN" "RGN" }
  { 0 "A" "RA5"  } { 1 "A" "RA3"  } { "A" "RA" } { "AN" "RAN" }
  { 0 "C" "RC5"  } { 1 "C" "RC3"  } { "C" "RC" } { "CN" "RCN" }
  { 0 "U" "RU5"  } { 1 "U" "RU3"  } { "U" "RU" } { "UN" "RUN" }
  { 0 "DG" "DG5"  } { 1 "DG" "DG3"  }  
  { 0 "DA" "DA5"  } { 1 "DA" "DA3"  }  
  { 0 "DC" "DC5"  } { 1 "DC" "DC3"  }  
  { 0 "DT" "DT5"  } { 1 "DT" "DT3"  }

}

addPdbAtomMap {
  { "O5*" "O5'" }
  { "C5*" "C5'" }
  { "C4*" "C4'" }
  { "O4*" "O4'" }
  { "C3*" "C3'" }
  { "O3*" "O3'" }
  { "C2*" "C2'" }
  { "C1*" "C1'" }
  { "C5M" "C7"  }
  { "O2*" "O2'" }
  { "H1*" "H1'" }
  { "H2*1" "H2'1" }
  { "H2*2" "H2'2" }
  { "H2'"  "H2'1" }
  { "H2''" "H2'2" }
  { "H3*" "H3'" }
  { "H4*" "H4'" }
  { "H5*1" "H5'1" }
  { "H5*2" "H5'2" }
  { "H5'"  "H5'1" }
  { "H5''" "H5'2" }
  { "HO2'" "HO'2" }
  { "HO5'" "H5T" }
  { "HO3'" "H3T" }
  { "O1'" "O4'" }
  { "OA"  "O1P" }
  { "OB"  "O2P" }
  { "OP1" "O1P" }
  { "OP2" "O2P" }
}


#
# assumed that most often proteins use HIE
#
NHIS = NHIE
HIS = HIE
CHIS = CHIE
"""

fout.close()

# Parse the OpenMM XML file.
OXML = ET.parse(sys.argv[1])

root = OXML.getroot()

# OpenMM Atom Types to Atom Class
OAtAc = OrderedDict()

# OpenMM Atom Classes to Masses
OAcMass = OrderedDict()

# OpenMM Residue-Atom Names to Atom Class
AA_OAc = OrderedDict()

# OpenMM Atom Class to Parameter Mapping 
# (vdW sigma and epsilon in AKMA)
OAcPrm = OrderedDict()
OBondPrm = OrderedDict()
OAnglePrm = OrderedDict()
ODihPrm = OrderedDict()
OImpPrm = OrderedDict()

Params = parameters.ParameterSet()

# Stage 1 processing: Read in force field parameters
for force in root:
    # Top-level tags in force field XML file are:
    # Forces
    # Atom types
    # Residues
    if force.tag == 'AtomTypes':
        for elem in force:
            OAtAc[elem.attrib['name']] = elem.attrib['class']
            mass = float(elem.attrib['mass'])
            if elem.attrib['class'] in OAcMass and mass != OAcMass[elem.attrib['class']]:
                print "Atom class mass not consistent"
                raise RuntimeError
            OAcMass[elem.attrib['class']] = mass
        # printcool_dictionary(OAtAc)
    # Harmonic bond parameters
    if force.tag == 'HarmonicBondForce':
        for elem in force:
            att = elem.attrib
            BC = (att['class1'], att['class2'])
            BCr = (att['class2'], att['class1'])
            acij = tuple(sorted(BC))
            if acij in OBondPrm:
                print acij, "already defined in OBndPrm"
                raise RuntimeError
            b = float(att['length'])*10
            k = float(att['k'])/10/10/2/4.184
            OBondPrm[acij] = (b, k)
            # Pass information to ParmEd
            Params._add_bond(acij[0], acij[1], rk=k, req=b)
            # New Params object can't write frcmod files.
            # Params.bond_types[acij] = pmd.BondType(k, b)
    # Harmonic angle parameters.  Same as for harmonic bonds.
    if force.tag == 'HarmonicAngleForce':
        for elem in force:
            att = elem.attrib
            AC = (att['class1'], att['class2'], att['class3'])
            ACr = (att['class3'], att['class2'], att['class1'])
            if AC[2] >= AC[0]:
                acijk = tuple(AC)
            else:
                acijk = tuple(ACr)
            if acijk in OAnglePrm:
                print acijk, "already defined in OAnglePrm"
                raise RuntimeError
            t = float(att['angle'])*180/np.pi
            k = float(att['k'])/2/4.184
            OAnglePrm[acijk] = (t, k)
            # Pass information to ParmEd
            Params._add_angle(acijk[0], acijk[1], acijk[2], thetk=k, theteq=t)
            # New Params object can't write frcmod files.
            # Params.angle_types[acijk] = pmd.AngleType(k, t)
    # Periodic torsion parameters.
    if force.tag == 'PeriodicTorsionForce':
        for elem in force:
            att = elem.attrib
            def fillx(strin):
                if strin == "" : return "X"
                else: return strin
            c1 = fillx(att['class1'])
            c2 = fillx(att['class2'])
            c3 = fillx(att['class3'])
            c4 = fillx(att['class4'])
            DC = (c1, c2, c3, c4) 
            DCr = (c4, c3, c2, c1)
            if c1 > c4:
                # Reverse ordering if class4 is alphabetically before class1
                acijkl = DCr
            elif c1 < c4:
                # Forward ordering if class1 is alphabetically before class4
                acijkl = DC
            else:
                # If class1 and class4 are the same, order by class2/class3
                if c2 > c3:
                    acijkl = DCr
                elif c3 > c2:
                    acijkl = DC
                else:
                    acijkl = DC
            keylist = sorted([i for i in att.keys() if 'class' not in i])
            dprms = OrderedDict()
            for p in range(1, 7):
                pkey = "periodicity%i" % p
                fkey = "phase%i" % p
                kkey = "k%i" % p
                if pkey in keylist:
                    dprms[int(att[pkey])] = (float(att[fkey])*180.0/np.pi, float(att[kkey])/4.184)
            dprms = OrderedDict([(p, dprms[p]) for p in sorted(dprms.keys())])
            # ParmEd dihedral list
            dihedral_list = []
            for p in dprms.keys():
                f, k = dprms[p]
                dihedral_list.append(pmd.DihedralType(k, p, f, 1.2, 2.0, list=dihedral_list))
                # Pass information to ParmEd
                dtyp = 'normal' if (elem.tag == 'Proper') else 'improper'
                Params._add_dihedral(acijkl[0], acijkl[1], acijkl[2], acijkl[3], 
                                     pk=k, phase=f, periodicity=p, dihtype=dtyp)
            if elem.tag == 'Proper':
                if acijkl in ODihPrm:
                    print acijkl, "already defined in ODihPrm"
                    raise RuntimeError
                ODihPrm[acijkl] = dprms
                # New Params object can't write frcmod files.
                # Params.dihedral_types[acijkl] = dihedral_list
            elif elem.tag == 'Improper':
                if acijkl in OImpPrm:
                    print acijkl, "already defined in OImpPrm"
                    raise RuntimeError
                OImpPrm[acijkl] = dprms
                if len(dihedral_list) > 1:
                    print acijkl, "more than one interaction"
                    raise RuntimeError
                # New Params object can't write frcmod files.
                # Params.improper_periodic_types[acijkl] = dihedral_list[0]
            else:
                raise RuntimeError
    # Nonbonded parameters
    if force.tag == 'NonbondedForce':
        for elem in force:
            sigma = float(elem.attrib['sigma'])*10
            epsilon = float(elem.attrib['epsilon'])/4.184
            atype = elem.attrib['type']
            aclass = OAtAc[atype]
            amass = OAcMass[aclass]
            msigeps = (amass, sigma, epsilon)
            if aclass in OAcPrm:
                if OAcPrm[aclass] != msigeps:
                    print 'mass, sigma and epsilon for atom class %s not uniquely defined:' % aclass
                    print msigeps, OAcPrm[aclass]
                    raise RuntimeError
            else:
                OAcPrm[aclass] = (amass, sigma, epsilon)
                # Params._add_atom(aclass, amass, sigma*2**(1./6)/2, epsilon)
                # New Params object can't write frcmod files.
                # atype = pmd.AtomType(aclass, None, amass, -1)
                # atype.set_lj_params(epsilon, sigma*2**(1./6)/2)
                # Params.atom_types[aclass] = atype

    # Residue definitions
    if force.tag == 'Residues':
        resnode = force

# Add atom types in the order of appearance in fb15.dat
for aclass in AtList:
    if aclass not in Type99 and aclass in OAcPrm:
        (amass, sigma, epsilon) = OAcPrm[aclass]
        Params._add_atom(aclass, amass, sigma*2**(1./6)/2, epsilon)

# Stage 2 processing: Read residue definitions
for elem in resnode:
    res = elem.attrib['name']
    for subelem in elem:
        # Atom tag: Create a NetworkX node
        att = subelem.attrib
        if subelem.tag == 'Atom':
            AA_OAc.setdefault(res, OrderedDict())[att['name']] = OAtAc[att['type']]

# Load AMBER OFF library
offinout = [('all_amino94ildn.lib', 'all_aminofb15.lib'),
            ('all_aminoct94ildn.lib', 'all_aminoctfb15.lib'),
            ('all_aminont94ildn.lib', 'all_aminontfb15.lib')]
for fin, fout in offinout:
    AOff = pmd.modeller.offlib.AmberOFFLibrary.parse(os.path.join(os.environ['AMBERHOME'], 'dat', 'leap', 'lib', fin))
    for AA in AOff.keys():
        for atom in AOff[AA].atoms:
            atom.type = AA_OAc[AA][atom.name]
    print "Writing .off library to %s" % fout
    pmd.modeller.offlib.AmberOFFLibrary.write(AOff,fout)

Params.write('frcmod.fb15')
# import IPython
# IPython.embed()
    

#print AOff

#import IPython
#IPython.embed()
#print Params
