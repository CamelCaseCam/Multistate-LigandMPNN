import PeptideBuilder
from PeptideBuilder import Geometry
from Bio.PDB import Structure, Model, Chain, Residue, Atom
import Bio.PDB.internal_coords


def _copy_gly(res : Residue, geo : Geometry.Geo):
    # Internal use only
    pass    # Nothing other than backbone atoms to copy for glycine

def _copy_ala(res : Residue, geo : Geometry.AlaGeo):
    ric = res.internal_coord
    # CA-CB length from residue
    geo.CA_CB_length = ric.get_length("CA:CB")
    # C-CA-CB angle
    geo.C_CA_CB_angle = ric.get_angle("C:CA:CB")
    # N-C-CA-CB angle
    geo.N_C_CA_CB_diangle = ric.get_angle("N:C:CA:CB")

def _copy_ser(res : Residue, geo : Geometry.SerGeo):
    ric = res.internal_coord
    # CA-CB length from residue
    geo.CA_CB_length = ric.get_length("CA:CB")
    # C-CA-CB angle
    geo.C_CA_CB_angle = ric.get_angle("C:CA:CB")
    # N-C-CA-CB angle
    geo.N_C_CA_CB_diangle = ric.get_angle("N:C:CA:CB")
    # CB-OG length
    geo.CB_OG_length = ric.get_length("CB:OG")
    # CA_CB_OG_angle
    geo.CA_CB_OG_angle = ric.get_angle("CA:CB:OG")
    # N-CA-CB-OG dihedral
    geo.N_CA_CB_OG_diangle = ric.get_angle("N:CA:CB:OG")

def _copy_cys(res : Residue, geo : Geometry.CysGeo):
    ric = res.internal_coord
    # CA-CB length from residue
    geo.CA_CB_length = ric.get_length("CA:CB")
    # C-CA-CB angle
    geo.C_CA_CB_angle = ric.get_angle("C:CA:CB")
    # N-C-CA-CB angle
    geo.N_C_CA_CB_diangle = ric.get_angle("N:C:CA:CB")
    # CB-SG length
    geo.CB_SG_length = ric.get_length("CB:SG")
    # CA_CB_SG_angle
    geo.CA_CB_SG_angle = ric.get_angle("CA:CB:SG")
    # N-CA-CB-SG dihedral
    geo.N_CA_CB_SG_diangle = ric.get_angle("N:CA:CB:SG")

def _copy_val(res : Residue, geo : Geometry.ValGeo):
    ric = res.internal_coord
    # CA-CB length from residue
    geo.CA_CB_length = ric.get_length("CA:CB")
    # C-CA-CB angle
    geo.C_CA_CB_angle = ric.get_angle("C:CA:CB")
    # N-C-CA-CB angle
    geo.N_C_CA_CB_diangle = ric.get_angle("N:C:CA:CB")
    # CB_CG1_length
    geo.CB_CG1_length = ric.get_length("CB:CG1")
    # CA_CB_CG1_angle
    geo.CA_CB_CG1_angle = ric.get_angle("CA:CB:CG1")
    # N_CA_CB_CG1_diangle
    geo.N_CA_CB_CG1_diangle = ric.get_angle("N:CA:CB:CG1")

    # CB_CG2_length
    geo.CB_CG2_length = ric.get_length("CB:CG2")
    # CA_CB_CG2_angle
    geo.CA_CB_CG2_angle = ric.get_angle("CA:CB:CG2")
    # N_CA_CB_CG2_diangle
    geo.N_CA_CB_CG2_diangle = ric.get_angle("N:CA:CB:CG2")

def _copy_ile(res : Residue, geo : Geometry.IleGeo):
    ric = res.internal_coord
    # CA-CB length from residue
    geo.CA_CB_length = ric.get_length("CA:CB")
    # C-CA-CB angle
    geo.C_CA_CB_angle = ric.get_angle("C:CA:CB")
    # N-C-CA-CB angle
    geo.N_C_CA_CB_diangle = ric.get_angle("N:C:CA:CB")
    # CB_CG1_length
    geo.CB_CG1_length = ric.get_length("CB:CG1")
    # CA_CB_CG1_angle
    geo.CA_CB_CG1_angle = ric.get_angle("CA:CB:CG1")
    # N_CA_CB_CG1_diangle
    geo.N_CA_CB_CG1_diangle = ric.get_angle("N:CA:CB:CG1")

    # CB_CG1_length
    geo.CB_CG1_length = ric.get_length("CB:CG1")
    # CA_CB_CG1_angle
    geo.CA_CB_CG1_angle = ric.get_angle("CA:CB:CG1")
    # N_CA_CB_CG1_diangle
    geo.N_CA_CB_CG1_diangle = ric.get_angle("N:CA:CB:CG1")

    # CB_CG2_length
    geo.CB_CG2_length = ric.get_length("CB:CG2")
    # CA_CB_CG2_angle
    geo.CA_CB_CG2_angle = ric.get_angle("CA:CB:CG2")
    # N_CA_CB_CG2_diangle
    geo.N_CA_CB_CG2_diangle = ric.get_angle("N:CA:CB:CG2")

    # CG1_CD1_length
    geo.CG1_CD1_length = ric.get_length("CG1:CD1")
    # CB_CG1_CD1_angle
    geo.CB_CG1_CD1_angle = ric.get_angle("CB:CG1:CD1")
    # CA_CB_CG1_CD1_diangle
    geo.CA_CB_CG1_CD1_diangle = ric.get_angle("CA:CB:CG1:CD1")

def _copy_leu(res : Residue, geo : Geometry.LeuGeo):
    ric = res.internal_coord
    # CA-CB length from residue
    geo.CA_CB_length = ric.get_length("CA:CB")
    # C-CA-CB angle
    geo.C_CA_CB_angle = ric.get_angle("C:CA:CB")
    # N-C-CA-CB angle
    geo.N_C_CA_CB_diangle = ric.get_angle("N:C:CA:CB")
    # CB_CG_length
    geo.CB_CG_length = ric.get_length("CB:CG")
    # CA_CB_CG_angle
    geo.CA_CB_CG_angle = ric.get_angle("CA:CB:CG")
    # N_CA_CB_CG_diangle
    geo.N_CA_CB_CG_diangle = ric.get_angle("N:CA:CB:CG")

    # CG_CD1_length
    geo.CG_CD1_length = ric.get_length("CG:CD1")
    # CB_CG_CD1_angle
    geo.CB_CG_CD1_angle = ric.get_angle("CB:CG:CD1")
    # CA_CB_CG_CD1_diangle
    geo.CA_CB_CG_CD1_diangle = ric.get_angle("CA:CB:CG:CD1")
    # CG_CD2_length
    geo.CG_CD2_length = ric.get_length("CG:CD2")
    # CB_CG_CD2_angle
    geo.CB_CG_CD2_angle = ric.get_angle("CB:CG:CD2")
    # CA_CB_CG_CD2_diangle
    geo.CA_CB_CG_CD2_diangle = ric.get_angle("CA:CB:CG:CD2")

def _copy_thr(res : Residue, geo : Geometry.ThrGeo):
    ric = res.internal_coord
    # CA-CB length from residue
    geo.CA_CB_length = ric.get_length("CA:CB")
    # C-CA-CB angle
    geo.C_CA_CB_angle = ric.get_angle("C:CA:CB")
    # N-C-CA-CB angle
    geo.N_C_CA_CB_diangle = ric.get_angle("N:C:CA:CB")
    # CB_OG1_length
    geo.CB_OG1_length = ric.get_length("CB:OG1")
    # CA_CB_OG1_angle
    geo.CA_CB_OG1_angle = ric.get_angle("CA:CB:OG1")
    # N_CA_CB_OG1_diangle
    geo.N_CA_CB_OG1_diangle = ric.get_angle("N:CA:CB:OG1")

    # CB_CG2_length
    geo.CB_CG2_length = ric.get_length("CB:CG2")
    # CA_CB_CG2_angle
    geo.CA_CB_CG2_angle = ric.get_angle("CA:CB:CG2")
    # N_CA_CB_CG2_diangle
    geo.N_CA_CB_CG2_diangle = ric.get_angle("N:CA:CB:CG2")

def _copy_arg(res : Residue, geo : Geometry.ArgGeo):
    ric = res.internal_coord
    # CA-CB length from residue
    geo.CA_CB_length = ric.get_length("CA:CB")
    # C-CA-CB angle
    geo.C_CA_CB_angle = ric.get_angle("C:CA:CB")
    # N-C-CA-CB angle
    geo.N_C_CA_CB_diangle = ric.get_angle("N:C:CA:CB")
    # CB_CG_length
    geo.CB_CG_length = ric.get_length("CB:CG")
    # CA_CB_CG_angle
    geo.CA_CB_CG_angle = ric.get_angle("CA:CB:CG")
    # N_CA_CB_CG_diangle
    geo.N_CA_CB_CG_diangle = ric.get_angle("N:CA:CB:CG")

    # CG_CD_length
    geo.CG_CD_length = ric.get_length("CG:CD")
    # CB_CG_CD_angle
    geo.CB_CG_CD_angle = ric.get_angle("CB:CG:CD")
    # CA_CB_CG_CD_diangle
    geo.CA_CB_CG_CD_diangle = ric.get_angle("CA:CB:CG:CD")

    # CD_NE_length
    geo.CD_NE_length = ric.get_length("CD:NE")
    # CG_CD_NE_angle
    geo.CG_CD_NE_angle = ric.get_angle("CG:CD:NE")
    # CB_CG_CD_NE_diangle
    geo.CB_CG_CD_NE_diangle = ric.get_angle("CB:CG:CD:NE")

    # NE_CZ_length
    geo.NE_CZ_length = ric.get_length("NE:CZ")
    # CD_NE_CZ_angle
    geo.CD_NE_CZ_angle = ric.get_angle("CD:NE:CZ")
    # CG_CD_NE_CZ_diangle
    geo.CG_CD_NE_CZ_diangle = ric.get_angle("CG:CD:NE:CZ")

    # CZ_NH1_length
    geo.CZ_NH1_length = ric.get_length("CZ:NH1")
    # NE_CZ_NH1_angle
    geo.NE_CZ_NH1_angle = ric.get_angle("NE:CZ:NH1")
    # CD_NE_CZ_NH1_diangle
    geo.CD_NE_CZ_NH1_diangle = ric.get_angle("CD:NE:CZ:NH1")

    # CZ_NH2_length
    geo.CZ_NH2_length = ric.get_length("CZ:NH2")
    # NE_CZ_NH2_angle
    geo.NE_CZ_NH2_angle = ric.get_angle("NE:CZ:NH2")
    # CD_NE_CZ_NH2_diangle
    geo.CD_NE_CZ_NH2_diangle = ric.get_angle("CD:NE:CZ:NH2")

def _copy_lys(res : Residue, geo : Geometry.LysGeo):
    ric = res.internal_coord
    # CA-CB length from residue
    geo.CA_CB_length = ric.get_length("CA:CB")
    # C-CA-CB angle
    geo.C_CA_CB_angle = ric.get_angle("C:CA:CB")
    # N-C-CA-CB angle
    geo.N_C_CA_CB_diangle = ric.get_angle("N:C:CA:CB")
    # CB_CG_length
    geo.CB_CG_length = ric.get_length("CB:CG")
    # CA_CB_CG_angle
    geo.CA_CB_CG_angle = ric.get_angle("CA:CB:CG")
    # N_CA_CB_CG_diangle
    geo.N_CA_CB_CG_diangle = ric.get_angle("N:CA:CB:CG")

    # CG_CD_length
    geo.CG_CD_length = ric.get_length("CG:CD")
    # CB_CG_CD_angle
    geo.CB_CG_CD_angle = ric.get_angle("CB:CG:CD")
    # CA_CB_CG_CD_diangle
    geo.CA_CB_CG_CD_diangle = ric.get_angle("CA:CB:CG:CD")

    # CD_CE_length
    geo.CD_CE_length = ric.get_length("CD:CE")
    # CG_CD_CE_angle
    geo.CG_CD_CE_angle = ric.get_angle("CG:CD:CE")
    # CB_CG_CD_CE_diangle
    geo.CB_CG_CD_CE_diangle = ric.get_angle("CB:CG:CD:CE")

    # CE_NZ_length
    geo.CE_NZ_length = ric.get_length("CE:NZ")
    # CD_CE_NZ_angle
    geo.CD_CE_NZ_angle = ric.get_angle("CD:CE:NZ")
    # CG_CD_CE_NZ_diangle
    geo.CG_CD_CE_NZ_diangle = ric.get_angle("CG:CD:CE:NZ")

def _copy_asp(res : Residue, geo : Geometry.AspGeo):
    ric = res.internal_coord
    # CA-CB length from residue
    geo.CA_CB_length = ric.get_length("CA:CB")
    # C-CA-CB angle
    geo.C_CA_CB_angle = ric.get_angle("C:CA:CB")
    # N-C-CA-CB angle
    geo.N_C_CA_CB_diangle = ric.get_angle("N:C:CA:CB")
    # CB_CG_length
    geo.CB_CG_length = ric.get_length("CB:CG")
    # CA_CB_CG_angle
    geo.CA_CB_CG_angle = ric.get_angle("CA:CB:CG")
    # N_CA_CB_CG_diangle
    geo.N_CA_CB_CG_diangle = ric.get_angle("N:CA:CB:CG")

    # CG_OD1_length
    geo.CG_OD1_length = ric.get_length("CG:OD1")
    # CB_CG_OD1_angle
    geo.CB_CG_OD1_angle = ric.get_angle("CB:CG:OD1")
    # CA_CB_CG_OD1_diangle
    geo.CA_CB_CG_OD1_diangle = ric.get_angle("CA:CB:CG:OD1")

    # CG_OD2_length
    geo.CG_OD2_length = ric.get_length("CG:OD2")
    # CB_CG_OD2_angle
    geo.CB_CG_OD2_angle = ric.get_angle("CB:CG:OD2")
    # CA_CB_CG_OD2_diangle
    geo.CA_CB_CG_OD2_diangle = ric.get_angle("CA:CB:CG:OD2")

def _copy_asn(res : Residue, geo : Geometry.AsnGeo):
    ric = res.internal_coord
    # CA-CB length from residue
    geo.CA_CB_length = ric.get_length("CA:CB")
    # C-CA-CB angle
    geo.C_CA_CB_angle = ric.get_angle("C:CA:CB")
    # N-C-CA-CB angle
    geo.N_C_CA_CB_diangle = ric.get_angle("N:C:CA:CB")
    # CB_CG_length
    geo.CB_CG_length = ric.get_length("CB:CG")
    # CA_CB_CG_angle
    geo.CA_CB_CG_angle = ric.get_angle("CA:CB:CG")
    # N_CA_CB_CG_diangle
    geo.N_CA_CB_CG_diangle = ric.get_angle("N:CA:CB:CG")

    # CG_OD1_length
    geo.CG_OD1_length = ric.get_length("CG:OD1")
    # CB_CG_OD1_angle
    geo.CB_CG_OD1_angle = ric.get_angle("CB:CG:OD1")
    # CA_CB_CG_OD1_diangle
    geo.CA_CB_CG_OD1_diangle = ric.get_angle("CA:CB:CG:OD1")

    # CG_ND2_length
    geo.CG_ND2_length = ric.get_length("CG:ND2")
    # CB_CG_ND2_angle
    geo.CB_CG_ND2_angle = ric.get_angle("CB:CG:ND2")
    # CA_CB_CG_ND2_diangle
    geo.CA_CB_CG_ND2_diangle = ric.get_angle("CA:CB:CG:ND2")

def _copy_glu(res : Residue, geo : Geometry.GluGeo):
    ric = res.internal_coord
    # CA-CB length from residue
    geo.CA_CB_length = ric.get_length("CA:CB")
    # C-CA-CB angle
    geo.C_CA_CB_angle = ric.get_angle("C:CA:CB")
    # N-C-CA-CB angle
    geo.N_C_CA_CB_diangle = ric.get_angle("N:C:CA:CB")
    # CB_CG_length
    geo.CB_CG_length = ric.get_length("CB:CG")
    # CA_CB_CG_angle
    geo.CA_CB_CG_angle = ric.get_angle("CA:CB:CG")
    # N_CA_CB_CG_diangle
    geo.N_CA_CB_CG_diangle = ric.get_angle("N:CA:CB:CG")

    # CG_CD_length
    geo.CG_CD_length = ric.get_length("CG:CD")
    # CB_CG_CD_angle
    geo.CB_CG_CD_angle = ric.get_angle("CB:CG:CD")
    # CA_CB_CG_CD_diangle
    geo.CA_CB_CG_CD_diangle = ric.get_angle("CA:CB:CG:CD")

    # CD_OE1_length
    geo.CD_OE1_length = ric.get_length("CD:OE1")
    # CG_CD_OE1_angle
    geo.CG_CD_OE1_angle = ric.get_angle("CG:CD:OE1")
    # CB_CG_CD_OE1_diangle
    geo.CB_CG_CD_OE1_diangle = ric.get_angle("CB:CG:CD:OE1")

    # CD_OE2_length
    geo.CD_OE2_length = ric.get_length("CD:OE2")
    # CG_CD_OE2_angle
    geo.CG_CD_OE2_angle = ric.get_angle("CG:CD:OE2")
    # CB_CG_CD_OE2_diangle
    geo.CB_CG_CD_OE2_diangle = ric.get_angle("CB:CG:CD:OE2")

def _copy_gln(res : Residue, geo : Geometry.GlnGeo):
    ric = res.internal_coord
    # CA-CB length from residue
    geo.CA_CB_length = ric.get_length("CA:CB")
    # C-CA-CB angle
    geo.C_CA_CB_angle = ric.get_angle("C:CA:CB")
    # N-C-CA-CB angle
    geo.N_C_CA_CB_diangle = ric.get_angle("N:C:CA:CB")
    # CB_CG_length
    geo.CB_CG_length = ric.get_length("CB:CG")
    # CA_CB_CG_angle
    geo.CA_CB_CG_angle = ric.get_angle("CA:CB:CG")
    # N_CA_CB_CG_diangle
    geo.N_CA_CB_CG_diangle = ric.get_angle("N:CA:CB:CG")

    # CG_CD_length
    geo.CG_CD_length = ric.get_length("CG:CD")
    # CB_CG_CD_angle
    geo.CB_CG_CD_angle = ric.get_angle("CB:CG:CD")
    # CA_CB_CG_CD_diangle
    geo.CA_CB_CG_CD_diangle = ric.get_angle("CA:CB:CG:CD")

    # CD_OE1_length
    geo.CD_OE1_length = ric.get_length("CD:OE1")
    # CG_CD_OE1_angle
    geo.CG_CD_OE1_angle = ric.get_angle("CG:CD:OE1")
    # CB_CG_CD_OE1_diangle
    geo.CB_CG_CD_OE1_diangle = ric.get_angle("CB:CG:CD:OE1")

    # CD_NE2_length
    geo.CD_NE2_length = ric.get_length("CD:NE2")
    # CG_CD_NE2_angle
    geo.CG_CD_NE2_angle = ric.get_angle("CG:CD:NE2")
    # CB_CG_CD_NE2_diangle
    geo.CB_CG_CD_NE2_diangle = ric.get_angle("CB:CG:CD:NE2")

def _copy_met(res : Residue, geo : Geometry.MetGeo):
    ric = res.internal_coord
    # CA-CB length from residue
    geo.CA_CB_length = ric.get_length("CA:CB")
    # C-CA-CB angle
    geo.C_CA_CB_angle = ric.get_angle("C:CA:CB")
    # N-C-CA-CB angle
    geo.N_C_CA_CB_diangle = ric.get_angle("N:C:CA:CB")
    # CB_CG_length
    geo.CB_CG_length = ric.get_length("CB:CG")
    # CA_CB_CG_angle
    geo.CA_CB_CG_angle = ric.get_angle("CA:CB:CG")
    # N_CA_CB_CG_diangle
    geo.N_CA_CB_CG_diangle = ric.get_angle("N:CA:CB:CG")

    # CG_SD_length
    geo.CG_SD_length = ric.get_length("CG:SD")
    # CB_CG_SD_angle
    geo.CB_CG_SD_angle = ric.get_angle("CB:CG:SD")
    # CA_CB_CG_SD_diangle
    geo.CA_CB_CG_SD_diangle = ric.get_angle("CA:CB:CG:SD")

    # SD_CE_length
    geo.SD_CE_length = ric.get_length("SD:CE")
    # CG_SD_CE_angle
    geo.CG_SD_CE_angle = ric.get_angle("CG:SD:CE")
    # CB_CG_SD_CE_diangle
    geo.CB_CG_SD_CE_diangle = ric.get_angle("CB:CG:SD:CE")

def _copy_his(res : Residue, geo : Geometry.HisGeo):
    ric = res.internal_coord
    # CA-CB length from residue
    geo.CA_CB_length = ric.get_length("CA:CB")
    # C-CA-CB angle
    geo.C_CA_CB_angle = ric.get_angle("C:CA:CB")
    # N-C-CA-CB angle
    geo.N_C_CA_CB_diangle = ric.get_angle("N:C:CA:CB")
    # CB_CG_length
    geo.CB_CG_length = ric.get_length("CB:CG")
    # CA_CB_CG_angle
    geo.CA_CB_CG_angle = ric.get_angle("CA:CB:CG")
    # N_CA_CB_CG_diangle
    geo.N_CA_CB_CG_diangle = ric.get_angle("N:CA:CB:CG")

    # CG_ND1_length
    geo.CG_ND1_length = ric.get_length("CG:ND1")
    # CB_CG_ND1_angle
    geo.CB_CG_ND1_angle = ric.get_angle("CB:CG:ND1")
    # CA_CB_CG_ND1_diangle
    geo.CA_CB_CG_ND1_diangle = ric.get_angle("N:CA:CB:CG:ND1")

    # CG_CD2_length
    geo.CG_CD2_length = ric.get_length("CG:CD2")
    # CB_CG_CD2_angle
    geo.CB_CG_CD2_angle = ric.get_angle("CB:CG:CD2")
    # CA_CB_CG_CD2_diangle
    geo.CA_CB_CG_CD2_diangle = ric.get_angle("CA:CB:CG:CD2")

    # ND1_CE1_length
    geo.ND1_CE1_length = ric.get_length("ND1:CE1")
    # CG_ND1_CE1_angle
    geo.CG_ND1_CE1_angle = ric.get_angle("CG:ND1:CE1")
    # CB_CG_ND1_CE1_diangle
    geo.CB_CG_ND1_CE1_diangle = ric.get_angle("CB:CG:ND1:CE1")

    # CD2_NE2_length
    geo.CD2_NE2_length = ric.get_length("CD2:NE2")
    # CG_CD2_NE2_angle
    geo.CG_CD2_NE2_angle = ric.get_angle("CG:CD2:NE2")
    # CB_CG_CD2_NE2_diangle
    geo.CB_CG_CD2_NE2_diangle = ric.get_angle("CB:CG:CD2:NE2")

def _copy_pro(res : Residue, geo : Geometry.ProGeo):
    ric = res.internal_coord
    # CA-CB length from residue
    geo.CA_CB_length = ric.get_length("CA:CB")
    # C-CA-CB angle
    geo.C_CA_CB_angle = ric.get_angle("C:CA:CB")
    # N-C-CA-CB angle
    geo.N_C_CA_CB_diangle = ric.get_angle("N:C:CA:CB")
    # CB_CG_length
    geo.CB_CG_length = ric.get_length("CB:CG")
    # CA_CB_CG_angle
    geo.CA_CB_CG_angle = ric.get_angle("CA:CB:CG")
    # N_CA_CB_CG_diangle
    geo.N_CA_CB_CG_diangle = ric.get_angle("N:CA:CB:CG")

    # CG_CD_length
    geo.CG_CD_length = ric.get_length("CG:CD")
    # CB_CG_CD_angle
    geo.CB_CG_CD_angle = ric.get_angle("CB:CG:CD")
    # CA_CB_CG_CD_diangle
    geo.CA_CB_CG_CD_diangle = ric.get_angle("CA:CB:CG:CD")

def _copy_phe(res : Residue, geo : Geometry.PheGeo):
    ric = res.internal_coord
    # CA-CB length from residue
    geo.CA_CB_length = ric.get_length("CA:CB")
    # C-CA-CB angle
    geo.C_CA_CB_angle = ric.get_angle("C:CA:CB")
    # N-C-CA-CB angle
    geo.N_C_CA_CB_diangle = ric.get_angle("N:C:CA:CB")
    # CB_CG_length
    geo.CB_CG_length = ric.get_length("CB:CG")
    # CA_CB_CG_angle
    geo.CA_CB_CG_angle = ric.get_angle("CA:CB:CG")
    # N_CA_CB_CG_diangle
    geo.N_CA_CB_CG_diangle = ric.get_angle("N:CA:CB:CG")

    # CG_CD1_length
    geo.CG_CD1_length = ric.get_length("CG:CD1")
    # CB_CG_CD1_angle
    geo.CB_CG_CD1_angle = ric.get_angle("CB:CG:CD1")
    # CA_CB_CG_CD1_diangle
    geo.CA_CB_CG_CD1_diangle = ric.get_angle("CA:CB:CG:CD1")

    # CG_CD2_length
    geo.CG_CD2_length = ric.get_length("CG:CD2")
    # CB_CG_CD2_angle
    geo.CB_CG_CD2_angle = ric.get_angle("CB:CG:CD2")
    # CA_CB_CG_CD2_diangle
    geo.CA_CB_CG_CD2_diangle = ric.get_angle("CA:CB:CG:CD2")

    # CD1_CE1_length
    geo.CD1_CE1_length = ric.get_length("CD1:CE1")
    # CG_CD1_CE1_angle
    geo.CG_CD1_CE1_angle = ric.get_angle("CG:CD1:CE1")
    # CB_CG_CD1_CE1_diangle
    geo.CB_CG_CD1_CE1_diangle = ric.get_angle("CB:CG:CD1:CE1")

    # CD2_CE2_length
    geo.CD2_CE2_length = ric.get_length("CD2:CE2")
    # CG_CD2_CE2_angle
    geo.CG_CD2_CE2_angle = ric.get_angle("CG:CD2:CE2")
    # CB_CG_CD2_CE2_diangle
    geo.CB_CG_CD2_CE2_diangle = ric.get_angle("CB:CG:CD2:CE2")

    # CE1_CZ_length
    geo.CE1_CZ_length = ric.get_length("CE1:CZ")
    # CD1_CE1_CZ_angle
    geo.CD1_CE1_CZ_angle = ric.get_angle("CD1:CE1:CZ")
    # CG_CD1_CE1_CZ_diangle
    geo.CG_CD1_CE1_CZ_diangle = ric.get_angle("CG:CD1:CE1:CZ")

def _copy_tyr(res : Residue, geo : Geometry.TyrGeo):
    ric = res.internal_coord
    # CA-CB length from residue
    geo.CA_CB_length = ric.get_length("CA:CB")
    # C-CA-CB angle
    geo.C_CA_CB_angle = ric.get_angle("C:CA:CB")
    # N-C-CA-CB angle
    geo.N_C_CA_CB_diangle = ric.get_angle("N:C:CA:CB")
    # CB_CG_length
    geo.CB_CG_length = ric.get_length("CB:CG")
    # CA_CB_CG_angle
    geo.CA_CB_CG_angle = ric.get_angle("CA:CB:CG")
    # N_CA_CB_CG_diangle
    geo.N_CA_CB_CG_diangle = ric.get_angle("N:CA:CB:CG")

    # CG_CD1_length
    geo.CG_CD1_length = ric.get_length("CG:CD1")
    # CB_CG_CD1_angle
    geo.CB_CG_CD1_angle = ric.get_angle("CB:CG:CD1")
    # CA_CB_CG_CD1_diangle
    geo.CA_CB_CG_CD1_diangle = ric.get_angle("CA:CB:CG:CD1")

    # CG_CD2_length
    geo.CG_CD2_length = ric.get_length("CG:CD2")
    # CB_CG_CD2_angle
    geo.CB_CG_CD2_angle = ric.get_angle("CB:CG:CD2")
    # CA_CB_CG_CD2_diangle
    geo.CA_CB_CG_CD2_diangle = ric.get_angle("CA:CB:CG:CD2")

    # CD1_CE1_length
    geo.CD1_CE1_length = ric.get_length("CD1:CE1")
    # CG_CD1_CE1_angle
    geo.CG_CD1_CE1_angle = ric.get_angle("CG:CD1:CE1")
    # CB_CG_CD1_CE1_diangle
    geo.CB_CG_CD1_CE1_diangle = ric.get_angle("CB:CG:CD1:CE1")

    # CD2_CE2_length
    geo.CD2_CE2_length = ric.get_length("CD2:CE2")
    # CG_CD2_CE2_angle
    geo.CG_CD2_CE2_angle = ric.get_angle("CG:CD2:CE2")
    # CB_CG_CD2_CE2_diangle
    geo.CB_CG_CD2_CE2_diangle = ric.get_angle("CB:CG:CD2:CE2")

    # CE1_CZ_length
    geo.CE1_CZ_length = ric.get_length("CE1:CZ")
    # CD1_CE1_CZ_angle
    geo.CD1_CE1_CZ_angle = ric.get_angle("CD1:CE1:CZ")
    # CG_CD1_CE1_CZ_diangle
    geo.CG_CD1_CE1_CZ_diangle = ric.get_angle("CG:CD1:CE1:CZ")

    # CZ_OH_length
    geo.CZ_OH_length = ric.get_length("CZ:OH")
    # CE1_CZ_OH_angle
    geo.CE1_CZ_OH_angle = ric.get_angle("CE1:CZ:OH")
    # CD1_CE1_CZ_OH_diangle
    geo.CD1_CE1_CZ_OH_diangle = ric.get_angle("CD1:CE1:CZ:OH")

def _copy_trp(res : Residue, geo : Geometry.TrpGeo):
    ric = res.internal_coord
    # CA-CB length from residue
    geo.CA_CB_length = ric.get_length("CA:CB")
    # C-CA-CB angle
    geo.C_CA_CB_angle = ric.get_angle("C:CA:CB")
    # N-C-CA-CB angle
    geo.N_C_CA_CB_diangle = ric.get_angle("N:C:CA:CB")
    # CB_CG_length
    geo.CB_CG_length = ric.get_length("CB:CG")
    # CA_CB_CG_angle
    geo.CA_CB_CG_angle = ric.get_angle("CA:CB:CG")
    # N_CA_CB_CG_diangle
    geo.N_CA_CB_CG_diangle = ric.get_angle("N:CA:CB:CG")

    # CG_CD1_length
    geo.CG_CD1_length = ric.get_length("CG:CD1")
    # CB_CG_CD1_angle
    geo.CB_CG_CD1_angle = ric.get_angle("CB:CG:CD1")
    # CA_CB_CG_CD1_diangle
    geo.CA_CB_CG_CD1_diangle = ric.get_angle("CA:CB:CG:CD1")

    # CG_CD2_length
    geo.CG_CD2_length = ric.get_length("CG:CD2")
    # CB_CG_CD2_angle
    geo.CB_CG_CD2_angle = ric.get_angle("CB:CG:CD2")
    # CA_CB_CG_CD2_diangle
    geo.CA_CB_CG_CD2_diangle = ric.get_angle("CA:CB:CG:CD2")

    # CD1_NE1_length
    geo.CD1_NE1_length = ric.get_length("CD1:NE1")
    # CG_CD1_NE1_angle
    geo.CG_CD1_NE1_angle = ric.get_angle("CG:CD1:NE1")
    # CB_CG_CD1_NE1_diangle
    geo.CB_CG_CD1_NE1_diangle = ric.get_angle("CB:CG:CD1:NE1")

    # CD2_CE2_length
    geo.CD2_CE2_length = ric.get_length("CD2:CE2")
    # CG_CD2_CE2_angle
    geo.CG_CD2_CE2_angle = ric.get_angle("CG:CD2:CE2")
    # CB_CG_CD2_CE2_diangle
    geo.CB_CG_CD2_CE2_diangle = ric.get_angle("CB:CG:CD2:CE2")

    # CD2_CE3_length
    geo.CD2_CE3_length = ric.get_length("CD2:CE3")
    # CG_CD2_CE3_angle
    geo.CG_CD2_CE3_angle = ric.get_angle("CG:CD2:CE3")
    # CB_CG_CD2_CE3_diangle
    geo.CB_CG_CD2_CE3_diangle = ric.get_angle("CB:CG:CD2:CE3")

    # CE2_CZ2_length
    geo.CE2_CZ2_length = ric.get_length("CE2:CZ2")
    # CD2_CE2_CZ2_angle
    geo.CD2_CE2_CZ2_angle = ric.get_angle("CD2:CE2:CZ2")
    # CG_CD2_CE2_CZ2_diangle
    geo.CG_CD2_CE2_CZ2_diangle = ric.get_angle("CG:CD2:CE2:CZ2")

    # CE3_CZ3_length
    geo.CE3_CZ3_length = ric.get_length("CE3:CZ3")
    # CD2_CE3_CZ3_angle
    geo.CD2_CE3_CZ3_angle = ric.get_angle("CD2:CE3:CZ3")
    # CG_CD2_CE3_CZ3_diangle
    geo.CG_CD2_CE3_CZ3_diangle = ric.get_angle("CG:CD2:CE3:CZ3")

    # CZ2_CH2_length
    geo.CZ2_CH2_length = ric.get_length("CZ2:CH2")
    # CE2_CZ2_CH2_angle
    geo.CE2_CZ2_CH2_angle = ric.get_angle("CE2:CZ2:CH2")
    # CD2_CE2_CZ2_CH2_diangle
    geo.CD2_CE2_CZ2_CH2_diangle = ric.get_angle("CD2:CE2:CZ2:CH2")

COPY_DICT = {
    "G": _copy_gly, "A": _copy_ala, "S": _copy_ser, "C": _copy_cys,
    "V": _copy_val, "I": _copy_ile, "L": _copy_leu, "T": _copy_thr,
    "R": _copy_arg, "K": _copy_lys, "D": _copy_asp, "N": _copy_asn,
    "E": _copy_glu, "Q": _copy_gln, "M": _copy_met, "H": _copy_his,
    "P": _copy_pro, "F": _copy_phe, "Y": _copy_tyr, "W": _copy_trp
}

def copy_all_geometry(residue : Residue, geo : Geometry.Geo):
    '''
    Copies non-backbone geometry from the Biopython structure to the PeptideBuilder geometry.
    '''
    # Get residue name to figure out what to do
    res_name = geo.residue_name
    if res_name not in COPY_DICT:
        raise ValueError(f"Unsupported residue type: {res_name}")
    # Call the appropriate copy function
    COPY_DICT[res_name](residue, geo)

