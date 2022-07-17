
import os
import time
import numpy    

from scipy.linalg import sqrtm

import pyscf
from pyscf.lib.logger import perf_counter
from pyscf.lib.logger import process_clock

from pyscf import gto
from pyscf import scf
from pyscf import tdscf
from pyscf import gto, scf

def get_r_ene(r, dr_max=0.1):
    dr_list = numpy.linspace(-dr_max, dr_max, 21)

    atm_list = []
    e_r_list = []
    e_u0_list = []
    e_u1_list = []
    e_casscf_list = []
    e_casci_list  = []

    dm0_rhf = None
    dm0_u0hf = None
    dm0_u1hf = None

    for dr in dr_list:
        rz = r + dr
        rxy = r - dr
        atoms = f'''
        Ni    0.0000000     0.0000000     0.0000000
        H     0.0000000     0.0000000    {rz: 10.7f}
        H     0.0000000    {rxy: 10.7f}     0.0000000
        H    {rxy: 10.7f}     0.0000000     0.0000000
        H     0.0000000     0.0000000    {-rz: 10.7f}
        H     0.0000000    {-rxy: 10.7f}     0.0000000
        H    {-rxy: 10.7f}     0.0000000     0.0000000
        '''
        atm_list.append(atoms)

        mol = gto.Mole()
        mol.verbose = 0
        mol.output = f"./log/{r}-{dr}-0.log"
        mol.atom = atoms
        mol.basis = {"H": "sto3g", 'Ni': "lanl2dz"}
        mol.ecp = {'Ni': "lanl2dz"}
        mol.symmetry = 0
        mol.charge = -4
        mol.build()

        mf = scf.RHF(mol)
        mf.max_cycle = 1000
        mf.diis_space = 10
        mf.verbose = 4
        mf.kernel(dm0=dm0_rhf)
        dm0_rhf = mf.make_rdm1()
        e_r_list.append(mf.e_tot)

        assert mf.converged

        ovlp = mf.get_ovlp()
        mo = mf.mo_coeff
        nmo = len(mf.mo_occ)

        ww = numpy.einsum("mp,mn->pn", mo, sqrtm(ovlp))
        w2 = ww ** 2

        ni_3d_idx = mol.search_ao_label("Ni 3d")
        tmp = numpy.einsum("pn->p", w2[:, ni_3d_idx])
        mo_list = numpy.where(tmp > 1e-1)[0]

        print(numpy.sort(tmp))
        print(mo_list)

        assert 1 == 2

        # ncas = len(mo_list)
        # nele = int(sum(mf.mo_occ[mo_list]))

        # print(ncas)
        # print(nele)

        # mycas = mf.CASCI(ncas, nele)
        # mycas.verbose = 4
        # mo = mycas.sort_mo(mo_list)
        # mycas.kernel(mo)
        # e_casci_list.append(mycas.e_tot)

        # mycas = mf.CASSCF(ncas, nele)
        # mycas.verbose = 4
        # mo = mycas.sort_mo(mo_list)
        # mycas.kernel(mo)
        # e_casscf_list.append(mycas.e_tot)

        mf = scf.UHF(mol)
        mf.max_cycle = 1000
        mf.diis_space = 10
        mf.verbose = 4
        mf.kernel(dm0=dm0_u0hf)
        dm0_u0hf = mf.make_rdm1()
        e_u0_list.append(mf.e_tot)

        assert mf.converged

        mol.spin = 2
        mol.build()

        mf = scf.UHF(mol)
        mf.max_cycle = 1000
        mf.diis_space = 10
        mf.verbose = 4
        mf.kernel(dm0=dm0_u1hf)
        dm0_u1hf = mf.make_rdm1()
        e_u1_list.append(mf.e_tot)

        assert mf.converged

        print(f"\ndr = {dr: 4.2f}, dr = {dr: 4.2f}")
        print(f"erhf = {e_r_list[-1]: 8.6f}, eu0hf = {e_u0_list[-1]: 8.6f}, eu1hf = {e_u1_list[-1]: 8.6f}")

    e_r_list = numpy.asarray(e_r_list)
    e_u0_list = numpy.asarray(e_u0_list)
    e_u1_list = numpy.asarray(e_u1_list)

    return dr_list, e_r_list, e_u0_list, e_u1_list

get_r_ene(1.4, dr_max=0.1)
