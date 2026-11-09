from   ase.md.velocitydistribution import MaxwellBoltzmannDistribution,Stationary
from   ase.md.nvtberendsen         import NVTBerendsen
from   ase.md.langevin             import Langevin
from   ase.io.trajectory           import Trajectory
from   ase.calculators.calculator import Calculator, all_changes
from   ase.io.lammpsdata          import read_lammps_data
from   ase    import Atoms, units
from   ase.md import MDLogger
from   time   import perf_counter
import numpy as np
import sys
import subprocess


#from chgnet.model import CHGNetCalculator
#from chgnet.model import CHGNet
#model           = CHGNet.from_file('./GBmodel.pth.tar') 
#calc            = CHGNetCalculator(model=model)
#XXX mace case
#from mace.calculators import MACECalculator
#calc     = MACECalculator(model_path='./MACE.model', device='cpu')
#calc = MACECalculator(model="mace-off23", device='cpu')  # または 'cuda'
#
# ASE >= 3.24 moved full_3x3_to_voigt_6_stress from ase.constraints to ase.stress,
# but mattersim 1.2.0 still imports it from the old location. Re-inject it so the
# script keeps working when the cluster's ASE is updated. No-op on older ASE.
try:
    import ase.constraints as _asec
    if not hasattr(_asec, "full_3x3_to_voigt_6_stress"):
        from ase.stress import full_3x3_to_voigt_6_stress as _f3, voigt_6_to_full_3x3_stress as _v6
        _asec.full_3x3_to_voigt_6_stress = _f3
        _asec.voigt_6_to_full_3x3_stress = _v6
except Exception:
    pass

from mattersim.forcefield import MatterSimCalculator
calc = MatterSimCalculator(load_path="best_model.pth", device='cpu')
#calc = MatterSimCalculator(device='cpu')
#
command         = "mkdir  output"
subprocess.call(command,shell=True)
output_filename = "./output/out"
log_filename    = output_filename + ".log"
traj_filename   = output_filename + ".traj"


atmtyp   = ["H","C","N","O"]
ntyp     = 4

kcal2ev  = 0.04337  


args   = sys.argv
file1  = args[1]    # lammps input file

def readcellInfo(inputdata):
   ip1    = open(inputdata,"r")
   cellIn = np.zeros(3)
   while True:
     line1 = ip1.readline()
     if not line1: break
     aa    = line1.split()
#
     if(len(aa) > 2):
       if(aa[2] == "xlo"): cellIn[0] = float(aa[1]) - float(aa[0])
       if(aa[2] == "ylo"): cellIn[1] = float(aa[1]) - float(aa[0])
       if(aa[2] == "zlo"): cellIn[2] = float(aa[1]) - float(aa[0])
#
#
   return cellIn



# Print statements
def print_dyn():
    imd          = dyn.get_number_of_steps()
    etot         = atoms.get_total_energy()
    temp_K       = atoms.get_temperature()
#   stress       = atoms.get_stress(include_ideal_gas=True)/units.GPa
#   stress_ave   = (stress[0]+stress[1]+stress[2])/3.0
    elapsed_time = perf_counter() - start_time
#   print(f"  {imd: >3}   {etot:.3f}    {temp_K:.2f}    {stress_ave:.2f}  {stress[0]:.2f}  {stress[1]:.2f}  {stress[2]:.2f}  {stress[3]:.2f}  {stress[4]:.2f}  {stress[5]:.2f}    {elapsed_time:.3f}")
    print(f"  {imd: >3}   {etot:.3f}    {temp_K:.2f}    {elapsed_time:.3f}")


def cartesian_to_spherical(x, y, z):
    r     = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi   = np.arctan2(y, x)
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
#   r     = np.asarray(r)
#   theta = np.asarray(theta)
#   phi   = np.asarray(phi)
    x     = r * np.sin(theta) * np.cos(phi)
    y     = r * np.sin(theta) * np.sin(phi)
    z     = r * np.cos(theta)

    return x, y, z





class LAMMPSInput():
    def __init__(self, file1):
       ip1   = open(file1,"r")
       lines = ip1.readlines()
       n     = len(lines)
#      --- defaults for the sampling / thermostat controls (all optional in in.test) ---
       self.thermostat   = "berendsen" # berendsen | langevin
       self.taut         = 100.0       # fs. Berendsen coupling time. MUST be >> dt
       self.friction     = 0.01        # 1/fs. Langevin friction
       self.dense        = 10          # dump interval [steps] while a bond is breaking
       self.nbreak_start = 1           # enter the dense-sampling phase at this many broken P-S
       self.nbreak_stop  = 8           # abort the run once this many P-S are broken
       self.rbreak       = 2.6         # P-S distance [A] above which a bond counts as broken
       self.hold_on_break= 1           # freeze the moving bias once dense sampling starts
       self.after        = 2000        # steps to keep sampling after the first bond breaks,
                                       # then stop. Without this a slow-pull run that breaks
                                       # early would dump thousands of correlated frames at
                                       # frozen bias. 0 = run to the end.
       self.equil        = 0           # steps of unbiased equilibration before the bias ramps.
                                       # Needed whenever the cell was rescaled: the structure
                                       # starts far from equilibrium and dumping there just
                                       # feeds the training set unrelaxed junk.
       for ii in range(n):
         aa = lines[ii].split()
         if(len(aa) > 0):
           if("atom_style" == aa[0]): self.atom_style = aa[1]
           if("read_data"  == aa[0]): self.inputdata  = aa[1]
           if("fix"  == aa[0]):
              if(aa[3] == "nvt"):     self.temperature = float(aa[5])
              if(aa[3] == "colvars"): self.colvars     = aa[4]
           if("run"      == aa[0]): self.nstep  = int(aa[1])
           if("timestep" == aa[0]): self.dt     = float(aa[1]) * 1000.0 # ps --> fs
           if("thermo"   == aa[0]): self.numint = int(aa[1])
#          --- extensions (ignored by LAMMPS, read only here) ---
           if("md_thermostat"  == aa[0]): self.thermostat    = aa[1].lower()
           if("md_taut"        == aa[0]): self.taut          = float(aa[1])
           if("md_friction"    == aa[0]): self.friction      = float(aa[1])
           if("sample_dense"   == aa[0]): self.dense         = int(aa[1])
           if("sample_start"   == aa[0]): self.nbreak_start  = int(aa[1])
           if("sample_stop"    == aa[0]): self.nbreak_stop   = int(aa[1])
           if("sample_rbreak"  == aa[0]): self.rbreak        = float(aa[1])
           if("hold_on_break"  == aa[0]): self.hold_on_break = int(aa[1])
           if("md_equil"       == aa[0]): self.equil         = int(aa[1])
           if("sample_after"   == aa[0]): self.after         = int(aa[1])



class HybridQMMLCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, MLcalc, ODPcalc, **kwargs):
        super().__init__(**kwargs)
        self.MLcalc     = MLcalc
        self.ODPcalc    = ODPcalc

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        # --- (1) MLIP で全体系を計算 ---
        self.MLcalc.calculate(atoms)
        energy_mm = self.MLcalc.results['energy']
        forces_mm = self.MLcalc.results['forces'].copy()

        # --- (2) ODP項を実行 ---
#       self.qm_calc.calculate(qm_atoms,0)
        self.ODPcalc.calc(atoms)
#       energy_qm = self.qm_calc.results['energy']
#       forces_qm = self.qm_calc.results['forces']

#       # --- (4) 力を合算 ---
        forces_mm = forces_mm + self.ODPcalc.gradODP01

        # --- (5) 合成結果を保存 ---
        self.results = {
            'energy': energy_mm,  # 必要なら QM 部分のエネルギー補正も可能
            'forces': forces_mm
        }



def printgeom(nat,XYZ,ATM,output,cell):
    output.write("%i\n%8.5f  %8.5f  %8.5f\n"%(nat,cell[0],cell[1],cell[2]))
    for ii in range(nat):
        output.write("%3s   %15.11f   %15.11f   %15.11f\n"%(ATM[ii],XYZ[ii][0],XYZ[ii][1],XYZ[ii][2]))


def readxyz(file1):
   input   = open(file1,"r")
   S       = input.readline()
   natom   = int(S)
   S       = input.readline()
   aa      = S.split()
   cell    = [0.0]*3
   cell[0] = float(aa[0])
   cell[1] = float(aa[1])
   cell[2] = float(aa[2])
#  XYZ   = natom*[3*[0.0]]
   XYZ     = [[0 for j in range(3)] for i in range(natom)]
   ATM     = natom*[0.0]
   for n in range(natom):
      S      = input.readline()
      L      = S.split()
      ATM[n] = L[0]
      for i in range(1,4): XYZ[n][i-1]=float(L[i]) # x
#
   XYZ   = np.asarray(XYZ)
   ATM   = np.asarray(ATM)
   cell  = np.asarray(cell)

   return natom,cell,ATM,XYZ


class unitcell:
  def __init__(self,nat,ATM,XYZ,cell):
    self.nat   = nat
    self.cell  = cell.copy()
    self.ATM   = ATM.copy()
    self.XYZ   = XYZ.copy()
    self.nbond = 0
    self.bond  = []
    self.ene   = 0.0  # eV
    self.ncomp = [0]*ntyp  #
    self.ncomp = np.asarray(self.ncomp)
    for  itype in range(ntyp):
       for iat in range(nat):
          if(ATM[iat] == atmtyp[itype]):
            self.ncomp[itype] = self.ncomp[itype] + 1
#                       if(atmtyp[itype] == "N" and self.ncomp[itype] == 2): self.ncomp[itype] = 1
    def dump(self,op):
        printgeom(self.nat,self.XYZ,self.ATM,op,self.cell)


def dis2(v1,v2,cell):
    v10 = v1.copy()
    v20 = v2.copy()
    v3  = v10 - v20
    for ixyz in range(3):
        while(abs(v3[ixyz]) > cell[ixyz]/2.0):
            if(v3[ixyz]  > 0): v10[ixyz] =  v10[ixyz]  - cell[ixyz]
            if(v3[ixyz]  < 0): v10[ixyz] =  v10[ixyz]  + cell[ixyz]
            v3  = v10 - v20
#   dis = np.sqrt((v3*v3).sum())
    return v3

def  dis3(v1,v2,cell):
    v10 = v1.copy()
    v20 = v2.copy()
    v3  = v10 - v20
    for ixyz in range(3):
        while(abs(v3[ixyz]) > cell[ixyz][ixyz]/2.0):
            if(v3[ixyz]  > 0): 
              for jxyz in range(3):
                  v10[jxyz] =  v10[jxyz]  - cell[ixyz][jxyz]
            if(v3[ixyz]  < 0): 
              for jxyz in range(3):
                  v10[jxyz] =  v10[jxyz]  + cell[ixyz][jxyz]
            v3  = v10 - v20
#    dis = np.sqrt((v3*v3).sum())
    return v3



class ODPcalculator:
  def __init__(self,file1):
    self.readODPinput(file1)
    self.RCmapping()
    self.P     = np.zeros(self.nbasis)
    self.R     = np.zeros(self.nbasis)
    self.T     = np.zeros(self.nbasis)
    self.X     = np.zeros(self.nbasis)
    self.XT    = np.zeros(self.nbasis)
#   calc ODP Reaction coordinate
    self.R     = [0.0]*self.nbasis
    self.P     = [self.tgtcenter]*self.nbasis
    self.R     = np.asarray(self.R)
    self.P     = np.asarray(self.P)
#   self.P     = self.calcRC(self.model01.XYZ)
    self.T     = self.P  - self.R
    self.norm  = np.sqrt((self.T*self.T).sum())
    self.cvlog = open("ODP.log","w")
    self.cvtrj = open("ODP.trj","a")
#   self.tgtcenter = 1.0
#
    for ii in range(self.nbasis):
       self.cvlog.write("The %ith R = %8.3f\n"%(ii,self.R[ii]))
    for ii in range(self.nbasis):
       self.cvlog.write("The %ith P = %8.3f\n"%(ii,self.P[ii]))
    self.cvlog.write("T vector length = %8.3f\n"%(self.norm))
#   
    self.ncc = 0
#   stage at which the moving bias was frozen (None = keep advancing)
    self.frozen_stage = None
#   stages elapsed during equilibration, subtracted so the ramp starts at 0
    self.stage_offset = 0
#
  def RCmapping(self):
     model00     = self.model00 
     nat         = model00.nat
     ind1        = np.arange(nat)
     logP        = model00.ATM == "P"
     indP        = ind1[logP]
     nP          = len(indP)
     self.nbasis = nP
#
     cell    = model00.cell.copy() 
     cell2   = model00.cell.copy() 
     cell2[2]= cell2[2]*2.0
     nat2    = model00.nat * 2
     ind2    = np.arange(nat2)
     ATM2    = [0]*nat2
     XYZ2    = np.zeros((nat2,3))
     for iat in range(nat):
       ATM2[iat]         = model00.ATM[iat]
       ATM2[iat+nat]     = model00.ATM[iat]
       XYZ2[iat,0]       = model00.XYZ[iat,0]
       XYZ2[iat,1]       = model00.XYZ[iat,1]
       XYZ2[iat,2]       = model00.XYZ[iat,2]
       XYZ2[iat+nat,0]   = model00.XYZ[iat,0]
       XYZ2[iat+nat,1]   = model00.XYZ[iat,1]
       XYZ2[iat+nat,2]   = model00.XYZ[iat,2]+cell[2]
#
     ATM2  = np.asarray(ATM2)
     logP2 = ATM2 == "P"
     indP2 = ind2[logP2]
     nP2   = len(indP2)
#    P-P position mapping
     RCmap1  = []
     map1xyz = np.zeros((12,3))
     for ii in range(nP):
        iat = indP[ii]
        dis = np.zeros(nP2)
#       get 1st nearest P
        for jj in range(nP2):
          jat     = indP2[jj]
          v3      = dis2(XYZ2[jat],XYZ2[iat],cell2)
          norm    = np.sqrt((v3*v3).sum())
          dis[jj] = norm
        argmin = np.argsort(dis)
        map1   = [0]*12
        subxyz = np.zeros((12,3))
        for jj in range(1,13):
          jat            = indP2[argmin[jj]]
          map1[jj-1]     = jat
          v3             = dis2(XYZ2[jat],XYZ2[iat],cell2)
          r,theta,phi    = cartesian_to_spherical(v3[0],v3[1],v3[2])
          subxyz[jj-1,0] = r
          subxyz[jj-1,1] = theta/np.pi*180
          subxyz[jj-1,2] = phi/np.pi*180
#         print(" %3i   %3i   %8.3f  %8.3f  %8.3f"%(ii,jj,r,theta/np.pi*180,phi/np.pi*180))
        argsph  = np.argsort(subxyz[:,1])
        subxyz  = subxyz[argsph]
        map1    = np.asarray(map1)
        map1    = map1[argsph]
#       print(subxyz)
        map1tmp = map1.copy()
#       1st layer
        argphi  = np.argsort(subxyz[0:4,2])
#       print(argphi)
        for jj in range(4):
          jat         = argphi[jj]
          map1xyz[jj] = map1xyz[jj] + subxyz[jat]
          map1[jj]    = map1tmp[argphi[jj]]
#       2nd layer
        argphi  = np.argsort(subxyz[4:8,2])
        argphi  = argphi + 4
        for jj in range(4):
          jat           = argphi[jj]
          map1xyz[jj+4] = map1xyz[jj+4] + subxyz[jat]
          map1[jj+4]    = map1tmp[jat]
#       3rd layer
        argphi  = np.argsort(subxyz[8:12,2])
        argphi  = argphi + 8
        for jj in range(4):
          jat           = argphi[jj]
          map1xyz[jj+8] = map1xyz[jj+8] + subxyz[jat]
          map1[jj+8]    = map1tmp[jat]
#
#
        RCmap1.append(map1)
        
#       map1xyz = map1xyz + subxyz[argsph]
#       print(map1)
#
     map1xyz       =  map1xyz/nP
     self.map1xyz  =  map1xyz.copy()
     self.RCmap1   =  np.asarray(RCmap1)
     print("Mapping 1 is for Crystal to amorphous control")
     print(map1xyz)
     print(self.RCmap1)
#
#    P-X position mapping
     RCmap2  = []
     map2xyz = np.zeros((5,3))
#
     logCl   = ATM2 == "Cl"
     logBr   = ATM2 == "Br"
     logS    = ATM2 == "S"
     logX    = logCl + logBr + logS
     nX      = int(logX.sum())
     indX    = ind2[logX]
#
     for ii in range(nP):
        iat = indP[ii]
        dis = np.zeros(nX)
#       get 1st nearest X
        for jj in range(nX):
          jat  = indX[jj]
          v3   = dis2(XYZ2[jat],XYZ2[iat],cell2)
          norm    = np.sqrt((v3*v3).sum())
          dis[jj] = norm
        argmin = np.argsort(dis)
#       print(dis[argmin])
        map1   = [0]*5 # 4
        subatm = [0]*5 # 4
        subxyz = np.zeros((5,3))
        for jj in range(5):
          jat            = indX[argmin[jj+4]]
          map1[jj  ]     = jat
          v3             = dis2(XYZ2[jat],XYZ2[iat],cell2)
          r,theta,phi    = cartesian_to_spherical(v3[0],v3[1],v3[2])
          subatm[jj]     = ATM2[jat]
          subxyz[jj,0]   = r
          subxyz[jj,1]   = theta/np.pi*180
          subxyz[jj,2]   = phi/np.pi*180
#       print(" %3i   %3i   %8.3f  %8.3f  %8.3f"%(ii,jj,r,theta/np.pi*180,phi/np.pi*180))
        subatm       = np.asarray(subatm)
        argsph       = np.argsort(subxyz[:,1])
        subatm       = subatm[argsph]
        subxyz       = subxyz[argsph]
        map1         = np.asarray(map1)
        map1         = map1[argsph]
        map1tmp      = map1.copy()
#
        map2xyz[0]   = map2xyz[0] + subxyz[0]
        map1[0]      = map1tmp[0]
#
#       print("wwwcheck =",ii)
#       for jj in range(5):
#         print("%s  %8.3f  %8.3f  %8.3f"%(subatm[jj],subxyz[jj,0],subxyz[jj,1],subxyz[jj,2]))
#       1st layer
        argphi  = np.argsort(subxyz[1:3,2])
#       print(argphi)
        for jj in range(2):
          jat           = argphi[jj]    + 1
          map2xyz[jj+1] = map2xyz[jj+1] + subxyz[jat]
          map1[jj+1]    = map1tmp[jat]
#       2nd layer
        argphi  = np.argsort(subxyz[3:5,2])
        for jj in range(2):
          jat           = argphi[jj]    + 3
          map2xyz[jj+3] = map2xyz[jj+3] + subxyz[jat]
          map1[jj+3]    = map1tmp[jat]
#
        RCmap2.append(map1)
#
#       print(subxyz)
     map2xyz = map2xyz /nP
     self.map2xyz  =  map2xyz.copy()
     self.RCmap2   =  np.asarray(RCmap2)
     print("Mapping 2 is just for monior")
     print(map2xyz)
     print(self.RCmap2)

#
  def calc(self,atoms):
#   get current stage for X
    nstep    = int(self.ncc   / self.targetNumSteps) - self.stage_offset
    if(nstep < 0):                     nstep = 0
    if(self.frozen_stage is not None): nstep = self.frozen_stage
    shift    = self.tgtcenter - self.centers
    dStep    = shift          / float(self.targetNumStages)
    ubiasX   = self.centers   + nstep*dStep
    ubiasT   = 0.0
    if(self.calcXT):
      shift    = self.tgtcenterT - self.centersT
      dStep    = shift           / float(self.targetNumStages)
      ubiasT   = self.centersT   + nstep*dStep
#   get current stage for XT if necessary
    cell       = atoms.cell.copy()
    self.calcX(atoms.positions,cell)
#   print(self.norm,self.ODP01,self.ODP02)
    if(self.ncc % 10 == 0):
      self.cvtrj.write(" %10i  %12.8f  %12.8f  #( %12.8f  %12.8f )\n"%(self.ncc,self.ODP01*self.norm,self.ODP02*self.norm,ubiasX,ubiasT)) 
#
#
    BumbX    = self.forceConstant * kcal2ev * (self.ODP01*self.norm - ubiasX)
    BumbT    = self.forceConstant * kcal2ev * (self.ODP02*self.norm - ubiasT)  # need to modify
    calcW    = False
    if(self.DoWall):
       if(self.ODP02 > self.upperWall):
          calcW = True
          BumbT = self.Wallconst * kcal2ev * (self.ODP02*self.norm - self.upperWall)  
         
      
#   gradient
    self.gradX(atoms.positions,BumbX,cell)
#
    if(self.calcXT or calcW):
      self.gradXT(atoms.positions,BumbT,cell)
      self.gradODP01 = self.gradODP01 + self.gradODP02
#
#   counter
    self.ncc = self.ncc + 1

#
  def calcX(self,xyzin,cell):
    val           = 0.0
    x             = 0.0
    xp            = 0.0
    tangent       = 0.0
    X             = self.calcRC(xyzin,cell)
#   print(X)   # debug: spammed output.txt to ~7 MB per run
    XT            = np.zeros(self.nbasis)
    for ia in range(self.nbasis):
        val   = (abs(X[ia]) - self.R[ia])*self.T[ia]/self.norm
        x     = x  + val
#
#   print("wwwcheck =",x,self.nbasis)   # debug
#   XT
    for ia in range(self.nbasis):
        XT[ia]         = abs(X[ia]) - self.R[ia]
        XT[ia]        -= x*self.T[ia]/self.norm
        xp            += XT[ia]*XT[ia]
#
    xp         = np.sqrt(xp)
#
    self.X     = X.copy()
    self.XT    = XT.copy()
    self.ODP01 = x   / self.norm  # normalize by distance (no unit)
    self.ODP02 = xp  / self.norm  # normalize by distance (no unit)
#
  def gradX(self,xyzin,Bumb,cell):
#   initial set up
    model00     = self.model00 
    nat         = model00.nat
    ind1        = np.arange(nat)
    logP        = model00.ATM == "P"
    indP        = ind1[logP]
    nP          = len(indP)
    cell1       = cell.copy()
    cell2       = cell1.copy() 
    cell2[2,:]  = cell2[2,:]*2.0
    nat2        = model00.nat * 2
    ind2        = np.arange(nat2)
    ATM2        = [0]*nat2
    XYZ2        = np.zeros((nat2,3))
    for iat in range(nat):
      ATM2[iat]         = model00.ATM[iat]
      ATM2[iat+nat]     = model00.ATM[iat]
      XYZ2[iat,0]       = xyzin[iat,0]
      XYZ2[iat,1]       = xyzin[iat,1]
      XYZ2[iat,2]       = xyzin[iat,2]
      XYZ2[iat+nat,0]   = xyzin[iat,0]+cell1[2,0]
      XYZ2[iat+nat,1]   = xyzin[iat,1]+cell1[2,1]
      XYZ2[iat+nat,2]   = xyzin[iat,2]+cell1[2,2]
#
    ATM2    = np.asarray(ATM2)
    RCmap1  = self.RCmap1.copy() 
#
    self.gradODP01 = np.zeros((self.nat,3))
    for ii in range(self.nbasis):
       iat    = indP[ii]
       sign   = -1.0
       if(self.X[ii] < 0.0): sign =  1.0
       for jj in range(12):
         jat   = RCmap1[ii,jj]
         for ix in range(3):
           val2   = Bumb*self.T[ii] * sign
 # 
           self.gradODP01[iat,ix]  +=-val2
           if(jat >= nat):
             self.gradODP01[jat-nat,ix]  += val2
           else:
             self.gradODP01[jat,ix]      += val2
## 
# 
  def gradXT(self,xyzin,Bumb,cell):
    thresh         = 1.0e-04
    self.gradODP02 = np.zeros((self.nat,3))
    if(self.ODP02 < thresh): return
#
#   initial set up
    model00     = self.model00 
    nat         = model00.nat
    ind1        = np.arange(nat)
    logP        = model00.ATM == "P"
    indP        = ind1[logP]
    nP          = len(indP)
    cell1       = cell.copy()
    cell2       = cell1.copy() 
    cell2[2,:]  = cell2[2,:]*2.0
    nat2        = model00.nat * 2
    ind2        = np.arange(nat2)
    ATM2        = [0]*nat2
    XYZ2        = np.zeros((nat2,3))
    for iat in range(nat):
      ATM2[iat]         = model00.ATM[iat]
      ATM2[iat+nat]     = model00.ATM[iat]
      XYZ2[iat,0]       = xyzin[iat,0]
      XYZ2[iat,1]       = xyzin[iat,1]
      XYZ2[iat,2]       = xyzin[iat,2]
      XYZ2[iat+nat,0]   = xyzin[iat,0]+cell1[2,0]
      XYZ2[iat+nat,1]   = xyzin[iat,1]+cell1[2,1]
      XYZ2[iat+nat,2]   = xyzin[iat,2]+cell1[2,2]
#
    ATM2    = np.asarray(ATM2)
    RCmap1  = self.RCmap1.copy() 
#
#
    coef           = 1.0 / self.ODP02
    for ii in range(self.nbasis):
       iat   = indP[ii]
       val1  = self.XT[ii]*coef
#
       sign   = 1.0
       if(self.X[ii] < 0.0): sign = -1.0
#
       val2  = 0.0
       for kk in range(self.nbasis):
           val2 = val2 + coef*self.XT[kk]*self.T[kk]/self.norm
       val2 = val2 * self.T[ii]/self.norm
#
       val2 = Bumb* (val1 - val2) * sign
#
       for jj in range(12):
         jat   = RCmap1[ii,jj]
         for ix in range(3):
#          
           self.gradODP02[iat,ix]  +=-val2
           if(jat >= nat):
             self.gradODP02[jat-nat,ix]  += val2
           else:
             self.gradODP02[jat,ix]      += val2

#      iat     = self.pair[ii,0]
#      jat     = self.pair[ii,1]
#      v1      = xyzin[iat].copy()
#      v2      = xyzin[jat].copy()
#      dist_v  = dis2(v1,v2,self.cell)
#      val1    = np.sqrt((dist_v*dist_v).sum())
## 
#      for ix in range(3):
#        dx     = dist_v[ix]
#        val2   = dx/val1
#        val3   = 0.0
#        for ia in range(self.nbasis):
#          val3 = val3 + val2*self.T[ii]/self.norm * self.T[ia]/self.norm*self.XT[ia]
##
#        val2 = coef*(val2*self.XT[ii]-val3)
## 
#        self.gradODP02[iat,ix]  +=-Bumb*val2
#        self.gradODP02[jat,ix]  += Bumb*val2
#    
#
  def calcRC(self,xyzin,cell):
    RC          = np.zeros(self.nbasis)
    model00     = self.model00 
    nat         = model00.nat
    ind1        = np.arange(nat)
    logP        = model00.ATM == "P"
    indP        = ind1[logP]
    nP          = len(indP)
#
#   if(len(cell) == 3):
#     cell1      = np.zeros((3,3)) 
#     cell1[0,0] = cell[0]
#     cell1[1,1] = cell[1]
#     cell1[2,2] = cell[2]
#   else:
    cell1       = cell.copy()
#
    cell2       = cell1.copy() 
    cell2[2,:]  = cell2[2,:]*2.0
#   print(cell2)
    nat2        = model00.nat * 2
    ind2        = np.arange(nat2)
    ATM2        = [0]*nat2
    XYZ2        = np.zeros((nat2,3))
    for iat in range(nat):
      ATM2[iat]         = model00.ATM[iat]
      ATM2[iat+nat]     = model00.ATM[iat]
      XYZ2[iat,0]       = xyzin[iat,0]
      XYZ2[iat,1]       = xyzin[iat,1]
      XYZ2[iat,2]       = xyzin[iat,2]
      XYZ2[iat+nat,0]   = xyzin[iat,0]+cell1[2,0]
      XYZ2[iat+nat,1]   = xyzin[iat,1]+cell1[2,1]
      XYZ2[iat+nat,2]   = xyzin[iat,2]+cell1[2,2]
#
    ATM2    = np.asarray(ATM2)
    logP2   = ATM2 == "P"
    indP2   = ind2[logP2]
    nP2     = len(indP2)
#   P-P position mapping
    RCmap1  = self.RCmap1.copy() 
    RCxyz   = np.zeros((nP,3))
    for ii in range(nP):
       iat    = indP[ii]
       subxyz = np.zeros((12,3))
       for jj in range(12):
         jat            = RCmap1[ii,jj]
         v3             = dis3(XYZ2[jat],XYZ2[iat],cell2)
#        r,theta,phi    = cartesian_to_spherical(v3[0],v3[1],v3[2])
         x,y,z          = spherical_to_cartesian(self.map1xyz[jj,0],self.map1xyz[jj,1]/180.0*np.pi,self.map1xyz[jj,2]/180.0*np.pi)
         subxyz[jj,0]   = v3[0] - x 
         subxyz[jj,1]   = v3[1] - y 
         subxyz[jj,2]   = v3[2] - z 
#        subxyz[jj,0]   = r     - self.map1xyz[jj,0] 
#        subxyz[jj,1]   = theta - self.map1xyz[jj,0] 
#        subxyz[jj,2]   = phi   - self.map1xyz[jj,0] 
       deltaX      = subxyz[:,0].sum() # / 12.0
       deltaY      = subxyz[:,1].sum() # / 12.0
       deltaZ      = subxyz[:,2].sum() # / 12.0
       RC[ii]      = (deltaX + deltaY + deltaZ ) / 36.0
#      print("%10.5f  %10.5f  %10.5f"%(deltaX,deltaY,deltaZ))
#      RCxyz[ii,0] = deltaX 
#      RCxyz[ii,1] = deltaY 
#      RCxyz[ii,2] = deltaZ 
#      print(subxyz)
#   for ii in range(self.nbasis):
#     iat            = self.pair[ii,0]
#     jat            = self.pair[ii,1]
#     v1             = xyzin[iat].copy()
#     v2             = xyzin[jat].copy()
#     dist_v         = dis2(v1,v2,self.cell)
#     val1           = np.sqrt((dist_v*dist_v).sum())
#     RC[ii]         = val1
#
    return RC
#


  def readODPinput(self,file1):
    ip1   = open(file1,"r")
    lines = ip1.readlines()
    nline = len(lines)
    for ii in range(nline):
      aa = lines[ii].split()
      if(len(aa) > 0): 
        if(aa[0] == "colvarsTrajFrequency"):
           self.dumpFreq = int(aa[1])
        if(aa[0] == "nbasis"):
           self.nbasis   = int(aa[1])
        if(aa[0] == "ODPRC"):
           pair          = [0]*self.nbasis*2
           bb            = aa[1].split(",")
           for ii in range(self.nbasis*2):
             pair[ii] = int(bb[ii])
           pair      = np.asarray(pair)
           self.pair = pair.reshape((self.nbasis,2))
        if(aa[0] == "refPositionsIni"):
           filetmp          = aa[1]
           nat,cell,ATM,XYZ = readxyz(filetmp)
           S00              = unitcell(nat,ATM,XYZ,cell)
           self.model00     = S00
           self.nat         = nat
           self.ATM         = ATM.copy()
           self.cell        = cell.copy()
        if(aa[0] == "refPositionsFin"):
           filetmp          = aa[1]
           nat,cell,ATM,XYZ = readxyz(filetmp)
           S00              = unitcell(nat,ATM,XYZ,cell)
           self.model01     = S00
        if(aa[0] == "colvars"):
           self.calcXT = False
           if(len(aa) == 3): self.calcXT = True 
        if(aa[0] == "targetCenters"):
           self.tgtcenter       = float(aa[1])
           if(self.calcXT): 
             self.tgtcenterT    = float(aa[2])
        if(aa[0] == "centers"):
           self.centers         = float(aa[1])
           if(self.calcXT): 
             self.centersT      = float(aa[2])
        if(aa[0] == "forceConstant"):
           self.forceConstant   = float(aa[1])
        if(aa[0] == "targetNumSteps"):
           self.targetNumSteps  = int(aa[1])
        if(aa[0] == "targetNumStages"):
           self.targetNumStages = int(aa[1])
        if(aa[0] == "harmonicWalls"):
           self.DoWall = True
        else:
           self.DoWall = False
        if(aa[0] == "lowerWalls"):
           self.lowerWall = float(aa[1])
        if(aa[0] == "upperWalls"):
           self.upperWall = float(aa[1])
        if(aa[0] == "upperWallConstant"):
           self.Wallconst = float(aa[1])
#       if(aa[0] == "ODPRC"):

    
# Read input keyword
LMPinp  = LAMMPSInput(file1)
# ODP set
ODPcal  = ODPcalculator(LMPinp.colvars)
# read initial coordinate from lammps data
atoms   = read_lammps_data(LMPinp.inputdata, atom_style=LMPinp.atom_style)    # ASE 2023+ 推奨
cellIn  = readcellInfo(LMPinp.inputdata)
atoms.set_cell([cellIn[0], cellIn[1], cellIn[2]])              # 単位はÅ（オングストローム）
# 周期境界条件を有効にする場合は
atoms.set_pbc([True, True, True])
# check
ATM     = atoms.get_chemical_symbols()
cell    = atoms.get_cell().lengths()
#print(ATM,cell)
#ODPcal.calc(atoms)
#ip1                 = open(file1,"r")
#
hybrid_calc   = HybridQMMLCalculator(MLcalc=calc, ODPcalc=ODPcal)
atoms.calc    = hybrid_calc
#natom,cell,ATM,XYZ  = readxyz(file1)
#print("Energy (eV):", atoms.get_potential_energy())
#print("Forces (eV/Å):", atoms.get_forces())
# input parameters
time_step    = LMPinp.dt          # fsec
temperature  = LMPinp.temperature # Kelvin
num_interval = LMPinp.numint
taut         = LMPinp.taut        # fs. was hard-coded to 1.0 = time_step, which turns the
                                  # Berendsen bath into a per-step velocity rescale: it drains
                                  # the work injected by the ODP bias, so Epot ran up ~120 eV
                                  # while T stayed pinned at 300 K.
#
print("output_filename = ",output_filename)
print("log_filename    = ",log_filename)
print("traj_filename   = ",traj_filename)
print("thermostat      = %s (taut=%.1f fs, friction=%.4f /fs, dt=%.2f fs)"%(
      LMPinp.thermostat,taut,LMPinp.friction,time_step))
# Set the momenta corresponding to the given "temperature"
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature,force_temp=True)
Stationary(atoms)  # Set zero total momentum to avoid drifting
# run NVT MD
if(LMPinp.thermostat == "langevin"):
   dyn = Langevin(atoms, time_step*units.fs, temperature_K = temperature,
                  friction=LMPinp.friction/units.fs, loginterval=num_interval)
else:
   if(taut < 10.0*time_step):
      print("  WARNING: taut=%.1f fs is not >> dt=%.2f fs. The bath will absorb the bias work."%(taut,time_step))
   dyn = NVTBerendsen(atoms, time_step*units.fs, temperature_K = temperature,
                      taut=taut*units.fs, loginterval=num_interval)
# run NPT MD
#dyn = NPT(atoms,
#      time_step*units.fs,
#      temperature_K  = temperature,
#      externalstress = sigma*units.bar,
#      ttime          = ttime*units.fs,
#      pfactor        = pfactor*units.GPa*(units.fs**2),
#      logfile        = log_filename,
#      trajectory     = traj_filename,
#      loginterval    = num_interval
#      )

#
# ---------------------------------------------------------------------------
#  PS4 monitor: adaptive sampling around the P-S dissociation
#
#  The old setup dumped every num_interval steps and the frames handed to DFT
#  stopped at the last frame BEFORE the first P-S broke.  Everything collected
#  was therefore intact crystal, which the training set already covers.  The
#  useful window - a few P-S broken but the lattice still standing - was thrown
#  away, and the run then wasted its remaining wall time tearing the cell apart
#  (50+ broken bonds, +120 eV).
#
#  Now: coarse dumping while intact, dense dumping once bonds start breaking,
#  and the run aborts before the structure collapses.
# ---------------------------------------------------------------------------
class StopMD(Exception):
    pass

class PS4Monitor():
    def __init__(self, atoms, dyn, traj_filename, dense, nbreak_start, nbreak_stop,
                 rbreak, coarse, hold_on_break, equil, after, odp):
        self.atoms   = atoms
        self.dyn     = dyn
        self.traj    = Trajectory(traj_filename, "w", atoms)
        self.dense   = dense
        self.equil   = equil
        self.after   = after
        self.brkstep = None   # step at which the breaking phase started
        self.nstart  = nbreak_start
        self.nstop   = nbreak_stop
        self.rbreak  = rbreak
        self.coarse  = coarse
        self.hold    = hold_on_break
        self.odp     = odp
        self.phase   = "intact"
        self.nwrite  = 0
        self.last    = -1     # last step written, guards against a duplicate frame 0
        sym          = np.asarray(atoms.get_chemical_symbols())
        self.indP    = np.arange(len(sym))[sym == "P"]
        self.indS    = np.arange(len(sym))[sym == "S"]
        # PS4 assignment fixed from the initial configuration: 4 nearest S per P
        d0           = self.psdist()
        self.ps4     = np.argsort(d0, axis=1)[:,:4]
        self.log     = open("PS4.log","w")
        self.log.write("# step  nbroken  maxPS[A]  phase  written\n")
        if(self.equil > 0):
            self.phase = "equil"
            self.odp.frozen_stage = 0     # hold the bias at its starting value
            print("  [PS4] equilibrating for %d steps with the bias held at stage 0"%(self.equil))
        else:
            self.write(0)                 # keep frame 0 only when there is nothing to equilibrate

    def psdist(self):
        cell = np.asarray(self.atoms.cell)
        inv  = np.linalg.inv(cell)
        f    = self.atoms.get_positions() @ inv
        d    = f[self.indP][:,None,:] - f[self.indS][None,:,:]
        d   -= np.round(d)
        return np.linalg.norm(d @ cell, axis=-1)

    def nbroken(self):
        ds = np.take_along_axis(self.psdist(), self.ps4, axis=1)
        return int((ds > self.rbreak).sum()), float(ds.max())

    def write(self, istep=0):
        if(istep == self.last): return 0
        self.traj.write()
        self.nwrite += 1
        self.last    = istep
        return 1

    def __call__(self):
        istep      = self.dyn.get_number_of_steps()
        nb, dmax   = self.nbroken()
        wrote      = 0
        if(self.phase == "equil"):
            if(istep < self.equil):
                if(istep % self.coarse == 0):
                    self.log.write(" %8d %6d %9.3f  %-8s %6d\n"%(istep,nb,dmax,self.phase,self.nwrite))
                    self.log.flush()
                return
            # equilibration done: release the bias and restart its ramp from stage 0
            self.phase = "intact"
            self.odp.stage_offset = int(self.odp.ncc / self.odp.targetNumSteps)
            self.odp.frozen_stage = None
            print("  [PS4] equilibration finished at step %d, bias ramp starts now"%(istep))
        if(nb >= self.nstart and self.phase == "intact"):
            self.phase   = "breaking"
            self.brkstep = istep
            print("  [PS4] step %d: first P-S broke (nbroken=%d). dense sampling every %d steps."%(istep,nb,self.dense))
            if(self.hold):
                # stop dragging the bias further: sample the transition instead of
                # blasting straight through it
                self.odp.frozen_stage = int(self.odp.ncc / self.odp.targetNumSteps)
                print("  [PS4] bias frozen at stage %d"%(self.odp.frozen_stage))
        if(self.phase == "breaking"):
            if(istep % self.dense == 0):  wrote = self.write(istep)
        else:
            if(istep % self.coarse == 0): wrote = self.write(istep)
        if(wrote or istep % self.coarse == 0):
            self.log.write(" %8d %6d %9.3f  %-8s %6d\n"%(istep,nb,dmax,self.phase,self.nwrite))
            self.log.flush()
        if(nb > self.nstop):
            print("  [PS4] step %d: nbroken=%d > %d - lattice is collapsing, stopping."%(istep,nb,self.nstop))
            raise StopMD()
        if(self.after > 0 and self.brkstep is not None and istep - self.brkstep >= self.after):
            print("  [PS4] step %d: %d steps sampled since the break, stopping."%(istep,self.after))
            raise StopMD()

monitor = PS4Monitor(atoms, dyn, traj_filename,
                     dense         = LMPinp.dense,
                     nbreak_start  = LMPinp.nbreak_start,
                     nbreak_stop   = LMPinp.nbreak_stop,
                     rbreak        = LMPinp.rbreak,
                     coarse        = num_interval,
                     hold_on_break = LMPinp.hold_on_break,
                     equil         = LMPinp.equil,
                     after         = LMPinp.after,
                     odp           = ODPcal)
dyn.attach(monitor, interval=LMPinp.dense)
dyn.attach(print_dyn, interval=num_interval)
dyn.attach(MDLogger(dyn, atoms, output_filename+".log", header=True,  peratom=True, mode="a"), interval=num_interval)
# Now run the dynamics
start_time = perf_counter()
print(f"    imd     Etot(eV)    T(K)    elapsed_time(sec)")
try:
    dyn.run(LMPinp.nstep)
except StopMD:
    print("  MD stopped early by the PS4 monitor at step %d"%(dyn.get_number_of_steps()))
print("  frames written: %d  (final phase: %s)"%(monitor.nwrite, monitor.phase))
elapsed_time = perf_counter() - start_time
print(" Total  wall  time:  %3f"%(elapsed_time))
