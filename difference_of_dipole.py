from pymatgen.io.vasp.outputs import Chgcar
import numpy as np

file_ML = "/home/luke/Desktop/MLDensity_tutorial-main/examples/benzene/chgcar_files_water/CHGCAR_1.2_final_0"

file_DFT = "/home/luke/Desktop/MLDensity_tutorial-main/examples/benzene/sfc_new/test/0/CHGCAR"

DFT_chg = Chgcar.from_file(
    file_DFT
)

ML_chg = Chgcar.from_file(
    file_ML
)

#we know the box is  15 x 15 x 15 Angstrom cubed and each axis has been divided into 224 grid points so. These are in cartesian coordinates 

x_DFT  = DFT_chg.get_axis_grid(0)   #these are in units of angstrom 
y_DFT  = DFT_chg.get_axis_grid(1)
z_DFT  = DFT_chg.get_axis_grid(2)



chg_DFT = DFT_chg.data['total'] #these are in units of electrons. they were already multiplied by volume of cell so are not in A^-3 



#sum alomng x and multiply and do element wise multiplication between x and chg 

#multiplying chg in electrons by a distance in angstrom gives units of eA


dx_DFT = np.einsum("x,xyz->",x_DFT,chg_DFT)/(len(x_DFT)*len(y_DFT)*len(z_DFT))      #x, y and z all have one axis
dy_DFT = np.einsum("y,xyz->",y_DFT,chg_DFT)/(len(x_DFT)*len(y_DFT)*len(z_DFT))      # chg is multi dimensional 
dz_DFT = np.einsum("z,xyz->",z_DFT,chg_DFT)/(len(x_DFT)*len(y_DFT)*len(z_DFT))      #x has been labelled as x. chg has 3 axes
print('for DFT', dx_DFT,dy_DFT,dz_DFT)                             #the first axis is x, the 2nd is y and the 3rd is z
                                            #dividing by the total number of grid points 


#we specify that the axes in chg are xyz, x is first axis, y is second axis etc 
#we do element wise multiplication of x and first axis of chg then y and second axis of chg etc



#we know the box is  15 x 15 x 15 Angstrom cubed and each axis has been divided into 224 grid points so. These are in cartesian coordinates 

x_ML  = ML_chg.get_axis_grid(0)   #these are in units of angstrom 
y_ML  = ML_chg.get_axis_grid(1)
z_ML  = ML_chg.get_axis_grid(2)



chg_ML = ML_chg.data['total'] #these are in units of electrons. they were already multiplied by volume of cell so are not in A^-3 



#sum alomng x and multiply and do element wise multiplication between x and chg 

#multiplying chg in electrons by a distance in angstrom gives units of eA


dx_ML = np.einsum("x,xyz->",x_ML,chg_ML)/(len(x_ML)*len(y_ML)*len(z_ML))      #x, y and z all have one axis
dy_ML = np.einsum("y,xyz->",y_ML,chg_ML)/(len(x_ML)*len(y_ML)*len(z_ML))      # chg is multi dimensional 
dz_ML = np.einsum("z,xyz->",z_ML,chg_ML)/(len(x_ML)*len(y_ML)*len(z_ML))      #x has been labelled as x. chg has 3 axes
print('for ML', dx_ML,dy_ML,dz_ML)                             #the first axis is x, the 2nd is y and the 3rd is z
                                            #dividing by the total number of grid points 


#we specify that the axes in chg are xyz, x is first axis, y is second axis etc 
#we do element wise multiplication of x and first axis of chg then y and second axis of chg etc
#These are in units of e*A lets convert them to C*m

e_charge = 1.60218e-19

conv = 3.33564e-30

dx_DFT = dx_DFT*(e_charge)*1e-10

dy_DFT = dy_DFT*(e_charge)*1e-10

dz_DFT = dz_DFT*(e_charge)*1e-10

#from wikipedia 1 Debye is 3.3e-30 C*m

#1C*m = 1/3.3e-30 D

dx_DFT = dx_DFT*(1/conv)

dy_DFT = dy_DFT*(1/conv)

dz_DFT = dz_DFT*(1/conv)


#In debye. the dipole moments are 

print('dipole moments in debye for DFT', dx_DFT/DFT_chg.data['total'].size, dy_DFT/DFT_chg.data['total'].size, dz_DFT/DFT_chg.data['total'].size)


#this gives us only the electronic. we will get the nuclear part later

#These are in units of e*A lets convert them to C*m

e_charge = 1.60218e-19

conv = 3.33564e-30

dx_ML = dx_ML*(e_charge)*1e-10

dy_ML = dy_ML*(e_charge)*1e-10

dz_ML = dz_ML*(e_charge)*1e-10

#from wikipedia 1 Debye is 3.3e-30 C*m

#1C*m = 1/3.3e-30 D

dx_ML = dx_ML*(1/conv)

dy_ML = dy_ML*(1/conv)

dz_ML = dz_ML*(1/conv)



#In debye. the dipole moments are 

print('dipole moments in debye for ML ', dx_ML/ML_chg.data['total'].size,dy_ML/ML_chg.data['total'].size,  dz_ML/ML_chg.data['total'].size)


#this gives us only the electronic. we will get the nuclear part later
dx_DFT = dx_DFT/DFT_chg.data['total'].size
dy_DFT = dy_DFT/DFT_chg.data['total'].size
dz_DFT = dz_DFT/DFT_chg.data['total'].size

dx_ML = dx_ML/ML_chg.data['total'].size
dy_ML = dy_ML/ML_chg.data['total'].size
dz_ML = dz_ML/ML_chg.data['total'].size

zval_dict = {
    'O': 6.0,       #dictionary of what the valencies are 
    'H': 1.0
    
}

from pymatgen.core.structure import Structure, Lattice
import pymatgen.core as pmg


#this is just a dictionary of what elements are present and the valencies that they have 

def calc_ionic(site, structure: Structure, zval):
    """
    Calculate the ionic dipole moment using ZVAL from pseudopotential.

    site: PeriodicSite
    structure: Structure
    zval: Charge value for ion (ZVAL for VASP pseudopotential)

    Returns polarization in electron Angstroms.
    """
    norms = structure.lattice.lengths       #this is just (15.0, 15.0, 15.0) Angstrom 
    print('norms are', norms)
    print(site.frac_coords, zval)
    return np.multiply(norms, -site.frac_coords * zval) #multiplying norms by (fractional coordinates*valency)

def get_total_ionic_dipole(structure, zval_dict):
    """
    Get the total ionic dipole moment for a structure.

    structure: pymatgen Structure
    zval_dict: specie, zval dictionary pairs
    center (np.array with shape [3,1]) : dipole center used by VASP
    tiny (float) : tolerance for determining boundary of calculation.
    """
    tot_ionic = []
    for site in structure:          #site is just coordinates in cartesian space ie 0.75 x 15 Angstrom 


        zval = zval_dict[str(site.specie)]          #this is just the valency of the atom at a site 

        print(calc_ionic(site, structure, zval))

        tot_ionic.append(calc_ionic(site, structure, zval)) #calculating the ionic dipole moment for that site, structure and valency 


    print(tot_ionic)
    return np.sum(tot_ionic, axis=0)    #summing them 

from pymatgen.core.structure import Structure, Lattice
import pymatgen.core as pmg

structure_DFT = Structure.from_file(file_DFT)

structure_ML = Structure.from_file(file_ML)

TOT_DFT_ID = get_total_ionic_dipole(structure_DFT, zval_dict)

TOT_ML_ID = get_total_ionic_dipole(structure_ML, zval_dict)

print('total ionic dipole for DFT', TOT_DFT_ID)
print('total ionic dipole for ML', TOT_ML_ID)

ix_DFT,iy_DFT,iz_DFT = TOT_DFT_ID



ix_DFT = ix_DFT*e_charge*(1e-10)*(1/conv)


iy_DFT = iy_DFT*e_charge*(1e-10)*(1/conv)


iz_DFT = iz_DFT*e_charge*(1e-10)*(1/conv)

print('ionic dipole in Debye for DFT', ix_DFT, iy_DFT,  iz_DFT)
print('electric dipole in Debye for DFT',dx_DFT,  dz_DFT,dz_DFT )

tx_DFT = ix_DFT+dx_DFT
ty_DFT = iy_DFT+dz_DFT
tz_DFT = iy_DFT+dz_DFT
print('total DFT dipole for water,' ,tx_DFT,ty_DFT,tz_DFT)


ix_ML,iy_ML,iz_ML = TOT_ML_ID



ix_ML = ix_ML*e_charge*(1e-10)*(1/conv)


iy_ML = iy_ML*e_charge*(1e-10)*(1/conv)


iz_ML = iz_ML*e_charge*(1e-10)*(1/conv)

print('ionic dipole in Debye for ML', ix_ML, iy_ML,  iz_ML)
print('electric dipole in debye for ML', dx_ML, dy_ML, dz_ML)

tx_ML = ix_ML+dx_ML
ty_ML = iy_ML+dz_ML
tz_ML = iy_ML+dz_ML
print('total ML dipole for water,' ,tx_ML,ty_ML,tz_ML)

from pymatgen.core.structure import Structure, Lattice
import pymatgen.core as pmg


structure_DFT = Structure.from_file(file_DFT)

co_ords_DFT = []

for site in structure_DFT:
    co_ords_DFT.append(site.frac_coords)


structure_ML = Structure.from_file(file_ML)

co_ords_ML = []

for site in structure_ML:
    co_ords_ML.append(site.frac_coords)


O_xyz_DFT = np.array(co_ords_DFT[2])

H1_xyz_DFT = np.array(co_ords_DFT[0])    

H2_xyz_DFT = np.array(co_ords_DFT[1])



O_xyz_ML = np.array(co_ords_ML[2])

H1_xyz_ML = np.array(co_ords_ML[0])    

H2_xyz_ML = np.array(co_ords_ML[1])


midpoint_H_DFT = (H1_xyz_DFT + H2_xyz_DFT)/2

midpoint_H_ML = (H1_xyz_ML + H2_xyz_ML)/2




import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

v1_DFT = np.array([tx_DFT,
ty_DFT,
tz_DFT])


v1_ML = np.array([tx_ML,
ty_ML,
tz_ML])

length_DFT = np.linalg.norm(v1_DFT)

v1_DFT_norm = v1_DFT*(1/length_DFT)*(1/15)

length_ML = np.linalg.norm(v1_ML)

v1_ML_norm = v1_ML*(1/length_ML)*(1/15)


ax.quiver(midpoint_H_DFT[0],midpoint_H_DFT[1],midpoint_H_DFT[2], v1_DFT_norm[0], v1_DFT_norm[1], v1_DFT_norm[2], label ='DFT dipole', color = 'red')

ax.quiver(midpoint_H_ML[0],midpoint_H_ML[1],midpoint_H_ML[2], v1_ML_norm[0], v1_ML_norm[1], v1_ML_norm[2], label ='ML dipole')

ax.scatter(*O_xyz_DFT, c='r',  label='O')
ax.scatter(*H1_xyz_DFT, c='g',  label='H1')     #O_xyz_DFT is actually the same as O_xyz_ML
ax.scatter(*H2_xyz_DFT, c='b',  label='H2')


ax.set_xlim([0.7, 0.85])
ax.set_ylim([0.7, 0.85])
ax.set_zlim([0.7, 0.85])



ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.legend()

plt.show()

from matplotlib.ticker import FuncFormatter


def custom_format(x, pos):
    # Multiply tick values by a factor of 224
    return '{:.2f}'.format(x * 224)

fig, ax = plt.subplots()

O_xy_DFT = O_xyz_DFT = O_xyz_DFT[:-1]


H1_xy_DFT =  H1_xyz_DFT[:-1]
    #From the CHGCAR file 

H2_xy_DFT = H2_xyz_DFT[:-1]


V_DFT = np.array([v1_DFT_norm[0],v1_DFT_norm[1]])


O_xy_ML = O_xyz_ML = O_xyz_ML[:-1]


H1_xy_ML =  H1_xyz_ML[:-1]
    #From the CHGCAR file 

H2_xy_ML = H2_xyz_ML[:-1]


V_ML = np.array([v1_ML_norm[0],v1_ML_norm[1]])



fig, ax = plt.subplots()


ax.quiver(midpoint_H_DFT[0], midpoint_H_DFT[1], V_DFT[0], V_DFT[1], angles='xy', scale_units='xy', scale=1, label = 'DFT unit dipole')

ax.quiver(midpoint_H_ML[0], midpoint_H_ML[1], V_ML[0], V_ML[1], angles='xy', scale_units='xy', scale=1, label = 'ML unit dipole', color = 'red')

ax.scatter(O_xy_DFT[0], O_xy_DFT[1], c='r', label='O')
ax.scatter(H1_xy_DFT[0], H1_xy_DFT[1], c='g', label='H')
ax.scatter(H2_xy_DFT[0], H2_xy_DFT[1], c='b', label='H')

ax.legend()
ax.set_title("Dipole direction")


ax.set_xlim([0.625, 0.89])
ax.set_ylim([0.625, 0.89])

# Apply the custom formatting to the x and y axes
ax.xaxis.set_major_formatter(FuncFormatter(custom_format))
ax.yaxis.set_major_formatter(FuncFormatter(custom_format))



# Add a title to your plot
plt.title('Dipole direction')

plt.grid()
plt.show()



import math

def angle_between_vectors(vector1, vector2):
    # Ensure both vectors have the same dimension
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same dimension")

    # Calculate the dot product of the two vectors
    dot_product = sum(x * y for x, y in zip(vector1, vector2))

    # Calculate the magnitude (norm) of each vector
    magnitude1 = math.sqrt(sum(x ** 2 for x in vector1))
    magnitude2 = math.sqrt(sum(y ** 2 for y in vector2))

    # Calculate the cosine of the angle between the vectors
    cosine_theta = dot_product / (magnitude1 * magnitude2)

    # Calculate the angle in radians using the arccosine function
    angle_radians = math.acos(cosine_theta)

    # Convert radians to degrees
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees

# Example usage:
vector1 = [3, 4]  # Replace with your first vector
vector2 = [1, 2]  # Replace with your second vector

angle = angle_between_vectors(V_DFT, V_ML)
print(f"The angle between the vectors is {angle} degrees.")
