"""
@uthor: Himaghna, 4th September 2019
Description: Write config file (json)
"""


import json

config_fname = 'config.json'
config_dict = {
    'xl_file': 'D:\\Research\\Phosphine-Ligands\\data_matrix_builder.xlsx',
    'target_column': 'E/Z Ratio',
    'descrptr_columns': ['Bond Lengths_angstrom1', 'Bond Lengths_angstrom2', 
                        'Bond Lengths_angstrom3', 'Bond Angles_degrees1', 
                        'Bond Angles_degrees2', 'Bond Angles_degrees3', 
                        'Mulliken_Charges1', 'Mulliken_Charges2',
                        'Mulliken_Charges3', 'Cone Angle _Ligand only', 
                        'Sterimol_Parameters1','Sterimol_Parameters2',
                        'Sterimol_Parameters3',
                        'Cone Angle - Dummy Pd - P = 2.28', 'IR Frequency1',
                        'IR Frequency2', 'IR Frequency3', 'IR Frequency4',
                        'IR Frequency5', 'IR Frequency6', 'APT Charge1', 
                        'APT Charge2', 'APT Charge3', 'APT Charge4', 
                        'APT Charge5', '31P-NMR Magnetic Shielding (ppm)'],
    'bins_upper_limit' : [700, 800, 900, 1000, 1100, 1200]
}
print(f"Writing to file {config_fname}")
json.dump( config_dict, open(config_fname, "w"))
print("Write Succesful")