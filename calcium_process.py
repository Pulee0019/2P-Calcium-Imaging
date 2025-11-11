# import suite2p

# fname = r"D:\Expriment\Calcium\animal1752\suite2p1\output\Image_scan_1_region_0_0_processed.tif" # Let's say input is of shape (4200, 325, 556)
# Lx, Ly = 512, 512 # Lx and Ly are the x and y dimensions of the imaging input
# # Read in our input tif and convert it to a BinaryRWFile
# f_input = suite2p.io.BinaryRWFile(Ly=Ly, Lx=Lx, filename=fname)

import numpy as np
data = np.load(r"D:\Expriment\Calcium\animal1752\suite2p1\output\suite2p\plane0\F.npy", allow_pickle=True)
print(data)