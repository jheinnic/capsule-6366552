from skimage.io import imread
import numpy as np
from varints import sqliteu

src = imread('/tmp/tmp4bzryghl/baseline_content.png')
flat = src.flatten()
mini = flat[0:64*1024]
val = sqliteu.encode(mini)

