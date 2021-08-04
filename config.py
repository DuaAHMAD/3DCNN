import socket
import os

# Global parameters
LEN_SAMPLE = 50
NONOVERLAP = 25

# IMAGE_HEIGHT = H  = 480
# IMAGE_WIDTH  = W = 640

N_CHANNELS = C = 3
ALIGNED_FACE_HEIGHT = F_H = 80
ALIGNED_FACE_WIDTH = F_W = 90


N_DIGITS = 6


CLASSES =    {
  "Controls": 0,
  "Patients": 1
}

# Machine-dependent paths
name = socket.gethostname()
if name == "wolfgang":
    RAW_DATA_PATH = "/timo/datasets/AV/BDI/Raw_Recordings"
    DERIVED_DATA_PATH = "/timo/datasets/AV/BDI/Temp_from_ext_HDD/"
else:
    raise Exception("Local data folder not specified for '{}'.".format(name))


# Sub-folder structures
ALIGNED_FACES_FOLDER = os.path.join(DERIVED_DATA_PATH, "OLD60AlignedFaces", "{exp}")