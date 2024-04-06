
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import model_to_dot
import visualkeras

# Load the model from the .h5 file
model = load_model("../smnist.h5")

from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
from collections import defaultdict

from tensorflow.keras import layers
from collections import defaultdict
color_map = defaultdict(dict)
color_map[layers.Conv2D]['fill'] = '#ff006d'
color_map[layers.MaxPooling2D]['fill'] = '#ffdd00'
color_map[layers.Dropout]['fill'] = '#ff7d00'
color_map[layers.Dense]['fill'] = '#01befe'
color_map[layers.Flatten]['fill'] = '#adff02'
color_map[layers.BatchNormalization]['fill'] = '#8f00ff'
visualkeras.layered_view(model, color_map=color_map, to_file='output.png').show()
