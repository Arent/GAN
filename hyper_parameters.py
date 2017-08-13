'paremeters regarding data'
PATH_FOLDER = 'data/image_locations'
LABEL_FOLDER = 'data/image_labels'
IMAGE_HEIGHT  = 64
IMAGE_WIDTH   = 64
NUM_CHANNELS  = 3

'parameters regarding saving and restoring'
model_folder = "saved_models/"
sample_folder = 'samples/'
model_identifier = "2017-Aug-08-16-52-09"
model_name = "Epoch_1_Batch_0.ckpt.meta"



'parameters regarding training'
TEST_SET_SIZE = 5
BATCH_SIZE    = 64
EPOCHS = 100
NORMALISATION_DECAY = 0.9
RELU_ALPHA = 1/6
BETA_ADAM = 0.5
LEARNING_RATE = 0.0002

'parameters regarding convolutions'
Z_DIMENSION = 100
KERNEL_WIDTH = 4
KERNEL_HEIGHT = 4
