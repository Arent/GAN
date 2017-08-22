#parameters that are actually just settings
run_type = 'train' # train or retrain
PATH_FOLDER = 'data/image_locations'
LABEL_FOLDER = 'data/image_labels'
model_folder = "saved_models/"
sample_folder = 'samples/'
model_identifier = "2017-Aug-19-15-00-48"
model_name = "0.meta"


#parameters regarding data'
IMAGE_HEIGHT  = 64
IMAGE_WIDTH   = 64
NUM_CHANNELS  = 3

#parameters regarding training'
TEST_SET_SIZE = 4
BATCH_SIZE    = 16
EPOCHS = 1
NORMALISATION_DECAY = 0.9
RELU_ALPHA = 1/6
BETA_ADAM = 0.5
LEARNING_RATE = 0.0002

#parameters regarding convolutions'
Z_DIMENSION = 100
KERNEL_WIDTH = 4
KERNEL_HEIGHT = 4
