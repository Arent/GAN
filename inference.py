import tensorflow as tf
import numpy as np
from PIL import Image
import hyper_parameters

BATCH_SIZE = 10  # overwrite batch size if needed


def rescale_and_save_images(images):
    for i, image_array in enumerate(images):  # first dimension in the image
        image_array = np.uint8((image_array + 1) *
                               127.5)  # rescale and make into integer\
        image = Image.fromarray(image_array)
        image.save("".join([model_folder, model_identifier,
                            "/", sample_folder, str(i), ".jpeg"]))


# Launch graph
with tf.Session() as sess:
    # Restore the graph and variables
    saver = tf.train.import_meta_graph(
        "".join([model_folder, model_identifier, "/", model_name]))
    saver.restore(sess, tf.train.latest_checkpoint(
        "".join([model_folder, model_identifier, "/"])))

    # retrieve operations to generate fake images using the variables
    graph = tf.get_default_graph()
    Z = graph.get_tensor_by_name('Z:0')
    images_tensor = graph.get_tensor_by_name('model/fake_images:0')

    # generate random z and use to generate images
    z_batch = np.random.uniform(-1, 1,
                                size=[BATCH_SIZE, Z_DIMENSION]).astype(np.float32)
    images = sess.run(images_tensor, feed_dict={Z: z_batch})
    rescale_and_save_images(images)
