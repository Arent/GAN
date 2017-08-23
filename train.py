import tensorflow as tf
import numpy as np
from PIL import Image
from hyper_parameters import *
from image_loader import *
from model import *
import datetime
import os
import math
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)
run_identifier = '{:%Y-%b-%d-%H-%M-%S}'.format(datetime.datetime.now())


def create_save_location():
    assert os.path.isdir("".join([model_folder, run_identifier])) == False
    os.mkdir("".join([model_folder, run_identifier]))
    os.mkdir("".join([model_folder, run_identifier, "/", sample_folder]))
    return "".join([model_folder, run_identifier, "/"])
save_location = create_save_location()


def rescale_and_save_image(image_array, iBatch):
    image_array = np.uint8((image_array + 1) *
                           127.5)  # rescale and make into integer\
    image = Image.fromarray(image_array)
    image.save("".join([model_folder, run_identifier,
                        "/", sample_folder, str(iBatch), ".jpeg"]))


def initialise_graph_train(session):
    # Builds the graph, create data loading operations and initialises variables
    # Get que with training files

    build_graph()  # this creates all variables and operations needed to train

    # initialize the variables
    session.run(tf.local_variables_initializer())
    session.run(tf.global_variables_initializer())

    return tf.get_default_graph()


def initialise_graph_retrain(session):
    # Restorres the graph, delete and recreate data loading operations
    # and initialises local variables

    saver = tf.train.import_meta_graph(
        "".join([model_folder, model_identifier, "/", model_name]))
    # restore variables
    saver.restore(sess, tf.train.latest_checkpoint(
        "".join([model_folder, model_identifier, "/"])))
    graph = tf.get_default_graph()
    # Delete old data loader objects and create new ones. (Cant find a better
    # way...)
    graph.clear_collection("queue_runners")
    graph.clear_collection("local_variables")


    # initialize the local variables, global variables have been restored
    sess.run(tf.local_variables_initializer())

    return graph

# Get que with training files
# n_train_files, n_test_files\
#     , train_image_batch, train_label_batch, test_image_batch, test_label_batch = get_batches()
batches_per_epoch =  500 #int(math.ceil(float(n_train_files) / BATCH_SIZE))



# Launch graph
with tf.Session() as sess:
    # get training operations and placeholders
    if run_type == "train":
        graph = initialise_graph_train(sess)
    elif run_type == "retrain":
        graph = initialise_graph_retrain(sess)
    else:
        raise NotImplementedError

    # Get training operations from graph
    train_op_discrim = graph.get_operation_by_name(
        "train_operation_discriminator")
    train_op_gen = graph.get_operation_by_name("train_operation_generator")
    images_tensor = graph.get_tensor_by_name('model/fake_images:0')

    # Create a saver object to check progress of training
    saver = tf.train.Saver(max_to_keep=3)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    summary_writer = tf.summary.FileWriter(
        "".join(['tensorboard_logs', "/", run_identifier]), graph)

    # Get placeholders
    Z = graph.get_tensor_by_name('Z:0')
    real_images = graph.get_tensor_by_name('real_images:0')

  # initialize the queue threads to start to shovel data
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord)
    iBatch = 0
    for iBatch in range(batches_per_epoch):
    # try:
    #     while not coord.should_stop():
        print(iBatch)

        # retrieve and display epoch and batch
        epoch = int(iBatch / batches_per_epoch) + 1
        batch_number = iBatch % batches_per_epoch
        print('Epoch: ' + str(epoch) + '/' + str(EPOCHS) + ' Batch: ' +
              str(batch_number) + '/' + str(batches_per_epoch - 1))

        # Get Z values and image batch
        z_batch = np.random.uniform(-1, 1,
                                    size=[BATCH_SIZE, Z_DIMENSION]).astype(np.float32)
        image_batch = mnist.train.next_batch(BATCH_SIZE)[0]# sess.run(train_image_batch, options=run_options)
        image_batch = image_batch.reshape([64,28,28,1])
        image_batch = np.pad(image_batch, pad_width=((0, 0), (2, 2), (2, 2), (0,0)), mode='constant')
        # Do the acual training
        _,  = sess.run([train_op_gen], options=run_options, feed_dict={
                       Z: z_batch, real_images: image_batch})
        _,  = sess.run([train_op_discrim], options=run_options, feed_dict={
                       Z: z_batch, real_images: image_batch})

        # Add summaries of variables to tensorboard
        merged_summary_op = tf.summary.merge_all()    
        summary_str = sess.run(merged_summary_op, options=run_options,
                            feed_dict={Z: z_batch, real_images: image_batch})
        summary_writer.add_summary(summary_str, iBatch)
        if iBatch == 0:  # Save metagraph only once
            saved_path = saver.save(
                sess, save_location + str(batch_number), write_meta_graph=True)

        if batch_number % 20 == 0:  # save variable state and make sample image
            images = sess.run(images_tensor, feed_dict={Z: z_batch})
            rescale_and_save_image(images[0, :, :, :], iBatch)
            saved_path = saver.save(sess, save_location + "Epoch_" + str(
                epoch) + "_Batch_" + str(batch_number), write_meta_graph=False)
            print("Model saved in file: %s" % saved_path)
        iBatch = iBatch + 1

    # except Exception, e:
    #     # Report exceptions to the coordinator.
    #     coord.request_stop(e)

    # finally:
    # stop our queue threads and properly close the session
    print("This run can be identified as {}".format(run_identifier))
    print("Run the command line:\n"
          "--> tensorboard --logdir=tensorboard_logs "
          "\nThen open http://0.0.0.0:6006/ into your web browser")

     # Add meta data such as computational time per operation
    summary_writer.add_run_metadata(run_metadata, 'Mysess')
    # coord.request_stop()
    # coord.join(threads)
    sess.close()
