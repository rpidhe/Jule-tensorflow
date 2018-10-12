import time

import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist
import os
import tensorflow.keras.backend as K
import numpy as np
flag = tf.app.flags
img_height = 28
img_width = 28
img_size = img_height * img_width
output_size = 10

flag.DEFINE_integer("batch_size",100,"batch size")
flag.DEFINE_integer("iteration_step",10000,"iteration step")
FLAGS = flag.FLAGS
def full_connect(input_data,output_size,scope="linear",stddev=0.02, bias_start=0.0):
    input_size = input_data.shape[-1]
    with tf.variable_scope(scope):
        w = tf.get_variable("Matrix",[input_size,output_size],tf.float32,tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("Bias",[output_size],tf.float32,tf.constant_initializer(value=bias_start))
        return tf.matmul(input_data,w) + bias

if __name__ == "__main__":
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(FLAGS.batch_size).repeat()
    x_data = tf.placeholder(tf.float32,[None,img_height,img_width],"x_data")
    y_data = tf.placeholder(tf.int32,[None],"y_data")
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[img_height,img_width]),
        tf.keras.layers.Lambda(lambda x:tf.expand_dims(x,axis=-1)),
        tf.keras.layers.Conv2D(3,kernel_size=3,strides=1,padding="same",activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(output_size,activation=tf.nn.softmax)]
    )
    arr = tf.constant(np.random.randn(100,2))
    save_weight = "simple_weight.h5"
    y_pred = model(x_data)
    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_data,y_pred))
    #loss = tf.keras.losses.categorical_crossentropy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1,output_type=tf.int32), y_data), dtype=tf.float32))
    global_step = tf.get_variable("globe_step",shape=(),dtype=tf.int32)
    learing_rate = tf.train.exponential_decay(
        0.01,  # Base learning rate.
        global_step,  # self.global_step,  # Current index into the dataset.
        10,  # Decay step.
        0.9,  # Decay rate.
        staircase=False)

    train_step = tf.train.AdamOptimizer(learing_rate).minimize(loss,global_step=global_step)
    iter = dataset.make_one_shot_iterator()
    next_data = iter.get_next()
    start = time.time()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        if os.path.exists(save_weight):
            model.load_weights(save_weight)
        # arr_slice_pre = session.run(arr_slice,feed_dict={"Direction:0":-1})
        # arr_slice_post = session.run(arr_slice, feed_dict={"Direction:0": 1})

        print('test accuracy %f' % accuracy.eval(feed_dict={x_data: x_test, y_data: y_test}))
        #y_tmp_pred = model.predict(x_test)
        #right = np.equal(np.argmax(y_tmp_pred, 1), y_test).astype(np.float32)
       # print('model predication test accuracy %f' % np.mean(right))
        for i in range(FLAGS.iteration_step):
            x,y = session.run(next_data)
            feed_dict = {x_data: x, y_data: y}
            _,loss_val,learning_phase,lr = session.run((train_step,loss,K.learning_phase(),learing_rate),feed_dict=feed_dict)
            print("Curing learning rate:",lr)
            if i%50 == 0:
                train_accuracy = session.run(accuracy,feed_dict=feed_dict)
                print("Accuracy: %f,Loss: %f ,Learning Phase %s" % (train_accuracy,loss_val,learning_phase))
        model.save_weights(save_weight,save_format='h5')
        print('test accuracy %f' % accuracy.eval(feed_dict={x_data: x_test, y_data: y_test}))
    print(time.time()-start,"sec")
