import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from sklearn.utils import linear_assignment_
from tensorflow.contrib.layers import convolution2d, batch_norm, fully_connected, max_pool2d, flatten
from tensorflow.examples.tutorials.mnist import input_data
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import normalize
from PIL import Image
from munkres import Munkres
import math
import timeit
import logging
import os.path
import tensorflow.keras.layers as keras_layers
import tensorflow.keras.backend as K
import pathlib
def bestMap(L1, L2):
    if L1.__len__() != L2.__len__():
        print('size(L1) must == size(L2)')

    Label1 = np.unique(L1)
    nClass1 = Label1.__len__()
    Label2 = np.unique(L2)
    nClass2 = Label2.__len__()

    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        for j in range(nClass2):
            G[i][j] = np.nonzero((L1 == Label1[i]) * (L2 == Label2[j]))[0].__len__()

    c = linear_assignment_.linear_assignment(-G.T)[:, 1]
    newL2 = np.zeros(L2.__len__())
    for i in range(nClass2):
        for j in np.nonzero(L2 == Label2[i])[0]:
            if len(Label1) > c[i]:
                newL2[j] = Label1[c[i]]

    return accuracy_score(L1, newL2)

def conv_block(input_tensor,
               filters,
               kernel_size,
               strides=(2, 2),
               padding="same"):
    x = keras_layers.Conv2D(filters, kernel_size=kernel_size, strides=strides,
                      kernel_initializer='he_normal',padding=padding,
                      )(input_tensor)
    x = keras_layers.BatchNormalization(axis=-1)(x)
    x = keras_layers.Activation('relu')(x)
    return x
class Jule():

    Ks = 20  # the number of nearest neighbours of a sample
    Kc = 5  # the number of nearest clusters of a cluster
    a = 1.0
    l = 1.0  # lambda
    alpha = 0  # -0.2
    epochs = 20  # 20
    batch_size = 100
    gamma_tr = 2  # weight of positive pairs in weighted triplet loss.
    margin = 0.2  # margin for weighted triplet loss
    num_nsampling = 20  # number of negative samples for each positive pairs to construct triplet.
    gamma_lr = 0.0001  # gamma for inverse learning rate policy
    power_lr = 0.75  # power for inverse learning rate policy
    p = 0
    iter_cnn = 0

    def __init__(self, dataset, RC=True, updateCNN=True, eta=0.9):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.dataset = dataset
        self.RC = RC
        self.updateCNN = updateCNN
        self.eta = eta
        self.tic = timeit.default_timer()
        self.output_path = pathlib.Path("./results/" + dataset )

        if not self.output_path.exists():
            self.output_path.mkdir()
        self.weight_save_path = self.output_path.joinpath("params/weight.h5")
        if not self.weight_save_path.parent.exists():
            self.weight_save_path.parent.mkdir()
        # set up logging to file - see previous section for more details
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename='./logfile/' + dataset + time.strftime("%d-%m-%Y_") + time.strftime("%H-%M-%S") +".log",
                            filemode='w')
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
        self.logger = logging.getLogger('')
        self.logger.info('%.2f s, Begin to extract dataset', timeit.default_timer() - self.tic)
        self.tic = timeit.default_timer()
        # Input data
        if 'mnist' in dataset:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            #mnist = input_data.read_data_sets('MNIST_data', dtype=tf.uint8, one_hot=False)
            self.image_size1 = 28
            self.image_size2 = 28
            self.channel = 1
            self.K = 10
            if "mnist-full" in dataset:
                self.images = np.reshape(x_train,[x_train.shape[0],-1])
                self.gnd = y_train
                self.logger.info('%.2f s, Finished extracting MNIST-full dataset', timeit.default_timer() - self.tic)
            else:
                self.images = np.reshape(x_test, [x_test.shape[0], -1])
                self.gnd = y_test
                self.logger.info('%.2f s, Finished extracting MNIST-test dataset', timeit.default_timer() - self.tic)
        # if 'mnist-test' in dataset:
        #     mnist = input_data.read_data_sets('MNIST_data', dtype=tf.uint8, one_hot=False)
        #     self.images = mnist.test.images
        #     self.gnd = mnist.test.labels
        #     self.image_size1 = 28
        #     self.image_size2 = 28
        #     self.channel = 1
        #     self.K = 10
        #     self.logger.info('%.2f s, Finished extracting MNIST-test dataset', timeit.default_timer() - self.tic)
        # elif 'mnist-full' in dataset:
        #     mnist = input_data.read_data_sets('MNIST_data', dtype=tf.uint8, one_hot=False)
        #     images = np.concatenate((mnist.test.images, mnist.validation.images), axis=0)
        #     gnd = np.concatenate((mnist.test.labels, mnist.validation.labels), axis=0)
        #     self.images = np.concatenate((mnist.train.images, images), axis=0)
        #     self.gnd = np.concatenate((mnist.train.labels, gnd), axis=0)
        #     self.image_size1 = 28
        #     self.image_size2 = 28
        #     self.channel = 1
        #     self.K = 10
        #     self.logger.info('%.2f s, Finished extracting MNIST-full dataset', timeit.default_timer() - self.tic)
        elif 'usps' in dataset:
            usps = fetch_mldata("USPS")
            self.images = usps.data
            self.gnd = usps.target.astype(np.int) - 1
            self.image_size1 = 16
            self.image_size2 = 16
            self.channel = 1
            self.K = 10
            self.logger.info('%.2f s, Finished extracting USPS dataset', timeit.default_timer() - self.tic)
        elif 'coil20' in dataset:
            self.image_size1 = 128
            self.image_size2 = 128
            self.channel =1
            self.images = np.zeros((20*72, self.image_size1 * self.image_size2), np.uint8)
            self.gnd = np.zeros(20*72, np.uint8)
            path = './dataset/coil-20-proc/obj'
            for i in range(20):
                for j in range(72):
                    img_name = path + str(i+1) + '__' + str(j) + '.png'
                    img = Image.open(img_name)
                    img.load()
                    img_data = np.asarray(img, dtype=np.uint8)
                    self.images[i*72+j, :] = np.reshape(img_data, (1, self.image_size1 * self.image_size2))
                    self.gnd[i*72+j] = i
            self.K = 20
            self.logger.info('%.2f s, Finished extracting COIL20 dataset', timeit.default_timer() - self.tic)
        elif 'coil100' in dataset:
            self.image_size1 = 128
            self.image_size2 = 128
            self.channel = 3
            self.images = np.zeros((100*72, self.image_size1 * self.image_size2 * self.channel), np.uint8)
            self.gnd = np.zeros(100*72, np.uint8)
            path = './dataset/coil-100/obj'
            for i in range(100):
                for j in range(72):
                    img_name = path + str(i+1) + '__' + str(j*5) + '.png'
                    img = Image.open(img_name)
                    img.load()
                    img_data = np.asarray(img, dtype=np.uint8)
                    self.images[i*72+j, :] = np.reshape(img_data, (1, self.image_size1 * self.image_size2 * self.channel))
                    self.gnd[i*72+j] = i
            self.K = 100
            self.logger.info('%.2f s, Finished extracting COIL100 dataset', timeit.default_timer() - self.tic)
        elif 'umist' in dataset:
            self.image_size1 = 112
            self.image_size2 = 92
            self.channel = 1
            self.images = np.zeros((575, self.image_size1 * self.image_size2), np.uint8)
            self.gnd = np.zeros(575, np.uint8)
            path = './dataset/UMist/'
            n = 0
            for i in range(20):
                j = 0
                while True:
                    img_name = path + chr(97 + i) + str(j + 1) + '.pgm' # ord('a')=97
                    if not os.path.isfile(img_name):
                        break
                    img = Image.open(img_name)
                    img.load()
                    img_data = np.asarray(img, dtype=np.uint8)
                    self.images[n, :] = np.reshape(img_data, (1, self.image_size1 * self.image_size2))
                    self.gnd[n] = i
                    j += 1
                    n += 1
            self.K = 20
            self.logger.info('%.2f s, Finished extracting UMist dataset', timeit.default_timer() - self.tic)

        self.Ns = np.size(self.images, 0)
        self.num_batch = int(np.ceil(1.0 * self.Ns / self.batch_size))

        # Optimizer: set up a variable that's incremented once per batch and controls the learning rate decay.
        self.global_step = tf.Variable(0)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        self.learning_rate = tf.train.exponential_decay(
            0.01,  # Base learning rate.
            self.iter_cnn, #self.global_step,  # Current index into the dataset.
            self.Ns,  # Decay step.
            0.99995,  # Decay rate.
            staircase=False)
    def build_model(self):
        # CNN
        self.input_x = tf.placeholder(tf.float32, shape=[None, self.image_size1 * self.image_size2 * self.channel])
        inputs = keras_layers.Input(tensor=self.input_x)
        x = keras_layers.Reshape([self.image_size1, self.image_size2, self.channel])(inputs)
        if 'mnist' in self.dataset: # 28 * 28
            x = conv_block(x,filters=50,kernel_size=(5, 5),strides=(1,1),padding="valid")
            x = keras_layers.MaxPooling2D((2,2),strides=(2,2),padding="same")(x)
            x = conv_block(x,filters=50,kernel_size=(5, 5),strides=(1,1),padding="valid")
        elif 'usps' in self.dataset: # 16 * 16
            x = conv_block(x, filters=50, kernel_size=(5, 5), strides=(1, 1), padding="valid")
            x = keras_layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)
        elif 'coil' in self.dataset: # 128 * 128
            x = conv_block(x, filters=50, kernel_size=(5, 5), strides=(1, 1), padding="valid")
            x = keras_layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)

            x = conv_block(x, filters=50, kernel_size=(5, 5), strides=(1, 1), padding="valid")
            x = keras_layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)

            x = conv_block(x, filters=50, kernel_size=(5, 5), strides=(1, 1), padding="valid")
            x = keras_layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)

            x = conv_block(x, filters=50, kernel_size=(5, 5), strides=(1, 1), padding="valid")
            x = keras_layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)

            # net = convolution2d(data, num_outputs=50, kernel_size=(5, 5), stride=(1, 1), padding='VALID',
            #                     normalizer_fn=batch_norm, activation_fn=tf.nn.relu)  # 124 * 124
            # net = max_pool2d(net, [2,2], [2,2], padding='SAME')  # 62 * 62
            # net = convolution2d(net, num_outputs=50, kernel_size=(5, 5), stride=(1, 1), padding='VALID',
            #                     normalizer_fn=batch_norm, activation_fn=tf.nn.relu)  # 58 * 58
            # net = max_pool2d(net, [2,2], [2,2], padding='SAME')  # 29 * 29
            # net = convolution2d(net, num_outputs=50, kernel_size=(5, 5), stride=(1, 1), padding='VALID',
            #                     normalizer_fn=batch_norm, activation_fn=tf.nn.relu)  # 25 * 25
            # net = max_pool2d(net, [2,2], [2,2], padding='SAME')  # 13 * 13
            # net = convolution2d(net, num_outputs=50, kernel_size=(5, 5), stride=(1, 1), padding='VALID',
            #                     normalizer_fn=batch_norm, activation_fn=tf.nn.relu)  # 9 * 9
            # net = max_pool2d(net, [2,2], [2,2], padding='SAME')  # 5 * 5
        elif 'umist' in self.dataset: # 112 * 92
            x = conv_block(x, filters=50, kernel_size=(5, 5), strides=(1, 1), padding="valid")
            x = keras_layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)

            x = conv_block(x, filters=50, kernel_size=(5, 5), strides=(1, 1), padding="valid")
            x = keras_layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)

            x = conv_block(x, filters=50, kernel_size=(5, 5), strides=(1, 1), padding="valid")
            x = keras_layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)

            # net = convolution2d(data, num_outputs=50, kernel_size=(5, 5), stride=(1, 1), padding='SAME',
            #                     normalizer_fn=batch_norm, activation_fn=tf.nn.relu)  # 112 * 92
            # net = max_pool2d(net, [2,2], [2,2], padding='SAME')  # 56 * 46
            # net = convolution2d(net, num_outputs=50, kernel_size=(5, 5), stride=(1, 1), padding='SAME',
            #                     normalizer_fn=batch_norm, activation_fn=tf.nn.relu)  # 56 * 46
            # net = max_pool2d(net, [2,2], [2,2], padding='SAME')  # 28 * 23
            # net = convolution2d(net, num_outputs=50, kernel_size=(5, 5), stride=(1, 1), padding='SAME',
            #                     normalizer_fn=batch_norm, activation_fn=tf.nn.relu)  # 28 * 23
            # net = max_pool2d(net, [2,2], [2,2], padding='SAME')  # 14 * 12

        print(x.get_shape())
        x = keras_layers.Flatten()(x)
        print(x.get_shape())
        x = keras_layers.Dense(160)(x)
        self.model = tf.keras.Model(inputs=inputs,outputs=x)
        # self.input_y = tf.placeholder(tf.int32, shape=[None])
        self.anc_idx = tf.placeholder(tf.int32, shape=[None])
        self.pos_idx = tf.placeholder(tf.int32, shape=[None])
        self.neg_idx = tf.placeholder(tf.int32, shape=[None])

        fea_batch = self.model(self.input_x)

        # Get trplets
        anc_idx, pos_idx, neg_idx = tf.py_func(self.get_triplet, [self.input_y], [tf.int64, tf.int64, tf.int64])
        anc = tf.gather(fea_batch, anc_idx)
        pos = tf.gather(fea_batch, pos_idx)
        neg = tf.gather(fea_batch, neg_idx)

        # Triplet loss fuction
        d_pos = tf.reduce_sum(tf.square(anc - pos), -1)
        d_neg = tf.reduce_sum(tf.square(anc - neg), -1)
        loss = tf.maximum(0., self.margin + self.gamma_tr * d_pos - d_neg)
        self.loss = tf.reduce_mean(loss)

        # Inverse learning rate policy
        learning_rate = self.learning_rate * np.power(1 + self.gamma_lr * self.iter_cnn, - self.power_lr)
        # train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=self.global_step)
        train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
        self.sess.run(tf.global_variables_initializer())

    def clusters_init(self, indices):
        # initialize labels for input data given knn indices
        start  = timeit.default_timer()
        labels = -np.ones(self.Ns, np.int)
        num_class = 0
        for i in range(self.Ns):
            pos = []
            cur_idx = i
            while labels[cur_idx] == -1:
                pos.append(cur_idx)
                neighbor = indices[cur_idx, 0]
                labels[cur_idx] = -2
                cur_idx = neighbor
                if np.size(pos) > 50:
                    break
            if labels[cur_idx] < 0:
                labels[cur_idx] = num_class
                num_class += 1
            for idx in pos:
                labels[idx] = labels[cur_idx]

        self.Nc = num_class
        self.logger.info('%.2f s, Finish clusters initialization,init clusters is %d, consuming: %.2fs', timeit.default_timer() - self.tic, self.Nc,timeit.default_timer() - start)
        return labels

    def get_Dis(self, fea, k):
        # Calculate Dis
        self.logger.info('%.2f s, Begin to fit neighbour graph', timeit.default_timer() - self.tic)
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(fea)
        self.logger.info('%.2f s, Finished fitting, begin to calculate Dis', timeit.default_timer() - self.tic)
        sortedDis, indexDis = neigh.kneighbors()
        self.logger.info('%.2f s, Finished the calculation of Dis', timeit.default_timer() - self.tic)

        return sortedDis, indexDis
    def get_asymA_fast(self,W, C):
        Nc = len(C)
        asymA = np.zeros((Nc, Nc))
        W_row_collapsed = np.zeros((Nc, W.shape[1]))
        W_col_collapsed = np.zeros((Nc, W.shape[1]))
        for i in range(Nc):
            W_row_collapsed[i] = np.sum(W[C[i]], 0, keepdims=False)
            W_col_collapsed[i] = np.sum(W[:, C[i]], 1, keepdims=False)
        collapsed_W = W_row_collapsed * W_col_collapsed
        for i in range(Nc):
            asymA[:, i] = np.sum(collapsed_W[:, C[i]], 1) / math.pow(
                        np.size(C[i], 0), 2)
        np.fill_diagonal(asymA,0)
        return asymA
    def get_asymA_slow(self,W,C):
        Nc = len(C)
        asymA = np.zeros((Nc, Nc))
        for i in range(Nc):
            #self.logger.info('%.2f s, Calculating A..., i: %d', timeit.default_timer() - self.tic, i)
            for j in range(i):
                if np.size(C[j], 0) != 0:
                    # asymA[i, j] = np.dot(np.sum(W[C[i], :][:, C[j]], 0), np.sum(W[C[j], :][:, C[i]], 1))  # A(Ci -> Cj)
                    asymA[i, j] = np.dot(np.sum(W[np.meshgrid(C[i], C[j],indexing='ij')], 0), np.sum(W[np.meshgrid(C[j], C[i],indexing='ij')], 1)) / math.pow(
                        np.size(C[j], 0), 2)  # A(Ci -> Cj)
                if np.size(C[i], 0) != 0:
                    # asymA[j, i] = np.dot(np.sum(W[C[j], :][:, C[i]], 0), np.sum(W[C[i], :][:, C[j]], 1))  # A(Cj -> Ci)
                    asymA[j, i] = np.dot(np.sum(W[np.meshgrid(C[j], C[i],indexing='ij')], 0), np.sum(W[np.meshgrid(C[i], C[j],indexing='ij')], 1)) / math.pow(
                        np.size(C[i], 0), 2)  # A(Cj -> Ci)
                # A[i, j] = asymA[i, j]/math.pow(np.size(C[j], 0), 2) + asymA[j, i]/ math.pow(np.size(C[i], 0), 2)
        return asymA
    def get_A(self, fea, sortedDis, indexDis, C):
        # Calculate W
        sortedDis = np.power(sortedDis, 2)
        sig2 = sortedDis.sum() / (self.Ks * self.Ns) * self.a
        XI = np.transpose(np.tile(range(self.Ns), (self.Ks, 1)))
        W = csr_matrix((np.exp(-sortedDis.flatten() * (1 / sig2)), (XI.flatten(), indexDis.flatten())),
                       shape=(self.Ns, self.Ns)).toarray()
        self.logger.info('%.2f s, Finished the calculation of W, sigma:%f', timeit.default_timer() - self.tic, np.sqrt(sig2))

        # Calculate A
        start = timeit.default_timer()
        self.logger.info('%.2f s, Calculating A...',start - self.tic)
        asymA = self.get_asymA_fast(W,C)
        A = asymA + asymA.T
        self.logger.info('%.2f s, Finishing  calculating A...,Consuming: %.2fs', timeit.default_timer() - self.tic, timeit.default_timer() - start)
        # Assert whether there are some self-contained clusters
        num_fea = np.size(fea, 1)
        if self.Nc > 20 * self.K:
            asymA_sum_row = np.sum(asymA, 0)
            asymA_sum_col = np.sum(asymA, 1)

            X_clusters = np.zeros((self.Nc, num_fea))
            for i in range(self.Nc):
                X_clusters[i, :] = np.mean(fea[C[i], :], 0)
            neigh = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(X_clusters)
            self.logger.info('%.2f s, Fit the nearest neighbor graoh of clusters', timeit.default_timer() - self.tic)

            i = 0
            while i < self.Nc:
                # Find the cluster ids whose affinities are both 0
                if asymA_sum_row[i] == 0 and asymA_sum_col[i] == 0:
                    indices = neigh.kneighbors(X_clusters[i, :].reshape(1, -1), return_distance=False)
                    for j in indices[0]:
                        if i != j:
                            break
                    self.logger.info('%.2f s, merge self-contained cluster %d with cluster %d', timeit.default_timer() - self.tic, i, j)
                    minIndex1 = min(i, j)
                    minIndex2 = max(i, j)
                    A, asymA, C = self.merge_cluster(A, asymA, C, minIndex1, minIndex2)
                    asymA_sum_row = np.sum(asymA, 0)
                    asymA_sum_col = np.sum(asymA, 1)

                    # update X_clusters
                    X_clusters[minIndex1, :] = np.mean(fea[C[minIndex1], :], 0)
                    X_clusters[minIndex2, :] = X_clusters[self.Nc, :]
                    X_clusters = X_clusters[0 : self.Nc, :]
                    neigh = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(X_clusters)
                else:
                    i += 1

        self.logger.info('%.2f s, Calculation of A based on clusters is completed', timeit.default_timer() - self.tic)

        return A, asymA, C

    def find_closest_clusters(self, A):
        # Find two clusters with the smallest loss
        np.fill_diagonal(A, - float('Inf'))
        indexA = A.argsort(axis=1)[:, ::-1][:, 0: self.Kc]
        sortedA = np.sort(A, axis=1)[:, ::-1][:, 0: self.Kc]

        minLoss = float('Inf')
        minIndex1 = -1
        minIndex2 = -2
        for i in range(self.Nc):
            loss = - (1 + self.l) * sortedA[i, 0] + sum(sortedA[i, 1: self.Kc]) * self.l / (self.Kc - 1)
            if loss < minLoss and i != indexA[i, 0]:
                minLoss = loss
                minIndex1 = min(i, indexA[i, 0])
                minIndex2 = max(i, indexA[i, 0])

        self.logger.info('%.2f s, number of clusters: %d, merge cluster %d and %d, loss: %f', timeit.default_timer() - self.tic, self.Nc, minIndex1, minIndex2, minLoss)
        return minIndex1, minIndex2

    def merge_cluster(self, A, asymA, C, minIndex1, minIndex2):

        # Merge
        cluster1 = C[minIndex1]
        cluster2 = C[minIndex2]
        new_cluster = cluster1 + cluster2

        # Update the merged cluster and its affinity
        C[minIndex1] = new_cluster
        asymA[minIndex1, 0: self.Nc] = asymA[minIndex1, 0: self.Nc] + asymA[minIndex2, 0: self.Nc]
        len1 = np.size(cluster1, 0)
        len2 = np.size(cluster2, 0)
        # asymA[0: self.Nc, minIndex1] = (asymA[0: self.Nc, minIndex1] * len1 + asymA[0: self.Nc, minIndex2] * len2) / (len1 + len2)
        asymA[0 : self.Nc, minIndex1] = asymA[0 : self.Nc, minIndex1] * (1 + self.alpha) * math.pow(len1, 2) / math.pow(len1 + len2, 2)\
                + asymA[0 : self.Nc, minIndex2] *(1 + self.alpha) * math.pow(len2, 2) / math.pow(len1 + len2, 2)
        asymA[minIndex1, minIndex1] = 0

        A[minIndex1, :] = asymA[minIndex1, :] + asymA[:, minIndex1]
        A[:, minIndex1] = A[minIndex1, :]

        # Replace the second cluster to be merged with the last cluster of the cluster array
        if (minIndex2 != self.Nc-1):
            C[minIndex2] = C[-1]
            asymA[0: self.Nc, minIndex2] = asymA[0: self.Nc, self.Nc - 1]
            asymA[minIndex2, 0: self.Nc] = asymA[self.Nc - 1, 0: self.Nc]
            asymA[minIndex2, minIndex2] = 0
            A[0: self.Nc, minIndex2] = A[0: self.Nc, self.Nc - 1]
            A[minIndex2, 0: self.Nc] = A[self.Nc - 1, 0: self.Nc]
            A[minIndex2, minIndex2] = 0

        # Remove the last cluster
        C.pop()
        asymA = asymA[0 : self.Nc - 1, 0 : self.Nc - 1]
        A = A[0 : self.Nc - 1, 0 : self.Nc - 1]
        self.Nc -= 1

        return A, asymA, C
    @staticmethod
    def get_C(labels):
        num_sam = np.size(labels)
        labels_from_one = np.zeros(num_sam, np.int)

        idx_sorted = labels.argsort(0)
        labels_sorted = labels[idx_sorted]
        nclusters = 0
        label = -1
        for i in range(num_sam):
            if labels_sorted[i] != label:
                label = labels_sorted[i]
                nclusters += 1
            labels_from_one[idx_sorted[i]] = nclusters

        C = []
        for i in range(nclusters):
            C.append([])
        for i in range(num_sam):
            C[labels_from_one[i] - 1].append(i)

        return C

    def get_labels(self, C):
        # Generate sample labels

        labels = np.zeros(self.Ns, np.int)
        for i in range(np.size(C)):
            labels[C[i]] = i

        return labels

    def evaluation(self, labels_pre):
        labels_gt = self.gnd
        unilabel_gt = np.unique(labels_gt)
        unilabel_pre = np.unique(labels_pre)

        num_gt = np.size(unilabel_gt)
        num_pre = np.size(unilabel_pre)

        G = np.zeros((num_gt, num_pre))
        for i in range(num_gt):
            for j in range(num_pre):
                G[i, j] = sum(np.logical_and(labels_gt == unilabel_gt[i], labels_pre == unilabel_pre[j]))

        m = Munkres()
        c = m.compute(-G)

        labels_pre_new = np.copy(labels_pre)
        for i in range(len(c)):
            labels_pre_new[labels_pre == unilabel_pre[c[i][1]]] = unilabel_gt[c[i][0]]
        AC = float(np.count_nonzero( labels_gt == labels_pre_new)) / self.Ns
        #NMI = normalized_mutual_info_score(labels_gt, labels_pre)
        # pr_gt = np.sum(G, 1)*1.0/self.Ns
        # pr_pre = np.sum(G, 0)*1.0/self.Ns
        #
        # # Entropy
        # H_gt = - np.sum(pr_gt * np.log(pr_gt))
        # H_pre = - np.sum(pr_pre * np.log(pr_pre))
        # H_gp = -np.sum(np.multiply(G*1.0/self.Ns, np.log(G*1.0/self.Ns+1e-10)))  # Joint entropy
        #
        # MI = H_gt + H_pre - H_gp
        # NMI = MI / np.sqrt(H_gt * H_pre)
        NMI = normalized_mutual_info_score(self.gnd, labels_pre)
        self.logger.info('%.2f s, AC: %f, NMI: %f', timeit.default_timer() - self.tic, AC, NMI)
        return AC, NMI


    def evaluation2(self, labels_pre):
        self.logger.info('Evaluatiing')
        #AC = bestMap(self.gnd,labels_pre)
        NMI = normalized_mutual_info_score(self.gnd, labels_pre)
        self.logger.info('%.2f s, NMI: %f', timeit.default_timer() - self.tic, NMI)
        return NMI
    
    def get_triplet(self, labels_batch):

        num_sam = np.size(labels_batch)
        C_batch = self.get_C(labels_batch)
        nclusters = np.size(C_batch,0)

        if nclusters <= self.num_nsampling:
            num_neg_sampling = nclusters - 1
        else:
            num_neg_sampling = self.num_nsampling

        num_triplet = 0
        for i in range(nclusters):
            num_Ci = np.size(C_batch[i])
            num_triplet += num_Ci * (num_Ci - 1) * num_neg_sampling // 2

        if num_triplet == 0:
            return 0,0,0

        anc = np.zeros(num_triplet, np.int32)
        pos = np.zeros(num_triplet, np.int32)
        neg = np.zeros(num_triplet, np.int32)

        id_triplet = 0
        for i in range(nclusters):
            if np.size(C_batch[i]) > 1:
                for m in range(np.size(C_batch[i])):
                    for n in range(m+1, np.size(C_batch[i])):
                        is_choosed = np.zeros(num_sam, np.int8)
                        while True:
                            id_s = np.random.randint(num_sam)
                            if is_choosed[id_s] == 0 and labels_batch[id_s] != i:
                                anc[id_triplet] = C_batch[i][m]
                                pos[id_triplet] = C_batch[i][n]
                                neg[id_triplet] = id_s
                                is_choosed[id_s] = 1
                                id_triplet += 1
                                if id_triplet % num_neg_sampling == 0:
                                    break

        # self.logger.info('%.2f s, get %d triplets, anc: %d, pos: %d, neg: %d ', timeit.default_timer() - self.tic, num_triplet, anc[-1],pos[-1],neg[-1])

        return anc, pos, neg

    def preidct(self,images):

        #Extract features, get the new feature representation
        fea = []
        x = tf.placeholder(tf.float32, shape=[None, self.image_size1 * self.image_size2 * self.channel])
        fea_batch = self.model(x)
        for i in range(self.num_batch):
            if i != self.num_batch - 1:
                batch = self.images[self.batch_size * i: self.batch_size * (i + 1), :]
                fea.append(self.sess.run(fea_batch, feed_dict={x: batch}))
            else:
                batch = self.images[self.batch_size * i:, :]
                fea.append(self.sess.run(fea_batch, feed_dict={x: batch}))
            self.logger.info('%.2f s, Period: %d, batch: %d, feature extracting...',
                             timeit.default_timer() - self.tic, self.p, i)
        return np.concatenate(fea)
    def train(self, labels):
        x = tf.placeholder(tf.float32, shape=[None, self.image_size1 * self.image_size2 * self.channel])
        #y = tf.placeholder(tf.int32, shape=[None])
        anc_idx = tf.placeholder(tf.int32,shape=[None])
        pos_idx = tf.placeholder(tf.int32, shape=[None])
        neg_idx = tf.placeholder(tf.int32, shape=[None])

        fea_batch = self.model(x)

        # Get trplets
        #anc_idx, pos_idx, neg_idx = tf.py_func(self.get_triplet, [y], [tf.int64, tf.int64, tf.int64])
        anc = tf.gather(fea_batch, anc_idx)
        pos = tf.gather(fea_batch, pos_idx)
        neg = tf.gather(fea_batch, neg_idx)

        # Triplet loss fuction
        d_pos = tf.reduce_sum(tf.square(anc - pos), -1)
        d_neg = tf.reduce_sum(tf.square(anc - neg), -1)
        loss = tf.maximum(0., self.margin + self.gamma_tr * d_pos - d_neg)
        loss = tf.reduce_mean(loss)

        # Inverse learning rate policy
        learning_rate = self.learning_rate * np.power(1 + self.gamma_lr * self.iter_cnn, - self.power_lr)

        # train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=self.global_step)
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        train_step = optimizer.minimize(loss)
        self.sess.run(tf.global_variables_initializer())
        for e in range(self.epochs):
            start = timeit.default_timer()
            self.logger.info('%.2f s, Period: %d, epoch: %d, training...', timeit.default_timer() - self.tic, self.p, e)
            index_rand = np.random.permutation(self.Ns)
            epoch_loss = 0
            for i in range(self.num_batch):
                if i != self.num_batch - 1:
                    index = index_rand[self.batch_size * i: self.batch_size * (i + 1)]
                    labels_batch = labels[index][:,None]
                    batch = self.images[index, :]
                else:
                    index = index_rand[self.batch_size * i: ]
                    labels_batch = labels[index][:,None]
                    batch = self.images[index]
                anc_idx_data, pos_idx_data, neg_idx_data = self.get_triplet(labels_batch)
                if not np.shape(anc_idx_data) == 0:
                    continue
                _, ln, triplet_loss, features = self.sess.run([train_step, learning_rate, loss, fea_batch],
                                                              feed_dict={x: batch,K.learning_phase():1,
                                                                         anc_idx: anc_idx_data,pos_idx:pos_idx_data,neg_idx:neg_idx_data})
                # print "fea_batch:", features
                self.iter_cnn += 1
                epoch_loss += triplet_loss
                # inverse learning rate policy
                learning_rate = self.learning_rate * np.power(1 + self.gamma_lr * self.iter_cnn, - self.power_lr)
                # self.logger.info('%.2f s, Period: %d, epoch: %d, batch: %d, leaning rate: %f, triplet loss: %f',
                #                 timeit.default_timer() - self.tic, self.p, e, i, ln, triplet_loss)
            self.logger.info('%.2f s, Epoch %d,leaning rate: %f, average loss: %f, consuming: %.2fs',
                             timeit.default_timer() - self.tic, e,epoch_loss/self.num_batch,timeit.default_timer() - start)
        self.logger.info('%.2f s, Period: %d, finished cnn training', timeit.default_timer() - self.tic, self.p)

        # Extract features, get the new feature representation
        # num_fea = fea_batch.get_shape().as_list()[1]
        # fea = np.zeros((self.Ns, num_fea))
        # for i in range(self.num_batch):
        #     if i != self.num_batch - 1:
        #         batch = self.images[self.batch_size * i: self.batch_size * (i + 1), :]
        #         fea[self.batch_size * i: self.batch_size * (i + 1), :] = self.sess.run(fea_batch, feed_dict={x: batch})
        #     else:
        #         batch = self.images[self.batch_size * i:, :]
        #         fea[self.batch_size * i:, :] = self.sess.run(fea_batch, feed_dict={x: batch})
        #     self.logger.info('%.2f s, Period: %d, epoch: %d, batch: %d, feature extracting...',
        #                      timeit.default_timer() - self.tic, self.p, e, i)
        self.logger.info('%.2f s, Period: %d, finished extraction of feature',
                         timeit.default_timer() - self.tic, self.p)
        self.p += 1

    def recurrent_process(self, features, updateCNN):

        fea = normalize(features, axis=1)

        if updateCNN:
            sortedDis, indexDis = self.get_Dis(fea, 1)

            labels = self.clusters_init(indexDis)
            self.evaluation2(labels)
            self.train(labels)
            #fea = self.preidct(self.images)
            fea = self.model.predict(self.images,self.batch_size)
            sortedDis, indexDis = self.get_Dis(fea, self.Ks)
        else:
            #with self.sess.as_default():
            self.model.load_weights(self.weight_save_path.__str__())
            # self.sess.run(tf.global_variables_initializer())
            #fea = self.preidct(self.images)
            fea = self.model.predict(fea,self.batch_size)
            sortedDis, indexDis = self.get_Dis(fea, self.Ks)
            labels = self.clusters_init(indexDis)
            self.evaluation2(labels)

        # sortedDis, indexDis = self.get_Dis(fea, self.Ks)
        # labels = self.clusters_init(indexDis)
        # self.evaluation2(labels)

        C = self.get_C(labels)
        A, asymA, C = self.get_A(fea, sortedDis, indexDis, C)

        t = 0
        ts = 0
        Np = np.ceil(self.eta * self.Nc)

        while self.Nc > self.K:
            t += 1
            index1, index2 = self.find_closest_clusters(A)
            A, asymA, C = self.merge_cluster(A, asymA, C, index1, index2)

            # if updateCNN and self.Nc == self.K:
            if updateCNN and t == ts + Np:
                labels = self.get_labels(C)
                self.evaluation2(labels)

                self.train(labels)
                #fea = self.preidct(self.images)
                fea = self.model.predict(self.images,self.batch_size)
                fea = normalize(fea, axis=1)

                # Update A based on the new feature representation
                sortedDis, indexDis = self.get_Dis(fea, self.Ks)
                A, asymA, C = self.get_A(fea, sortedDis, indexDis, C)

                ts = t
                Np = np.ceil(self.eta * self.Nc)

        return C, fea

    def run(self):
        C, fea = self.recurrent_process(self.images, self.updateCNN)
        labels = self.get_labels(C)
        acc_sf,nmi_sf = self.evaluation(labels)
        if self.updateCNN:
            np.savez_compressed(self.output_path.joinpath('features_'+ self.dataset), fea, 'features')
            self.logger.info('%.2f s, deep representations saved', timeit.default_timer() - self.tic)
            self.model.save_weights(self.weight_save_path.__str__())

        if self.RC and self.updateCNN:
            self.logger.info('%.2f s, begin to re-run clustering',  timeit.default_timer() - self.tic)
            C, fea = self.recurrent_process(fea, updateCNN = False)
            labels = self.get_labels(C)
            acc_sr, nmi_sr = self.evaluation2(labels)
            print("Final num_clusters: %d,ACC_SR: %f,NMI_SR :%f" % (self.Nc,acc_sr, nmi_sr))

        print("Final num_clusters: %d,ACC_SF: %f,NMI_SF :%f" % (self.Nc,acc_sf,nmi_sf))

if __name__ == "__main__":
    jule = Jule('mnist-test', RC = True, updateCNN = False, eta = 0.2)
    jule.build_model()
    jule.run()