import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tensorflow as tf
import GPy
import GPyOpt
import keras
from keras.layers import Input, Dense, Lambda, InputLayer, concatenate, Activation, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Deconv2D
from keras.losses import MSE
from keras.models import Model, Sequential
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.utils import np_utils
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import utils
import os

sess = tf.InteractiveSession()
K.set_session(sess)

latent_size = 8

vae, encoder, decoder = utils.create_vae(batch_size=128, latent=latent_size)
sess.run(tf.global_variables_initializer())
vae.load_weights('CelebA_VAE_small_8.h5')

K.set_learning_phase(False)

latent_placeholder = tf.placeholder(tf.float32, (1, latent_size))
decode = decoder(latent_placeholder)


class FacialComposit:
    def __init__(self, decoder, latent_size):
        self.latent_size = latent_size
        self.latent_placeholder = tf.placeholder(tf.float32, (1, latent_size))
        self.decode = decoder(self.latent_placeholder)
        self.samples = None
        self.images = None
        self.rating = None

    def _get_image(self, latent):
        img = sess.run(self.decode,
                       feed_dict={self.latent_placeholder: latent[None, :]})[0]
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def _show_images(images, titles, clear=True):
        assert len(images) == len(titles)
        if clear:
            clear_output()
        plt.figure(figsize=(3 * len(images), 3))
        n = len(titles)
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.imshow(images[i])
            plt.title(str(titles[i]))
            plt.axis('off')
        plt.show()

    @staticmethod
    def _draw_border(image, w=2):
        bordred_image = image.copy()
        bordred_image[:, :w] = [1, 0, 0]
        bordred_image[:, -w:] = [1, 0, 0]
        bordred_image[:w, :] = [1, 0, 0]
        bordred_image[-w:, :] = [1, 0, 0]
        return bordred_image

    def query_initial(self, n_start=5, select_top=None):
        '''
        Creates initial points for Bayesian optimization
        Generate *n_start* random images and asks user to rank them.
        Gives maximum score to the best image and minimum to the worst.
        :param n_start: number of images to rank initialy.
        :param select_top: number of images to keep
        '''
        if select_top is None:
            select_top = n_start
        self.samples = np.random.normal(size=(select_top, self.latent_size))
        self.images = np.zeros(shape=(select_top, 64, 64, 3))
        self.rating = np.zeros(select_top)

        for i in range(select_top):
            self.images[i] = sess.run(self.decode, feed_dict={self.latent_placeholder:
                                                                  self.samples[i].reshape((1, latent_size))})[0]

        ### Show user some samples (hint: use self._get_image and input())
        ratings = []
        FacialComposit._show_images(self.images, range(select_top))
        print("Enter image by order of similarity (most similar first):")
        order_str = input()
        order = [int(i) for i in order_str.split()]

        images = np.zeros(shape=(select_top, 64, 64, 3))
        samples = np.zeros(shape=(len(self.images), self.latent_size))

        for i, rating in zip(order, range(select_top)):
            images[rating] = self.images[i]
            samples[rating] = self.samples[i]

        self.images = images
        self.samples = samples
        self.rating = np.array([float(i) for i in reversed(range(1, len(self.images) + 1))])

        print("Ordered images:")
        self._show_images(self.images, self.rating, clear=False)

        print(self.rating)

        # Check that tensor sizes are correct
        np.testing.assert_equal(self.rating.shape, [select_top])
        np.testing.assert_equal(self.images.shape, [select_top, 64, 64, 3])
        np.testing.assert_equal(self.samples.shape, [select_top, self.latent_size])

    def recreate_ratings(self):
        self.rating = np.zeros(len(self.images))

        rating_sum = 0.
        for i in range(len(self.images)):
            rating_sum += i
        for i in range(len(self.images)):
            self.rating[i] = (len(self.images) - i)

    def evaluate(self, candidate):
        """
        Queries candidate vs known image set.
        Adds candidate into images pool.
        :param candidate: latent vector of size 1xlatent_size
        """
        print("candidate", candidate.shape)
        initial_size = len(self.images)

        ## Show user an image and ask to assign score to it.
        ## You may want to show some images to user along with their scores
        ## You should also save candidate, corresponding image and rating
        choices = np.random.choice(np.array(range(1, len(self.images) - 1)), size=3, replace=False)
        choices = np.insert(choices, 0, 0)
        choices = np.append(choices, len(self.images) - 1)
        images = self.images[choices]
        ratings = self.rating[choices]

        image = self._get_image(candidate.reshape(self.latent_size))

        irs = sorted(zip(images, ratings, choices), key=lambda x: -x[1])
        sorted_images = [ir[0] for ir in irs]
        sorted_ratings = [ir[1] for ir in irs]
        sorted_choices = [ir[2] for ir in irs]

        print("best ratings: {}".format(self.rating))
        self._show_images(self.images[:5], self.rating[:5], clear=False)

        #         print("samples: {}".format(self.samples))
        #         print("Candidate: {}".format(candidate))
        self._show_images([image], ["candidate"], clear=False)
        print("sorted ratings: {}".format(sorted_ratings))
        print("Lineup:")
        titles = ["{}: score: {}".format(i, sorted_ratings[i - 1]) for i in range(1, len(sorted_ratings) + 1)]
        self._show_images(sorted_images, titles=titles, clear=False)

        print("Give image score:")
        candidate_rating = float(input())
        candidate_order = self.insert_candidate(image, candidate, candidate_rating)

        # print("insert at: {}".format(candidate_order))
        # print("candidate_rating: {} sorted ratings: {}".format(candidate_rating, self.rating))
        # #         print("candidate: {}".format(candidate))
        # #         print("samples: {}".format(self.samples))
        # print("Lineup:")
        # self._show_images(self.images, titles=self.rating, clear=False)
        # input()

        assert len(self.images) == initial_size + 1
        assert len(self.rating) == initial_size + 1
        assert len(self.samples) == initial_size + 1
        return candidate_rating

    def insert_candidate(self, image, candidate, rating):
        i = 0
        while i < len(self.rating):
            if rating > self.rating[i]:
                break
            i += 1
        self.images = np.insert(self.images, i, image, axis=0)
        self.rating = np.insert(self.rating, i, rating, axis=0)
        self.samples = np.insert(self.samples, i, candidate, axis=0)
        return i

    def optimize(self, n_iter=10, w=4, acquisition_type='MPI', acquisition_par=0.3):
        if self.samples is None:
            self.query_initial()

        bounds = [{'name': 'z_{0:03d}'.format(i),
                   'type': 'continuous',
                   'domain': (-w, w)}
                  for i in range(self.latent_size)]
        initial_rating = -self.rating[:, None]

        optimizer = GPyOpt.methods.BayesianOptimization(f=self.evaluate, domain=bounds,
                                                        acquisition_type=acquisition_type,
                                                        acquisition_par=acquisition_par,
                                                        exact_eval=False,  # Since we are not sure
                                                        model_type='GP',
                                                        X=self.samples,
                                                        Y=initial_rating,
                                                        maximize=True)
        optimizer.run_optimization(max_iter=n_iter, eps=-1)

    def get_best(self):
        index_best = np.argmax(self.rating)
        return self.images[index_best]

    def draw_best(self, title=''):
        index_best = np.argmax(self.rating)
        image = self.images[index_best]
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()


composit = FacialComposit(decoder, 8)
composit.optimize()
