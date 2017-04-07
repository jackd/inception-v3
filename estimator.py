"""Estimator implementation of inception-v3 model."""
import os
import tensorflow as tf
import tensorflow.contrib.learn as learn


class InceptionV3Estimator(learn.BaseEstimator):
    """
    Estimator implementation of inception-v3 model.

    See https://arxiv.org/pdf/1512.00567.pdf for details.

    Reproduction of models/slim/nets/inception_v3.py, from
    https://github.com/tensorflow/models
    """

    def __init__(self, model_id='base'):
        """Initialize with a model_id."""
        super(InceptionV3Estimator, self).__init__(
            os.path.join(os.path.dirname(__file__), 'models', model_id))

    def _conv(self,
              inputs,
              num_outputs,
              kernel_size,
              stride,
              padding,
              training):
        scale = False
        momentum = 0.99
        x = tf.layers.conv2d(
            inputs, num_outputs, kernel_size, stride, padding,
            activation=tf.nn.relu)
        return tf.layers.batch_normalization(
            x, training=training, scale=scale, momentum=momentum)
        # normalizer_params = {
        #     'is_training': training,
        #     'scale': scale,
        #     'decay': momentum,
        # }
        # return tf.contrib.layers.conv2d(
        #     inputs, num_outputs, kernel_size, stride, padding,
        #     activation_fn=tf.nn.relu,
        #     normalizer_fn=tf.contrib.layers.batch_norm,
        #     normalizer_params=normalizer_params)

    def _max_pool(self, inputs, kernel_size, stride, padding):
        return tf.layers.max_pooling2d(inputs, kernel_size, stride, padding)

    def _average_pool(self, inputs, kernel_size, stride, padding):
        return tf.layers.average_pooling2d(
            inputs, kernel_size, stride, padding)

    def _check_shape(self, x, shape):
        if x.get_shape().as_list()[1:] != shape:
            raise Exception('Shape check failed: expected %s, got %s' %
                            (str(shape), str(x.get_shape())))

    def _fig5(self, x, training):
        with tf.name_scope('fig5'):
            with tf.name_scope('branch0'):
                b0 = x
                b0 = self._conv(b0, 64, 1, 1, "SAME", training)
            with tf.name_scope('branch1'):
                b1 = x
                b1 = self._conv(b1, 48, 1, 1, "SAME", training)
                b1 = self._conv(b1, 64, 5, 1, "SAME", training)
            with tf.name_scope('branch2'):
                b2 = x
                b2 = self._conv(b2, 64, 1, 1, "SAME", training)
                b2 = self._conv(b2, 96, 3, 1, "SAME", training)
                b2 = self._conv(b2, 96, 3, 1, "SAME", training)
            with tf.name_scope('branch3'):
                b3 = x
                b3 = self._average_pool(b3, 3, 1, "SAME")
                b3 = self._conv(b3, 64, 1, 1, "SAME", training)
            x = tf.concat([b0, b1, b2, b3], axis=3)
        return x

    # def _fig5b(self, x, training):
    #     with tf.name_scope('fig5b'):
    #         with tf.name_scope('branch0'):
    #             b0 = x
    #             b0 = self._conv(b0, 64, 1, 1, "SAME", training)
    #         with tf.name_scope('branch1'):
    #             b1 = x
    #             b1 = self._conv(b1, 48, 1, 1, "SAME", training)
    #             b1 = self._conv(b1, 64, 5, 1, "SAME", training)
    #         with tf.name_scope('branch2'):
    #             b2 = x
    #             b2 = self._conv(b2, 64, 1, 1, "SAME", training)
    #             b2 = self._conv(b2, 96, 3, 1, "SAME", training)
    #             b2 = self._conv(b2, 96, 3, 1, "SAME", training)
    #         with tf.name_scope('branch3'):
    #             b3 = x
    #             b3 = self._average_pool(b3, 3, 1, "SAME")
    #             b3 = self._conv(b3, 64, 1, 1, "SAME", training)
    #         x = tf.concat([b0, b1, b2, b3], axis=3)
    #     return x
    #
    # def _fig5c(self, x, training):
    #     with tf.name_scope('fig5c'):
    #         with tf.name_scope('branch0'):
    #             b0 = x
    #             b0 = self._conv(b0, 64, 1, 1, "VALID", training)
    #         with tf.name_scope('branch1'):
    #             b1 = x
    #             b1 = self._max_pool(b1, 2, 1, "VALID")
    #             b1 = self._conv(b1, 192, 1, 1, "VALID", training)
    #         with tf.name_scope('branch2'):
    #             b2 = x
    #             b2 = self._conv(b2, 192, 1, 1, "VALID", training)
    #             b2 = self._conv(b2, 192, 3, 1, "VALID", training)
    #         with tf.name_scope('branch3'):
    #             b3 = x
    #             b3 = self._conv(b3, 192, 1, 1, "VALID", training)
    #             b3 = self._conv(b3, 192, 3, 1, "SAME", training)
    #             b3 = self._conv(b3, 192, 3, 1, "VALID", training)
    #         x = tf.concat([b0, b1, b2, b3], axis=3)
    #     return x

    def _fig5_6_bridge(self, x, training):
        with tf.name_scope('bridge5_6'):
            with tf.name_scope('branch0'):
                b0 = x
                b0 = self._max_pool(b0, 3, 2, "VALID")
            with tf.name_scope('branch1'):
                b1 = x
                b1 = self._conv(b1, 384, 1, 1, "VALID", training)
                b1 = self._conv(b1, 384, 3, 2, "VALID", training)
            with tf.name_scope('branch2'):
                b2 = x
                b2 = self._conv(b2, 64, 1, 1, "VALID", training)
                b2 = self._conv(b2, 96, 1, 1, "VALID", training)
                b2 = self._conv(b2, 96, 3, 2, "VALID", training)
            return tf.concat([b0, b1, b2], axis=3)

    def _fig6(self, x, training):
        n = 7
        with tf.name_scope('fig6'):
            with tf.name_scope('branch0'):
                b0 = x
                b0 = self._conv(b0, 192, 1, 1, "SAME", training)
            with tf.name_scope('branch1'):
                b1 = x
                b1 = self._average_pool(b1, 1, 1, "SAME")
                b1 = self._conv(b1, 128, 1, 1, "SAME", training)
                b1 = self._conv(b1, 192, [1, n], 1, "SAME", training)
                b1 = self._conv(b1, 192, [n, 1], 1, "SAME", training)
            with tf.name_scope('branch2'):
                b2 = x
                b2 = self._conv(b2, 128, 1, 1, "SAME", training)
                b2 = self._conv(b2, 128, [1, n], 1, "SAME", training)
                b2 = self._conv(b2, 128, [n, 1], 1, "SAME", training)
                b2 = self._conv(b2, 192, [1, n], 1, "SAME", training)
                b2 = self._conv(b2, 192, [n, 1], 1, "SAME", training)
            with tf.name_scope('branch3'):
                b3 = x
                b3 = self._average_pool(b3, 3, 1, "SAME")
                b3 = self._conv(b3, 192, 1, 1, "SAME", training)
            x = tf.concat([b0, b1, b2, b3], axis=3)
        return x

    def _fig6_7_bridge(self, x, training):
        with tf.name_scope('bridge6_7'):
            with tf.name_scope('branch0'):
                b0 = x
                b0 = self._max_pool(b0, 3, 2, "VALID")
            with tf.name_scope('branch1'):
                b1 = x
                b1 = self._average_pool(b1, 1, 1, "VALID")
                b1 = self._conv(b1, 320, 3, 2, "VALID", training)
            with tf.name_scope('branch2'):
                b2 = x
                for i in range(3):
                    b2 = self._conv(b2, 192, 1, 1, "VALID", training)
                b2 = self._conv(b2, 192, 3, 2, "VALID", training)
            x = tf.concat([b0, b1, b2], axis=3)
        return x

    # made it up...
    def _fig7(self, x, training):
        n = 3
        with tf.name_scope('fig7'):
            with tf.name_scope('branch0'):
                b0 = x
                b0 = self._conv(b0, 320, 1, 1, "SAME", training)
            with tf.name_scope('branch1'):
                b1 = x
                b1 = self._average_pool(b1, 3, 1, "SAME")
                b1 = self._conv(b1, 320, 1, 1, "SAME", training)
            with tf.name_scope('branch2'):
                b2 = x
                b2 = self._conv(b2, 320, 1, 1, "SAME", training)
                with tf.name_scope('branch2a'):
                    b2a = self._conv(b2, 160, [1, n], 1, "SAME", training)
                with tf.name_scope('branch2b'):
                    b2b = self._conv(b2, 160, [n, 1], 1, "SAME", training)
            with tf.name_scope('branch3'):
                b3 = x
                b3 = self._conv(b3, 320, 1, 1, "SAME", training)
                b3 = self._conv(b3, 320, 3, 1, "SAME", training)
                with tf.name_scope('branch2a'):
                    b3a = self._conv(b3, 160, [1, n], 1, "SAME", training)
                with tf.name_scope('branch2b'):
                    b3b = self._conv(b3, 160, [n, 1], 1, "SAME", training)

            x = tf.concat([b0, b1, b2a, b2b, b3a, b3b], axis=3)
        return x

    def _fig7_out_bridge(self, x, training):
        with tf.name_scope('bridge7_out'):
            with tf.name_scope('branch0'):
                b0 = x
                b0 = self._conv(b0, 1024, 3, 1, "SAME", training)
            with tf.name_scope('branch1'):
                b1 = x
                b1 = self._conv(b1, 1024, 1, 1, "SAME", training)
                b1 = self._max_pool(b1, 3, 1, "SAME")
            x = tf.concat([b0, b1], axis=3)
        return x

    def get_logits(self, image, training):
        """Build the inception_v3 model up to the logits."""
        x = image

        self._check_shape(x, [299, 299, 3])
        x = self._conv(x, 32, 3, 2, "VALID", training)

        self._check_shape(x, [149, 149, 32])
        x = self._conv(x, 32, 3, 1, "VALID", training)

        self._check_shape(x, [147, 147, 32])
        x = self._conv(x, 64, 3, 1, "SAME", training)

        self._check_shape(x, [147, 147, 64])
        x = self._max_pool(x, 3, 2, "VALID")

        self._check_shape(x, [73, 73, 64])
        x = self._conv(x, 80, 3, 1, "VALID", training)

        self._check_shape(x, [71, 71, 80])
        x = self._conv(x, 192, 3, 2, "VALID", training)

        self._check_shape(x, [35, 35, 192])
        x = self._conv(x, 288, 3, 1, "SAME", training)

        self._check_shape(x, [35, 35, 288])
        # x = self._fig5a(x, training)
        # x = self._fig5b(x, training)
        # x = self._fig5c(x, training)
        for i in range(3):
            x = self._fig5(x, training)

        self._check_shape(x, [35, 35, 288])
        x = self._fig5_6_bridge(x, training)

        self._check_shape(x, [17, 17, 768])
        # x = self._fig6a(x, training)
        # x = self._fig6b(x, training)
        # x = self._fig6c(x, training)
        # x = self._fig6d(x, training)
        # x = self._fig6e(x, training)
        for i in range(5):
            x = self._fig6(x, training)

        self._check_shape(x, [17, 17, 768])
        x = self._fig6_7_bridge(x, training)

        self._check_shape(x, [8, 8, 1280])
        # x = self._fig7a(x, training)
        # x = self._fig7b(x, training)
        for i in range(2):
            x = self._fig7(x, training)

        self._check_shape(x, [8, 8, 1280])
        x = self._fig7_out_bridge(x, training)

        self._check_shape(x, [8, 8, 2048])
        x = self._average_pool(x, 8, 1, "VALID")

        self._check_shape(x, [1, 1, 2048])
        x = tf.squeeze(x, axis=[1, 2])

        self._check_shape(x, [2048])
        x = tf.layers.dense(x, 1000, activation=None)

        self._check_shape(x, [1000])
        return x

    def _check_images(self, images):
        if images.get_shape().as_list()[1:] != [299, 299, 3]:
            raise ValueError('image shape must be (299, 299, 3)')
        if images.dtype != tf.float32:
            raise ValueError('image dtype must be tf.float32')

    def _check_labels(self, labels):
        if len(labels.get_shape()) != 1:
            raise ValueError('labels must be 1D')

    def _get_loss(self, logits, labels):
        with tf.name_scope('loss'):
            loss = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits, name='cross_entroy'),
                name='loss')
        return loss

    def _get_train_ops(self, features, labels):
        """
        Get training ops.

        Inputs:
            `features`: per-image standardized image batch,
                dtype tf.float32, shape (None, 299, 299, 3)
            `labels`: int32 label, shape (None,)

        Returns:
            `ModelFnOps` object.
        """
        images = features
        self._check_images(images)
        self._check_labels(labels)
        logits = self.get_logits(images, True)
        loss = self._get_loss(logits, labels)

        tf.summary.scalar('loss', loss)

        steps = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)
        if len(steps) == 1:
            step = steps[0]
        else:
            raise Exception('Multiple global steps disallowed')

        with tf.name_scope('train_op_generation'):
            # train_op = tf.contrib.layers.optimize_loss(
            #     loss, step, self.learning_rate, self.optimizer)
            train_op = tf.train.AdamOptimizer().minimize(loss, step)

        predictions = tf.nn.softmax(logits)
        return learn.ModelFnOps(
            learn.ModeKeys.TRAIN,
            predictions=predictions, loss=loss, train_op=train_op)

    def _get_eval_ops(self, features, labels, metrics):
        """
        Get evaluation ops.

        Inputs:
            `features`: per-image standardized images batch,
                dtype tf.float32, shape (None, 299, 299, 3)
            `labels`: int32 label, shape (None,)

        Returns:
          `ModelFnOps` object.
        """
        images = features
        self._check_images(images)
        self._check_labels(labels)
        raise NotImplementedError()
        predictions = None
        loss = None
        return learn.ModelFnOps(
            learn.ModeKeys.EVAL, predictions=predictions, loss=loss)

    def _get_predict_ops(self, features):
        """
        Get ops for prediction.

        Inputs:
            `features`: image batch

        Returns:
            `ModelFnOps` object. Predictions is dict with
                `heatmaps`: final heatmap inference
                (sometimes) `example_id`: same as `example_id` passed in via
                    features if it is.

        """
        images = features
        self._check_images(images)
        raise NotImplementedError()
        predictions = None
        return learn.ModelFnOps(learn.ModeKeys.INFER, predictions=predictions)


if __name__ == '__main__':
    training = True
    estimator = InceptionV3Estimator('test')
    image = tf.placeholder(shape=(None, 299, 299, 3), dtype=tf.float32)
    estimator.get_logits(image, training)
    print('woot!')
