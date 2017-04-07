"""Provides preprocessor implementations for model training/evaluation."""
import tensorflow as tf
from imagenet.preprocessor import ImageClassificationPreprocessor
from imagenet.data import Imagenet2012EvalDataset, Imagenet2012TrainDataset


class ImagenetPreprocessor(ImageClassificationPreprocessor):
    """Preprocessor for inception_v3 estimator for Imagenet dataset."""

    def __init__(self, eval_data):
        """Initialize with bool `eval_data`."""
        super(ImagenetPreprocessor, self).__init__(
            Imagenet2012EvalDataset() if eval_data else
            Imagenet2012TrainDataset(), image_format='jpeg')

    def preprocess_single_image(self, image):
        """Resize image to 299 * 299."""
        image = tf.image.random_flip_left_right(image)
        shape, = tf.shape_n([image])
        h = shape[0]
        w = shape[1]
        d = tf.maximum(h, w)
        image = tf.image.resize_image_with_crop_or_pad(image, d, d)
        image = tf.image.resize_images(image, (299, 299))
        image.set_shape((299, 299, 3))
        image = tf.image.per_image_standardization(image)
        return image


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from preprocess import get_batch_data
    from imagenet.data import label_name

    preprocessor = ImagenetPreprocessor(False)
    images, labels = get_batch_data(preprocessor, batch_size=64, shuffle=True)

    for image, label in zip(images, labels):
        image -= np.min(image)
        image /= np.max(image)
        # image *= 255
        # image = image.astype(np.uint8)
        plt.imshow(image)
        plt.title(label_name(label))
        plt.show()
