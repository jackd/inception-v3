"""Fit the estimator model to imagenet data."""

from inception_v3.estimator import InceptionV3Estimator
from inception_v3.preprocessor import ImagenetPreprocessor


def fit(batch_size, max_steps=1000000, model_id='base'):
    """Fit the estimator model to imagenet data."""
    def input_fn():
        preprocessor = ImagenetPreprocessor(eval_data=False)

        images, labels = preprocessor.get_preprocessed_batch(
            batch_size, shuffle=True)

        return images, labels

    estimator = InceptionV3Estimator(model_id=model_id)
    estimator.fit(input_fn=input_fn, max_steps=max_steps)


if __name__ == '__main__':
    batch_size = 16

    fit(batch_size)
