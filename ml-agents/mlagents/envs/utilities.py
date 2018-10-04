from PIL import Image
import numpy as np
import io


def process_pixels(image_bytes, gray_scale):
    """
    Converts byte array observation image into numpy array, re-sizes it,
    and optionally converts it to grey scale
    :param image_bytes: input byte array corresponding to image
    :return: processed numpy array of observation from environment
    """
    s = bytearray(image_bytes)
    image = Image.open(io.BytesIO(s))
    s = np.array(image) / 255.0
    if gray_scale:
        s = np.mean(s, axis=2)
        s = np.reshape(s, [s.shape[0], s.shape[1], 1])
    return s
