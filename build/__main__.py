import argparse
import requests
from scipy import fftpack
import numpy
from PIL import Image

URL = 'http://www2.mps.mpg.de/homes/heller/downloads/files/SETI_message.txt'
BYTE_SIZE = 359

def get_data(url=URL):
    """Downloads SETI data from provided URL and returns raw content. """
    return requests.get(URL).content


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]


def dct(b):
    """
    Apply DCT to all elements in b

    :param b: A list of data to apply DCT
    :return: numpy.ndarray DCT applied data
    """
    return fftpack.dct(b, norm='ortho')


def idct(c):
    """
    Apply inverce DCT to all elements in b

    :param c:
    :return:
    """
    return fftpack.idct(c, norm='ortho')


def main():
    content = get_data()
    bytes = [int(chunk, 2) for chunk in chunks(content, BYTE_SIZE)]
    bytes = numpy.asarray(bytes)

    one = -1
    len_bytes = len(bytes)
    for i in range(10, len_bytes):
        if (len_bytes % i) == 0:
            one = i
            break

    if one < 0:
        raise AssertionError

    bytes.shape = (i, len_bytes/i)

    # byte_set = set(bytes)
    # freq = [(b, bytes.count(b), b) for b in byte_set]
    print("{}  elements found".format(len(bytes)))
    # print(sorted(freq, reverse=True))

    coefs = dct(bytes)
    img = Image.fromarray(coefs)
    img.show()

    # for i in range(10):
    #     coefs_copy = coefs.copy()
    #     coefs_copy[i:,:] = 0
    #     coefs_copy[:,i:] = 0
    #
    #     new_bytes = idct(coefs_copy)
    #     img = Image.fromarray(new_bytes)
    #     img.show()

if __name__ == '__main__':
    main()