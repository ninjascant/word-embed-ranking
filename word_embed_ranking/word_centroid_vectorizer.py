import logging
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from tqdm.auto import tqdm
import scipy.sparse as sparse
from .utils import read_word_vector_file, get_word_vector_size, get_default_word_vector, EN_STOP_WORDS

tqdm.pandas()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
c_handler.setFormatter(f_format)
logger.addHandler(c_handler)



