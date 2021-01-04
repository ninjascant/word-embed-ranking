import numpy as np

EN_STOP_WORDS = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during',
                 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours',
                 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from',
                 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through',
                 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their',
                 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any',
                 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then',
                 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few',
                 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further',
                 'was', 'here', 'than'}


def read_word_vector_file(file_path):
    embedding_dict = {}
    with open(file_path) as file:
        for i, line in enumerate(file):
            # if i > 100:
            #     break
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], 'float32')
            embedding_dict[word] = vector
    return embedding_dict


def get_word_vector_size(word_embed_dict):
    embeds = iter(word_embed_dict.values())
    return next(embeds).shape[0]


def get_default_word_vector(vector_shape):
    return np.zeros(vector_shape, dtype=np.float32)
