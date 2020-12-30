from annoy import AnnoyIndex


def build_annoy_index(vectors, num_trees=50):
    vec_size = vectors.shape[1]
    ann_index = AnnoyIndex(vec_size, 'angular')
    for i in range(vectors.shape[0]):
        vector = vectors[i]
        ann_index.add_item(i, vector)

    ann_index.build(num_trees)
    ann_index.save('model/search_index.ann')
    return ann_index
