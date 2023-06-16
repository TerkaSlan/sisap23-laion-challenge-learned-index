from sklearn.metrics.pairwise import cosine_similarity


def pairwise_cosine(x, y):
    return 1-cosine_similarity(x, y)
