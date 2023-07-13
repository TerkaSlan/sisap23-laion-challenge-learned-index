from sklearn.metrics.pairwise import cosine_similarity


def pairwise_cosine(x, y):
    return 1-cosine_similarity(x, y)


def pairwise_cosine_threshold(x, y, threshold):
    return 1-cosine_similarity(x, y)


def save_as_pickle(filename: str, obj):
    """
    Saves an object as a pickle file.
    Expects that the directory with the file exists.

    Parameters
    ----------
    filename : str
        Path to the file to write to.
    obj: object
        The object to save.
    """
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
