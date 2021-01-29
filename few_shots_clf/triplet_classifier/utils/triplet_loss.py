##########################
# Imports
##########################


import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa


##########################
# Function
##########################


def _pairwise_distances(embeddings: tf.Tensor):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    dot_product = K.dot(embeddings, K.transpose(embeddings))
    square_norm = tf.linalg.diag_part(dot_product)
    distances = K.expand_dims(square_norm, 1) - 2.0 * \
        dot_product + K.expand_dims(square_norm, 0)
    distances = K.maximum(distances, 0.0)
    mask = K.equal(distances, 0.0)
    mask = K.cast(mask, tf.float32)
    distances = K.sqrt(distances + mask * K.epsilon())
    distances = distances * (1.0 - mask)
    return distances


def _get_triplet_mask(labels: tf.Tensor):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    indices_equal = K.cast(tf.eye(K.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.math.logical_not(indices_equal)
    i_not_equal_j = K.expand_dims(indices_not_equal, 2)
    i_not_equal_k = K.expand_dims(indices_not_equal, 1)
    j_not_equal_k = K.expand_dims(indices_not_equal, 0)
    distinct_indices = tf.math.logical_and(tf.math.logical_and(i_not_equal_j,
                                                               i_not_equal_k),
                                           j_not_equal_k)
    label_equal = K.equal(K.expand_dims(labels, 0),
                          K.expand_dims(labels, 1))
    i_equal_j = K.expand_dims(label_equal, 2)
    i_equal_k = K.expand_dims(label_equal, 1)
    valid_labels = tf.math.logical_and(
        i_equal_j, tf.math.logical_not(i_equal_k))
    mask = tf.math.logical_and(distinct_indices, valid_labels)
    mask = K.cast(mask, tf.float32)
    return mask


def invalid_triplets_ratio(labels: tf.Tensor, embeddings: tf.Tensor, margin: float) -> tf.Tensor:
    """[summary]

    Args:
        labels (tf.Tensor): [description]
        embeddings (tf.Tensor): [description]
        margin (float): [description]

    Returns:
        tf.Tensor: [description]
    """
    pairwise_dist = _pairwise_distances(embeddings)
    anchor_positive_dist = K.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = K.expand_dims(pairwise_dist, 1)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    mask = _get_triplet_mask(labels)
    triplet_loss = tf.math.multiply(mask, triplet_loss)
    triplet_loss = K.maximum(triplet_loss, 0.0)
    valid_triplets = K.cast(tf.math.greater(triplet_loss, K.epsilon()),
                            tf.float32)
    num_positive_triplets = tf.math.reduce_sum(valid_triplets)
    num_valid_triplets = tf.math.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / num_valid_triplets
    return fraction_positive_triplets


def triplet_loss_function(margin: float, mining_strategy: str):
    """[summary]

    Args:
        margin (float): Margin of the triplet loss.
        mining_strategy (str): Strategy used to create batches
    """
    if mining_strategy == "soft":
        return tfa.losses.TripletSemiHardLoss(margin=margin)
    if mining_strategy == "hard":
        return tfa.losses.TripletHardLoss(margin=margin)
    return tfa.losses.TripletSemiHardLoss(margin=margin)


def triplet_loss_metric(margin: float):
    """[summary]

    Args:
        margin (float): Margin of the triplet loss.
    """
    @tf.function
    def metric(y_true, y_pred):
        return invalid_triplets_ratio(y_true, y_pred, margin)
    return metric
