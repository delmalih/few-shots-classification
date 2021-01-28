##########################
# Imports
##########################


import tensorflow as tf
import tensorflow.keras.backend as K


##########################
# Function
##########################


def _pairwise_distances(embeddings: tf.Tensor, squared: bool = False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    dot_product = K.dot(embeddings, K.transpose(embeddings))
    square_norm = tf.linalg.diag_part(dot_product)
    distances = K.expand_dims(square_norm, 1) - 2.0 * \
        dot_product + K.expand_dims(square_norm, 0)
    distances = K.maximum(distances, 0.0)
    if not squared:
        mask = K.equal(distances, 0.0)
        mask = K.cast(mask, tf.float32)
        distances = K.sqrt(distances + mask * K.epsilon())
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels: tf.Tensor):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    indices_equal = K.cast(tf.eye(K.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.math.logical_not(indices_equal)
    labels_equal = K.equal(K.expand_dims(labels, 0), K.expand_dims(labels, 1))
    mask = tf.math.logical_and(indices_not_equal, labels_equal)
    return mask


def _get_anchor_negative_triplet_mask(labels: tf.Tensor):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    labels_equal = K.equal(K.expand_dims(labels, 0), K.expand_dims(labels, 1))
    mask = tf.math.logical_not(labels_equal)
    return mask


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
    distinct_indices = tf.math.logical_and(
        tf.math.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)
    label_equal = K.equal(K.expand_dims(labels, 0),
                          K.expand_dims(labels, 1))
    i_equal_j = K.expand_dims(label_equal, 2)
    i_equal_k = K.expand_dims(label_equal, 1)
    valid_labels = tf.math.logical_and(
        i_equal_j, tf.math.logical_not(i_equal_k))
    mask = tf.math.logical_and(distinct_indices, valid_labels)
    return mask


def batch_all_triplet_loss(
        labels: tf.Tensor,
        embeddings: tf.Tensor,
        margin: float,
        squared: bool = False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    anchor_positive_dist = K.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(
        anchor_positive_dist.shape)
    anchor_negative_dist = K.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(
        anchor_negative_dist.shape)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    mask = _get_triplet_mask(labels)
    mask = K.cast(mask, tf.float32)
    triplet_loss = tf.math.multiply(mask, triplet_loss)
    triplet_loss = K.maximum(triplet_loss, 0.0)
    valid_triplets = K.cast(tf.math.greater(
        triplet_loss, K.epsilon()), tf.float32)
    num_positive_triplets = tf.math.reduce_sum(valid_triplets)
    num_valid_triplets = tf.math.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / \
        (num_valid_triplets + K.epsilon())
    triplet_loss = tf.math.reduce_sum(
        triplet_loss) / (num_positive_triplets + K.epsilon())
    return triplet_loss, fraction_positive_triplets


def batch_hard_triplet_loss(
        labels: tf.Tensor,
        embeddings: tf.Tensor,
        margin: float,
        squared: bool = False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = K.cast(mask_anchor_positive, tf.float32)
    anchor_positive_dist = tf.math.multiply(
        mask_anchor_positive, pairwise_dist)
    hardest_positive_dist = tf.math.reduce_max(
        anchor_positive_dist, axis=1, keepdims=True)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = K.cast(mask_anchor_negative, tf.float32)
    max_anchor_negative_dist = tf.math.reduce_max(
        pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + \
        max_anchor_negative_dist * (1.0 - mask_anchor_negative)
    hardest_negative_dist = tf.math.reduce_min(
        anchor_negative_dist, axis=1, keepdims=True)
    triplet_loss = K.maximum(hardest_positive_dist -
                             hardest_negative_dist + margin, 0.0)
    triplet_loss = tf.math.reduce_mean(triplet_loss)
    return triplet_loss


def triplet_loss_function(
        margin: float,
        mining_strategy: str,
        squared: bool = False):
    """[summary]

    Args:
        labels (tf.Tensor): [description]
        embeddings (tf.Tensor): [description]
        margin (float): [description]
        mining_strategy (str): [description]
        squared (bool, optional): [description]. Defaults to False.
    """
    @tf.function
    def loss(y_true, y_pred):
        if mining_strategy == "soft":
            return batch_all_triplet_loss(y_true, y_pred, margin, squared)[0]
        if mining_strategy == "hard":
            return batch_hard_triplet_loss(y_true, y_pred, margin, squared)
        return 0
    return loss


def triplet_loss_metric(margin: float, squared: bool = False):
    """[summary]

    Args:
        labels (tf.Tensor): [description]
        embeddings (tf.Tensor): [description]
        margin (float): [description]
        mining_strategy (str): [description]
        squared (bool, optional): [description]. Defaults to False.
    """
    @tf.function
    def metric(y_true, y_pred):
        return batch_all_triplet_loss(y_true, y_pred, margin, squared)[1]
    return metric
