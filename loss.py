import tensorflow as tf
from tensorflow.keras import backend as K

def gradient(image):
    """
    Computes the horizontal and vertical gradients of an image.
    """
    dx = image[:, :, 1:, :] - image[:, :, :-1, :]
    dy = image[:, 1:, :, :] - image[:, :-1, :, :]
    return dx, dy

def gradient_contrastive_loss(y_true, y_pred, epsilon=1e-6):
    """
    Gradient-Based Contrastive Loss (L_CGC) – Encourages local contrast preservation
    by enforcing similarity in gradient structures.
    """
    # Compute image gradients
    dx_true, dy_true = gradient(y_true)
    dx_pred, dy_pred = gradient(y_pred)
    
    # Crop dimensions to ensure shape compatibility
    min_h = min(dx_true.shape[1], dy_true.shape[1])
    min_w = min(dx_true.shape[2], dy_true.shape[2])
    
    dx_true = dx_true[:, :min_h, :min_w, :]
    dy_true = dy_true[:, :min_h, :min_w, :]
    dx_pred = dx_pred[:, :min_h, :min_w, :]
    dy_pred = dy_pred[:, :min_h, :min_w, :]

    # Compute pairwise contrast differences
    contrast_true = K.abs(dx_true - dy_true) + epsilon
    contrast_pred = K.abs(dx_pred - dy_pred)
    
    # Loss is the absolute difference between normalized contrast values
    return K.mean(K.abs(contrast_true - contrast_pred))

def multi_scale_adaptive_contrast_loss(y_true, y_pred, epsilon=1e-6):
    """
    Multi-Scale Adaptive Contrast Loss (MAC) – Preserves local and global contrast
    by normalizing pixel intensities based on their standard deviation.
    """
    mu_true = K.mean(y_true, axis=[1,2,3], keepdims=True)
    mu_pred = K.mean(y_pred, axis=[1,2,3], keepdims=True)
    
    sigma_true = K.std(y_true, axis=[1,2,3], keepdims=True)
    sigma_pred = K.std(y_pred, axis=[1,2,3], keepdims=True)
    
    contrast_true = (y_true - mu_true) / (sigma_true + epsilon)
    contrast_pred = (y_pred - mu_pred) / (sigma_pred + epsilon)
    
    return K.mean(K.abs(contrast_true - contrast_pred))

def mae_loss(y_true, y_pred):
    """
    Mean Absolute Error (MAE) – Ensures pixel-wise accuracy.
    """
    return K.mean(K.abs(y_true - y_pred))

def ssim_loss(y_true, y_pred):
    """
    SSIM-Based Loss – Encourages perceptual similarity.
    """
    return 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)

def custom_loss(y_true, y_pred, alpha=0.3, beta=0.3, gamma=0.2, delta=0.2):
    """
    Final Loss Function: A weighted sum of Gradient Contrastive Loss (L_CGC),
    MAC Loss, MAE Loss, and SSIM Loss.
    """
    l1 = multi_scale_adaptive_contrast_loss(y_true, y_pred)
    l2 = mae_loss(y_true, y_pred)
    l3 = ssim_loss(y_true, y_pred)
    # l4 = gradient_contrastive_loss(y_true, y_pred)  # Fixed contrastive loss term
    
    return alpha * l1 + beta * l2 + gamma * l3
    # + delta * l4
