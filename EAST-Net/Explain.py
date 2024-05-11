import numpy as np
import torch


def getShapValue(model, data, ht_pair, data_input):
    def MSE(y_pred: np.array, y_true: np.array):
        return np.mean(np.square(y_pred - y_true))
    mse_loss = torch.nn.MSELoss()
    y_pred = []
    with torch.no_grad():
        X_seq, t_x, t_y, y_true = data
        z_X_seq = X_seq.mean(dim=-1).unsqueeze(-1).expand_as(X_seq)
        z_t_x = t_x.mean(dim=-1).unsqueeze(-1).expand_as(t_x)
        z_t_y = t_y.mean(dim=-1).unsqueeze(-1).expand_as(t_y)
        combinations = [
            (X_seq, t_x, t_y),
            (X_seq, t_x, z_t_y),
            (X_seq, z_t_x, t_y),
            (X_seq, z_t_x, z_t_y),
            (z_X_seq, t_x, t_y),
            (z_X_seq, t_x, z_t_y),
            (z_X_seq, z_t_x, t_y),
            (z_X_seq, z_t_x, z_t_y)
        ]
        for X, tX, tY in combinations:
            pred, _ = model(X, tX, tY, ht_pair)
            pred = data_input.channel_wise_denormalize(pred)
            y_pred.append(pred) #可以加上denormalize

    shap_X = MSE(y_pred[1], y_pred[5])+MSE(y_pred[2], y_pred[6])\
             +MSE(y_pred[0],y_pred[4])+MSE(y_pred[3], y_pred[7])
    shap_tx = MSE(y_pred[1], y_pred[3])+MSE(y_pred[4], y_pred[6])+\
              MSE(y_pred[0], y_pred[2])+MSE(y_pred[5], y_pred[7])
    shap_ty = MSE(y_pred[0], y_pred[1])+MSE(y_pred[2], y_pred[3])+\
              MSE(y_pred[4], y_pred[5])+MSE(y_pred[6], y_pred[7])
    return shap_X, shap_tx, shap_ty
