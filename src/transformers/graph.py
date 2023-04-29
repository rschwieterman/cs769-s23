import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import tensor



TL = torch.zeros([12,40])

TL[0] =  tensor([0.6622, 0.6906, 0.6796, 0.6871, 0.6767, 0.6774, 0.6692, 0.6879, 0.6118,
        0.7102, 0.7110, 0.6609, 0.5957, 0.7031, 0.7176, 0.6420, 0.6013, 0.7027,
        0.7095, 0.6296, 0.6256, 0.7039, 0.6980, 0.6636, 0.6123, 0.7066, 0.7087,
        0.6589, 0.6255, 0.6957, 0.7027, 0.6682, 0.6753, 0.6826, 0.6693, 0.6920,
        0.6389, 0.6942, 0.6939, 0.6715])
TL[1] =  tensor([0.6951, 0.7294, 0.7484, 0.6995, 0.6801, 0.7178, 0.7160, 0.6965, 0.7335,
        0.7147, 0.7035, 0.6937, 0.7039, 0.6866, 0.6684, 0.6765, 0.6736, 0.6683,
        0.6339, 0.6638, 0.7232, 0.7182, 0.7171, 0.6964, 0.7305, 0.7118, 0.7191,
        0.6908, 0.7182, 0.7073, 0.6925, 0.6970, 0.6806, 0.7261, 0.7499, 0.7144,
        0.7118, 0.7195, 0.7055, 0.6915])
TL[2] =  tensor([0.7081, 0.6955, 0.7145, 0.7331, 0.7128, 0.6868, 0.7004, 0.7278, 0.6717,
        0.6980, 0.7116, 0.7230, 0.6137, 0.6868, 0.6799, 0.6842, 0.5808, 0.6541,
        0.6403, 0.6536, 0.7010, 0.6959, 0.7051, 0.7304, 0.6709, 0.6870, 0.7055,
        0.7262, 0.6791, 0.6909, 0.6978, 0.7331, 0.7224, 0.6870, 0.7103, 0.7253,
        0.6978, 0.6985, 0.7035, 0.7373])
TL[3] =  tensor([0.7443, 0.7283, 0.9885, 0.9690, 0.7428, 0.7075, 0.9885, 0.9686, 0.7148,
        0.7002, 0.9873, 0.9630, 0.6602, 0.6346, 0.9866, 0.9536, 0.6386, 0.6233,
        0.9860, 0.9500, 0.7442, 0.7190, 0.9881, 0.9667, 0.7268, 0.7094, 0.9878,
        0.9641, 0.7243, 0.6966, 0.9880, 0.9639, 0.7695, 0.7267, 0.9888, 0.9707,
        0.7350, 0.7033, 0.9883, 0.9657])
TL[4] =  tensor([0.7092, 0.9624, 0.6907, 0.9884, 0.6972, 0.9630, 0.6861, 0.9885, 0.7089,
        0.9599, 0.6918, 0.9870, 0.7034, 0.9573, 0.6713, 0.9860, 0.6843, 0.9555,
        0.6490, 0.9861, 0.7148, 0.9606, 0.6875, 0.9878, 0.7082, 0.9599, 0.6788,
        0.9874, 0.7049, 0.9607, 0.6835, 0.9879, 0.6953, 0.9601, 0.6845, 0.9888,
        0.7050, 0.9598, 0.6920, 0.9880])
TL[5] =  tensor([0.6882, 0.7201, 0.7173, 0.9871, 0.6761, 0.7063, 0.6929, 0.9871, 0.7629,
        0.7159, 0.7026, 0.9845, 0.7403, 0.7016, 0.6648, 0.9815, 0.7237, 0.6891,
        0.6261, 0.9799, 0.7353, 0.7097, 0.7030, 0.9859, 0.7597, 0.7161, 0.6864,
        0.9854, 0.7461, 0.7033, 0.6940, 0.9859, 0.6770, 0.7138, 0.7096, 0.9884,
        0.7239, 0.7061, 0.7019, 0.9862])
TL[6] =  tensor([0.6969, 0.9735, 0.7676, 0.7226, 0.6791, 0.9739, 0.7173, 0.7070, 0.7132,
        0.9727, 0.8317, 0.7080, 0.6899, 0.9726, 0.8189, 0.6920, 0.6782, 0.9722,
        0.7876, 0.6554, 0.7056, 0.9731, 0.8082, 0.7128, 0.7015, 0.9732, 0.8214,
        0.7070, 0.7035, 0.9740, 0.8008, 0.6953, 0.6991, 0.9744, 0.7435, 0.7251,
        0.6971, 0.9734, 0.7912, 0.7129])
TL[7] =  tensor([0.9684, 0.7092, 0.7078, 0.9916, 0.9687, 0.6826, 0.6866, 0.9916, 0.9653,
        0.7641, 0.6966, 0.9881, 0.9620, 0.7636, 0.6857, 0.9839, 0.9630, 0.7263,
        0.6543, 0.9814, 0.9676, 0.7386, 0.7111, 0.9893, 0.9665, 0.7521, 0.6863,
        0.9889, 0.9678, 0.7466, 0.7067, 0.9899, 0.9694, 0.7073, 0.7028, 0.9932,
        0.9684, 0.7458, 0.7225, 0.9905])
TL[8] =  tensor([0.9191, 0.8800, 0.6852, 0.9829, 0.9235, 0.8840, 0.6836, 0.9832, 0.9145,
        0.8699, 0.7033, 0.9808, 0.9088, 0.8596, 0.7180, 0.9779, 0.9074, 0.8543,
        0.7122, 0.9748, 0.9192, 0.8734, 0.7081, 0.9817, 0.9156, 0.8744, 0.7175,
        0.9813, 0.9187, 0.8730, 0.6877, 0.9822, 0.9254, 0.8819, 0.6884, 0.9847,
        0.9202, 0.8718, 0.6850, 0.9822])
TL[9] =  tensor([0.7844, 0.7101, 0.7442, 0.9839, 0.7686, 0.6872, 0.7102, 0.9798, 0.7332,
        0.8018, 0.7830, 0.9801, 0.6760, 0.7972, 0.7864, 0.9721, 0.6691, 0.7635,
        0.7589, 0.9648, 0.7618, 0.7690, 0.7660, 0.9817, 0.7304, 0.7875, 0.7775,
        0.9810, 0.7297, 0.7723, 0.7493, 0.9803, 0.8012, 0.6939, 0.7296, 0.9855,
        0.7522, 0.7623, 0.7472, 0.9826])
TL[10] =  tensor([0.9627, 0.7307, 0.9676, 0.8228, 0.9624, 0.7416, 0.9609, 0.8056, 0.9629,
        0.7569, 0.9677, 0.8023, 0.9602, 0.7648, 0.9619, 0.8041, 0.9632, 0.7422,
        0.9585, 0.8079, 0.9660, 0.7603, 0.9710, 0.8112, 0.9633, 0.7581, 0.9674,
        0.8184, 0.9645, 0.7575, 0.9667, 0.8072, 0.9700, 0.7292, 0.9722, 0.8154,
        0.9675, 0.7363, 0.9731, 0.8022])
TL[11] =  tensor([0.9347, 0.6840, 0.9618, 0.9551, 0.9133, 0.7510, 0.9506, 0.9293, 0.9348,
        0.7782, 0.9540, 0.9542, 0.9336, 0.7758, 0.9547, 0.9542, 0.9395, 0.8263,
        0.9626, 0.9537, 0.9285, 0.7504, 0.9628, 0.9528, 0.9466, 0.7702, 0.9606,
        0.9595, 0.9298, 0.7626, 0.9634, 0.9618, 0.9293, 0.6864, 0.9680, 0.9664,
        0.9261, 0.7156, 0.9724, 0.9655])

print(TL.sum())