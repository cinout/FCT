import torch

base_classes = [
    "aeroplane",
    "bicycle",
    "boat",
    "bottle",
    "car",
    "cat",
    "chair",
    "diningtable",
    "dog",
    "horse",
    "person",
    "pottedplant",
    "sheep",
    "train",
    "tvmonitor",
]
novel_classes = ["nectarine", "orange", "cereal", "almond_mix", "dummy"]
all_classes = [
    "aeroplane",
    "bicycle",
    "boat",
    "bottle",
    "car",
    "cat",
    "chair",
    "diningtable",
    "dog",
    "horse",
    "person",
    "pottedplant",
    "sheep",
    "train",
    "tvmonitor",
    "nectarine",
    "orange",
    "cereal",
    "almond_mix",
    "dummy",
]


pred_classes = torch.tensor(
    [
        18,
        17,
        16,
        16,
        15,
        16,
        15,
        15,
        14,
        12,
        11,
        3,
        5,
        7,
        11,
        18,
        15,
        17,
        14,
        16,
        3,
        6,
        12,
        0,
        6,
        14,
        15,
        17,
        17,
        16,
        13,
        10,
        8,
        16,
        15,
        4,
        14,
        5,
        7,
        13,
        17,
        6,
        6,
        0,
        2,
        6,
        0,
        17,
        1,
        1,
        18,
        12,
        10,
        14,
        12,
        2,
        8,
        11,
        8,
        2,
        5,
        8,
        10,
        6,
        9,
        5,
        2,
        9,
        3,
        4,
        10,
        7,
        1,
        4,
        13,
        4,
        10,
        11,
        5,
        0,
        18,
        3,
        0,
        6,
        10,
        4,
        7,
        9,
        12,
        3,
        8,
        18,
        10,
        10,
        1,
        18,
        15,
        2,
        7,
        3,
    ]
)

scores = torch.tensor(
    [
        1.0000e00,
        1.0000e00,
        9.9974e-01,
        9.9925e-01,
        9.9803e-01,
        1.6772e-01,
        1.0330e-01,
        1.3426e-02,
        5.6555e-03,
        5.4483e-03,
        2.8356e-03,
        1.5515e-03,
        1.0994e-03,
        6.9731e-04,
        5.5212e-04,
        5.5116e-04,
        5.3646e-04,
        2.4959e-04,
        2.4049e-04,
        2.3380e-04,
        2.3294e-04,
        1.1169e-04,
        8.0154e-05,
        5.8849e-05,
        4.9775e-05,
        4.2369e-05,
        3.5292e-05,
        3.2201e-05,
        2.2297e-05,
        1.9726e-05,
        1.7766e-05,
        1.7680e-05,
        1.7531e-05,
        1.5974e-05,
        1.3187e-05,
        1.1512e-05,
        1.0165e-05,
        1.0009e-05,
        9.7372e-06,
        9.0024e-06,
        8.8134e-06,
        8.0140e-06,
        7.6155e-06,
        7.2084e-06,
        4.6917e-06,
        4.0298e-06,
        3.5110e-06,
        3.2252e-06,
        3.0301e-06,
        2.4681e-06,
        2.2153e-06,
        2.0290e-06,
        1.0845e-06,
        1.0572e-06,
        9.6077e-07,
        7.3332e-07,
        6.9393e-07,
        6.6324e-07,
        6.3313e-07,
        4.7465e-07,
        4.5776e-07,
        4.0541e-07,
        3.9554e-07,
        3.2602e-07,
        3.2106e-07,
        2.6278e-07,
        2.3872e-07,
        2.2207e-07,
        2.0342e-07,
        1.6582e-07,
        1.5820e-07,
        1.5487e-07,
        1.5442e-07,
        1.1569e-07,
        1.1268e-07,
        9.4383e-08,
        9.3539e-08,
        8.4513e-08,
        8.2109e-08,
        7.8749e-08,
        7.2834e-08,
        6.2798e-08,
        5.1139e-08,
        4.1488e-08,
        3.1447e-08,
        2.7770e-08,
        2.4510e-08,
        2.1184e-08,
        1.5659e-08,
        1.1959e-08,
        8.3822e-09,
        6.9462e-09,
        6.3466e-09,
        5.4866e-09,
        5.1137e-09,
        3.7378e-09,
        2.9960e-09,
        2.8375e-09,
        2.7418e-09,
        2.4583e-09,
    ]
)

final_preds = []

for c in novel_classes:
    c_ordinal = all_classes.index(c)
    c_pred_idx = torch.nonzero(pred_classes == c_ordinal).squeeze()
    c_pred_scores = torch.index_select(scores, 0, c_pred_idx)

    # FIXME: other categories
    if c == "orange":
        topk = 2
    else:
        topk = 1

    (values, indices) = torch.topk(
        c_pred_scores,
        k=(c_pred_idx.shape[0] if topk > c_pred_idx.shape[0] else topk),
        dim=0,
    )
    final_preds_4c = torch.index_select(c_pred_idx, 0, indices)
    final_preds.append(final_preds_4c)

final_preds = torch.cat(final_preds, dim=0)

print(final_preds)


# novel_classes_ordinal = [all_classes.index(c) for c in novel_classes]

# hoho = sum(pred_classes == i for i in novel_classes_ordinal)

# valid_preds = scores > 0.2

# print(hoho & valid_preds)


# exit()


# novel_predictions_idx = torch.nonzero(
#     sum(pred_classes == i for i in novel_classes_ordinal)
# ).squeeze()

# print(novel_predictions_idx.shape[0])
