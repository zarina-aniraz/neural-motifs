#%%
from config import VG_SGG_FN, VG_SGG_DICT_FN, IM_DATA_FN
from dataloaders import visual_genome
import numpy as np

split_mask_1, gt_boxes_1, gt_classes_1, relationships_1 = visual_genome.load_graphs(VG_SGG_FN, mode='train', filter_empty_rels=False)
split_mask_2, gt_boxes_2, gt_classes_2, relationships_2 = visual_genome.load_graphs(VG_SGG_FN, mode='test', filter_empty_rels=False)

split_mask = split_mask_1 | split_mask_2
# gt_boxes = gt_boxes_1 + gt_boxes_2 + gt_boxes_3
gt_classes = gt_classes_1 + gt_classes_2
relationships = relationships_1 + relationships_2

i2c, i2p = visual_genome.load_info(VG_SGG_DICT_FN)
filenames = visual_genome.load_image_filenames(IM_DATA_FN)
filenames = [filenames[i] for i in np.where(split_mask)[0]]

#%%
idx = 0
classes = gt_classes[idx]
relations = relationships[idx]
print('Classes:', [i2c[c] for c in classes])
print('Relations:', [(i2c[classes[c1]], i2p[r], i2c[classes[c2]]) for (c1, c2, r) in relations])


#%%
import numpy as np
from itertools import product

co_count = np.zeros([len(i2c), len(i2c), len(i2p)])
count = np.zeros(len(i2c))
co_occur = np.zeros([len(i2c), len(i2c)])

for classes, relations in zip(gt_classes, relationships):
    # classes = info['gt_classes']
    # relations = info['gt_relations']
    count_check = {}
    co_occur_check = {}

    for o1, o2, r in relations:
        c1, c2 = classes[o1], classes[o2]
        co_count[c1, c2, r] += 1

    for c1, c2 in product(classes, classes):
        if c1 != c2:
            if (c1, c2) not in co_occur_check:
                co_occur_check[(c1, c2)] = True
                co_occur_check[(c2, c1)] = True
                co_occur[c1, c2] += 1
                co_occur[c2, c1] += 1
            if c1 not in count_check:
                count_check[c1] = True
                count[c1] += 1
            if c2 not in count_check:
                count_check[c2] = True
                count[c2] += 1