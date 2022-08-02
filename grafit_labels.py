import pickle
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from fathomnet.api import images

def match_fine_grained_label(entry):
    '''Get the fine labels by using the image url, and matching the boundingBoxes using the FathomNet API and the entry coordinates.'''
    try:
        fn_entry = images.find_by_url(entry[7])
    except:
        matched_concept = ''
        print(f'Fetch failed for {entry[7]}')
        return matched_concept
    bboxes = fn_entry.boundingBoxes
    matched_concept = ''
    for bbox in bboxes:
        x_match = entry[2] == bbox.x
        y_match = entry[3] == bbox.y
        w_match = entry[4] - entry[2] == bbox.width
        h_match = entry[5] - entry[3] == bbox.height
        if x_match + y_match + w_match + h_match >= 3:
            matched_concept = bbox.concept
            break
    return matched_concept


if __name__ == "__main__":
    '''
    Load in train_df.csv, and get the fine labels by using the image url, and matching the boundingBoxes using the FathomNet API.

    Call is /image/query/url/{url} (figure out the fathomnet-py call). Match the boundingBoxes using the coordinates
    '''
    train_entries_df = pd.read_csv('/mnt/md0/Projects/Fathomnet/Training_Files/2022-03-11-Detectron/train_file_v2_df.csv')

    fine_grained_labels = []
    unmatched = 0
    idx = 0
    unmatched_entries = []

    idx = 0
    for row in train_entries_df.itertuples(name=None):
        if idx < 0:
            idx += 1
            continue
        label = match_fine_grained_label(row)
        idx += 1
        if label == '':
            unmatched += 1
            print(f"{unmatched} of {idx} unmatched")
            unmatched_entries.append(row)
        fine_grained_labels.append(label)

    print("Processing complete!")
    pickle.dump(fine_grained_labels, open('fine-grained-labels.pkl','wb'))