import os
import sys
import traceback
import argparse
import numpy as np
import pandas as pd
from sklearn import neighbors
import pickle


def train(input_files):
    print('Starting the training.')
    try:
        features = input_files
        
        model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
        model.fit(features)
        dist, idlist = model.kneighbors(features)
        
        # save the model
        filename = "knn_model.sav"
        pickle.dump(model, os.path.join(args.model_dir, filename))
        print('Training complete.')

    except Exception as e:
        trc = traceback.format_exc()
        with open(os.path.join(args.model_dir, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        sys.exit(255)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]

    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ np.load(file) for file in input_files ]
    train(raw_data)
    sys.exit(0)