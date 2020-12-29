import os
import argparse
import numpy as np
from os.path import join

from core.dataio import Database


def main(args_):

    db_root = args.folder
    db_names = os.listdir(db_root)
    db_names = [db_name for db_name in db_names if '.h5' in db_name]

    databases = [Database(join(db_root, db_name), mode='r', title=None)
                 for db_name in db_names]
    expected_rows = np.sum([db.codes.shape[0] for db in databases])

    # WARNING: opens the first database and will IMMEDIATELY begin writing to it, so will get screwed up if interrupted!
    # TODO: get url_max_len from Database attrs!
    database_final = Database(args.outname,
                              data_root=db_root,
                              url_max_len=64,
                              mode='w',
                              expected_rows=expected_rows)
    for i, database_other in enumerate(databases):
        print(f'\r{i}/{len(databases[1:])}', end='')
        database_final.cat(database_other)

    [db.close() for db in databases]
    database_final.close()


if __name__ == '__main__':

    default_folder = '/home/heka/database'
    default_outname = 'test.h5'

    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", type='str', default=default_folder)
    parser.add_argument("--outname", type='str', default=default_outname)

    args = parser.parse_args()
    main(args)
