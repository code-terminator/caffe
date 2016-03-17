import sys
import lmdb
import StringIO
import numpy as np


def set_caffe_path(caffe_path):
    # Remove all other caffe paths, if any.
    sys.path = [p for p in sys.path if p.find('caffe') == -1]
    sys.path.insert(0, caffe_path)

    import caffe
    reload(caffe)


class lmdb_npy(object):
    """
    Helper class to read/write np arrays on lmdb
    """

    def __init__(self, lmdb_path):
        import lmdb
        self.env = lmdb.open(lmdb_path, map_size=1099511627776)

    def keys(self):
        with self.env.begin() as txn:
            cursor = txn.cursor()
            keys = [key for key, _ in cursor]
        return keys
    
    def get(self, key):
        with self.env.begin() as txn:
            bstring = txn.get(key)
            mfile = StringIO.StringIO(bstring)
            arr = np.load(mfile)
            return arr

    def set(self, key, nparr):
        with self.env.begin(write=True) as txn:
            mfile = StringIO.StringIO()
            np.save(mfile, nparr)
            txn.put(key, mfile.getvalue())

    def set_multiple(self, key, np_dict):
        with self.env.begin(write=True) as txn:
            mfile = StringIO.StringIO()
            np.savez_compressed(mfile, **np_dict)
            txn.put(key, mfile.getvalue())
