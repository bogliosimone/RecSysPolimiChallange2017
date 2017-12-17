import numpy as np
import scipy.sparse as sps
from sklearn.utils.sparsefuncs import inplace_row_scale
from sklearn.utils.sparsefuncs import inplace_column_scale

class CB_asymmetric_cosine:
    def __init__(self,urm, icm, k=100, alpha=0.5, q=1, m=0, name=""):
        urm = sps.csr_matrix(urm, dtype=np.int)
        icm = sps.csc_matrix(icm, dtype=np.int)
        if m == 0:
            m = 1e-6

        self.urm = urm
        self.k = k
        self.alpha = alpha
        self.q = q
        self.m = m
        self.MAP = 0.0
        self.name = name

        n_user = urm.shape[0]
        n_item = urm.shape[1]

        self.n_tracks = n_item
        self.n_playlists = n_user

        print("Start  CB asy model..")
        s = (icm * icm.T).tocsr()
        s_diagonal = s.diagonal()
        s = s - sps.dia_matrix((s.diagonal()[np.newaxis, :], [0]), shape=s.shape)
        s.eliminate_zeros()
        s_one = sps.csr_matrix((np.ones(s.data.shape[0]), s.indices, s.indptr), shape=(n_item, n_item),
                               dtype=np.float32)
        norm1 = s_diagonal ** (1 - alpha)
        norm2 = s_diagonal ** alpha
        inplace_row_scale(s_one, norm1)
        inplace_column_scale(s_one, norm2)
        s_one.data = (s_one.data + m)
        s.data = s.data / s_one.data
        assert sps.isspmatrix_csr(s)
        data, rows, cols = [], [], []
        print("Keep only k-similar item..")
        for i_row in range(n_item):
            row = s[i_row, :]
            idxs = np.array(row.indices)
            values = np.array(row.data)
            k_top = min(k, values.shape[0])
            topk_idxs_values = np.argpartition(values, -(k_top))[-(k_top):]
            n = topk_idxs_values.shape[0]
            # create incrementally the similarity matrix
            data.extend(values[topk_idxs_values])
            cols.extend(idxs[topk_idxs_values])
            rows.extend(np.full(n, i_row))
        print("Building sparse matrix..")
        s = sps.csr_matrix((data, (rows, cols)), shape=(n_item, n_item), dtype=np.float32)
        # apply q
        s.data = s.data ** q
        # do rating
        print("Start building rating matrix..")
        # important!!! multiply for s, not for s.T!
        r = urm * s
        print("Normalizing rating matrix..")
        r = r.tocsr()
        normv = []
        for i_row in range(n_user):
            norm = urm[i_row, :].nnz
            if norm == 0:
                normv.append(1.0)
            else:
                normv.append(1.0 / norm)
        normv = np.array(normv)
        inplace_row_scale(r, normv)

        self.s = s
        self.r = r
        print("Model done")
        return

    def getRM_csr(self, targetPlaylists= [], targetTracks=[]):
        r = self.r
        if len(targetPlaylists) == 0 or len(targetTracks) == 0:
            return r
        n_playlist = self.n_playlists
        n_track = self.n_tracks

        target_p = np.unique(targetPlaylists)
        target_t = np.unique(targetTracks)

        diag_t = sps.csr_matrix((np.ones(target_t.shape[0]), (target_t, target_t)), shape=(n_track, n_track),
                                dtype=np.float32)
        diag_p = sps.csr_matrix((np.ones(target_p.shape[0]), (target_p, target_p)), shape=(n_playlist, n_playlist),
                                dtype=np.float32)

        r_cut = r * diag_t
        r_cut = ((r_cut.T) * diag_p).T
        self.r = r_cut
        return r_cut

    def print_setting(self):
        info = ("CB "+str(self.name)+" ASY:" +
                "\tmap = " + str(round(self.MAP, 4))
                + "\tk = " + str(self.k)
                + "\talpha = " + str(self.alpha)
                + "\tq = " + str(self.q)
                + "\tm = " + str(round(self.m, 1)))
        print info
        return info
