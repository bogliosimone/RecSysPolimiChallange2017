import numpy as np
import scipy.sparse as sps
from sklearn.utils.sparsefuncs import inplace_row_scale
from sklearn.utils.sparsefuncs import inplace_column_scale


class CF_IB_asymmetric_cosine:
    def __init__(self, urm, k=100, alpha=0.5, q=1, m=0):
        urm = sps.csr_matrix(urm, dtype=np.float32)
        if m == 0:
            m = 1e-6

        self.urm = urm
        self.k = k
        self.alpha = alpha
        self.q = q
        self.m = m
        self.MAP = 0.0

        n_user = urm.shape[0]
        n_item = urm.shape[1]
        self.n_playlists = n_user
        self.n_tracks = n_item

        print("Start asymmetric cosine item-based model..")
        s = (urm.T * urm).tocsr()
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
        print("Model asymmetric cosine item-based done")

        self.s = s
        self.r = r
        return

    def getRM_csr(self, targetPlaylists = [], targetTracks = []):
        r = self.r
        if len(targetPlaylists)==0 or len(targetTracks)==0:
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

    def evaluate(self, targetPlaylists, targetTracks, urm_test, n_rec=5):
        # cut the matrix
        self.r = self.getRM_csr(targetPlaylists, targetTracks)

        cumulative_MAP = 0.0
        num_eval = 0
        n_target_playlists = len(targetPlaylists)
        for i, user_id in enumerate(targetPlaylists):
            if i % 2000 == 0:
                print("User %d of %d" % (i, n_target_playlists))
            relevant_items = urm_test[user_id].indices
            if len(relevant_items) > 0:
                recommended_items = self._recommend(user_id, n_rec)
                num_eval += 1
                cumulative_MAP += self._MAP(recommended_items, relevant_items)
        cumulative_MAP /= num_eval
        print("Recommender performance is: MAP = {:.4f}".format(cumulative_MAP))
        self.MAP = cumulative_MAP
        return cumulative_MAP

    def _recommend(self, targetPlaylist, n_rec=5):
        scores = (self.r[targetPlaylist, :]).toarray().ravel()
        ranking = self._filter_seen(targetPlaylist, scores, n_rec)
        return ranking

    def _MAP(self, recommended_items, relevant_items):
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
        # Cumulative sum: precision at 1, at 2, at 3 ...
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1.0 + np.arange(is_relevant.shape[0]))
        map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
        return map_score

    def _filter_seen(self, playlist_id, scores, n_rec=5):
        ranking = np.array([])
        k = 70  # adjust this value
        count = 0
        while (ranking.shape[0] < n_rec):
            top_scores_idx_no_ordered = np.argpartition(scores, -k)[-k:]
            top_scores_value_no_ordered = scores[top_scores_idx_no_ordered]
            top_scores_idx_value_ordered = top_scores_value_no_ordered.argsort()[::-1]
            top_scores_idx_ordered = top_scores_idx_no_ordered[top_scores_idx_value_ordered]
            ranking = top_scores_idx_ordered
            user_profile = self.urm[playlist_id]
            seen = user_profile.indices
            unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
            ranking = ranking[unseen_mask]
            k = k + 50  # adjust this value
            count = count + 1
        # if count > 0:
        #    print ("re-sample argpartition: " + str(count) + " time")
        return ranking[:n_rec]

    def print_setting(self):
        info = ("CF IB ASY:" +
                "\tmap = " + str(round(self.MAP, 4))
                + "\tk = " + str(self.k)
                + "\talpha = " + str(self.alpha)
                + "\tq = " + str(self.q)
                + "\tm = " + str(round(self.m, 1)))
        print info
        return info
