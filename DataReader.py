import pandas as pd
import numpy as np
import scipy.sparse as sp
import cPickle as pickle
import random
import os
from numpy import bincount, log, sqrt
from sklearn.utils.sparsefuncs import inplace_row_scale

class DataReaderChallangePolimi2017:
    folder = "./CsvFiles/"
    f_train = folder + "train_final.csv"
    f_target_tracks = folder + "target_tracks.csv"
    f_target_playlist = folder + "target_playlists.csv"
    f_playlists = folder + "playlists_final.csv"
    f_tracks = folder + "tracks_final.csv"
    folder_dump = "./DumpData/"
    folder_submissions = "./Submissions/"
    f_dump_train_test = folder_dump + "DataReaderTrainTestDump"
    f_dump_full = folder_dump + "DataReaderFullDump"

    dict_tracks = {}
    dict_playlists = {}

    dict_tracks_reverse = {}
    dict_playlists_reverse = {}

    dict_attribute_reverse = {}
    dict_artists = {}
    dict_albums = {}
    dict_tags = {}
    range_artists = []
    range_albums = []
    range_tags = []

    playlists = np.array([])
    tracks = np.array([])
    target_playlists = np.array([])
    target_tracks = np.array([])

    full_train = np.array([], [])
    train = np.array([[], []])
    test = np.array([[], []])

    artists = np.array([])
    albums = np.array([])
    tags = np.array([])

    n_playlists = 0
    n_tracks = 0
    n_attributes = 0

    urm = sp.coo_matrix(([], (train[0], train[1])), shape=(n_playlists, n_tracks)).tocsr()
    urm_test = sp.coo_matrix(([], (test[0], test[1])), shape=(n_playlists, n_tracks)).tocsr()
    norm_vect_urm = np.array([])


    def __init__(self, evaluation=True, rebuild=False, fraction_train=0.8):
        print ("Date Reader Challange Polimi 2017")
        if rebuild == True:
            self._buildData()
            if evaluation == True:
                print ("Split dataset in train-test, update target tracks and urm..")
                self._splitTrainTest(fraction_train)
                self._save(self.f_dump_train_test)
                print ("Dataset train-test created")
            else:
                self._save(self.f_dump_full)
                print ("Dataset full created")
        if rebuild == False:
            if evaluation == True:
                self._load(self.f_dump_train_test)
                print ("Dataset train-test loaded")
            else:
                self._load(self.f_dump_full)
                print ("Dataset full loaded")
        return

    def _buildData(self):
        self._buildPlaylists()
        print ("Playlists created..")
        self._buildTracks()
        print ("Tracks created..")
        self._buildTargetPlaylists()
        print ("Target playlists created..")
        self._buildTargetTracks()
        print ("Target tracks created..")
        self._buildTrain()
        self._buildURM_csr()
        print ("Train and URM created..")
        self._buildArtistsAlbumTags()
        print ("Attributes tracks created..")
        return

    def _save(self, file_dump):
        print ("Saving data in " + str(file_dump))
        if not os.path.exists(self.folder_dump):
            print ("Folder not found, creating folder " + self.folder_dump)
            os.makedirs(self.folder_dump)
        pickle.dump(self.__dict__, open(file_dump, "wb"))
        # print ("Data saved")

        return

    def _load(self, file_dump):
        print ("Loading data from " + str(file_dump))
        tmp = pickle.load(open(file_dump, "rb"))
        self.__dict__.update(tmp)
        # print ("Data loaded")
        return

    def _buildPlaylists(self):
        df = self._readPlaylists()
        self.dict_playlists, self.dict_playlists_reverse = self._buildDictAligned(df["playlist_id"])
        self.playlists = np.array(self.dict_playlists.keys())
        self.n_playlists = self.playlists.shape[0]
        return

    def _buildTracks(self):
        df = self._readTracks()
        self.dict_tracks, self.dict_tracks_reverse = self._buildDictAligned(df["track_id"])
        self.tracks = np.array(self.dict_tracks.keys())
        self.n_tracks = self.tracks.shape[0]
        return

    def _buildTargetPlaylists(self):
        df = self._readTargetPlaylists()
        self.target_playlists = np.array(map(self.dict_playlists.get, df["playlist_id"]))
        return

    def _buildTargetTracks(self):
        df = self._readTargetTracks()
        self.target_tracks = np.array(map(self.dict_tracks.get, df["track_id"]))
        return

    def _buildTrain(self):
        df = self._readTrain()
        playlists = np.array(map(self.dict_playlists.get, df["playlist_id"]))
        tracks = np.array(map(self.dict_tracks.get, df["track_id"]))
        self.train = np.array([playlists, tracks])
        self.full_train = self.train
        return

    def _buildURM_csr(self):
        v = np.ones(self.train[0].shape[0])
        urm = sp.coo_matrix((v, (self.train[0], self.train[1])), shape=(self.n_playlists, self.n_tracks),dtype=np.float).tocsr()
        self.urm = urm
        normv = []
        for i_row in range(self.n_playlists):
            norm = self.urm[i_row, :].nnz
            if norm == 0:
                normv.append(1.0)
            else:
                normv.append(1.0 / norm)
        self.norm_vect_urm = np.array(normv)
        return

    def _buildTestURM_csr(self):
        if self.test[0].shape[0] == 0:
            print("--- !!!Warning!!! There is not test set here! ---")
        v = np.ones(self.test[0].shape[0])
        urm_t = sp.coo_matrix((v, (self.test[0], self.test[1])), shape=(self.n_playlists, self.n_tracks),dtype=np.float).tocsr()
        self.urm_test = urm_t
        return

    def _buildDictAligned(self, values):
        values = np.unique(values)
        new_values = range(0, values.shape[0])
        dic_r = dict(zip(new_values, values))
        dic = dict(zip(values, new_values))
        return dic, dic_r

    def _buildDictsArtistsAlbumsTags(self, df):
        l = []
        for row in df["tags"]:
            l.extend(row)
        tags = np.unique(np.array(l))
        l = []
        for a in df["artist_id"]:
            l.append(a)
        artists = np.unique(np.array(l))
        l = []
        for row in df["album"]:
            l.extend(row)
        albums = np.unique(np.array(l))
        r1 = artists.shape[0]
        r2 = r1 + albums.shape[0]
        r3 = r2 + tags.shape[0]
        self.range_artists = [0, r1]
        self.range_albums = [r1, r2]
        self.range_tags = [r2, r3]
        self.n_attributes = r3
        self.dict_attribute_reverse = dict(zip(range(0, r3), np.append(np.append(artists, albums), tags)))
        self.dict_artists = dict(zip(artists, range(0, r1)))
        self.dict_albums = dict(zip(albums, range(r1, r2)))
        self.dict_tags = dict(zip(tags, range(r2, r3)))
        return

    def _buildArtistsAlbumTags(self):
        df = self._readTracks()

        self._buildDictsArtistsAlbumsTags(df)

        artists = []
        albums = []
        tags = []
        tracks = df["track_id"]

        ar = df["artist_id"]
        for i in range(tracks.shape[0]):
            t = self.dict_tracks[tracks[i]]
            a = self.dict_artists[ar[i]]
            artists.append([t, a])
        self.artists = np.array(artists).T

        al = df["album"]
        for i in range(tracks.shape[0]):
            t = self.dict_tracks[tracks[i]]
            for a in al[i]:
                a = self.dict_albums[a]
                albums.append([t, a])
        self.albums = np.array(albums).T

        ta = df["tags"]
        for i in range(tracks.shape[0]):
            t = self.dict_tracks[tracks[i]]
            for tag in ta[i]:
                tag = self.dict_tags[tag]
                tags.append([t, tag])
        self.tags = np.array(tags).T
        return

    def _readTrain(self):
        # playlist_id - track_id
        df = pd.read_csv(self.f_train, sep="\t", header=0, dtype={"playlist_id": np.int, "track_id": np.int})
        # print("Interaction imported..")
        return df

    def _readTargetPlaylists(self):
        # playlist_id
        df = pd.read_csv(self.f_target_playlist, sep="\t", header=0, dtype={"playlist_id": np.int})
        print("Target playlists imported..")
        return df

    def _readTargetTracks(self):
        # track_id
        df = pd.read_csv(self.f_target_tracks, sep="\t", header=0, dtype={"track_id": np.int})
        # print("Target tracks imported..")
        return df

    def _readPlaylists(self):
        # created_at - playlist_id - title - numtracks - duration - owner
        df = pd.read_csv(self.f_playlists, sep="\t", header=0, dtype={"playlist_id": np.int, "created_at": np.int,
                                                                      "numtracks": np.int, "title": str,
                                                                      "duration": np.int, "owner": np.int
                                                                      })
        df['title'] = self._vectStringToVectArray(df['title'])
        # print("Playlists imported..")
        return df

    def _readTracks(self):
        # track_id - artist_id - duration - playcount - album - tags
        df = pd.read_csv(self.f_tracks, sep="\t", header=0, dtype={"track_id": np.int, "artist_id": np.int,
                                                                   "playcount": np.float, "duration": np.int,
                                                                   "album": str, "tags": str
                                                                   })
        df['album'] = self._vectStringToVectArray(df['album'])
        df['tags'] = self._vectStringToVectArray(df['tags'])
        # print("Tracks imported..")
        return df

    def _vectStringToVectArray(self, vect):
        new_vect = []
        for a in vect:
            try:
                a = a.replace("[", "").replace("]", "").replace(" ", "")
                a = map(np.int, a.split(','))
            except (ValueError, IndexError) as e:
                a = []
            new_vect.append(np.array(a))
        return new_vect

    def _splitTrainTest(self, fraction_train=0.8):
        p_all = self.train[0]
        t_all = self.train[1]
        target_playlist = self.target_playlists

        train_p, train_t = [], []
        test_p, test_t = [], []
        dict_p = {}

        n = len(p_all)
        for i in range(0, n):
            if p_all[i] in target_playlist:
                if p_all[i] in dict_p:
                    dict_p[p_all[i]].append(t_all[i])
                else:
                    dict_p[p_all[i]] = []
                    dict_p[p_all[i]].append(t_all[i])
            else:
                train_p.append(p_all[i])
                train_t.append(t_all[i])

        for p in dict_p:
            tracks = dict_p[p]
            inds = set(random.sample(list(range(len(tracks))), int(fraction_train * len(tracks))))  # maybe add -1
            train_tracks = [n for i, n in enumerate(tracks) if i in inds]
            test_tracks = [n for i, n in enumerate(tracks) if i not in inds]
            for t in train_tracks:
                train_p.append(p)
                train_t.append(t)
            for t in test_tracks:
                test_p.append(p)
                test_t.append(t)

        train_p = np.array(train_p, dtype=np.int)
        train_t = np.array(train_t, dtype=np.int)
        test_p = np.array(test_p, dtype=np.int)
        test_t = np.array(test_t, dtype=np.int)

        self.train = np.array([train_p, train_t])
        self.test = np.array([test_p, test_t])
        self.target_tracks = np.unique(test_t)
        self._buildURM_csr()
        self._buildTestURM_csr()
        return

    def _recommend(self, targetPlaylist, n_rec=5):
        scores = (self.r[targetPlaylist, :]).toarray().ravel()
        ranking, scores = self._filter_seen(targetPlaylist, scores, n_rec)
        return ranking, scores

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
            r_scores = scores[ranking[:n_rec]]
            k = k + 50  # adjust this value
            count = count + 1
        # if count > 0:
        #    print ("re-sample argpartition: " + str(count) + " time")
        return ranking[:n_rec],r_scores

    def getURM_csr(self):
        return self.urm

    def getTestURM_csr(self):
        if self.test[0].shape[0] == 0:
            print("--- !!!Warning!!! There is not test set here! ---")
        return self.urm_test

    def getTrain(self):
        return self.train

    def getTest(self):
        if self.test[0].shape[0] == 0:
            print("--- !!!Warning!!! There is not test set here! ---")
        return self.test

    def getTargetPlaylists(self):
        return self.target_playlists

    def getTargetTracks(self):
        return self.target_tracks

    def getICM_csr(self, artists=False, albums=False, tags=False):
        tracks = []
        attributes = []
        if artists == True:
            tracks.extend(self.artists[0])
            attributes.extend(self.artists[1])
        if albums == True:
            tracks.extend(self.albums[0])
            attributes.extend(self.albums[1])
        if tags == True:
            tracks.extend(self.tags[0])
            attributes.extend(self.tags[1])

        tracks = np.array(tracks)
        attributes = np.array(attributes)
        v = np.ones(attributes.shape[0])
        icm = sp.coo_matrix((v, (tracks, attributes)), shape=(self.n_tracks, self.n_attributes),dtype=np.float).tocsr()
        # set to 1 because there are more or less 15 tracks with a tags repeated 2 time (nice dataset!)
        icm.data = np.ones(icm.data.shape[0])       
        '''
        #not usefull cut matrix atm
        n_track = self.n_tracks
        t1 = self.train[1]
        t2 = self.test[1]
        target_t = np.unique(np.append(t1,t2))
        diag_t = sp.csr_matrix((np.ones(target_t.shape[0]), (target_t, target_t)), shape=(n_track, n_track),
                               dtype=np.float32)
        icm = ((icm.T) * diag_t).T 
        '''
        return icm

    def getUCM_csr(self, artists = False, albums = False):
        n_playlists = self.n_playlists
        n_attributes = self.n_attributes
        train = self.train.T
        r,c,v = [],[],[]
        
        if (artists):
            dic_ar = {}
            tr_ar = self.artists.T
            for row in tr_ar:
                dic_ar[row[0]] = row[1]
            for row in train:
                r.append(row[0])
                c.append(dic_ar[row[1]])
                v.append(1)
        
        if (albums):
            dic_al = {}
            tr_al = self.albums.T
            for row in tr_al:
                dic_al[row[0]] = row[1]
            for row in train:
                if row[1] not in dic_al: continue #these track doesn't have an album
                r.append(row[0])
                c.append(dic_al[row[1]])
                v.append(1)

        ucm = sp.coo_matrix((v, (r, c)), shape=(n_playlists, n_attributes),dtype=np.float).tocsr()
        return ucm
    
    
    def getOwnersUCM_csr(self):
        #owner ucm, since the max id is not too big, we don't use a dict for owners
        df = self._readPlaylists()

        owners = []
        playlists = df["playlist_id"]
        ow = df["owner"]
        
        for i in range(playlists.shape[0]):
            p = self.dict_playlists[playlists[i]]
            o = ow[i]
            owners.append([p, o])
            
        owners = np.array(owners).T        
        n_playlists = self.n_playlists
        n_owners = max(owners[1]) +1
        
        r,c,v = [],[],[]
        for row in owners.T:
            r.append(row[0])
            c.append(row[1])
            v.append(1)
        ucm_owners = sp.coo_matrix((v, (r, c)), shape=(n_playlists, n_owners),dtype=np.float).tocsr()
        return ucm_owners
    
    
    def getOwners(self):
        #owner ucm, since the max id is not too big, we don't use a dict for owners
        df = self._readPlaylists()

        owners = []
        playlists = df["playlist_id"]
        ow = df["owner"]
        
        for i in range(playlists.shape[0]):
            p = self.dict_playlists[playlists[i]]
            o = ow[i]
            owners.append([p, o])
            
        owners = np.array(owners).T        
        n_owners = max(owners[1]) +1
        return owners, n_owners
    
    
    
    def getTitlesUCM_csr(self):
        #titles ucm, since the max id is not too big, we don't use a dict for each word of titles
        df = self._readPlaylists()
        
        titles = []
        playlists = df["playlist_id"]
        t = df["title"]
        
        for i in range(playlists.shape[0]):
            p = self.dict_playlists[playlists[i]]
            for word in t[i]:
                titles.append([p, word])
                
        titles = np.array(titles).T
        n_playlists = self.n_playlists
        n_titles = max(titles[1]) +1
        
        r,c,v = [],[],[]
        for row in titles.T:
            r.append(row[0])
            c.append(row[1])
            v.append(1)
        ucm_titles = sp.coo_matrix((v, (r, c)), shape=(n_playlists, n_titles),dtype=np.float).tocsr()
        return ucm_titles
    
    
    def getOwnersICM_csr(self, ones=True):
        train = self.train.T
        owners, n_owners = self.getOwners()
        n_tracks = self.n_tracks
        
        d = {}
        for row in train:
            p = row[0]
            t = row[1]
            if p in d:
                d[p].append(t)
            else:
                d[p] = []
                d[p].append(t)
        t_o = []
        for row in owners.T:
            o = row[1]
            p = row[0]
            if p not in d:
                continue #we don't have this playlist in train
            t_p = d[p]
            for t in t_p:
                t_o.append([t,o])
        t_o = np.array(t_o)
        
        r,c,v = [],[],[]
        for row in t_o:
            r.append(row[0])
            c.append(row[1])
            v.append(1)

        icm = sp.coo_matrix((v, (r, c)), shape=(n_tracks, n_owners ),dtype=np.float).tocsr()
        if ones: icm.data = np.ones(len(icm.data))
        return icm
    
    
    def buildSubmissionFile(self, recommendation, file_name):
        # format recommendation: [ [p1,[t1,t2,t3,t4,t5]] , [p2,[t1,t2,t3,t4,t5]] , ... ]
        file_name = self.folder_submissions + file_name
        print ("Creating submission file in: " + file_name)
        if not os.path.exists(self.folder_submissions):
            print ("Folder not found, creating folder " + self.folder_submissions)
            os.makedirs(self.folder_submissions)
        with open(file_name, "w") as f:
            f.write("playlist_id,track_ids\n")
            for row in recommendation:
                p = str(self.dict_playlists_reverse[row[0]])
                t1 = str(self.dict_tracks_reverse[row[1][0]])
                t2 = str(self.dict_tracks_reverse[row[1][1]])
                t3 = str(self.dict_tracks_reverse[row[1][2]])
                t4 = str(self.dict_tracks_reverse[row[1][3]])
                t5 = str(self.dict_tracks_reverse[row[1][4]])
                f.write(p + "," + t1 + " " + t2 + " " + t3 + " " + t4 + " " + t5 + "\n")
        print("Done")
        return

    def reduceRM(self, r):
        n_playlist = self.n_playlists
        n_track = self.n_tracks

        target_p = self.target_playlists
        target_t = self.target_tracks

        diag_t = sp.csr_matrix((np.ones(target_t.shape[0]), (target_t, target_t)), shape=(n_track, n_track),
                                dtype=np.float32)
        diag_p = sp.csr_matrix((np.ones(target_p.shape[0]), (target_p, target_p)), shape=(n_playlist, n_playlist),
                                dtype=np.float32)

        r_cut = r * diag_t
        r_cut = ((r_cut.T) * diag_p).T
        return r_cut.tocsr()

    def normalizeRMCountURM_inplace(self,r):
        inplace_row_scale(r, self.norm_vect_urm)
        return r

    def evaluateMAP(self, r, n_rec=5, allmaps = False, verbose=True):
        targetPlaylists = self.target_playlists
        # cut the matrix
        self.r = self.reduceRM(r)
        allmap = []
        cumulative_MAP = 0.0
        num_eval = 0
        n_target_playlists = len(targetPlaylists)
        for i, user_id in enumerate(targetPlaylists):
            if i % 2500 == 0:
                if verbose : print("User %d of %d" % (i, n_target_playlists))
            relevant_items = self.urm_test[user_id].indices
            if len(relevant_items) > 0:
                recommended_items, _ = self._recommend(user_id, n_rec)
                num_eval += 1
                score = self._MAP(recommended_items, relevant_items)
                allmap.append([user_id,score,recommended_items])
                cumulative_MAP +=  score
        cumulative_MAP /= num_eval
        if verbose :print("Recommender performance is: MAP = {:.4f}".format(cumulative_MAP))
        if allmaps == True: return allmap, num_eval
        return cumulative_MAP

    def evaluateMAPfromRecommendations(self, allRecs,n_rec = 5, verbose = True):
        targetPlaylists = self.target_playlists
        cumulative_MAP = 0.0
        num_eval = 0
        n_target_playlists = len(targetPlaylists)
        for i, row in enumerate(allRecs):
            user_id = row[0]
            if i % 2500 == 0:
                if verbose :print("User %d of %d" % (i, n_target_playlists))
            relevant_items = self.urm_test[user_id].indices
            if len(relevant_items) > 0:
                recommended_items = row[1][:n_rec]
                num_eval += 1
                cumulative_MAP += self._MAP(recommended_items, relevant_items)
        cumulative_MAP /= num_eval
        if verbose :print("Recommender performance is: MAP = {:.4f}".format(cumulative_MAP))
        return cumulative_MAP


    def getAllRecommendations(self, r, n_rec = 5, verbose = True):
        # format recommendation: [ [p1,[t1,t2,t3,t4,t5],[r1,r2,r3,r4,r5]], [p2,[t1,t2,t3,t4,t5],[r1,r2,r3,r4,r5]] , ... ]
        targetPlaylists = self.target_playlists
        # cut the matrix
        self.r = self.reduceRM(r)
        n_target_playlists = len(targetPlaylists)
        allRec = []
        for i, user_id in enumerate(targetPlaylists):
            if i % 2500 == 0:
                if verbose :print("User %d of %d" % (i, n_target_playlists))
            recommended_items, scores = self._recommend(user_id, n_rec)
            allRec.append([user_id,recommended_items,scores])
        if verbose :print("Recommendations done")
        return allRec

    #######
    #
    # Utility functions
    #
    #######
    
    
    def save_sparse_csr(self,filename,array):
        np.savez(filename,data = array.data ,indices=array.indices,indptr =array.indptr, shape=array.shape )


    def load_sparse_csr(self,filename):
        loader = np.load(filename)
        return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),shape = loader['shape'])
    
    def knn_csr(self,r, k, verbose=True):
        if verbose: print ("!!!Care, maybe this function is broken!!!")
        data, rows, cols = [], [], []
        for i_row in range(r.shape[0]):
            row = r[i_row, :]
            idxs = np.array(row.indices)
            values = np.array(row.data)
            k_top = min(k, values.shape[0])
            topk_idxs_values = np.argpartition(values, -(k_top))[-(k_top):]
            n = topk_idxs_values.shape[0]
            # create incrementally the similarity matrix
            data.extend(values[topk_idxs_values])
            cols.extend(idxs[topk_idxs_values])
            rows.extend(np.full(n, i_row))
        s = sp.csr_matrix((data, (rows, cols)), shape=(r.shape[0],r.shape[1]), dtype=np.float32)
        return s.tocsr()
    
    def bm25_row(self, X, K1=100, B=0.8):
        #Weighs each row of a sparse matrix by OkapiBM25 weighting
        # calculate idf per term (user)
        X = sp.coo_matrix(X)
        N = float(X.shape[0])
        idf = log(N / (1 + bincount(X.col)))
        
        # calculate length_norm per document (artist)
        row_sums = np.ravel(X.sum(axis=1))
        average_length = row_sums.mean()
        length_norm = (1.0 - B) + B * row_sums / average_length
        
        # weight matrix rows by bm25
        X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.row] + X.data) * idf[X.col]
        return X.tocsr()
    
    def tfidf_row(self, X):
        #TFIDF each row of a sparse amtrix
        X = sp.coo_matrix(X)

        # calculate IDF
        N = float(X.shape[0])
        idf = log(N / (1 + bincount(X.col)))

        # apply TF-IDF adjustment
        X.data = sqrt(X.data) * idf[X.col]
        return X.tocsr()


        
        
        
        
        
        
        
        
