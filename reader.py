import numpy as np
import pandas as pd
import tensorflow as tf

class Reader(object):

    def read_data(self):
        pass

    def merge_id(self):
        self._train_data['h_id'] = self._e_id[self._train_data.h].values
        self._train_data['r_id'] = self._r_id[self._train_data.r].values
        self._train_data['t_id'] = self._e_id[self._train_data.t].values

        self._test_data['h_id'] = self._e_id[self._test_data.h].values
        self._test_data['r_id'] = self._r_id[self._test_data.r].values
        self._test_data['t_id'] = self._e_id[self._test_data.t].values

        self._valid_data['h_id'] = self._e_id[self._valid_data.h].values
        self._valid_data['r_id'] = self._r_id[self._valid_data.r].values
        self._valid_data['t_id'] = self._e_id[self._valid_data.t].values
    
    def gen_t_label(self):
        full = pd.concat([self._train_data, self._test_data, self._valid_data], ignore_index=True)
        f_t_labels = full['t_id'].groupby([full['h_id'], full['r_id']]).apply(lambda x: pd.unique(x.values))
        f_t_labels.name = 't_label'

#         f_h_labels = full['h_id'].groupby([full['t_id'], full['r_id']]).apply(lambda x: pd.unique(x.values))
#         f_h_labels.name = 'h_label'

        self._test_data = self._test_data.join(f_t_labels, on=['h_id', 'r_id'])
#         self._test_data = self._test_data.join(f_h_labels, on=['t_id', 'r_id'])

        self._valid_data = self._valid_data.join(f_t_labels, on=['h_id', 'r_id'])
#         self._valid_data = self._valid_data.join(f_h_labels, on=['t_id', 'r_id'])

    def add_reverse(self):
        def add_reverse_for_data(data):
            reversed_data = data.rename(columns={'h_id': 't_id', 't_id': 'h_id'})
            reversed_data.r_id += self._relation_num
            data = pd.concat(([data, reversed_data]), ignore_index=True)
            return data

        self._train_data = add_reverse_for_data(self._train_data)
        self._test_data = add_reverse_for_data(self._test_data)
        self._valid_data = add_reverse_for_data(self._valid_data)
        self._relation_num_for_eval = self._relation_num
        self._relation_num *= 2
        print (self._relation_num, self._relation_num_for_eval)

    def reindex_kb(self):
        train_data = self._train_data
        test_data = self._test_data
        valid_data = self._valid_data
        eids = pd.concat([train_data.h_id, train_data.t_id,], ignore_index=True)

        tv_eids = np.unique(pd.concat([test_data.h_id, test_data.t_id, valid_data.t_id, valid_data.h_id]))
        not_train_eids = tv_eids[~np.in1d(tv_eids, eids)]
        print(not_train_eids)

        rids = pd.concat([train_data.r_id,],ignore_index=True)
        

        def gen_map(eids, rids):
            e_num = eids.groupby(eids.values).size().sort_values()[::-1]
            not_train = pd.Series(np.zeros_like(not_train_eids), index=not_train_eids)
            e_num = pd.concat([e_num, not_train])

            r_num = rids.groupby(rids.values).size().sort_values()[::-1]
            e_map = pd.Series(range(e_num.shape[0]), index=e_num.index)
            r_map = pd.Series(range(r_num.shape[0]), index=r_num.index)
            return e_map, r_map
        
        def remap_kb(kb, e_map, r_map):
            kb.loc[:, 'h_id'] = e_map.loc[kb.h_id.values].values
            kb.loc[:, 'r_id'] = r_map.loc[kb.r_id.values].values
            kb.loc[:, 't_id'] = e_map.loc[kb.t_id.values].values
            return kb
        
        def remap_id(s, rm):
            s = rm.loc[s.values].values
            return s
        
        e_map, r_map = gen_map(eids, rids)
        self._e_map, self._r_map = e_map, r_map
        
        self._train_data = remap_kb(train_data, e_map, r_map)
        self._valid_data = remap_kb(self._valid_data, e_map, r_map)
        self._test_data = remap_kb(self._test_data, e_map, r_map)
        
        self._e_id = remap_id(self._e_id, e_map)
        self._r_id = remap_id(self._r_id, r_map)
        
        return not_train_eids
    
    
    def in2d(self, arr1, arr2):
        """Generalisation of numpy.in1d to 2D arrays"""

        assert arr1.dtype == arr2.dtype

        arr1_view = np.ascontiguousarray(arr1).view(np.dtype((np.void,
                                                              arr1.dtype.itemsize * arr1.shape[1])))
        arr2_view = np.ascontiguousarray(arr2).view(np.dtype((np.void,
                                                              arr2.dtype.itemsize * arr2.shape[1])))
        intersected = np.in1d(arr1_view, arr2_view)
        return intersected.view(np.bool).reshape(-1)





    def gen_filter_mat(self):
        def gen_filter_vector(r):
            v = np.ones(self._entity_num)
            v[r] = -1
            return v

        print('start gen filter mat')



        self._tail_valid_filter_mat = np.stack(self._valid_data.t_label.apply(gen_filter_vector).values)
        self._tail_test_filter_mat = np.stack(self._test_data.t_label.apply(gen_filter_vector).values)

#         self._head_valid_filter_mat = np.stack(self._valid_data.h_label.apply(gen_filter_vector).values)
#         self._head_test_filter_mat = np.stack(self._test_data.h_label.apply(gen_filter_vector).values)





    def gen_label_mat_for_train(self):
        def gen_train_relation_label_vac(r):
            c = pd.value_counts(r)
            values = 1. * c.values / c.sum()
            return np.stack([c.index, values], axis=1)

        def gen_train_entity_label_vac(r):
            indices = np.stack([r.label_id.values, r.values], axis=1)
            values = np.ones_like(r.values, dtype=np.int)
            return tf.SparseTensor(indices=indices, values=values, dense_shape=[1, self._entity_num])

        tr = self._train_data
        print('start gen t_label')
        labels = tr['t_id'].groupby([tr['h_id'], tr['r_id']]).size()
        labels = pd.Series(range(labels.shape[0]), index=labels.index)
        labels.name = 'label_id'
        tr = tr.join(labels, on=['h_id', 'r_id'])

        self._train_data = tr
        sp_tr = tf.SparseTensor(tr[['label_id', 't_id']].values, np.ones([len(tr)], dtype=np.float32), dense_shape=[len(tr), self._entity_num])

        self._label_indices, self._label_values = sp_tr.indices[:], sp_tr.values[:]





class FreeBaseReader(Reader):

    def read_data(self):
        path = self._options.data_path
        tr = pd.read_csv(path + 'train.txt', header=None, sep='\t', names=['h', 't', 'r'])
        te = pd.read_csv(path + 'test.txt', header=None, sep='\t', names=['h', 't', 'r'])
        val = pd.read_csv(path + 'valid.txt', header=None, sep='\t', names=['h', 't', 'r'])

        e_id = pd.read_csv(path + 'entity2id.txt', header=None, sep='\t', names=['e', 'eid'])
        e_id = pd.Series(e_id.eid.values, index=e_id.e.values)
        r_id = pd.read_csv(path + 'relation2id.txt', header=None, sep='\t', names=['r', 'rid'])
        r_id = pd.Series(r_id.rid.values, index=r_id.r.values)
        
        

        self._entity_num = e_id.shape[0]
        self._relation_num = r_id.shape[0]

        self._train_data = tr
        self._test_data = te
        self._valid_data = val

        self._e_id, self._r_id = e_id, r_id



class WordNetReader(Reader):

    def read_data(self):
        path = self._options.data_path
        tr = pd.read_csv(path+'train.txt', header=None, sep='\t', names=['h', 'r', 't'])
        te = pd.read_csv(path + 'test.txt', header=None, sep='\t', names=['h', 'r', 't'])
        val = pd.read_csv(path + 'valid.txt', header=None, sep='\t', names=['h', 'r', 't'])
#         de = pd.read_csv(path + 'wordnet-mlj12-definitions.txt', header=None, sep='\t',
#                          names=['h', 'name', 'description'])
        
        r_list = pd.unique(pd.concat([tr.r, te.r, val.r]))
        r_list = pd.Series(r_list, index=np.arange(r_list.shape[0]))
        
        e_list = pd.unique(pd.concat([tr.h, te.h, val.h, tr.t, te.t, val.t, ]))
        e_list = pd.Series(e_list, index=np.arange(e_list.shape[0]))
        
        
        e_id = pd.Series(e_list.index, index=e_list.values)
        r_id = pd.Series(r_list.index, index=r_list.values)




        self._entity_num = e_id.shape[0]
        self._relation_num = r_id.shape[0]

        self._train_data = tr
        self._test_data = te
        self._valid_data = val
#         self._definition_data = de

        self._e_id, self._r_id = e_id, r_id

class FB13Reader(Reader):
    def read_data(self):
        path = self._options.data_path
        tr = pd.read_csv(path+'train.txt', header=None, sep='\t', names=['h', 'r', 't'])
        te = pd.read_csv(path + 'test.txt', header=None, sep='\t', names=['h', 'r', 't', 'l'])
        val = pd.read_csv(path + 'dev.txt', header=None, sep='\t', names=['h', 'r', 't', 'l'])

        r_list = pd.unique(pd.concat([tr.r, te.r, val.r]))
        r_list = pd.Series(r_list, index=np.arange(r_list.shape[0]))

        e_list = pd.unique(pd.concat([tr.h, tr.t, te.h, te.t, val.h, val.t], ignore_index=True))
        e_list = pd.Series(e_list, index=np.arange(e_list.shape[0]))

        e_id = pd.Series(e_list.index, index=e_list.values)
        r_id = pd.Series(r_list.index, index=r_list.values)

        self._entity_num = e_id.shape[0]
        self._relation_num = r_id.shape[0]

        self._test_classification_label = te.l
        self._valid_classification_label = val.l

        self._train_data = tr
        self._test_data = te[['h', 'r', 't']]
        self._valid_data = val[['h', 'r', 't']]

        self._e_id, self._r_id = e_id, r_id


class WN11Reader(FB13Reader):
    pass

