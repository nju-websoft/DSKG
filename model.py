import os, random, datetime
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import gc
from reader import *

flags = tf.app.flags

flags.DEFINE_string("save_path", 'ckpt/RNNKM', "Directory to write the model and "
                                               "training summaries.")

flags.DEFINE_string("data_path", 'data/FB15k/', "Training text file. "
                                                "E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")

flags.DEFINE_integer("hidden_size", 512, 'the hidden size')

flags.DEFINE_integer("batch_size", 2048,
                     "Number of training examples processed per step "
                     "(size of a minibatch)."
                     )
flags.DEFINE_float("keep_prob", 0.5, '')

flags.DEFINE_integer("num_layers", 2, '')

flags.DEFINE_integer("epochs_to_train", 1000,
                     "Number of epochs to train. Each epoch processes the training data once "
                     "completely."
                     )
flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples",
                     512,
                     "Negative samples per training example."
                     )

flags.DEFINE_integer("predict_tail",
                     1,
                     "predict_tail=1, head=0"
                     )

FLAGS = flags.FLAGS


class Options(object):
    """Options used by our model."""

    def __init__(self):
        # Model options.

        # Embedding dimension.
        self.hidden_size = FLAGS.hidden_size

        self.keep_prob = FLAGS.keep_prob

        self.num_layers = FLAGS.num_layers

        self.data_path = FLAGS.data_path

        self.num_samples = FLAGS.num_neg_samples

        self.learning_rate = FLAGS.learning_rate

        self.epochs_to_train = FLAGS.epochs_to_train

        self.batch_size = FLAGS.batch_size

        self.predict_tail = FLAGS.predict_tail

        # Where to write out summaries.
        self.save_path = FLAGS.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


class Printer(object):
    '''
    basic printer
    '''

    def print_result(self, r, data, epoch, f=None):
        print(
            'epoch:%s data:%s rank_method:%s MR:%f H@10:%f FMR:%f FH@10:%f MRR:%f' %
            (epoch, data, r[0], r[1], r[2], r[3], r[4], r[5])
        )
        if f is not None:
            print(
                'epoch:%s data:%s rank_method:%s MR:%f H@10:%f FMR:%f FH@10:%f MRR:%f' %
                (epoch, data, r[0], r[1], r[2], r[3], r[4], r[5]),
                file=f
            )


class Model(object):
    '''RNNKM basic model'''

    def __init__(self, options: Options, session, init_tensor=True):
        self._options = options
        self._session = session

        super(Model, self).__init__(options, session)

    def init_variables(self):
        options = self._options
        size = options.hidden_size

        self._entity_embedding = tf.get_variable(
            'entity_embedding',
            [self._entity_num, size],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        self._relation_embedding = tf.get_variable(
            'relation_embedding',
            [self._relation_num, size],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        self._relation_softmax_w = tf.get_variable(
            "relation_softmax_w",
            [self._relation_num, self._options.hidden_size],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        self._relation_softmax_b = tf.get_variable(
            "relation_softmax_b",
            [self._relation_num],
            initializer=tf.constant_initializer(0)
        )
        self._entity_softmax_w = tf.get_variable(
            "entity_softmax_w",
            [self._entity_num, self._options.hidden_size],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        self._entity_softmax_b = tf.get_variable(
            "entity_softmax_b",
            [self._entity_num],
            initializer=tf.constant_initializer(0)
        )

        self._lr = tf.Variable(options.learning_rate, trainable=False)

        self.build_graph()
        self.build_eval_graph()

        self._last_mean_loss = 100000

    def lstm_cell(self, drop=True, keep_prob=0.5, num_layers=2, hidden_size=None):
        if not hidden_size:
            hidden_size = self._options.hidden_size

        def basic_lstm_cell():
            return tf.contrib.rnn.LSTMCell(
                hidden_size,
#                 initializer=tf.orthogonal_initializer,
                state_is_tuple=True,
                reuse=tf.get_variable_scope().reuse
            )

        def drop_cell():
            return tf.contrib.rnn.DropoutWrapper(
                basic_lstm_cell(),
                output_keep_prob=keep_prob
            )

        if drop:
            gen_cell = drop_cell
        else:
            gen_cell = basic_lstm_cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [gen_cell() for _ in range(num_layers)],
            state_is_tuple=True,
        )
        return cell

    def logits(self, input, predict_relation=True):

        if not predict_relation:
            w = self._entity_softmax_w
            b = self._entity_softmax_b
        else:
            w = self._relation_softmax_w
            b = self._relation_softmax_b
        return tf.nn.bias_add(tf.matmul(input, tf.transpose(w)), b)

    def softmax(self, logits):
        return tf.nn.softmax(logits)
        # return tf.nn.sigmoid(logits)

    def loss(self, logits, weights, labels):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return loss

    def sampled_loss(self, inputs, labels, predict_relation=True, weights=None, tail=False):
        if predict_relation:
            w, b, n = self._relation_softmax_w, self._relation_softmax_b, self._relation_num
        else:
            w, b, n = self._entity_softmax_w, self._entity_softmax_b, self._entity_num

        num_sampled = min(self._options.num_samples, w.shape[0])
        #sampled_softmax_loss
        losses = tf.nn.sampled_softmax_loss(
            weights=w,
            biases=b,
            labels=tf.reshape(labels, [-1, 1]),
            inputs=inputs,
            num_sampled=num_sampled,
            num_classes=n,
            partition_strategy='div'
        )

        if weights is not None:
            losses = losses * weights
        return tf.reduce_sum(losses)
        # return losses

    def get_optimizer(self):
        return tf.train.AdamOptimizer(self._lr)

    def input_to_rnn(self, cell, size, inputs, reuse=False, state=None):
        if state == None:
            state = cell.zero_state(size, tf.float32)
        outputs = []
        with tf.variable_scope('RNN'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            for i in range(len(inputs)):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                (output, state) = cell(inputs[i], state)
                outputs.append(output)
        return outputs, state



class Trainer(object):
    '''
    basic version
    '''

    def __init__(self, options: Options, session, init_tensor=True):
        data_path = self._options.data_path

        handled_path = data_path + 'basic_trainer_saved.pkl'

        if os.path.exists(handled_path):
            print('load file from local')
            (self._entity_num, self._relation_num, self._relation_num_for_eval, self._train_data, self._test_data,
             self._valid_data) = pickle.load(open(handled_path, 'rb'))
        else:
            self.read_data()
            self.merge_id()
            self.add_reverse()
            self.reindex_kb()
            self.gen_t_label()
            # self.merge_path()

            print('start save dfs')
            saved = (
                self._entity_num, self._relation_num, self._relation_num_for_eval, self._train_data, self._test_data,
                self._valid_data)
            pickle.dump(saved, open(handled_path, 'wb'))

        self.gen_filter_mat()

        self.init_variables()
        if init_tensor:
            tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

    def bn(self, inputs, is_train=True, reuse=True):
        return tf.contrib.layers.batch_norm(inputs,
                                            center=True,
                                            scale=True,
                                            is_training=is_train,
                                            reuse=reuse,
                                            scope='bn',
                                            )

    def full_conn(self, inputs, out_size, ac=None, bi=None, scope=None):
        return tf.contrib.layers.fully_connected(inputs,
                                                 out_size,
                                                 activation_fn=ac,
                                                 biases_initializer=bi,
                                                 scope=scope
                                                 )
    
    def sample(self, data):
        return data[np.random.choice(len(data), size=len(data), replace=False)]
    
    def padding_data(self, data):
        padding_num = self._options.batch_size - len(data) % self._options.batch_size
        data = np.concatenate([data, np.zeros((padding_num, data.shape[1]), dtype=np.int32)])
        return data, padding_num

    def train(self, t_labels=None, r_labels=None):
        opts = self._options
        data = self._train_data[['h_id', 'r_id', 't_id']].values
        data = self.sample(data)

        num_batch = len(data) // opts.batch_size
        print(opts.batch_size, num_batch, self._session.run(self._lr), self._last_mean_loss)

        fetches = {
            "loss": self._loss,
            "train_op": self._train_op
        }
        losses = 0.0
        for i in range(num_batch):
            one_batch_data = data[i * opts.batch_size: (i + 1) * opts.batch_size]
            if i > 0: tf.get_variable_scope().reuse_variables()

            feed_dict = {}
            feed_dict[self._e] = one_batch_data[:, 0]
            feed_dict[self._r] = one_batch_data[:, 1]
            feed_dict[self._label] = one_batch_data[:, 2]
            vals = self._session.run(fetches, feed_dict)

            del one_batch_data

            loss = vals["loss"]
            losses += loss
            print(i, loss, end='\r')
        self._last_mean_loss = losses / num_batch

        del data
        return self._last_mean_loss



class RespectiveTrainer(Trainer):
    def build_graph(self, use_bn=True):
        options = self._options
        size = options.batch_size
        hidden_size = options.hidden_size

        e = tf.placeholder(tf.int32, [size])
        r = tf.placeholder(tf.int32, [size])
        label = tf.placeholder(tf.int32, [size])

        e_embedding = tf.nn.embedding_lookup(self._entity_embedding, e)
        r_embedding = tf.nn.embedding_lookup(self._relation_embedding, r)

        if use_bn:
            with tf.variable_scope('input_bn'):
                e_embedding = self.bn(e_embedding, reuse=False)
                # with tf.variable_scope('input_bn_relation'):
                r_embedding = self.bn(r_embedding, reuse=True)

        with tf.variable_scope('rnn_entity', reuse=False):
            cell = self.lstm_cell(True, options.keep_prob, options.num_layers)
            relation_outputs, state = self.input_to_rnn(cell, size, [e_embedding, ])

        with tf.variable_scope('rnn_relation', reuse=False):
            cell = self.lstm_cell(True, options.keep_prob, options.num_layers)
            entity_outputs, state = self.input_to_rnn(cell, size, [r_embedding, ], state=state)

        relation_output = relation_outputs[-1]
        entity_output = entity_outputs[-1]

        if use_bn:
            with tf.variable_scope('output_bn'):
                relation_output = self.bn(relation_output, reuse=False)
                # with tf.variable_scope('output_bn_entity'):
                entity_output = self.bn(entity_output, reuse=True)

        relation_loss = self.sampled_loss(relation_output, r, predict_relation=True)
        entity_loss = self.sampled_loss(entity_output, label, predict_relation=False)

        r_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

        loss = (relation_loss + entity_loss) / size

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5.)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self._train_op = self.get_optimizer().apply_gradients(
                zip(grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step()
            )

        self._loss = loss
        self._e, self._r, self._label = e, r, label
        
        
        
class RespectiveTester(object):
    def build_eval_graph(self, use_bn=True):
        options = self._options
        batch_size = options.batch_size
        hidden_size = options.hidden_size

        e = tf.placeholder(tf.int32, [None], name='eval_entity')
        r = tf.placeholder(tf.int32, [None], name='eval_relation')

        e_embedding = tf.nn.embedding_lookup(self._entity_embedding, e)
        r_embedding = tf.nn.embedding_lookup(self._relation_embedding, r)

        if use_bn:
            with tf.variable_scope('input_bn'):
                e_embedding = self.bn(e_embedding, is_train=False, reuse=True)
                # with tf.variable_scope('input_bn_relation'):
                r_embedding = self.bn(r_embedding, is_train=False, reuse=True)

        with tf.variable_scope('rnn_entity', reuse=True):
            cell = self.lstm_cell(False, 1., options.num_layers)
            relation_outputs, state = self.input_to_rnn(cell, batch_size, [e_embedding, ])

        with tf.variable_scope('rnn_relation', reuse=True):
            cell = self.lstm_cell(False, 1., options.num_layers)
            entity_outputs, state = self.input_to_rnn(cell, batch_size, [r_embedding, ], reuse=True, state=state)

        relation_output = relation_outputs[-1]
        entity_output = entity_outputs[-1]

        if use_bn:
            with tf.variable_scope('output_bn'):
                relation_output = self.bn(relation_output, is_train=False, reuse=True)
                # with tf.variable_scope('output_bn_entity'):
                entity_output = self.bn(entity_output, is_train=False, reuse=True)

        relation_logits = self.logits(relation_output, predict_relation=True)
        entity_logits = self.logits(entity_output, predict_relation=False)

        relation_probs = self.softmax(relation_logits)
        entity_probs = self.softmax(entity_logits)

        self._eval_e, self._eval_r = e, r
        self._relation_probs, self._entity_probs = relation_probs, entity_probs




class FBRespective(Model, FreeBaseReader, RespectiveTrainer, RespectiveTester, Printer):
    pass



class WNRespective(Model, WordNetReader, RespectiveTrainer, RespectiveTester, Printer):
    pass