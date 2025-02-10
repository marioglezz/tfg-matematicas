import tensorflow as tf
import os
import sys
import time
from tqdm import tqdm
import random as rd
import numpy as np

SEED = 42 
rd.seed(SEED)
np.random.seed(SEED)
tf.compat.v1.set_random_seed(SEED)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.compat.v1.disable_eager_execution()

from src.utility.helper import *
from src.utility.batch_test import *

class NGCF(object):
    def __init__(self, data_config, pretrain_data):
        self.model_type = 'ngcf'

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 100

        self.norm_adj = data_config['norm_adj']
        self.norm_adj_loops = data_config['norm_adj_loops']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.n_nonzero_elems_loops = self.norm_adj_loops.count_nonzero()

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.model_type += '_l%d' % (self.n_layers)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.verbose = args.verbose

        # definciion de vectores auxiliares para la entrada y el dropout
        self.users = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.compat.v1.placeholder(tf.int32, shape=(None,))

        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.compat.v1.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.compat.v1.placeholder(tf.float32, shape=[None])

        # Inicialización de los embeddings y de los pesos de la red, usando Xavier por defecto
        self.weights = self._init_weights()

        # Se aplica la arquitectura Neural Graph Collaborative Filtering para calcular los embeddings en base a los mecanismos de agregación y paso de mensajes
        self.u_embeddings, self.i_embeddings = self._create_ngcf_embed()

        
        self.u_g_embeddings = tf.nn.embedding_lookup(self.u_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.i_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.i_embeddings, self.neg_items)

        #Calculo de predicciones ustilizando el producto escalar entre los embeddings de los usuarios y los items
        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)

        #Calcular la pérdida aplicando la función BPR
        self.mf_loss, self.emb_loss = self.create_bpr_loss(self.u_g_embeddings, self.pos_i_g_embeddings, self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss

        #Optimizador Adam para el entrenamiento del modelo
        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.keras.initializers.GlorotUniform()

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            all_weights['W_1_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_1_%d' % k)
            all_weights['b_1_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_1_%d' % k)

            all_weights['W_2_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_2_%d' % k)
            all_weights['b_2_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_2_%d' % k)
            
        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []

        #Divide la matriz de adyacencia normalizada en n_folds partes, facilitando trabajar con matrices grandes dispersas
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        #Divide la matriz de adyacencia normalizada en n_folds partes
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            # Convierte las filas correspondientes al fold en un tensor para poder trabajar con TensorFlow
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp)) #aplica el dropout

        return A_fold_hat

    def _create_ngcf_embed(self):
        if self.node_dropout_flag:
            # node dropout
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
            A_I_fold_hat = self._split_A_hat_node_dropout(self.norm_adj_loops)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)
            A_I_fold_hat = self._split_A_hat(self.norm_adj_loops)

        #Estabalece la matriz E concatenando los embeddings de usuarios e ítems
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        #Guarda los embeddings de las distintas capas como lista de listas
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed_1, temp_embed_2 = [], []

            # Se divide A en en folds para facilitar el procesamiento y se multiplica cada entrada siguiendo la ecuacion matricial de la GNN
            for f in range(self.n_fold):
                #(Ã+I)*E^(k-1)      Ã*E^(k-1)
                temp_embed_1.append(tf.sparse.sparse_dense_matmul(A_I_fold_hat[f], ego_embeddings))
                temp_embed_2.append(tf.sparse.sparse_dense_matmul(A_fold_hat[f], ego_embeddings))

            # Se concatenan los subvectores de salida de la multiplicación para recuperar el tamaño original
            side_embeddings_1 = tf.concat(temp_embed_1, 0)
            side_embeddings_2 = tf.concat(temp_embed_2, 0)
            # (Ã+I)*E^(k-1)*W_1^(k) + B_1^(k)
            sum_embeddings = tf.matmul(side_embeddings_1, self.weights['W_1_%d' % k]) + self.weights['b_1_%d' % k]

            # E^(k-1)W_2^(k)
            embeddings_w2 = tf.matmul(ego_embeddings, self.weights['W_2_%d' % k])

            # Ã*E^(k-1) \cdot E^(k-1)W_2^(k) + B_2^(k)
            sum_embeddings_2 = tf.multiply(side_embeddings_2, embeddings_w2) + self.weights['b_2_%d' % k]

            # LeakyReLU((Ã+I)*E^(k-1)*W_1^(k) + B_1^(k) + Ã*E^(k-1) \cdot E^(k-1)W_2^(k) + B_2^(k))
            ego_embeddings = tf.nn.leaky_relu(sum_embeddings + sum_embeddings_2)

            # Message dropout
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # Normalizamos los vectores de embeddings
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1) #Ultima capa donde se agregan los embeddings de todas las capas para cada vertice
        u_embeddings, i_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0) #dividimos por usuarios e items
        return u_embeddings, i_embeddings


    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1) #suma predicciones positivas
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1) #suma predicciones negativas

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items) #normalizan los vectores
        regularizer = regularizer/self.batch_size #tamaño del regularizador promedido por el batch
        
        # Calculo de la BPR promedio del batch
        maxi = tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores))
        mf_loss = tf.negative(tf.reduce_mean(maxi))

        emb_loss = self.decay * regularizer #Termino de regularización correspondiente a la función de pérdida \lambda ||O||_2

        return mf_loss, emb_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.asmatrix([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        #Aplica la tecnica de dropout para tensores dispersos

        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random.uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse.retain(X, dropout_mask)

        return pre_out * tf.divide(1., keep_prob)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    #Generamos las matrices de adyacencia normalizadas y la matriz de adyacencia normalizada con lazos que se usaran en la GNN
    norm_adj, norm_adj_loops = data_generator.get_adj_mat()
    config['norm_adj'] = norm_adj
    config['norm_adj_loops'] = norm_adj_loops
    t0 = time()
    pretrain_data = None

    model = NGCF(data_config=config, pretrain_data=pretrain_data)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    
    sess.run(tf.compat.v1.global_variables_initializer())
    cur_best_pre_0 = 0.
    print("-------------------------------------------------------")
    print('Pesos y embeddings inicializados')

    #Entrenamiento
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        pbar = tqdm(range(n_batch), desc=f"Epoch {epoch}", ncols=100)
        for idx in pbar:
            users, pos_items, neg_items = data_generator.sample()
            _, batch_loss, batch_mf_loss, batch_emb_loss = sess.run(
                [model.opt, model.loss, model.mf_loss, model.emb_loss],
                feed_dict={
                    model.users: users,
                    model.pos_items: pos_items,
                    model.neg_items: neg_items,
                    model.node_dropout: eval(args.node_dropout),
                    model.mess_dropout: eval(args.mess_dropout)
                }
            )
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

            avg_loss = loss / (idx + 1)
            elapsed = time() - t1
            avg_batch_time = elapsed / (idx + 1)
            remaining_batches = n_batch - (idx + 1)
            est_time_remaining = avg_batch_time * remaining_batches

            pbar.set_postfix({
                "avg_loss": f"{float(avg_loss):.5f}",
                "est_remain": f"{float(est_time_remaining):.2f}s"
            })

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # Se evalua cada 10 epocas de entrenamiento del modelo
        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train_loss==[%.5f=%.5f + %.5f]' % (epoch, time() - t1, float(loss), float(mf_loss), float(emb_loss))
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=True)
        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            print("Epoch {} - Ks = {}: recall = {}, precision = {}, ndcg = {}, hit ratio = {}"
                  .format(epoch+1, Ks, ret['recall'], ret['precision'], ret['ndcg'], ret['hit_ratio']))

        mid_K = len(Ks) // 2
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][mid_K], cur_best_pre_0, stopping_step, expected_order='acc', flag_step=5)
        if should_stop == True:
            break

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hits = np.array(hit_loger)
    best_rec_0 = max(recs[:, 0]) #mejor iteración según el recall
    best_index = list(recs[:, 0]).index(best_rec_0)
    
    final_perf = ("Mejores metricas en test: Ks = {}\n\trecall = {},\n\tprecision = {},\n\tndcg = {},\n\thit ratio = {}"
                  .format(Ks, recs[best_index], pres[best_index], ndcgs[best_index],hits[best_index]))
    print(final_perf)
    
    # Guardar resultados y parámetros de configuración
    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    with open(save_path, 'a') as f:
        f.write('embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s\n\t%s\n'
            % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs, final_perf))
