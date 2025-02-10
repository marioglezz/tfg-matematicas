import tensorflow as tf
import os
import time
from tqdm import tqdm
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()

from src.utility.helper import *
from src.utility.batch_test import *

EVALUATION_FREQUENCY = 2

class NeuMF(object):
    def __init__(self, data_config):
        self.model_type = 'neumf'
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.lr = args.lr
        self.batch_size = args.batch_size
        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.gmf_emb_dim = args.embed_size
        self.mlp_layers = eval(args.layer_size)

        # Placeholders para entrenamiento
        self.users = tf.compat.v1.placeholder(tf.int32, shape=[None], name='users')
        self.pos_items = tf.compat.v1.placeholder(tf.int32, shape=[None], name='pos_items')
        self.ratings = tf.compat.v1.placeholder(tf.float32, shape=[None], name='ratings')

        # Inicialización de embeddings usando el inicializador Glorot
        initializer = tf.keras.initializers.GlorotUniform()

        # GMF
        self.gmf_user_embedding = tf.Variable(initializer([self.n_users, self.gmf_emb_dim]), name='gmf_user_embedding')
        self.gmf_item_embedding = tf.Variable(initializer([self.n_items, self.gmf_emb_dim]), name='gmf_item_embedding')

        # MLP 
        mlp_embedding_dim = self.mlp_layers[0]
        self.mlp_user_embedding = tf.Variable(initializer([self.n_users, mlp_embedding_dim]), name='mlp_user_embedding')
        self.mlp_item_embedding = tf.Variable(initializer([self.n_items, mlp_embedding_dim]), name='mlp_item_embedding')

        # GMF: se calcula la prediccion
        gmf_u = tf.nn.embedding_lookup(self.gmf_user_embedding, self.users)
        gmf_i = tf.nn.embedding_lookup(self.gmf_item_embedding, self.pos_items)
        gmf_vector = tf.multiply(gmf_u, gmf_i)

        # MLP: se obtiene la representación concatenada y se la pasa por las capas densas
        mlp_u = tf.nn.embedding_lookup(self.mlp_user_embedding, self.users)
        mlp_i = tf.nn.embedding_lookup(self.mlp_item_embedding, self.pos_items)
        mlp_vector = tf.concat([mlp_u, mlp_i], axis=1)

        # Construcción de las capas ocultas del MLP
        self.mlp_layers_list = []  # se guardan las capas para reutilizarlas en test
        vector = mlp_vector
        for idx in range(len(self.mlp_layers)):
            dense_layer = tf.keras.layers.Dense(
                units=self.mlp_layers[idx],
                activation=tf.nn.relu,
                kernel_regularizer=tf.keras.regularizers.l2(self.regs[0]),
                name=f"mlp_layer{idx}"
            )
            vector = dense_layer(vector)
            self.mlp_layers_list.append(dense_layer)
        mlp_output = vector  # salida final del MLP

        # Se concatenan las salidas de GMF y MLP
        neumf_vector = tf.concat([gmf_vector, mlp_output], axis=1)

        # Capa de salida
        self.pred_layer = tf.keras.layers.Dense(
            units=1,
            activation=None,
            name="prediction_logits"
        )
        logits = self.pred_layer(neumf_vector)
        self.logits = tf.squeeze(logits)
        self.prediction = tf.nn.sigmoid(self.logits)

        # Función de pérdida BCE
        self.loss = tf.reduce_mean(- (self.ratings * tf.math.log(self.prediction + 1e-8) + (1 - self.ratings) * tf.math.log(1 - self.prediction + 1e-8)))
        # Regularización L2 sobre los embeddings
        reg_loss = self.decay * (tf.nn.l2_loss(gmf_u) + tf.nn.l2_loss(gmf_i))
        reg_loss += self.decay * (tf.nn.l2_loss(mlp_u) + tf.nn.l2_loss(mlp_i))
        self.loss += reg_loss

        # Optimización con Adam
        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        # Dado un usuario, calcula la predicción para todos los ítems.
        def predict_for_user(u):

            # Parte GMF
            u_gmf = tf.nn.embedding_lookup(self.gmf_user_embedding, u)
            all_gmf = self.gmf_item_embedding
            gmf_vec = tf.multiply(tf.tile(tf.expand_dims(u_gmf, 0), [self.n_items, 1]), all_gmf)

            # Parte MLP
            u_mlp = tf.nn.embedding_lookup(self.mlp_user_embedding, u)
            all_mlp = self.mlp_item_embedding
            u_mlp_tile = tf.tile(tf.expand_dims(u_mlp, 0), [self.n_items, 1])
            mlp_vec = tf.concat([u_mlp_tile, all_mlp], axis=1)

            # Se pasan las salidas por las capas ocultas del MLP (reutilizando las mismas capas)
            for dense in self.mlp_layers_list:
                mlp_vec = dense(mlp_vec)

            # Se concatenan ambas partes
            final_vec = tf.concat([gmf_vec, mlp_vec], axis=1)
            logits = self.pred_layer(final_vec)
            preds = tf.nn.sigmoid(tf.squeeze(logits, axis=1))
            return preds

        # Dado un batch de usuarios de test, se calcula la predicción para cada uno, cada fila contendrá las predicciones para todos los ítems
        self.batch_ratings = tf.map_fn(predict_for_user, self.users, dtype=tf.float32)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    t0 = time()
    model = NeuMF(data_config=config)

    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=sess_config)
    sess.run(tf.compat.v1.global_variables_initializer())
    print('Se entrena el modelo NeuMF desde cero.')

    # Parámetros para early stopping
    patience = 3
    stopping_step = 0
    should_stop = False
    cur_best_pre_0 = 0.0

    # Listas para almacenar métricas en cada evaluación
    loss_loger, rec_loger, pre_loger, ndcg_loger, hit_loger = [], [], [], [], []
    Ks = eval(args.Ks)
    epochs = args.epoch

    for epoch in range(epochs):
        t1 = time()
        epoch_loss = 0.0
        n_batch = data_generator.n_train // args.batch_size + 1

        pbar = tqdm(range(n_batch), desc=f"Epoch {epoch+1}", ncols=100)
        for idx in pbar:
            # Se muestrean positivos y negativos
            users, pos_items, neg_items = data_generator.sample()
            combined_users = users + users
            combined_items = pos_items + neg_items
            combined_ratings = [1.0] * len(pos_items) + [0.0] * len(neg_items)

            feed_dict = {
                model.users: combined_users,
                model.pos_items: combined_items,
                model.ratings: combined_ratings
            }
            _, batch_loss = sess.run([model.opt, model.loss], feed_dict=feed_dict)
            epoch_loss += batch_loss

            avg_loss = epoch_loss / (idx + 1)
            elapsed = time() - t1
            avg_batch_time = elapsed / (idx + 1)
            remaining_batches = n_batch - (idx + 1)
            est_time_remaining = avg_batch_time * remaining_batches
            pbar.set_postfix({"avg_loss": f"{avg_loss:.5f}", "est_remain": f"{est_time_remaining:.2f}s"})

        loss_loger.append(epoch_loss)

        # Evaluación cada EVALUATION_FREQUENCY epochs
        if (epoch + 1) % EVALUATION_FREQUENCY == 0:
            users_to_test = list(data_generator.test_set.keys())
            ret = test(sess, model, users_to_test)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])
            print("Epoch {} - Ks = {}: recall = {}, precision = {}, ndcg = {}, hit ratio = {}"
                  .format(epoch+1, Ks, ret['recall'], ret['precision'], ret['ndcg'], ret['hit_ratio']))

            mid_K = len(Ks) // 2
            current_recall = ret['recall'][mid_K]
            cur_best_pre_0, stopping_step, should_stop = early_stopping(current_recall, cur_best_pre_0, stopping_step, expected_order='acc', flag_step=patience)
            if should_stop:
                print("Early stopping en epoch %d" % (epoch+1))
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
        f.write('embed_size=%d, lr=%.4f, layers=%s, regs=%.6f\n\t%s\n' % (args.embed_size, args.lr, model.mlp_layers, model.decay, final_perf))