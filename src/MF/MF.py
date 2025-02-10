import tensorflow as tf
import os
import time
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()

from src.utility.helper import *
from src.utility.batch_test import *

EVALUATION_FREQUENCY = 2

class MF(object):
    def __init__(self, data_config, pretrain_data):
        self.model_type = 'mf'
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        
        self.emb_dim = args.embed_size
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.decay = eval(args.regs)[0]

        self.users = tf.compat.v1.placeholder(tf.int32, shape=[None], name='users')
        self.pos_items = tf.compat.v1.placeholder(tf.int32, shape=[None], name='pos_items')
        self.ratings = tf.compat.v1.placeholder(tf.float32, shape=[None], name='ratings') #valor real de si hay o no interacción
        
        # Inicialización de parámetros: embeddings y biases
        self.weights = self._init_weights()
        
        # Obtención de embeddings y biases
        self.u_embed = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)    
        self.pos_i_embed = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        
        self.u_bias = tf.nn.embedding_lookup(self.weights['user_bias'], self.users)      
        self.pos_i_bias = tf.nn.embedding_lookup(self.weights['item_bias'], self.pos_items)
        
        # Cálculo de la predicción para cada par (u, i) siguiendo el algoritmo de MF: pred = b_u + b_i + ⟨p_u, q_i⟩ sin μ  ya que al ser interacciones no aporta
        self.prediction = self.u_bias + self.pos_i_bias + tf.reduce_sum(tf.multiply(self.u_embed, self.pos_i_embed), axis=1)
        
        # Función de pérdida:  
        # Se calcula el error cuadrático entre la etiqueta real y la predicción, (r_{u,i} - pred)^2 y se le suma la regularización L2 sobre embeddings y biases.
        self.mse_loss = tf.reduce_mean(tf.square(self.ratings - self.prediction))
        
        self.reg_loss = self.decay * ( tf.nn.l2_loss(self.u_embed) + tf.nn.l2_loss(self.pos_i_embed) + tf.nn.l2_loss(self.u_bias) + tf.nn.l2_loss(self.pos_i_bias))
        self.loss = self.mse_loss + self.reg_loss
        
        # Optimización: se utiliza Adam para minimizar la pérdida total.
        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
       
        # Operación para evaluación
        test_u_embed  = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        test_u_bias   = tf.nn.embedding_lookup(self.weights['user_bias'], self.users)
        test_i_embed  = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        test_i_bias   = tf.nn.embedding_lookup(self.weights['item_bias'], self.pos_items)
        self.batch_ratings = tf.expand_dims(test_u_bias, 1) + tf.expand_dims(test_i_bias, 0) + tf.matmul(test_u_embed, test_i_embed, transpose_b=True)
    
    def _init_weights(self):

        all_weights = dict()
        initializer = tf.keras.initializers.GlorotUniform()
        
        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
        all_weights['user_bias'] = tf.Variable(tf.zeros([self.n_users]), name='user_bias')
        all_weights['item_bias'] = tf.Variable(tf.zeros([self.n_items]), name='item_bias')
        print('Usando inicialización aleatoria para MF.')
            
        return all_weights

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    
    t0 = time()
    pretrain_data = None
        
    # Crear el modelo MF
    model = MF(data_config=config, pretrain_data=pretrain_data)
    
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=sess_config)
    
    sess.run(tf.compat.v1.global_variables_initializer())
    print('Se entrena el modelo MF (MSE) desde cero.')
    
    # Entrenamiento
    patience = 5
    stopping_step = 0
    should_stop = False
    cur_best_pre_0 = 0.0

    loss_loger, rec_loger, pre_loger, ndcg_loger, hit_loger = [], [], [], [], []
    for epoch in range(args.epoch):
        t1 = time()
        epoch_loss = 0.0
        n_batch = data_generator.n_train // args.batch_size + 1
        
        pbar = tqdm(range(n_batch), desc=f"Epoch {epoch+1}", ncols=100)
        for idx in pbar:

            # Construimos el batch
            users, pos_items, neg_items = data_generator.sample()
            combined_users = users + users
            combined_items = pos_items + neg_items
            combined_ratings = [1.0] * len(pos_items) + [0.0] * len(neg_items) #ejemplos tantos positivos como negativos para el entrenamiento
            
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
        
        # Cada EVALUATION_FREQUENCY epochs se evalúa el modelo utilizando las métricas de ranking.
        if (epoch + 1) % EVALUATION_FREQUENCY== 0:
            users_to_test = list(data_generator.test_set.keys())

            # En test se evalúa puntando cada usuario contra todos los ítems, mismo marco de evaluacion que en los otros modelos
            ret = test(sess, model, users_to_test)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])
            print("Epoch {} - Ks = {}: recall = {}, precision = {}, ndcg = {}, hit ratio = {}"
                  .format(epoch+1, Ks, ret['recall'], ret['precision'], ret['ndcg'], ret['hit_ratio']))
            
            # Se utiliza el valor de recall en el índice medio de Ks para early stopping, se podría modificar pero es representativo recall@10
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
    
    final_perf = ("Mejores metricas en test: Ks = {}\n\t recall = {},\n\tprecision = {},\n\tndcg = {},\n\thit ratio = {}"
                  .format(Ks, recs[best_index], pres[best_index], ndcgs[best_index],hits[best_index]))
    print(final_perf)
    
    # Guardar resultados y parámetros de configuración
    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    with open(save_path, 'a') as f:
        f.write('embed_size=%d, lr=%.4f, regs=%.6f\n\t%s\n' % (args.embed_size, args.lr, model.decay, final_perf))
