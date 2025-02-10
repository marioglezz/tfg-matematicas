import os
import random

def process_ratings(ratings_file, train_output_file, test_output_file, train_ratio=0.8):
    """
    Procesa el archivo ratings.dat para MovieLens 1M para adapatarlo a la estructura definida en la implementación original de NGCF en
    Wang Xiang et al. Neural Graph Collaborative Filtering y genera:
      - Un diccionario de usuario -> [lista de id de películas] (todos 0-indexados)
      - Divide, para cada usuario, las películas en dos conjuntos: train_ratio para el entrenamiento y 1 - train_ratio para la evaluación.
      - Escribe el archivo train.txt con el formato: userID movieID1 movieID2 ... que muestra las interacciones entre usuarios e items
      - Escribe test.txt con el mismo formato.
    """
    user_movies = {}

    # Procesa ratings.dat que es donde se encuentran las valoraciones en el dataset MOvieLens-1M
    with open(ratings_file, 'r') as f:
        for line in f:
            # Formato: userID::movieID::rating::timestamp
            parts = line.strip().split("::")
            if len(parts) < 2:
                continue
            # Restar 1 a userID y movieID para pasarlos a 0-indexedados
            user_id = int(parts[0]) - 1
            movie_id = int(parts[1]) - 1
            if user_id not in user_movies:
                user_movies[user_id] = []
            user_movies[user_id].append(movie_id)

    # Dividir las películas de cada usuario en train y test
    train_data = {}
    test_data = {}
    for user, movies in user_movies.items():
        random.shuffle(movies)
        n = len(movies)
        n_train = int(n * train_ratio)
        # Asegurarse de que haya al menos 1 película en test para poder evaluar 
        # sino el usuario siempre dará 0 en las métricas a pesar de que el modelo prediga a la perfección sus gustos
        if n_train == n and n > 1:
            n_train = n - 1
        train_data[user] = movies[:n_train]
        test_data[user] = movies[n_train:]

    with open(train_output_file, 'w') as f:
        for user, movies in train_data.items():
            movies_str = " ".join(str(m) for m in movies)
            f.write(f"{user} {movies_str}\n")
    print(f"Archivo de entrenamiento escrito en: {train_output_file}")

    with open(test_output_file, 'w') as f:
        for user, movies in test_data.items():
            movies_str = " ".join(str(m) for m in movies)
            f.write(f"{user} {movies_str}\n")
    print(f"Archivo de prueba escrito en: {test_output_file}")

if __name__ == "__main__":
    ratings_file = "ratings.dat"
    train_output_file = "train.txt"
    test_output_file = "test.txt"
    
    process_ratings(ratings_file, train_output_file, test_output_file)
