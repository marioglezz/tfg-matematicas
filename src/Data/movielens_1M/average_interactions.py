def main():
    """
    Script generado para calcular el promedio del número de valoraciones en el conjunto de evaluaci´n, usado en la memoria asociada al proyeto
    """
    file_path = "test.txt"
    total_interactions = 0
    user_count = 0

    try:
        with open(file_path, "r") as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue
                num_interactions = len(tokens) - 1
                total_interactions += num_interactions
                user_count += 1

        if user_count > 0:
            average = total_interactions / user_count
            print("Número total de usuarios:", user_count)
            print("Media de interacciones por usuario en el test set:", average)
        else:
            print("No se encontró ningún usuario en el archivo.")

    except FileNotFoundError:
        print(f"No se encontró el archivo {file_path}.")

if __name__ == "__main__":
    main()