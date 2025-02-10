import matplotlib.pyplot as plt
import re
import os
import argparse
import glob

plt.style.use('seaborn-v0_8-paper')

def parse_file(filepath):
    with open(file=filepath, mode='r') as f:
        content = f.read()
    
    raw_blocks = [block.strip() for block in content.split("\n\n") if block.strip()]
    blocks = []

    for block in raw_blocks:
        lines = block.splitlines()
        if not lines:
            continue
    
        context = lines[0].strip()
        ks = []
        metrics = {}
        
        for i, line in enumerate(lines):
            if "Ks" in line:
                ks_match = re.search(r'\[([^\]]+)\]', line)
                if ks_match:
                    try:
                        ks = [int(x.strip()) for x in ks_match.group(1).split(',')]
                    except Exception as e:
                        print("Error al parsear los valores de Ks:", e)

                metric_lines = lines[i+1:]
                for mline in metric_lines:
                    mline = mline.strip().rstrip(',')
                    if '=' in mline and '[' in mline:
                        key, rest = mline.split("=", 1)
                        key = key.strip().lower()
                        value_match = re.search(r'\[([^\]]+)\]', rest)
                        if value_match:
                            try:
                                values = [float(x.strip()) for x in value_match.group(1).split(',')]
                                metrics[key] = values
                            except Exception as e:
                                print(f"Error al parsear la métrica '{key}':", e)
                break

        blocks.append({"context": context, "ks": ks, "metrics": metrics})
    return blocks

def aggregate_models_from_folder(folder):
    aggregated_data = {}
    pattern = os.path.join(folder, "*_best.txt")
    files = glob.glob(pattern)
    for filepath in files:
        blocks = parse_file(filepath)
        base_name = os.path.basename(filepath)
        model_name = base_name.split("_")[0]
        if model_name in aggregated_data:
            aggregated_data[model_name].extend(blocks)
        else:
            aggregated_data[model_name] = blocks
    return aggregated_data

def plot_metric_all_models(aggregated_data, metric_name, output_dir="plots"):
    plt.figure(figsize=(8, 6))
    canonical_ks = None

    for model_name, blocks in aggregated_data.items():
        for idx, block in enumerate(blocks, start=1):
            ks = block.get("ks", [])
            metrics = block.get("metrics", {})
            if metric_name not in metrics:
                print(f"El modelo {model_name} (Bloque {idx}) no tiene la métrica '{metric_name}'.")
                continue
            values = metrics[metric_name]
            indices = list(range(len(ks)))
            plt.plot(indices, values, marker='o', linestyle='-', linewidth=1.5, markersize=6,
                     label=f"{model_name.upper()}")
            if canonical_ks is None and ks:
                canonical_ks = ks
    
    plt.xlabel("K", fontsize=12)
    plt.ylabel(metric_name.upper(), fontsize=12)
    plt.title(f"Comparativa de la métrica {metric_name.upper()}", fontsize=14)
    plt.legend()
    plt.grid(True)
    
    if canonical_ks:
        plt.xticks(range(len(canonical_ks)), canonical_ks)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    safe_metric = metric_name.replace(" ", "_")
    output_file = os.path.join(output_dir, f"all_models_{safe_metric}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Guardado gráfico comparativo: {output_file}")


def plot_all_metrics(aggregated_data, output_dir="plots"):
    all_metrics = set()
    for blocks in aggregated_data.values():
        for block in blocks:
            all_metrics.update(block.get("metrics", {}).keys())
    for metric in all_metrics:
        plot_metric_all_models(aggregated_data, metric, output_dir)


def main():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    aggregated_data = aggregate_models_from_folder(current_folder)
    
    if not aggregated_data:
        print("No se encontraron archivos *_best.txt en el directorio.")
        return

    plot_all_metrics(aggregated_data, output_dir="plots")

if __name__ == "__main__":
    main()
