import pandas as pd
import matplotlib.pyplot as plt


def plot_benchmark(
    benchmark: dict,
    metric: str = 'accuracy',
    as_percentage: bool = True
):
    df = pd.DataFrame.from_dict(benchmark, orient='index')
    for idx in df.index:
        model_stats = df.loc[idx]
        x = model_stats['latency']['mean']
        y = model_stats['metrics'][metric]
        if as_percentage:
            y *= 100
        s = model_stats['size']

        plt.scatter(x=x, y=y, s=s, label=idx, alpha=0.5)

    legend = plt.legend(bbox_to_anchor=(1, 1))
    for handle in legend.legendHandles:
        handle.set_sizes([20])
    plt.xlabel('Latency (ms)')
    plt.ylabel(f'{metric.title()} (%)')
    plt.show()


if __name__ == '__main__':
    import json

    benchmark = json.load(open('../benchmark_results.json'))
    plot_benchmark(benchmark)
