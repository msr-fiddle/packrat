from argparse import ArgumentParser
from enum import Enum
import sys
import urllib
import pandas as pd
from plotnine.themes import theme_gray, theme
from plotnine.themes.elements import (element_line, element_rect,
                                      element_text, element_blank)
from plotnine import ggplot
from plotnine.labels import ggtitle
from plotnine.mapping import aes
from plotnine.labels import labs
from plotnine.scales import scale_x_continuous, scale_y_continuous, scale_color_discrete, scale_x_log10
from plotnine.geoms import geom_point, geom_line, geom_text
from io import BytesIO


TORCHSERVE_TAG = "torchserve-opt"
PYTORCH_TAG = "large-batches-tcmalloc"


def get_remote_data(type: str, model: str, tag: str):
    if tag.__contains__("torchserve"):
        filename = "result.csv"
    else:
        filename = f"{model}_{type}.csv"
    url: str = f"https://msr-fiddle.github.io/naf/{model}/cloudlab/{tag}/{filename}"
    print("Fetch {}".format(url))
    response = urllib.request.urlopen(url)
    buf = BytesIO(response.read())
    return pd.read_csv(buf)


class MyTheme(theme_gray):
    "Set the theme for the plot"

    def __init__(self, base_size=11, base_family='DejaVu Sans'):
        theme_gray.__init__(self, base_size, base_family)
        bgcolor = '#FFFFFF'
        self.add_theme(
            theme(
                axis_text=element_text(size=base_size + 3),
                axis_ticks=element_blank(),
                title=element_text(color='#3C3C3C'),
                legend_background=element_rect(fill='None'),
                legend_key=element_rect(fill='#FFFFFF', colour=None),
                panel_background=element_rect(fill=bgcolor),
                panel_border=element_blank(),
                panel_grid_major=element_line(
                    color='#D5D5D5', linetype='solid', size=1),
                panel_grid_minor=element_blank(),
                plot_background=element_rect(
                    fill=bgcolor, color=bgcolor, size=1),
                strip_background=element_rect(size=0)),
            inplace=True)


def scaling_plot_per_batch(benchmarks, yaxis: str, label: str):
    "Plots a latency graph for every data-structure in the results file"

    ylabel = ""
    if yaxis.__contains__('latency'):
        ylabel = "Latency per batch (ms)"
    elif yaxis.__contains__('throughput'):
        ylabel = "Throughput (Inferences/sec)"

    for batch_size in benchmarks['batch_size'].unique():
        data_set = []
        for model in benchmarks.model.unique():
            opt = "Script"
            bench = model
            benchmark = benchmarks.loc[(benchmarks['model'] == model) &
                                       (benchmarks['batch_size'] == batch_size)]
            benchmark.insert(0, 'config', "opt={}".format(opt))
            benchmark.insert(0, 'change', benchmark[yaxis].astype(int))
            data_set.append(benchmark)
        data = pd.concat(data_set)

        plot = ggplot(data=data, mapping=aes(x='cores', y=yaxis, color='config', ymin=0)) + \
            MyTheme(base_size=10) + labs(y=f"{ylabel}") + \
            theme(legend_position='top', legend_title=element_blank(), legend_box_spacing=0.0) + \
            ggtitle("Threads vs {} (BS={})".format(label, batch_size)) + \
            scale_x_continuous(
                breaks=data['cores'].unique(), labels=["{}".format(thr) for thr in data['cores'].unique()], name='# Threads') + \
            scale_y_continuous(labels=lambda lst: ["{:.1f}".format(x) for x in lst]) + \
            scale_color_discrete(breaks=data['config'].unique()) + \
            geom_point() + \
            geom_text(aes(label='change'), size=13, ha="left", va="bottom") + \
            geom_line()

        plot.save("{}-{}-{}.png".format(bench, label.lower(), batch_size), dpi=300,
                  width=12, height=5, verbose=False)


def scaling_plot_per_thread(benchmarks, yaxis: str, label: str):
    "Plots a latency or throughput graph for every data-structure in the results file"

    ylabel = ""
    if yaxis.__contains__('latency'):
        ylabel = "Latency per batch (ms)"
    elif yaxis.__contains__('throughput'):
        ylabel = "Throughput (Inferences/sec)"

    for model in benchmarks.model.unique():
        data_set = []
        for cores in benchmarks['cores'].unique():
            if cores % 2 != 0 and cores != 1:
                continue
            opt = "Script"
            bench = model
            benchmark = benchmarks.loc[(benchmarks['model'] == model) &
                                       (benchmarks['cores'] == cores)]
            benchmark.insert(0, 'config', "Threads={}".format(cores))
            benchmark.insert(0, 'change', benchmark[yaxis].astype(int))
            data_set.append(benchmark)
        data = pd.concat(data_set)

        plot = ggplot(data=data, mapping=aes(x='batch_size', y=yaxis, color='config')) + \
            MyTheme(base_size=10) + labs(y=f"{ylabel}") + \
            theme(legend_position='top', legend_title=element_blank()) + \
            ggtitle("Batch-Size vs {} (Opt={})".format(label, opt.upper())) + \
            scale_x_log10(
            breaks=data['batch_size'].unique(), labels=["{}".format(thr) for thr in data['batch_size'].unique()], name='Batch Size') + \
            scale_y_continuous(labels=lambda lst: ["{:.1f}".format(x) for x in lst]) + \
            scale_color_discrete(breaks=data['config'].unique()) + \
            geom_point() + \
            geom_line()

        plot.save("{}-{}-{}-perBS.png".format(bench, label.lower(), opt.lower()), dpi=300,
                  width=12, height=5, verbose=False)


def scaling_per_batch_comparison(yaxis: str, label: str):
    "Plots a latency graph for every data-structure in the results file"

    ylabel = ""
    type = ""
    if yaxis.__contains__('latency'):
        ylabel = "Latency per batch (ms)"
        type = Plots.latency.name
    elif yaxis.__contains__('throughput'):
        ylabel = "Throughput (Inferences/sec)"
        type = Plots.throughput.name

    df_pytorch = get_remote_data(type, "resnet", PYTORCH_TAG)
    df_pytorch = df_pytorch.rename(columns={"intraop_threads": "cores"})
    df_torchserve = get_remote_data(type, "resnet", TORCHSERVE_TAG)

    # Make the column names uniform across dataframes
    if yaxis.__contains__('latency'):
        df_pytorch = df_pytorch.rename(columns={"latency(avg)": "latency"})
        df_torchserve = df_torchserve.rename(columns={yaxis: "latency"})
        yaxis = "latency"

    for batch_size in df_torchserve['batch_size'].unique():
        data_set = []
        for model in df_torchserve.model.unique():
            bench = model
            benchmark_ts = df_torchserve.loc[(df_torchserve['model'] == model) &
                                             (df_torchserve['batch_size'] == batch_size)]
            benchmark_ts.insert(0, 'config', "opt=Script, Backend=TorchServe")
            benchmark_ts.insert(0, 'change', df_torchserve[yaxis].astype(int))
            data_set.append(benchmark_ts)

        for name in df_pytorch.benchmark.unique():
            bench = name.split('_')[0]
            benchmark_py = df_pytorch.loc[(df_pytorch['benchmark'] == name) &
                                          (df_pytorch['batch_size'] == batch_size) &
                                          (df_pytorch['topology'] == "sequential")]
            benchmark_py.insert(0, 'config', "opt=Script, Backend=PyTorch")
            benchmark_py.insert(0, 'change', benchmark_py[yaxis].astype(int))
            data_set.append(benchmark_py)

        data = pd.concat(data_set)

        plot = ggplot(data=data, mapping=aes(x='cores', y=yaxis, color='config', ymin=0)) + \
            MyTheme(base_size=10) + labs(y=f"{ylabel}") + \
            theme(legend_position='top', legend_title=element_blank(), legend_box_spacing=0.0) + \
            ggtitle("Threads vs {} (BS={})".format(label, batch_size)) + \
            scale_x_continuous(
                breaks=data['cores'].unique(), labels=["{}".format(thr) for thr in data['cores'].unique()], name='# Threads') + \
            scale_y_continuous(labels=lambda lst: ["{:.1f}".format(x) for x in lst]) + \
            scale_color_discrete(breaks=data['config'].unique()) + \
            geom_point() + \
            geom_text(aes(label='change'), size=13, ha="left", va="bottom") + \
            geom_line()

        print(
            f"Plotting for Model={bench}, Type={label.lower()}, BS={batch_size}")
        plot.save("{}-{}-comparison-{}.png".format(bench, label.lower(), batch_size), dpi=300,
                  width=12, height=5, verbose=False)


class Plots(Enum):
    latency = 1
    throughput = 2
    compare = 3


def args_parser():
    args = ArgumentParser(description="Run the plot script")
    args.add_argument("-p", "--plot", type=str, default=Plots.latency.name,
                      help="Script supports the following plot types {}".format([plot.name for plot in Plots]))
    args.add_argument("-f", "--file", type=str, required=False,
                      help="Path to the date file.")
    arguments = args.parse_args()

    if (arguments.plot != Plots.compare.name) and len(sys.argv) < 2:
        args.print_help()
        sys.exit(1)
    return arguments


if __name__ == '__main__':
    args = args_parser()

    if args.plot == Plots.latency.name:
        DF = pd.read_csv(args.file, skip_blank_lines=True)
        scaling_plot_per_batch(DF, "mean_predict_latency", "Latency")
    elif args.plot == Plots.throughput.name:
        DF = pd.read_csv(args.file, skip_blank_lines=True)
        scaling_plot_per_batch(DF, "throughput", "Throughput")
        scaling_plot_per_thread(DF, "throughput", "Throughput")
    elif args.plot == Plots.compare.name:
        scaling_per_batch_comparison("throughput", "Throughput")
        scaling_per_batch_comparison("mean_predict_latency", "Latency")
    else:
        print("Unknown plot type")
        sys.exit(1)
