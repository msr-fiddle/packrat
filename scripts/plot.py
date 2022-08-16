"""
Script that plots benchmark data-visualizations.
"""

from argparse import ArgumentParser
from enum import Enum
import sys
import pandas as pd
from plotnine.themes import theme_gray, theme
from plotnine.themes.elements import (element_line, element_rect,
                                      element_text, element_blank)
from plotnine.stats import stat_function
from plotnine import ggplot
from plotnine.labels import ggtitle
from plotnine.mapping import aes
from plotnine.labels import labs
from plotnine.scales import scale_x_continuous, scale_y_continuous, scale_color_discrete, scale_fill_gradientn, scale_x_log10, scale_y_log10
from plotnine.geoms import geom_point, geom_line, geom_tile, geom_text


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

    if yaxis not in ['latency(avg)', 'throughput', 'flops']:
        print("Invalid y-axis: {}".format(yaxis))
        sys.exit(1)

    ylabel = ""
    if yaxis == 'latency(avg)':
        ylabel = "Latency per batch (ms)"
    elif yaxis == 'throughput':
        ylabel = "Throughput (Inferences/sec)"
    elif yaxis == 'flops':
        ylabel = "GFLOPs/sec"

    for batch_size in benchmarks['batch_size'].unique():
        data_set = []
        for name in benchmarks.benchmark.unique():
            opt = name.split('_')[2].upper()
            bench = name.split('_')[0]
            benchmark = benchmarks.loc[(benchmarks['benchmark'] == name) &
                                       (benchmarks['batch_size'] == batch_size) &
                                       (benchmarks['topology'] == "sequential")]
            benchmark.insert(0, 'config', "opt={}".format(opt))
            benchmark['change'] = benchmark[yaxis].astype(int)
            data_set.append(benchmark)
        data = pd.concat(data_set)

        plot = ggplot(data=data, mapping=aes(x='intraop_threads', y=yaxis, color='config')) + \
            MyTheme(base_size=10) + labs(y=f"{ylabel}") + \
            theme(legend_position='top', legend_title=element_blank(), legend_box_spacing=0.0) + \
            ggtitle("Threads vs {} (BS={})".format(label, batch_size)) + \
            scale_x_continuous(
                breaks=data['intraop_threads'].unique(), labels=["{}".format(thr) for thr in data['intraop_threads'].unique()], name='# threads') + \
            scale_y_continuous(labels=lambda lst: ["{:.1f}".format(x) for x in lst]) + \
            scale_color_discrete(breaks=data['config'].unique()) + \
            geom_point() + \
            geom_text(aes(label='change'), size=13, ha="left", va="bottom") + \
            geom_line()

        plot.save("{}-{}-{}.png".format(bench, label.lower(), batch_size), dpi=300,
                  width=12, height=5, verbose=False)


def scaling_plot_per_opt(benchmarks, yaxis: str, label: str):
    "Plots a latency graph for every data-structure in the results file"

    if yaxis not in ['latency(avg)', 'throughput', 'flops']:
        print("Invalid y-axis: {}".format(yaxis))
        sys.exit(1)

    ylabel = ""
    if yaxis == 'latency(avg)':
        ylabel = "Latency per batch (ms)"
    elif yaxis == 'throughput':
        ylabel = "Throughput (Inferences/sec)"
    elif yaxis == 'flops':
        ylabel = "GFLOPs/sec"

    for name in benchmarks.benchmark.unique():
        data_set = []
        for batch_size in benchmarks['batch_size'].unique():
            opt = name.split('_')[2]
            bench = name.split('_')[0]
            benchmark = benchmarks.loc[(benchmarks['benchmark'] == name) &
                                       (benchmarks['batch_size'] == batch_size) &
                                       (benchmarks['topology'] == "sequential")]
            benchmark.insert(0, 'config', "Batch={}".format(batch_size))
            benchmark['change'] = benchmark[yaxis].astype(int)
            data_set.append(benchmark)
        data = pd.concat(data_set)

        plot = ggplot(data=data, mapping=aes(x='intraop_threads', y=yaxis, color='config')) + \
            MyTheme(base_size=10) + labs(y=f"{ylabel}") + \
            theme(legend_position='top', legend_title=element_blank(), legend_box_spacing=0.0) + \
            ggtitle("Threads vs {} (Opt={})".format(label, opt.upper())) + \
            scale_x_continuous(
                breaks=data['intraop_threads'].unique(), labels=["{}".format(thr) for thr in data['intraop_threads'].unique()], name='# threads') + \
            scale_y_continuous(labels=lambda lst: ["{:.1f}".format(x) for x in lst]) + \
            scale_color_discrete(breaks=data['config'].unique()) + \
            geom_point() + \
            geom_line()

        plot.save("{}-{}-{}.png".format(bench, label.lower(), opt), dpi=300,
                  width=12, height=5, verbose=False)


def scaling_plot_per_thread(benchmarks, yaxis: str, label: str):
    "Plots a latency or throughput graph for every data-structure in the results file"

    if yaxis not in ['latency(avg)', 'throughput']:
        print("Invalid y-axis: {}".format(yaxis))
        sys.exit(1)

    ylabel = ""
    if yaxis == 'latency(avg)':
        ylabel = "Latency per batch (ms)"
    elif yaxis == 'throughput':
        ylabel = "Throughput (Inferences/sec)"

    for name in benchmarks.benchmark.unique():
        data_set = []
        for intraop_threads in benchmarks['intraop_threads'].unique():
            if intraop_threads % 2 != 0 and intraop_threads != 1:
                continue
            opt = name.split('_')[2].upper()
            bench = name.split('_')[0]
            benchmark = benchmarks.loc[(benchmarks['benchmark'] == name) &
                                       (benchmarks['intraop_threads'] == intraop_threads) &
                                       (benchmarks['topology'] == "sequential")]
            benchmark.insert(0, 'config', "Threads={}".format(intraop_threads))
            benchmark['change'] = benchmark[yaxis].astype(int)
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


def plot_scaling_heatmap(benchmarks, yaxis: str, label: str):
    "Plots a latency graph for every data-structure in the results file"

    if yaxis not in ['latency(avg)', 'throughput']:
        print("Invalid y-axis: {}".format(yaxis))
        sys.exit(1)

    for batch_size in benchmarks['batch_size'].unique():
        benchmark = benchmarks.loc[benchmarks['batch_size'] == batch_size]
        benchmark[yaxis] = benchmark[yaxis].astype(int)

        p = ggplot(data=benchmark,
                   mapping=aes(x='intraop_threads', y='interop_threads', fill=yaxis)) + \
            labs(x="#Intraop Threads") + \
            labs(y="#Interop Threads") + \
            theme(legend_position="right", legend_title=element_blank()) +\
            scale_x_continuous(breaks=benchmark['intraop_threads'].unique(), labels=["{}".format(thr) for thr in benchmark['intraop_threads'].unique()]) + \
            scale_y_continuous(breaks=benchmark['interop_threads'].unique(), labels=["{}".format(thr) for thr in benchmark['interop_threads'].unique()]) + \
            geom_tile(aes(width=.95, height=.95)) + \
            ggtitle("{} for BS = {}".format(label, batch_size)) + \
            geom_text(aes(label=yaxis), size=7, color='black') + \
            theme(axis_text_x=element_text(size=10),
                  axis_text_y=element_text(size=10),
                  panel_background=element_rect(fill="white"),
                  panel_grid_major=element_line(colour="white"),
                  strip_background=element_rect(colour="orange", fill="orange")) + \
            scale_fill_gradientn(colors=[
                "#0080FF", "#00FFFF", "#00FF80", "#FFFF00", "#FF8000", "#FF0000", "#800000"])

        p.save('{}_heatmap_{}.png'.format(label.lower(), batch_size), dpi=300)


def plot_throughput_scaleup(benchmarks):
    "plots a throughput graph for every data-structure in the results file"

    for name in benchmarks.benchmark.unique():
        data_set = []
        for batch_size in benchmarks['batch_size'].unique():
            opt = name.split('_')[2]
            bench = name.split('_')[0]
            benchmark = benchmarks.loc[(benchmarks['benchmark'] == name) &
                                       (benchmarks['batch_size'] == batch_size) &
                                       (benchmarks['topology'] == "sequential")]
            benchmark.insert(0, 'config', "Batch={}".format(batch_size))
            benchmark['speedup'] = (
                benchmark['throughput'] / benchmark['throughput'].iloc[0]).round(1)
            data_set.append(benchmark)
        data = pd.concat(data_set)

        plot = ggplot(data=data, mapping=aes(x='intraop_threads', y='speedup', color='config')) + \
            MyTheme(base_size=10) + \
            labs(y="Speedup") + \
            theme(legend_position='top', legend_title=element_blank(), legend_box_spacing=0.0) + \
            ggtitle("Throughput Speedup (Opt={})".format(opt.upper())) + \
            scale_x_continuous(
                breaks=data['intraop_threads'].unique(), labels=["{}".format(thr) for thr in data['intraop_threads'].unique()], name='# threads') + \
            scale_y_continuous(labels=lambda lst: ["{:,}".format(x) for x in lst]) + \
            scale_color_discrete(breaks=data['config'].unique()) + \
            geom_point() + \
            geom_line() + \
            stat_function(geom="line", color='black',
                          size=1, fun=lambda x: x.mean())

        plot.save("{}-throughput-speedup-{}.png".format(bench, opt), dpi=300,
                  width=12, height=5, verbose=False)


def plot_sequential_and_interleaved_comparison_per_batch(benchmarks, yaxis: str, label: str):
    "Plotting sequential and interleaved comparison per batch"

    if yaxis not in ['latency(avg)', 'throughput']:
        print("Invalid y-axis: {}".format(yaxis))
        sys.exit(1)

    ylabel = ""
    if yaxis == 'latency(avg)':
        ylabel = "Latency per batch (ms)"
    elif yaxis == 'throughput':
        ylabel = "Throughput (Inferences/sec)"

    for batch_size in benchmarks['batch_size'].unique():
        data_set = []
        for name in benchmarks.benchmark.unique():
            for topo in benchmarks['topology'].unique():
                opt = name.split('_')[2].upper()
                if opt != "SCRIPT":
                    continue
                bench = name.split('_')[0]
                benchmark = benchmarks.loc[(benchmarks['benchmark'] == name) &
                                           (benchmarks['batch_size'] == batch_size) &
                                           (benchmarks['topology'] == topo)]
                benchmark.insert(
                    0, 'config', "opt={},topo={}".format(opt, topo))
                benchmark['change'] = benchmark[yaxis].astype(int)
                data_set.append(benchmark)
        data = pd.concat(data_set)

        plot = ggplot(data=data, mapping=aes(x='intraop_threads', y=yaxis, color='config')) + \
            MyTheme(base_size=10) + labs(y=f"{ylabel}") + \
            theme(legend_position='top', legend_title=element_blank(), legend_box_spacing=0.0) + \
            ggtitle("Threads vs {} (BS={})".format(label, batch_size)) + \
            scale_x_continuous(
                breaks=data['intraop_threads'].unique(), labels=["{}".format(thr) for thr in data['intraop_threads'].unique()], name='# threads') + \
            scale_y_continuous(labels=lambda lst: ["{:.1f}".format(x) for x in lst]) + \
            scale_color_discrete(breaks=data['config'].unique()) + \
            geom_point() + \
            geom_text(aes(label='change'), size=13, ha="left", va="bottom") + \
            geom_line()

        plot.save("{}-{}-numa-{}.png".format(bench, label.lower(), batch_size), dpi=300,
                  width=12, height=5, verbose=False)


def plot_sequential_and_interleaved_ratio_per_batch(benchmarks, yaxis: str, label: str):
    "Plotting sequential and interleaved comparison per batch"

    if yaxis not in ['latency(avg)', 'throughput']:
        print("Invalid y-axis: {}".format(yaxis))
        sys.exit(1)

    ylabel = ""
    if yaxis == 'latency(avg)':
        ylabel = "% Latency Loss"
    elif yaxis == 'throughput':
        ylabel = "% Throughput Loss"

    topologies = benchmarks['topology'].unique()
    if len(topologies) != 2:
        print("Not enough topologies to compare")
        return

    for batch_size in benchmarks['batch_size'].unique():
        data_set = []
        for threads in benchmarks['intraop_threads'].unique():
            for name in benchmarks.benchmark.unique():
                opt = name.split('_')[2].upper()
                if opt != "SCRIPT":
                    continue
                bench = name.split('_')[0]
                benchmark = benchmarks.loc[(benchmarks['benchmark'] == name) &
                                           (benchmarks['batch_size'] == batch_size) &
                                           (benchmarks['intraop_threads'] == threads)]

                topology_sequential = benchmark.loc[benchmark['topology']
                                                    == 'sequential'][yaxis].iloc[0]
                topology_interleaved = benchmark.loc[benchmark['topology']
                                                     == 'interleave'][yaxis].iloc[0]
                if yaxis == 'latency(avg)':
                    loss = int(((topology_interleaved - topology_sequential) /
                                topology_sequential) * 100)
                elif yaxis == 'throughput':
                    loss = int(((topology_sequential - topology_interleaved) /
                                topology_sequential) * 100)
                new_row = pd.DataFrame.from_records([{
                    "batch_size": int(batch_size),
                    "intraop_threads": int(threads),
                    "sequential": topology_sequential,
                    "interleave": topology_interleaved,
                    "loss": loss
                }])
                data_set.append(new_row)

        data = pd.concat(data_set)

        plot = ggplot(data=data, mapping=aes(x='intraop_threads', y='loss')) + \
            MyTheme(base_size=10) + labs(y=f"{ylabel}") + \
            theme(legend_position='top', legend_title=element_blank(), legend_box_spacing=0.0) + \
            ggtitle("Threads vs {} (BS={})".format(label, batch_size)) + \
            scale_x_continuous(
                breaks=data['intraop_threads'].unique(), labels=["{}".format(thr) for thr in data['intraop_threads'].unique()], name='# threads') + \
            scale_y_continuous(labels=lambda lst: ["{:.1f}".format(x) for x in lst]) + \
            geom_point() + \
            geom_text(aes(label='loss'), size=13, ha="left", va="bottom") + \
            geom_line()

        plot.save("{}-{}-ratio-{}.png".format(bench, label.lower(), batch_size), dpi=300,
                  width=12, height=5, verbose=False)


def plot_multiinstance_comparison_per_batch(benchmarks, yaxis: str, label: str):
    maxthreads = 16
    instances = [1, 2, 4, 8]
    bench = benchmarks['benchmark'].iloc[0].split('_')[0]

    ylabel = ""
    if yaxis == 'latency(avg)':
        ylabel = "Latency per batch (ms)"
    elif yaxis == 'throughput':
        ylabel = "Throughput (Inferences/sec)"

    data = pd.DataFrame()
    for batch_size in benchmarks['batch_size'].unique():
        if batch_size < 8:
            continue

        for instance in instances:
            t = maxthreads / instance
            b = batch_size / instance
            point = benchmarks.loc[(benchmarks['topology'] == "sequential") &
                                   (benchmarks['batch_size'] == b) &
                                   (benchmarks['intraop_threads'] == t)][yaxis]

            if yaxis == 'latency(avg)':
                point = max(point)
            elif yaxis == 'throughput':
                point = sum(point)
            # print(f"{instance},{b},{t},{point}")

            new_row = pd.DataFrame.from_records([{"B": int(batch_size), "i": int(instance), "b": b,
                                                  "t": int(t), yaxis: int(point)}])
            data = pd.concat([data, new_row], ignore_index=True)

    data['B'] = data['B'].astype(int)
    data['B'] = data['B'].astype('category')
    data['change'] = data[yaxis].astype(int)

    plot = ggplot(data=data, mapping=aes(x='i', y=yaxis, color='B')) + \
        MyTheme(base_size=10) + labs(y=f"{ylabel}") + \
        theme(legend_position='top', legend_title=element_blank()) + \
        ggtitle("Instances vs {}".format(label)) + \
        scale_x_log10(
        breaks=data['i'].unique(), labels=["I={}, t={}".format(thr, int(maxthreads/thr)) for thr in data['i'].unique()], name='# Instances') + \
        scale_y_log10(labels=lambda lst: ["{:.1f}".format(y) for y in lst]) + \
        scale_color_discrete(breaks=data['B'].unique(), labels=["B={}".format(b) for b in data['B'].unique()]) + \
        geom_point(size=5) + \
        geom_text(aes(label='change'), size=13, ha="left", va="bottom") +\
        geom_line()

    plot.save("{}-{}-multiinstance.png".format(bench, label.lower()),
              dpi=300, width=12, height=5, verbose=False)


class Plots(Enum):
    latency = 1
    throughput = 2


class PType(Enum):
    normal = 1  # Normal scaleup plot #Threads vs Throughput or Latency
    heatmap = 2  # Heatmap plot for latency and throughput
    scaleup = 3  # Scaleup plot for throughput
    flops = 4  # Threads vs GFLOPs/sec
    numa = 5  # Threads vs latency or throughput comparison between sequential and interleaved case
    batch = 6  # Threads vs throughput or latency comparison per thread
    multiinstance = 7  # Threads vs throughput or latency comparison for multiple instances
    all = 255


def args_parser():
    args = ArgumentParser(description="Run the plot script")
    args.add_argument("-p", "--plot", type=str, default=Plots.latency.name,
                      help="Script supports the following plot types {}".format([plot.name for plot in Plots]))
    args.add_argument("-t", "--type", type=str, default=PType.normal.name,
                      help="Script supports the following plot types {}".format([t.name for t in PType]))
    args.add_argument("-f", "--file", type=str, required=True,
                      help="Path to the date file.")

    if len(sys.argv) < 2:
        args.print_help()
        sys.exit(1)
    return args.parse_args()


def plot_latency(df, ptype):
    if ptype == PType.normal or ptype == PType.all:
        scaling_plot_per_batch(df, 'latency(avg)', 'Latency')
        scaling_plot_per_opt(df, 'latency(avg)', 'Latency')
    elif ptype == PType.heatmap or ptype == PType.all:
        plot_scaling_heatmap(df, 'latency(avg)', 'Latency')
    elif ptype == PType.flops or ptype == PType.all:
        scaling_plot_per_batch(df, 'flops', 'Flops')
        scaling_plot_per_opt(df, 'flops', 'Flops')
    elif ptype == PType.numa or ptype == PType.all:
        plot_sequential_and_interleaved_comparison_per_batch(
            df, 'latency(avg)', 'Latency')
        plot_sequential_and_interleaved_ratio_per_batch(
            df, 'latency(avg)', 'Latency')
    elif ptype == PType.batch or ptype == PType.all:
        scaling_plot_per_thread(df, 'latency(avg)', 'Latency')
    elif ptype == PType.multiinstance or ptype == PType.all:
        plot_multiinstance_comparison_per_batch(df, 'latency(avg)', 'Latency')
    else:
        raise Exception("Unknown plot type")


def plot_throughput(df, ptype):
    if ptype == PType.normal or ptype == PType.all:
        scaling_plot_per_batch(df, 'throughput', 'Throughput')
        scaling_plot_per_opt(df, 'throughput', 'Throughput')
    elif ptype == PType.heatmap or ptype == PType.all:
        plot_scaling_heatmap(df, 'throughput', 'Throughput')
    elif ptype == PType.scaleup or ptype == PType.all:
        plot_throughput_scaleup(df)
    elif ptype == PType.numa or ptype == PType.all:
        plot_sequential_and_interleaved_comparison_per_batch(
            df, 'throughput', 'Throughput')
        plot_sequential_and_interleaved_ratio_per_batch(
            df, 'throughput', 'Throughput')
    elif ptype == PType.batch or ptype == PType.all:
        scaling_plot_per_thread(df, 'throughput', 'Throughput')
    elif ptype == PType.multiinstance or ptype == PType.all:
        plot_multiinstance_comparison_per_batch(df, 'throughput', 'Throughput')
    else:
        raise Exception("Unknown plot type")


if __name__ == '__main__':
    args = args_parser()

    DF = pd.read_csv(args.file, skip_blank_lines=True)

    if args.plot == Plots.latency.name:
        plot_latency(DF, PType[args.type])
    elif args.plot == Plots.throughput.name:
        plot_throughput(DF, PType[args.type])
    else:
        print("Unknown plot type")
        sys.exit(1)
