"""
Script that plots benchmark data-visualizations.
"""

import sys
from turtle import color
import pandas as pd
from plotnine.themes import theme_gray, theme
from plotnine.themes.elements import (element_line, element_rect,
                                      element_text, element_blank)
from plotnine import ggplot
from plotnine.labels import ggtitle
from plotnine.mapping import aes
from plotnine.labels import labs
from plotnine.scales import scale_x_continuous, scale_y_continuous, scale_color_discrete, scale_y_log10, scale_fill_gradientn 
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


def plot_benchmark_latency_per_batch(benchmarks):
    "Plots a latency graph for every data-structure in the results file"

    for batch_size in benchmarks['batch_size'].unique():
        data_set = []
        for name in benchmarks.benchmark.unique():
            opt = name.split('_')[2].upper()
            benchmark = benchmarks.loc[(benchmarks['benchmark'] == name) &
                                       (benchmarks['batch_size'] == batch_size)]
            benchmark.insert(0, 'config', "opt={}".format(opt))
            data_set.append(benchmark)
        data = pd.concat(data_set)

        plot = ggplot(data=data, mapping=aes(x='threads', y='latency(avg)', color='config')) + \
            MyTheme(base_size=10) + labs(y="Latency [ms]") + \
            theme(legend_position='top', legend_title=element_blank(), legend_box_spacing=-0.35) + \
            ggtitle("Threads vs Latency (BS={})".format(batch_size)) + \
            scale_x_continuous(
                breaks=data['threads'].unique(), labels=["{}".format(thr) for thr in data['threads'].unique()], name='# threads') + \
            scale_y_log10(labels=lambda lst: ["{:.1f}".format(x) for x in lst]) + \
            scale_color_discrete(breaks=data['config'].unique()) + \
            geom_point() + \
            geom_line()

        plot.save("resnet-latency-{}.png".format(batch_size), dpi=300,
                  width=12, height=5, verbose=False)


def plot_benchmark_latency_per_opt(benchmarks):
    "Plots a latency graph for every data-structure in the results file"

    for name in benchmarks.benchmark.unique():
        data_set = []
        for batch_size in benchmarks['batch_size'].unique():
            opt = name.split('_')[2]
            benchmark = benchmarks.loc[(benchmarks['benchmark'] == name) &
                                       (benchmarks['batch_size'] == batch_size)]
            benchmark.insert(0, 'config', "Batch={}".format(batch_size))
            data_set.append(benchmark)
        data = pd.concat(data_set)

        plot = ggplot(data=data, mapping=aes(x='threads', y='latency(avg)', color='config')) + \
            MyTheme(base_size=10) + labs(y="Latency [ms]") + \
            theme(legend_position='top', legend_title=element_blank(), legend_box_spacing=-0.35) + \
            ggtitle("Threads vs Latency (Opt={})".format(opt.upper())) + \
            scale_x_continuous(
                breaks=data['threads'].unique(), labels=["{}".format(thr) for thr in data['threads'].unique()], name='# threads') + \
            scale_y_log10(labels=lambda lst: ["{:.1f}".format(x) for x in lst]) + \
            scale_color_discrete(breaks=data['config'].unique()) + \
            geom_point() + \
            geom_line()

        plot.save("resnet-latency-{}.png".format(opt), dpi=300,
                  width=12, height=5, verbose=False)


def plot_benchmark_throughputs_per_batch(benchmarks):
    "plots a throughput graph for every data-structure in the results file"

    for batch_size in benchmarks['batch_size'].unique():
        data_set = []
        for name in benchmarks.benchmark.unique():
            opt = name.split('_')[2].upper()
            benchmark = benchmarks.loc[(benchmarks['benchmark'] == name) &
                                       (benchmarks['batch_size'] == batch_size)]
            benchmark.insert(0, 'config', "opt={}".format(opt))
            data_set.append(benchmark)
        data = pd.concat(data_set)

        plot = ggplot(data=data, mapping=aes(x='threads', y='throughput', color='config')) + \
            MyTheme(base_size=10) + \
            labs(y="throughput [ops]") + \
            theme(legend_position='top', legend_title=element_blank(), legend_box_spacing=-0.35) + \
            ggtitle("Threads vs Throughput (BS={})".format(batch_size)) + \
            scale_x_continuous(
                breaks=data['threads'].unique(), labels=["{}".format(thr) for thr in data['threads'].unique()], name='# threads') + \
            scale_y_continuous(labels=lambda lst: ["{:,}".format(x) for x in lst]) + \
            scale_color_discrete(breaks=data['config'].unique()) + \
            geom_point() + \
            geom_line()

        plot.save("resnet-throughput-{}.png".format(batch_size), dpi=300,
                  width=12, height=5, verbose=False)


def plot_benchmark_throughputs_per_opt(benchmarks):
    "plots a throughput graph for every data-structure in the results file"

    for name in benchmarks.benchmark.unique():
        data_set = []
        for batch_size in benchmarks['batch_size'].unique():
            opt = name.split('_')[2]
            benchmark = benchmarks.loc[(benchmarks['benchmark'] == name) &
                                       (benchmarks['batch_size'] == batch_size)]
            benchmark.insert(0, 'config', "Batch={}".format(batch_size))
            data_set.append(benchmark)
        data = pd.concat(data_set)

        plot = ggplot(data=data, mapping=aes(x='threads', y='throughput', color='config')) + \
            MyTheme(base_size=10) + \
            labs(y="throughput [ops]") + \
            theme(legend_position='top', legend_title=element_blank(), legend_box_spacing=-0.35) + \
            ggtitle("Threads vs Throughput (Opt={})".format(opt.upper())) + \
            scale_x_continuous(
                breaks=data['threads'].unique(), labels=["{}".format(thr) for thr in data['threads'].unique()], name='# threads') + \
            scale_y_continuous(labels=lambda lst: ["{:,}".format(x) for x in lst]) + \
            scale_color_discrete(breaks=data['config'].unique()) + \
            geom_point() + \
            geom_line()

        plot.save("resnet-throughput-{}.png".format(opt), dpi=300,
                  width=12, height=5, verbose=False)


def plot_latency_heatmap(benchmarks):
    for batch_size in benchmarks['batch_size'].unique():
        benchmark = benchmarks.loc[benchmarks['batch_size'] == batch_size]
        benchmark['latency(avg)'] = benchmark['latency(avg)'].astype(int)

        p = ggplot(data=benchmark,
                   mapping=aes(x='intraop_threads', y='interop_threads', fill='latency(avg)')) + \
            labs(x="#Intraop Threads") + \
            labs(y="#Interop Threads") + \
            theme(legend_position="right", legend_title=element_blank()) +\
            scale_x_continuous(breaks=benchmark['intraop_threads'].unique(), labels=["{}".format(thr) for thr in benchmark['intraop_threads'].unique()]) + \
            scale_y_continuous(breaks=benchmark['interop_threads'].unique(), labels=["{}".format(thr) for thr in benchmark['interop_threads'].unique()]) + \
            geom_tile(aes(width=.95, height=.95)) + \
            ggtitle("Latency for BS = {}".format(batch_size)) + \
            geom_text(aes(label='latency(avg)'), size=7, color='black') + \
            theme(axis_text_x=element_text(size=10),
                  axis_text_y=element_text(size=10),
                  panel_background=element_rect(fill="white"),
                  panel_grid_major=element_line(colour="white"),
                  strip_background=element_rect(colour="orange", fill="orange")) + \
            scale_fill_gradientn(colors=["#0080FF", "#00FFFF", "#00FF80", "#FFFF00", "#FF8000", "#FF0000", "#800000"])

        p.save('latency_heatmap_{}.png'.format(batch_size), dpi=300)


def plot_throughput_heatmap(benchmarks):
    for batch_size in benchmarks['batch_size'].unique():
        benchmark = benchmarks.loc[benchmarks['batch_size'] == batch_size]
        benchmark['throughput'] = benchmark['throughput'].astype(int)

        p = ggplot(data=benchmark,
                   mapping=aes(x='intraop_threads', y='interop_threads', fill='throughput')) + \
            labs(x="#Intraop Threads") + \
            labs(y="#Interop Threads") + \
            theme(legend_position="right", legend_title=element_blank()) +\
            scale_x_continuous(breaks=benchmark['intraop_threads'].unique(), labels=["{}".format(thr) for thr in benchmark['intraop_threads'].unique()]) + \
            scale_y_continuous(breaks=benchmark['interop_threads'].unique(), labels=["{}".format(thr) for thr in benchmark['interop_threads'].unique()]) + \
            geom_tile(aes(width=.95, height=.95)) + \
            ggtitle("Throughput for BS = {}".format(batch_size)) + \
            geom_text(aes(label='throughput'), size=7, color='black') + \
            theme(axis_text_x=element_text(size=10),
                  axis_text_y=element_text(size=10),
                  panel_background=element_rect(fill="white"),
                  panel_grid_major=element_line(colour="white"),
                  strip_background=element_rect(colour="orange", fill="orange")) + \
            scale_fill_gradientn(colors=["#0080FF", "#00FFFF", "#00FF80", "#FFFF00", "#FF8000", "#FF0000", "#800000"])

        p.save('throughput_heatmap_{}.png'.format(batch_size), dpi=300)

def plot_latency(df):
    if 'interop_threads' in df.columns.to_list() or 'intraop_threads' in df.columns.to_list():
        plot_latency_heatmap(df)
    else:
        plot_benchmark_latency_per_batch(df)
        plot_benchmark_latency_per_opt(df)


def plot_throughput(df):
    if 'interop_threads' in df.columns.to_list() or 'intraop_threads' in df.columns.to_list():
        plot_throughput_heatmap(df)
    else:
        plot_benchmark_throughputs_per_batch(df)
        plot_benchmark_throughputs_per_opt(df)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: latency.py latency/throughput <results-file>")
        sys.exit(1)

    DF = pd.read_csv(sys.argv[2], skip_blank_lines=True)

    if sys.argv[1] == "latency":
        plot_latency(DF)
    elif sys.argv[1] == "throughput":
        plot_throughput(DF)

    else:
        print("Unknown benchmark type")
        sys.exit(1)
