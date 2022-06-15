"""
Script that plots benchmark data-visualizations.
"""

import sys
import pandas as pd
from plotnine.themes import theme_gray, theme
from plotnine.themes.elements import (element_line, element_rect,
                                      element_text, element_blank)
from plotnine import ggplot
from plotnine.mapping import aes
from plotnine.labels import labs
from plotnine.scales import scale_x_continuous, scale_y_continuous, scale_color_discrete, scale_y_log10
from plotnine.geoms import geom_point, geom_line


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


def plot_benchmark_latency(benchmarks):
    "Plots a latency graph for every data-structure in the results file"

    benchmarks['config'] = benchmarks.apply(
        lambda row: "Batch={:d}".format(int(row.batch_size)), axis=1)

    plot = ggplot(data=benchmarks, mapping=aes(x='threads', y='latency', color='config')) + \
        MyTheme(base_size=10) + labs(y="Latency [ms]") + \
        theme(legend_position='top', legend_title=element_blank()) + \
        scale_x_continuous(
            breaks=benchmarks['threads'].unique(), labels=["{}".format(thr) for thr in benchmarks['threads'].unique()], name='# Threads') + \
        scale_y_log10(labels=lambda lst: ["{:.1f}".format(x) for x in lst]) + \
        scale_color_discrete(breaks=benchmarks['config'].unique()) + \
        geom_point() + \
        geom_line()

    plot.save("resnet-latency.png", dpi=300, width=12, height=5, verbose=False)


def plot_benchmark_throughputs(benchmarks):
    "Plots a throughput graph for every data-structure in the results file"
    # Fail if we have more than one experiment duration

    benchmarks['config'] = benchmarks.apply(
        lambda row: "Batch={:d}".format(int(row.batch_size)), axis=1)

    plot = ggplot(data=benchmarks, mapping=aes(x='threads', y='throughput', color='config')) + \
        MyTheme(base_size=10) + \
        labs(y="Throughput [ops]") + \
        theme(legend_position='top', legend_title=element_blank()) + \
        scale_x_continuous(
            breaks=benchmarks['threads'].unique(), labels=["{}".format(thr) for thr in benchmarks['threads'].unique()], name='# Threads') + \
        scale_y_continuous(labels=lambda lst: ["{:,}".format(x) for x in lst]) + \
        scale_color_discrete(breaks=benchmarks['config'].unique()) + \
        geom_point() + \
        geom_line()

    plot.save("resnet-throughput.png", dpi=300,
              width=12, height=5, verbose=False)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: latency.py latency/throughput <results-file>")
        sys.exit(1)

    DF = pd.read_csv(sys.argv[2], skip_blank_lines=True)

    if sys.argv[1] == "latency":
        plot_benchmark_latency(DF)
    elif sys.argv[1] == "throughput":
        plot_benchmark_throughputs(DF)
    else:
        print("Unknown benchmark type")
        sys.exit(1)
