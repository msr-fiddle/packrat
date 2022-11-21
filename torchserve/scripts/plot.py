from argparse import ArgumentParser
from enum import Enum
import sys
import pandas as pd
from plotnine.themes import theme_gray, theme
from plotnine.themes.elements import (element_line, element_rect,
                                      element_text, element_blank)
from plotnine import ggplot
from plotnine.labels import ggtitle
from plotnine.mapping import aes
from plotnine.labels import labs
from plotnine.scales import scale_x_continuous, scale_y_continuous, scale_color_discrete
from plotnine.geoms import geom_point, geom_line, geom_text


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

    ylabel = "Latency per batch (ms)"

    for batch_size in benchmarks['batch_size'].unique():
        data_set = []
        for model in benchmarks.model.unique():
            opt = "Script"
            bench = "resnet"
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


class Plots(Enum):
    latency = 1
    throughput = 2


def args_parser():
    args = ArgumentParser(description="Run the plot script")
    args.add_argument("-p", "--plot", type=str, default=Plots.latency.name,
                      help="Script supports the following plot types {}".format([plot.name for plot in Plots]))
    args.add_argument("-f", "--file", type=str, required=True,
                      help="Path to the date file.")

    if len(sys.argv) < 2:
        args.print_help()
        sys.exit(1)
    return args.parse_args()


if __name__ == '__main__':
    args = args_parser()

    DF = pd.read_csv(args.file, skip_blank_lines=True)

    if args.plot == Plots.latency.name:
        scaling_plot_per_batch(DF, "mean_predict_latency", "Latency")
    elif args.plot == Plots.throughput.name:
        raise Exception("TODO")
    else:
        print("Unknown plot type")
        sys.exit(1)
