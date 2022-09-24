import sys
import pandas as pd
from plotnine import *
from plotnine.data import *


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


def parse_allocator_multiinstance_results(data, yaxis: str):
    ylabel = ""
    if yaxis == 'latency(avg)':
        ylabel = "Latency per batch (ms)"
    elif yaxis == 'throughput':
        ylabel = "Throughput (Inferences/sec)"

    per_allocator = []
    for allocator in data.allocator.unique():
        for instance in data.instances.unique():
            df = data[(data.allocator == allocator) &
                      (data.instances == instance)]
            if yaxis == 'latency(avg)':
                per_instance = df.groupby(
                    ['allocator', 'instances', 'intraop_threads', 'batch_size'], as_index=False).agg({yaxis: max})
            elif yaxis == 'throughput':
                per_instance = df.groupby(
                    ['allocator', 'instances', 'intraop_threads', 'batch_size'], as_index=False).agg({yaxis: sum})
            if instance == 2:
                if yaxis == 'latency(avg)':
                    one_instance = df.groupby(
                        ['allocator', 'instances', 'intraop_threads', 'batch_size'], as_index=False).agg({yaxis: min})
                elif yaxis == 'throughput':
                    one_instance = df.groupby(
                        ['allocator', 'instances', 'intraop_threads', 'batch_size'], as_index=False).agg({yaxis: max})
                one_instance['instances'] = 1
                per_allocator.append(one_instance)
            per_allocator.append(per_instance)
    plot_data = pd.concat(per_allocator)

    plot = ggplot(plot_data, mapping=aes(x='instances', y=yaxis, color='allocator')) + \
        ggtitle(f"Instances vs {yaxis} <1, 16>") + \
        MyTheme(base_size=6) + labs(y=f"{ylabel}") + \
        theme(legend_position='top', legend_title=element_blank(), legend_box_spacing=0.0) + \
        scale_x_continuous(
        breaks=[max(1, thr) for thr in range(0, 17, 2)], labels=["{}".format(max(1, thr)) for thr in range(0, 17, 2)], name='# Instances') + \
        geom_point() + \
        geom_line()
    plot.save(f"plot-multiinstance-{yaxis}.png",
              dpi=300, width=84, height=32, units='mm')


def parse_multiinstance_results(df, yaxis: str):
    df.loc[df['intraop_threads'] == 16, 'topology'] = 'single'
    if yaxis == 'latency(avg)':
        data = df.groupby(
            ['topology', 'intraop_threads', 'batch_size'], as_index=False).agg({'interop_threads': sum, yaxis: max})
    if yaxis == "throughput":
        data = df.groupby(
            ['topology', 'intraop_threads', 'batch_size'], as_index=False).agg({'interop_threads': sum, yaxis: sum})
    data = data.rename(
        columns={"intraop_threads": "threads", "interop_threads": "instances"})
    data['effective_bs'] = data['instances'] * data['batch_size']
    data = data.sort_values(by=['effective_bs', 'threads'])

    print(data.to_string(index=False))

    p = ggplot(data, aes(x='effective_bs', weight=yaxis, fill='topology', xticks=False)) + \
        labs(y="Latency") + \
        scale_y_log10(labels=lambda lst: ["{:,.0f}".format(y) for y in lst]) + \
        scale_x_log10(
        breaks=data['batch_size'].unique(), labels=["{}".format(thr) for thr in data['batch_size'].unique()], name='Batch Size') + \
        geom_bar(position=position_dodge())
    p.save("plot.png",
           dpi=300, width=84, height=32, units='mm')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: <Path to the data file>.")
        sys.exit(1)
    df = pd.read_csv(sys.argv[1], skip_blank_lines=True)
    parse_multiinstance_results(df, 'latency(avg)')
