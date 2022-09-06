import sys
import pandas as pd
from plotnine import *
from plotnine.data import *


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
