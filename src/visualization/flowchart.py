# -*- coding: utf-8 -*-
import logging
from pathlib import Path

from schemdraw import flow
import click


@click.command()
@click.argument('figure_filepath', type=click.Path(exists=True))
def main(
        figure_filepath,
):
    """
    Draw a flowchart and save it to image file
    """
    logger = logging.getLogger(__name__)
    figure_fp = Path(figure_filepath)

    with schemdraw.Drawing(file=figure_fp / 'flowchart.svg') as d:
        d.config(fontsize=12)
        d += (sd := flow.Data().label('Spatial data'))
        d += flow.Arrow().down()
        d += (id := flow.Data().label('Income data'))
        d += flow.Arrow().down()
        d += (dd := flow.Data().label('Demographic data'))
        d += flow.Arrow().down()

        d += (cd := flow.Box().label('Combine data'))
        d += flow.Arrow().down()
        d += (cl := flow.Box().label('Clean data'))
        d += flow.Arrow().down()
        d += (clust := flow.Box().label('Create clusters'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
