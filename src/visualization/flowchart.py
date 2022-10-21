# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import graphviz
import click


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('figure_filepath', type=click.Path())
def main(
        input_filepath,
        figure_filepath,
):
    """
    Draw a flowchart from a dot file and save it to image file
    """
    logger = logging.getLogger(__name__)
    input_fp = Path(input_filepath)
    figure_fp = Path(figure_filepath)

    dot = graphviz.Source.from_file(input_fp / 'flowchart.dot')
    graph = graphviz.DiGraph(dot)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
