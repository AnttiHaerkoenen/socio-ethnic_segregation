# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import schemdraw
from schemdraw import flow
import click


@click.command()
@click.argument("figure_filepath", type=click.Path(exists=True))
def main(
    figure_filepath,
):
    """
    Draw a flowchart and save it to image file
    """
    logger = logging.getLogger(__name__)
    figure_fp = Path(figure_filepath)

    with schemdraw.Drawing(file=figure_fp / "flowchart.svg", show=False) as d:
        d.config(fontsize=12)
        d += (cd := flow.Box(w=4).label("Combine data"))
        d += flow.Arrow().down(d.unit / 2).at(cd.S)
        d += flow.Box(w=4).label("Clean data")
        d += flow.Arrow().down(d.unit / 2)
        d += flow.Box(w=4).label("Factor analysis")
        d += flow.Arrow().down(d.unit / 2)
        d += flow.Box(w=4).label("Create clusters")
        d += flow.Arrow().down(d.unit / 2)
        d += flow.Box(w=4).label("Multilevel regression")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
