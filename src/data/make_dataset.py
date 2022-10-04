# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import numpy as np
import geopandas as gpd


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making interim data set from raw data')
    input_fp = Path(input_filepath)
    output_fp = Path(output_filepath)

    plot_data_fp = input_fp / 'spatial_income_1880.gpkg'
    old_areas_fp = input_fp / 'old_districts.gpkg'
    plot_output_fp = output_fp / 'spatial_income_1880.gpkg'

    logger.info(f'Reading data from {plot_data_fp}')
    logger.info('Checking coordinates')
    data = gpd.read_file(plot_data_fp).set_crs(epsg=3067)
    old_areas = gpd.read_file(old_areas_fp).set_crs(epsg=3067)

    assert data.crs == old_areas.crs == "epsg:3067"

    data['is_old'] = data.geometry.within(old_areas.unary_union)
    data.rename(columns={
        'lutheran_density': 'lutheran',
        'orthodox_density': 'orthodox',
        'total_density': 'population',
    }, inplace=True)

    logger.info(f'Saving data to {plot_output_fp}')
    data.to_file(plot_output_fp)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
