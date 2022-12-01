# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import geopandas as gpd


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path(exists=True))
@click.option(
    "--min_density",
    "min_density",
    type=click.FloatRange(0, 20),
    help="lots with population density lower than this are dropped, default 5",
    default=5,
)
@click.option(
    "--districts",
    "districts",
    type=click.STRING,
    default="all",
    help="districts kept, in format 'district1 district2', default 'all'",
)
def main(input_filepath, output_filepath, min_density, districts):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    interim data (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info("Making interim data set from raw data")

    input_fp = Path(input_filepath)
    output_fp = Path(output_filepath)
    plot_data_fp = input_fp / "spatial_income_1880.gpkg"
    old_areas_fp = input_fp / "old_districts.gpkg"
    water_fp = input_fp / "water_1913.gpkg"
    churches_fp = input_fp / "churches.gpkg"
    plot_output_fp = output_fp / "spatial_income_1880.gpkg"
    water_output_fp = output_fp / "water_1913.gpkg"
    churches_output_fp = output_fp / "churches.gpkg"

    logger.info(f"Reading data from {plot_data_fp}")
    data = gpd.read_file(plot_data_fp).set_crs(epsg=3067)
    water = gpd.read_file(water_fp).set_crs(epsg=3067)
    old_areas = gpd.read_file(old_areas_fp).set_crs(epsg=3067)
    churches = gpd.read_file(churches_fp).set_crs(epsg=3067)

    assert data.crs == old_areas.crs == water.crs == churches.crs == "epsg:3067", ValueError("mismatching coordinate types")

    if districts.lower() == "all":
        districts = list(data.district.unique())
    else:
        districts = districts.split()

    data["is_old"] = data.geometry.within(old_areas.unary_union)
    logger.info("data.is_old created")

    data.rename(
        columns={
            "lutheran_density": "lutheran",
            "orthodox_density": "orthodox",
            "total_density": "population",
        },
        inplace=True,
    )

    data = (
        data.drop(
            index=data.query(
                f"(population < {min_density}) | (district != {districts})"
            ).index
        )
        .dropna()
        .reset_index()
    )
    logger.info(f"Dropped plots with lowest density and selected districts not in {districts}")

    logger.info(f"Saving data to {plot_output_fp}")
    data.to_file(plot_output_fp)
    water.to_file(water_output_fp)
    churches.to_file(churches_output_fp)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
