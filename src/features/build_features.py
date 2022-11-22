# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.cluster import KMeans


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path(exists=True))
@click.option(
    "--n_clusters",
    default=10,
    type=click.IntRange(5, 25),
    help="Number of clusters",
)
@click.option(
    "--seed",
    default=42,
    type=click.IntRange(0, 1000),
    help="Seed for pseudorandom elements",
)
def main(
    input_filepath,
    output_filepath,
    n_clusters,
    seed,
):
    """Runs data processing scripts to turn interim data from (../interim) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Making final data set from interim data")
    input_fp = Path(input_filepath)
    output_fp = Path(output_filepath)

    plot_data_fp = input_fp / "spatial_income_1880.gpkg"
    plot_output_fp = output_fp / "spatial_income_1880.gpkg"

    income_data_fp = input_fp / "income_tax_record_1880.csv"
    income_output_fp = output_fp / "income_tax_record_1880.csv"

    logger.info(f"Reading data from {plot_data_fp}")
    data = gpd.read_file(plot_data_fp).set_crs(epsg=3067)

    for col in [
        "total_income",
        "estate_income",
        "salary_pension_income",
        "business_income",
    ]:
        data[f"{col}_ln"] = data[col].apply(np.log)
        data.loc[np.isneginf(data[f"{col}_ln"]), f"{col}_ln"] = None
        logger.info(f"{col}_ln created")

    data["lutheran_ln"] = data.lutheran.apply(np.log)
    logger.info("lutheran_ln created")

    data["orthodox_ln"] = data.orthodox.apply(np.log)
    logger.info("orthodox_ln created")

    data["population_ln"] = data.population.apply(np.log)
    logger.info("population_ln created")

    data["orthodox_proportion"] = data.orthodox / data.population
    logger.info("orthodox_proportion created")

    data["orthodox_proportion_ln"] = data.orthodox_proportion.apply(np.log)
    data.loc[np.isneginf(data.orthodox_proportion_ln), "orthodox_proportion_ln"] = None
    logger.info("orthodox_proportion_ln created")

    data["income_per_capita"] = data.total_income / data.population
    logger.info("income_per_capita created")

    data["income_per_capita_ln"] = data.total_income_ln - data.population_ln
    logger.info("income_per_capita_ln created")

    xy = np.array([data.geometry.x, data.geometry.y]).T
    groups = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(xy)
    data["group"] = groups
    logger.info(f"data divided into {n_clusters} clusters")

    logger.info(f"Saving data to {plot_output_fp}")
    data.to_file(plot_output_fp)

    logger.info(f"Reading data from {income_data_fp}")
    tax = pd.read_csv(income_data_fp, index_col=0)

    tax['total_income'] = tax.loc[:, ["estate_income", "business_income", "salary_pension_income"]].sum(axis=1)
    logger.info("total_income created in tax data")

    logger.info(f"Saving data to {income_output_fp}")
    tax.to_csv(income_output_fp)

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
