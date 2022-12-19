# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.spatial import distance_matrix


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path(exists=True))
def main(
    input_filepath,
    output_filepath,
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

    churches_data_fp = input_fp / "churches.gpkg"

    logger.info(f"Reading data from {plot_data_fp} and {churches_data_fp}")
    data = gpd.read_file(plot_data_fp)
    churches = gpd.read_file(churches_data_fp)

    data["group"] = data.district.factorize()[0]
    logger.info("grouping based on district created")

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
    data = data.drop(
        index=data[np.isneginf(data.orthodox_proportion_ln)].index
    ).reset_index()
    logger.info("orthodox_proportion_ln created, negative infinities removed")

    data["income_per_capita"] = data.total_income / data.population
    logger.info("income_per_capita created")

    data["income_per_capita_ln"] = data.total_income_ln - data.population_ln
    logger.info("income_per_capita_ln created")

    distances = distance_matrix(
        pd.DataFrame({"x": data.geometry.x, "y": data.geometry.y}),
        pd.DataFrame({"x": churches.geometry.x, "y": churches.geometry.y}),
    )
    data["distance_from_orthodox_church"] = distances.min(axis=1).round()
    logger.info("distance_from_orthodox_church created")

    logger.info(f"Saving data to {plot_output_fp}")
    data.to_file(plot_output_fp)

    logger.info(f"Reading data from {income_data_fp}")
    tax = pd.read_csv(income_data_fp, index_col=0)

    tax["total_income"] = tax.loc[
        :, ["estate_income", "business_income", "salary_pension_income"]
    ].sum(axis=1)
    logger.info("total_income created in tax data")
    ordering, _ = pd.factorize(tax.total_income, sort=True)
    tax["order"] = ordering.max() - ordering + 1
    logger.info("order created in tax data")

    logger.info(f"Saving data to {income_output_fp}")
    tax.to_csv(income_output_fp)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
