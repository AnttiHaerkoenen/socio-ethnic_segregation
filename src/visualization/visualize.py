# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import geopandas as gpd
import arviz as az
import matplotlib.pyplot as plt


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("model_filepath", type=click.Path())
@click.argument("figure_filepath", type=click.Path())
def main(
    input_filepath,
    model_filepath,
    figure_filepath,
):
    """
    Draw figures and save them to 'figure_filepath'
    """
    logger = logging.getLogger(__name__)
    data_fp = Path(input_filepath)
    model_fp = Path(model_filepath)
    figure_fp = Path(figure_filepath)

    data = gpd.read_file(data_fp / "spatial_income_1880.gpkg")
    water = gpd.read_file(data_fp / "water_1913.gpkg")

    prior = az.InferenceData.from_netcdf(model_fp / "prior")
    posterior = az.InferenceData.from_netcdf(model_fp / "posterior")
    posterior_prediction = az.InferenceData.from_netcdf(
        model_fp / "posterior_prediction"
    )

    logger.info("Plotting posterior distribution")
    az.plot_posterior(
        posterior, 
        var_names=["β"],
        grid=(4, 3),
        figsize=(12, 16.5),
    )
    plt.tight_layout()
    plt.savefig(figure_fp / "posterior", dpi=300)

    logger.info("Plotting trace plot")
    az.plot_trace(posterior)
    plt.tight_layout()
    plt.savefig(figure_fp / "model_trace.png", dpi=300)

    logger.info("Plotting forest plot of the posterior")
    az.plot_forest(posterior, combined=True, hdi_prob=0.95)
    plt.tight_layout()
    plt.savefig(figure_fp / "model_forest_plot", dpi=300)

    logger.info("Saving summary data for model")
    posterior_summary = az.summary(posterior, hdi_prob=0.95)
    posterior_summary.to_csv(figure_fp / "posterior_summary.csv")

    logger.info("Saving posterior predictive checks")
    ppc = az.plot_ppc(
        posterior_prediction,
        legend=False,
    )
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(figure_fp / "posterior_predictive_check.png", dpi=300)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
