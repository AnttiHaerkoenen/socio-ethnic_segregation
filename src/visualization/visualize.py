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

    model_1_dir = model_fp / "model_1"
    prior_1 = az.InferenceData.from_netcdf(model_1_dir / "prior")
    posterior_1 = az.InferenceData.from_netcdf(model_1_dir / "posterior")
    posterior_prediction_1 = az.InferenceData.from_netcdf(
        model_1_dir / "posterior_prediction"
    )

    model_2_dir = model_fp / "model_2"
    prior_2 = az.InferenceData.from_netcdf(model_2_dir / "prior")
    posterior_2 = az.InferenceData.from_netcdf(model_2_dir / "posterior")
    posterior_prediction_2 = az.InferenceData.from_netcdf(
        model_2_dir / "posterior_prediction"
    )

    logger.info("Saving posterior predictive checks")
    ppc_1 = az.plot_ppc(
        posterior_prediction_1,
        legend=False,
        num_pp_samples=1000,
    )
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(figure_fp / "model_1_posterior_predictive_check.png", dpi=300)
    ppc_2 = az.plot_ppc(
        posterior_prediction_2,
        legend=False,
        num_pp_samples=1000,
    )
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(figure_fp / "model_2_posterior_predictive_check.png", dpi=300)

    logger.info("Plotting posterior distributions")
    az.plot_posterior(posterior_2, var_names="β_O", grid=(12, 3), figsize=(12, 36))
    plt.savefig(figure_fp / "model_2_posterior", dpi=300)

    logger.info("Plotting trace plot")
    az.plot_trace(posterior_2)
    plt.tight_layout()
    plt.savefig(figure_fp / "model_2_trace.png", dpi=300)

    logger.info("Plotting forest plot of the posterior")
    az.plot_forest(posterior_2, combined=True, hdi_prob=0.95)
    plt.tight_layout()
    plt.savefig(figure_fp / "model_2_forest_plot", dpi=300)

    logger.info("Saving summary data for model")
    posterior_summary = az.summary(posterior_2, hdi_prob=0.95)
    posterior_summary.to_csv(figure_fp / "posterior_summary.csv")

    logger.info("Plotting a map of clusters")
    cluster_colors = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(12, 10))
    data.plot(
        ax=ax,
        column="group",
        categorical=True,
        cmap=cluster_colors,
        markersize=25,
        edgecolor="black",
        alpha=0.8,
        legend=True,
        legend_kwds={
            "title": "Clusters",
            "loc": "lower left",
            "frameon": True,
            "framealpha": 1,
            "edgecolor": "black",
        },
    )
    offset = 100
    x_min, y_min, x_max, y_max = data.total_bounds
    water.plot(
        ax=ax,
        color="#1fb9db",
    )
    ax.set_ylim(y_min - offset, y_max + offset)
    ax.set_xlim(x_min - offset, x_max + offset)
    plt.xticks([])
    plt.yticks([])
    fig.tight_layout()
    plt.savefig(figure_fp / "cluster_map.png", dpi=300)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
