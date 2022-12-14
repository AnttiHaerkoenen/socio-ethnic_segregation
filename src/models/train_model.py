# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
import pymc as pm
import pandas as pd
import geopandas as gpd
import aesara.tensor as at
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("model_filepath", type=click.Path())
@click.argument("figure_filepath", type=click.Path())
@click.option(
    "--seed",
    default=42,
    type=click.IntRange(0, 1000),
    help="Seed for pseudorandom elements",
)
@click.option(
    "--prior_samples",
    default=100,
    type=click.IntRange(10, 2000),
    help="Number of samples from prior distribution",
)
@click.option(
    "--draws",
    default=1000,
    type=click.IntRange(10, 2000),
    help="Number of draws from posterior distribution",
)
@click.option(
    "--tune",
    default=1000,
    type=click.IntRange(10, 2000),
    help="Number of tuning samples",
)
@click.option(
    "--target_accept",
    default=0.95,
    type=click.FloatRange(0.5, 0.99),
    help="Target accept threshold for NUTS",
)
def main(
    input_filepath,
    model_filepath,
    figure_filepath,
    seed,
    prior_samples,
    draws,
    tune,
    target_accept,
):
    """Train models and save them to 'model_filepath'"""
    logger = logging.getLogger(__name__)
    data_fp = Path(input_filepath)
    model_fp = Path(model_filepath)
    figure_fp = Path(figure_filepath)

    logger.info("Preparing data")
    data = gpd.read_file(data_fp / "spatial_income_1880.gpkg")
    data = data.loc[data.is_old]
    data['distance_from_church_km'] = data['distance_from_orthodox_church'] / 1000
    N_CLUSTERS = len(data.group.unique())
    N = data.shape[0]
    O_norm = (
        StandardScaler()
        .fit_transform(data.orthodox_proportion_ln.values.reshape(-1, 1))
        .flatten()
    )
    logger.info("Calculating distance matrix")
    xy = pd.DataFrame({"x": data.geometry.x, "y": data.geometry.y, "group": data.group})
    d = distance_matrix(xy, xy)

    logger.info("Training model")

    with pm.Model() as model:
        idx = data.group
        W = pm.ConstantData("W", data.total_income_ln)
        C = pm.ConstantData("C", data.distance_from_church_km)

        ?? = pm.Normal("??", [0, 0, 0], [0.1, 0.1, 0.1], shape=3)
        ?? = pm.MvNormal(
            "??", mu=??, cov=np.diagflat(np.array([0.1, 0.1, 0.1])), shape=(N_CLUSTERS, 3)
        )

        ??2 = pm.Normal("????", 1, 0.2)
        ??2_std = pm.Normal("????_scaled", 1, 0.2) # scaled by 75
        K = ??2 * at.exp(-75 * ??2_std * at.power(d, 2)) + np.diag([0.01] * N)
        ?? = ??[idx, 0] + ??[idx, 1] * W + ??[idx, 2] * C
        O = pm.MvNormal("O", mu=??, cov=K, shape=N, observed=O_norm)

        logger.info(f"Drawing {prior_samples} samples from prior distribution")
        prior = pm.sample_prior_predictive(samples=prior_samples, random_seed=seed)
        logger.info(
            f"Drawing {draws} samples from posterior distribution with {tune} tuning samples, target_accept={target_accept}"
        )
        posterior = pm.sample(
            draws=draws,
            tune=tune,
            init="adapt_diag",
            return_inferencedata=True,
            target_accept=target_accept,
            random_seed=seed,
        )
        logger.info("Sampling posterior predictive distribution")
        posterior_prediction = pm.sample_posterior_predictive(
            posterior,
            random_seed=seed,
        )

    logger.info("Saving model to netcdf files")
    for file in model_fp.glob("*"):
        file.unlink(missing_ok=True)
    prior.to_netcdf(model_fp / "prior")
    posterior.to_netcdf(model_fp / "posterior")
    posterior_prediction.to_netcdf(model_fp / "posterior_prediction")
    logger.info("Model saved")

    logger.info("Saving model as plate diagram")
    graph = pm.model_to_graphviz(model)
    graph.format = "svg"
    graph.render(figure_fp / "plate_diagram")
    logger.info("Plate diagram saved")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
