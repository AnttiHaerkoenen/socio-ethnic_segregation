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
def main(
    input_filepath,
    model_filepath,
    figure_filepath,
    seed,
):
    """Train models and save them to 'model_filepath'"""
    logger = logging.getLogger(__name__)
    data_fp = Path(input_filepath)
    model_fp = Path(model_filepath)
    figure_fp = Path(figure_filepath)

    logger.info("Preparing data")
    data = gpd.read_file(data_fp / "spatial_income_1880.gpkg")
    data = data.drop(index=data[data.population < 5].index).dropna().reset_index()
    N_CLUSTERS = len(data.group.unique())
    N = data.shape[0]
    O_norm = StandardScaler().fit_transform(data.orthodox_proportion_ln.values.reshape(-1, 1)).flatten()

    logger.info("Calculating distance matrix")
    xy = pd.DataFrame({"x": data.geometry.x, "y": data.geometry.y, "group": data.group})
    d = distance_matrix(xy, xy)
    
    logger.info("Training model 1")

    with pm.Model() as model_1:
        idx = data.group
        W = pm.MutableData("W", data.total_income_ln)

        θ = pm.Normal("θ", [0, 0], [0.1, 0.1], shape=2)

        β = pm.MvNormal(
            "β", mu=θ, cov=np.diagflat(np.array([ 0.1, 0.1])), shape=(N_CLUSTERS, 2)
        )
        μ = at.math.sigmoid(β[idx, 0] + β[idx, 1] * W)
        σ = pm.HalfNormal("σ", 0.01)

        O = pm.Beta("O", mu=μ, sigma=σ, observed=data.orthodox_proportion)

        prior_1 = pm.sample_prior_predictive(random_seed=seed)
        posterior_1 = pm.sample(init="adapt_diag", return_inferencedata=True, target_accept=0.95, random_seed=seed)
        posterior_prediction_1 = pm.sample_posterior_predictive(posterior_1, random_seed=seed)

    logger.info("Saving model 1 to netcdf files")

    model_1_dir = model_fp / "model_1"
    model_1_dir.mkdir(exist_ok=True)

    for file in model_1_dir.glob("*"):
        file.unlink(missing_ok=True)

    prior_1.to_netcdf(model_1_dir / "prior")
    posterior_1.to_netcdf(model_1_dir / "posterior")
    posterior_prediction_1.to_netcdf(model_1_dir / "posterior_prediction")

    logger.info("Model 1 saved")

    logger.info("Saving model 1 as plate diagram")
    graph_1 = pm.model_to_graphviz(model_1)
    graph_1.format = "svg"
    graph_1.render(figure_fp / "model_1")
    logger.info("Plate diagram saved")

    logger.info("Training model 2")

    with pm.Model() as model_2:
        idx = data.group.loc[::5]
        W = pm.MutableData("W", data.total_income_ln.loc[::5])
        N = 170
        d = d[::5, ::5]

        θ = pm.Normal("θ", [0, 0], [0.1, 0.1], shape=2)
        β = pm.MvNormal(
            "β", mu=θ, cov=np.diagflat(np.array([ 0.01, 0.01])), shape=(N_CLUSTERS, 2)
        )

        η2 = pm.Exponential('η²', 1)
        ρ2 = pm.Exponential('ρ²', 1)
        K = η2 * (at.exp(-ρ2 * at.power(d, 2)) + np.diag([0.01] * N))

        γ = pm.MvNormal("γ", mu=np.zeros(N), cov=K, shape=N)
        μ = β[idx, 0] + β[idx, 1] * W + γ
        σ = pm.HalfNormal("σ", 0.01)
        O = pm.Normal("O", mu=μ, sigma=σ, observed=O_norm[::5])

        prior_2 = pm.sample_prior_predictive(samples=100, random_seed=seed)
        posterior_2 = pm.sample(
            draws=1000,
            tune=1000,
            init="adapt_diag",
            return_inferencedata=True,
            target_accept=0.95,
            random_seed=seed,
        )
        posterior_prediction_2 = pm.sample_posterior_predictive(
            posterior_2, 
            random_seed=seed,
        )

    logger.info("Saving model 2 to netcdf files")

    model_2_dir = model_fp / "model_2"
    model_2_dir.mkdir(exist_ok=True)

    for file in model_2_dir.glob("*"):
        file.unlink(missing_ok=True)

    prior_2.to_netcdf(model_2_dir / "prior")
    posterior_2.to_netcdf(model_2_dir / "posterior")
    posterior_prediction_2.to_netcdf(model_2_dir / "posterior_prediction")

    logger.info("Model 2 saved")

    logger.info("Saving model 2 as plate diagram")
    graph_2 = pm.model_to_graphviz(model_2)
    graph_2.format = "svg"
    graph_2.render(figure_fp / "model_2")
    logger.info("Plate diagram saved")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
