# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
import pymc as pm
import pandas as pd
import geopandas as gpd


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
@click.argument('figure_filepath', type=click.Path())
@click.option('--seed', default=42, type=click.IntRange(0, 1000), help='Seed for pseudorandom elements')
def main(
        input_filepath,
        model_filepath,
        figure_filepath,
        seed,
):
    """ Train models and save them to 'model_filepath'
    """
    logger = logging.getLogger(__name__)
    data_fp = Path(input_filepath)
    model_fp = Path(model_filepath)
    figure_fp = Path(figure_filepath)

    data = gpd.read_file(data_fp / 'spatial_income_1880.gpkg')
    n_clusters = len(data.group.unique())

    pca_transformed = pd.read_csv(data_fp / 'pca_transformed.csv', index_col=0).drop(columns=['geometry'])
    data = data.join(pca_transformed)
    data = data.drop(index=data[data.population < 5].index).dropna().reset_index()

    logger.info('Training model 1')

    with pm.Model() as model_1:
        W = pm.MutableData('W', -data['3'])

        β_P = pm.MvNormal('β_P', np.array([0, 0]), np.array(np.diagflat([0.1, 0.1])), shape=2)

        μ_P = β_P[0] + β_P[1] * W
        σ_P = pm.Exponential('σ_P', 1)
        P = pm.Normal('P', mu=μ_P, sigma=σ_P, observed=-data['1'])

        β_O = pm.MvNormal('β_O', np.array([0, 0, 0]), np.array(np.diagflat([0.1, 0.1, 0.1])), shape=3)

        μ_O = β_O[0] + β_O[1] * W + β_O[2] * μ_P
        σ_O = pm.Exponential('σ_O', 1)
        O = pm.Normal('O', mu=μ_O, sigma=σ_O, observed=data['2'])

        prior_1 = pm.sample_prior_predictive(samples=1000, random_seed=seed)
        posterior_1 = pm.sample(draws=1000, tune=1000, init="adapt_diag", return_inferencedata=True, target_accept=0.9,
                                random_seed=seed)
        posterior_prediction_1 = pm.sample_posterior_predictive(posterior_1, random_seed=seed)

    logger.info('Saving model 1 to netcdf files')

    model_1_dir = model_fp / 'model_1'
    model_1_dir.mkdir(exist_ok=True)

    for file in model_1_dir.glob('*'):
        file.unlink(missing_ok=True)

    prior_1.to_netcdf(model_1_dir / 'prior')
    posterior_1.to_netcdf(model_1_dir / 'posterior')
    posterior_prediction_1.to_netcdf(model_1_dir / 'posterior_prediction')

    logger.info('Model 1 saved')

    logger.info('Saving model 1 as plate diagram')
    graph_1 = pm.model_to_graphviz(model_1)
    graph_1.format = 'svg'
    graph_1.render(figure_fp / 'model_1')
    logger.info('Plate diagram saved')

    logger.info('Training model 2')
    with pm.Model() as model_2:
        W = pm.MutableData('W', -data['3'])
        idx = data.group

        θ_P = pm.MvNormal('θ_P', np.array([0, 0]), np.array(np.diagflat([0.1, 0.1])), shape=2)
        θ_O = pm.MvNormal('θ_O', np.array([0, 0, 0]), np.array(np.diagflat([0.1, 0.1, 0.01])), shape=3)

        β_P = pm.MvNormal('β_P', θ_P, np.array(np.diagflat([0.01, 0.01])), shape=(n_clusters, 2))
        β_O = pm.MvNormal('β_O', θ_O, np.array(np.diagflat([0.01, 0.01, 0.01])), shape=(n_clusters, 3))

        μ_P = β_P[idx, 0] + β_P[idx, 1] * W
        σ_P = pm.Exponential('σ_P', 1)
        P = pm.Normal('P', mu=μ_P, sigma=σ_P, observed=-data['1'])

        μ_O = β_O[idx, 0] + β_O[idx, 1] * W + β_O[idx, 2] * μ_P
        σ_O = pm.Exponential('σ_O', 1)
        O = pm.Normal('O', mu=μ_O, sigma=σ_O, observed=data['2'])

        prior_2 = pm.sample_prior_predictive(samples=1000, random_seed=seed)
        posterior_2 = pm.sample(draws=1000, tune=1000, init="adapt_diag", return_inferencedata=True, target_accept=0.9,
                                random_seed=seed)
        posterior_prediction_2 = pm.sample_posterior_predictive(posterior_2, random_seed=seed)

    logger.info('Saving model 2 to netcdf files')

    model_2_dir = model_fp / 'model_2'
    model_2_dir.mkdir(exist_ok=True)

    for file in model_2_dir.glob('*'):
        file.unlink(missing_ok=True)

    prior_2.to_netcdf(model_2_dir / 'prior')
    posterior_2.to_netcdf(model_2_dir / 'posterior')
    posterior_prediction_2.to_netcdf(model_2_dir / 'posterior_prediction')

    logger.info('Model 2 saved')

    logger.info('Saving model 2 as plate diagram')
    graph_2 = pm.model_to_graphviz(model_2)
    graph_2.format = 'svg'
    graph_2.render(figure_fp / 'model_2')
    logger.info('Plate diagram saved')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
