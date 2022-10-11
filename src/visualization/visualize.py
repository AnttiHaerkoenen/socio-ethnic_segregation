# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import arviz as az
import matplotlib.pyplot as plt


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
@click.argument('figure_filepath', type=click.Path())
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

    model_1_dir = model_fp / 'model_1'
    prior_1 = az.InferenceData.from_netcdf(model_1_dir / 'prior')
    posterior_1 = az.InferenceData.from_netcdf(model_1_dir / 'posterior')
    posterior_prediction_1 = az.InferenceData.from_netcdf(model_1_dir / 'posterior_prediction')

    model_2_dir = model_fp / 'model_2'
    prior_2 = az.InferenceData.from_netcdf(model_2_dir / 'prior')
    posterior_2 = az.InferenceData.from_netcdf(model_2_dir / 'posterior')
    posterior_prediction_2 = az.InferenceData.from_netcdf(model_2_dir / 'posterior_prediction')

    logger.info('Saving posterior predictive checks')
    ppc_1 = az.plot_ppc(posterior_prediction_1)
    plt.savefig(figure_fp / 'model_1_posterior_predictive_check.png', dpi=300)
    ppc_2 = az.plot_ppc(posterior_prediction_2)
    plt.savefig(figure_fp / 'model_2_posterior_predictive_check.png', dpi=300)

    logger.info('Plotting posterior distributions')
    az.plot_posterior(posterior_2, var_names='β_O', grid=(12, 3), figsize=(20, 20))
    plt.savefig(figure_fp / 'model_2_posterior_beta.png', dpi=300)

    logger.info('Plotting trace plot')
    az.plot_trace(posterior_2)
    plt.savefig(figure_fp / 'model_2_trace.png', dpi=300)

    logger.info('Plotting forest plot of the posterior')
    az.plot_forest(posterior_2, combined=True, hdi_prob=0.94)
    plt.savefig(figure_fp / 'model_2_forest_plot', dpi=300)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
