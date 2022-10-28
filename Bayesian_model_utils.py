# Util functions for Bayesian modeling of recall/HFA data

import arviz as az
import bambi as bmb
import argparse
from os.path import join
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import matplotlib.pyplot as plt 

# import pymc3 as pm
# import pymc3.sampling_jax
# import theano

def run_model(df, y=None, X=None, Intx=None, rand_effect=None, rand_slopes=False, priors=None, categorical=None,
              cores=2, chains=2, tune=1500, draws=2000, target_accept=0.95, model_fam='bernoulli', 
              output_dir='/home1/salman.qasim/Salman_Project/FR_Emotion/BayesModels', return_model=False,
              categorical_baseline=True, save_model_res = True, label=None):
    """
    Run the model: 
    
    Parameters
    ----------
    df: pandas dataframe 
        All the variables 
    y: str
        Name of the dependent variable
    X: list of strings 
        Names of the independent variables 
    Intx: list of strings
        Names of the interactions, formatted as 'a:b' 
    rand_effect: str
        Name of variable for random effects (i.e. 'subject') 
    rand_slopes: bool 
        Should we fit random slopes in addition to random intercepts? 
    priors: dict 
        Prior distribution for any independent variables (i.e. Intercept, random intercept) 
        Default to weakly informative priors (Gelman et al. 2008) 
    categorical: list of strings 
        Names of categorical variables going into model 
    
    Returns
    -------
    results: model
    
    """
    
    # set seed
#     SEEDS = [7, 8, 1111, 42]
#     SEEDS = [1, 2, 1111, 42]
# For use with stim HippOnly on behavior
#     SEEDS = [12, 19, 11, 41]
    
    # if no label provided, make your own
    if not label: 
        label = (f"{y}" + "_{}"*len(X)).format(*X)
        
    
    if categorical_baseline: # this means we set a baseline for categorical effects 
        # format random effects properly 
        rand_term = [f'(1|{x})' for x in rand_effect]
        if rand_slopes: # do we want a more complex mixed-effects model with random slopes as well as intercepts?
            formula = f'{y} ~ 1+'+'+'.join(rand_term)+'+'+'+'.join(X)+'+'+'+'.join([f'({x}|{rand_effect})' for x in X])+'+'+'+'.join(Intx)+'+'+'+'.join([f'({intx}|{rand_effect})' for intx in Intx])
        else:
            if not Intx: 
                formula = f'{y} ~ 1+'+'+'.join(rand_term)+'+'+'+'.join(X)
            else:
                formula = f'{y} ~ 1+'+'+'.join(rand_term)+'+'+'+'.join(X)+'+'+'+'.join(Intx)
    else: # this means all levels of categorical effect are judged absent or present 
         # format random effects properly 
        rand_term = [f'(0+1|{x})' for x in rand_effect]
        if rand_slopes: 
            formula = f'{y} ~ 0+'+'+'.join(rand_term)+'+'+'+'.join(X)+'+'+'+'.join([f'(0 + {x}|{rand_effect})' for x in X])+'+'+'+'.join(Intx)+'+'+'+'.join([f'(0 + {intx}|{rand_effect})' for intx in Intx])
        else:
            if not Intx:
                formula = f'{y} ~ 0+'+'+'.join(rand_term)+'+'+'+'.join(X)
            else:
                formula = f'{y} ~ 0+'+'+'.join(rand_term)+'+'+'+'.join(X)+'+'+'+'.join(Intx)        

    # construct the model 
    model = bmb.Model(formula=formula, 
                  data=df[rand_effect + [y] + X],
                 family=model_fam,
                 priors=priors,
                 categorical=categorical)
    
    # Future steps for speedup:
    
#     if jax: 
#         model.build() # before fitting using bambi get pymc backend model 
#         results =  model.backend.model.sampling_jax.sample_numpyro_nuts(draws, tune=tune, target_accept=target_accept)
#     else:

    # fit the model 
    results=model.fit(cores=cores, 
                          chains=chains, 
                          tune=tune, 
                          draws=draws, 
                          target_accept=target_accept,
                          init='advi+adapt_diag')

    # save out the CSV with the results 
    summary = az.summary(results, hdi_prob=0.95)
    for var in summary.index.values:
        if var.endswith("]"):  # skip individual subject parameters
            continue
        if var.startswith("1"):  # skip random effects.
            continue
        summary.loc[var, "P>0"] = np.mean(results.posterior[var].values > 0)
    summary.to_csv(join(output_dir , f"{label}_summary.csv"))

    # save out the HDI plot of all the fixed effects 
    axes = az.plot_forest(results,
                           kind='ridgeplot',
                           var_names=[f'^{x}' for x in X],
                           filter_vars="regex",
                           colors='black',
                           combined=True,
                           hdi_prob=0.95,
                           figsize=(9, 7))
    plt.vlines(0, plt.ylim()[0], plt.ylim()[1], color = 'black')
    # forestplot
#         plt.xlim([-0.2, 0.1])

    plt.savefig(join(output_dir, f'{label}_HDIplot.pdf'), dpi=300)
    plt.close()

    # print out the latex table for the cell output
    results_df = summary.loc[:,
                   ['mean', 'sd', 'hdi_2.5%', 'hdi_97.5%']
                  ]
    all_tests = [] 

    for x in X: 
        temp = results_df.filter(like=f'{x}', axis=0)
        all_tests.append(temp)

    results_df = pd.concat(all_tests)

    c_string = '|c'*results_df.shape[0] + '|'
    print(results_df.reset_index().to_latex(index=False, 
                                            column_format=c_string).replace("\\\n", "\\ \hline\n"))  

    if save_model_res:
        # save out the model for model comparison (THIS WILL BE GB-sized, so really only needed for model comparison)
        az.to_netcdf(results, join(output_dir , f"{label}_model"))
    
    if return_model:
        return model, results 
    
    
def plot_res(fitted_model, X, save_label=None): 
    """
    Code for plotting select beta weights in a publication friendly manner from saved csv
    """

    axes = az.plot_forest(fitted_model,
                           kind='ridgeplot',
                           var_names=[f'^{x}' for x in X],
                           filter_vars="regex",
                           colors='black',
                           combined=True,
                           hdi_prob=0.95,
                           figsize=(9, 7))
    # 'forestplot'
    plt.vlines(0, plt.ylim()[0], plt.ylim()[1], color = 'black')
    plt.savefig(save_label, dpi=300)
    plt.close()

def print_latex_table(fitted_model, X):
    """
    Code for printing the pub-friendly table of model details from model csv table
    """
    summary = az.summary(fitted_model, hdi_prob=0.95)
    
    # print out the latex table for the cell output
    results_df = summary.loc[:,
                   ['mean', 'sd', 'hdi_2.5%', 'hdi_97.5%']
                  ]
    all_tests = [] 
    for x in X: 
        temp = results_df.filter(like=f'{x}', axis=0)
        all_tests.append(temp)

    results_df = pd.concat(all_tests)

    c_string = '|c'*results_df.shape[0] + '|'
    print(results_df.reset_index().to_latex(index=False, 
                                            column_format=c_string).replace("\\\n", "\\ \hline\n"))  
    
def plot_predictions(df, model, fitted_model, X, y='recalled',  
                    save_dir='/home1/salman.qasim/Salman_Project/FR_Emotion/BayesModels'):
    """
    Code for plotting the model predictions 
    """
    
    # sample from posterior-predictive distribution to make in-sample predictions 
    model.predict(fitted_model, kind="pps")
    y_posterior = fitted_model.posterior[f"{y}_mean"].stack(samples=("chain", "draw")).values
    # Select 25% of the values in the posterior, making sure we take values from both chains.
    y_posterior = recall_posterior[:, ::4]

    for x in X:
        save_file = join(save_dir, f'{x}_PredictRecallplot.pdf')
        with PdfPages(save_file) as pdf:
            f, ax = plt.subplots(1, 1, figsize=[3,3], dpi=300)
            for pps in y_posterior.T:
                sns.regplot(data=df, x=x, y=pps, ci=None, scatter=False, line_kws={'alpha':0.1}, color='k')
            sns.regplot(data=df, x=x, y=y_posterior.mean(axis=1), ci=None, scatter=False, color='r')
            pdf.savefig()
            plt.close()