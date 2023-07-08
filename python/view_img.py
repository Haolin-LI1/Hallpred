import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import urllib
import seaborn as sns
import pandas as pd
from scipy import stats
from diptest import diptest

def bar_plot(gen_mutation,dataset,threshold,figsize=(11.2,5),color='skyblue',edgecolor='black',title=None,xlabel=None,ylabel=None):

    """ 
    ================================== [ Introduction ] ======================================

    This function is used to generate a bar plot. Visually compare the predicted ddg value for
    each variation.
    
    ===================================== [ Input ] ==========================================

    gen_mutation:
        A list of the corresponding variation type of each dataset.

    dataset:
        A list of the predictive ddg value of each variation.
    
    threshold:
        This threshold determines whether a variation is at risk of causing a protein misfolding.

    figsize: 
        The figure size of bar plot. The default size is set as (11.2,5).

    color: 
        The color of bar. The default color is set as 'skyblue'.
    
    edgecolor:
        The color of the edge of bar. The default color is set as 'black'.

    title:
        The title of plot. The default title is set as None.
    
    xlabel:
        The x-label of plot. The default label is set as None.

    ylabel:
        The y-label of plot. The default label is set as None.
    
    ===================================== [ Output ] =========================================

    img_url:
        The url of bar plot. It is used to pass the plot to the front-end django page.

    ===========================================================================================
    """

    plt.figure(figsize=figsize)
    plt.barh(gen_mutation,dataset,color=color,edgecolor=edgecolor)
    plt.axvline(x=threshold, color='r', linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xlim([min(dataset) - 0.1, max(2.6,max(dataset) + 0.1)])

    plt.legend([f'Threshold = {threshold} (kcal/mol)'])

    img_url = save_encode_img_base64()

    return img_url

def generate_kdeplot(figsize,dataset,fill=True,color='royalblue',title='Density Plot',xlabel=None,ylabel=None):

    """ 
    ================================== [ Introduction ] ======================================

    This function is used to generate a kernel density estimate plot. Visualize the distribution
    of dataset.
    
    ===================================== [ Input ] ==========================================

    figsize: 
        The figure size of kde plot.

    dataset:
        Gene mutation dataset of ddg value. Type is QuerySet and obtained from database.
    
    fill:
        The figure will be filled in with a lighter color. The default flag is True.

    color: 
        The color of plot. The default color is set as 'royalblue'.

    title:
        The title of plot. The default title is set as 'Density Plot'.
    
    xlabel:
        The x-label of plot. The default label is set as None.

    ylabel:
        The y-label of plot. The default label is set as None.
    
    ===================================== [ Output ] =========================================

    img_url:
        The url of kde plot. It is used to pass the plot to the front-end django page.

    ===========================================================================================
    """

    plt.figure(figsize=figsize)
    sns.kdeplot(dataset, fill=fill, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    img_url = save_encode_img_base64()
    
    return img_url

def generate_boxplot(figsize,dataset,color='cornflowerblue',title='Box Plot',xlabel=None,ylabel=None,stripflag=True):

    """ 
    ================================== [ Introduction ] ======================================

    This function is used to generate a boxplot. Visualize the statistical analysis of dataset.
    
    ===================================== [ Input ] ==========================================

    figsize: 
        The figure size of box plot.

    dataset:
        Gene mutation dataset of ddg value. Type is QuerySet and obtained from database.
    
    color: 
        The color of box. The default color is set as 'cornflowerblue'.

    title:
        The title of plot. The default title is set as 'Box Plot'.
    
    xlabel:
        The x-label of plot. The default label is set as None.

    ylabel:
        The y-label of plot. The default label is set as None.
    
    stripflag:
        Draw each specific data on the boxplot. The default flag is True.
    
    ===================================== [ Output ] =========================================

    img_url:
        The url of box plot. It is used to pass the plot to the front-end django page.

    ===========================================================================================
    """
    
    plt.figure(figsize=figsize)
    sns.boxplot(dataset,color=color,width=0.45)
    plt.axhline(np.mean(dataset), color='firebrick', linestyle='dashed', linewidth=1, label='Mean')
    plt.legend()
    if stripflag:
        sns.stripplot(data=dataset, color="orange", jitter=0.2, size=2.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    img_url = save_encode_img_base64()
    
    return img_url

def generate_cdfplot(figsize,dataset,threshold,title='Cumulative Distribution Function',xlabel='ddg (kcal/mol)',ylabel='Probability Density'):

    """ 
    ================================== [ Introduction ] ======================================

    This function is used to generate a cumulative distribution function plot. Visualize the
    relationship between the data set and the ddg threshold.
    
    ===================================== [ Input ] ==========================================

    figsize: 
        The figure size of cdf plot.

    dataset:
        Gene mutation dataset of ddg value. Type is QuerySet and obtained from database.
    
    threshold:
        This threshold determines whether a variation is at risk of causing a protein misfolding.

    title:
        The title of plot. The default title is set as 'Cumulative Distribution Function'.
    
    xlabel:
        The x-label of plot. The default label is set as 'ddg (kcal/mol)'.

    ylabel:
        The y-label of plot. The default label is set as 'CDF'.
    
    ===================================== [ Output ] =========================================

    img_url:
        The url of cdf plot. It is used to pass the plot to the front-end django page.
    
    probability:
        The percentage of the dataset that exceeds the threshold.

    ===========================================================================================
    """

    combined_ddgs_array = np.array(dataset)
    len_combined_ddgs = len(combined_ddgs_array)
    combined_ddgs_array.sort()
    cumulative_probability = 1. * np.arange(len_combined_ddgs) / (len_combined_ddgs - 1)
    
    probability = np.sum(combined_ddgs_array > threshold) / len_combined_ddgs

    plt.figure(figsize=figsize)
    plt.plot(combined_ddgs_array, cumulative_probability)

    y_corresponding_threshold = np.interp(threshold, combined_ddgs_array, cumulative_probability)
    
    plt.fill_between(combined_ddgs_array, y_corresponding_threshold, cumulative_probability, where=(cumulative_probability>=y_corresponding_threshold), interpolate=True, color='red', alpha=0.3, label=f'Prob > threshold = {probability:.2%}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    img_url = save_encode_img_base64()

    return img_url,probability

def generate_pieplot(figsize,dataset,labels,title=None,xlabel=None,ylabel=None):

    """ 
    ================================== [ Introduction ] ======================================

    This function is used to generate a pie plot.
    
    ===================================== [ Input ] ==========================================

    figsize: 
        The figure size of cdf plot.

    dataset:
        A list of the number of each label.
    
    labels:
        A list of labels.

    title:
        The title of plot. The default title is set as None.
    
    xlabel:
        The x-label of plot. The default label is set as None.

    ylabel:
        The y-label of plot. The default label is set as None.
    
    ===================================== [ Output ] =========================================

    img_url:
        The url of pie plot. It is used to pass the plot to the front-end django page.
    
    ===========================================================================================
    """

    plt.figure(figsize=figsize)
    plt.pie(dataset, labels=labels, autopct='%1.1f%%')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    img_url = save_encode_img_base64()

    return img_url

def generate_heatmap(dataset,figsize=(12,10),title='Heatmap',xlabel='mut_from',ylabel='mut_to'):

    """ 
    ================================== [ Introduction ] ======================================

    This function is used to generate a heatmap. Visualize the relationship between mutants
    and mutated genes in a protein.
    
    ===================================== [ Input ] ==========================================

    dataset:
        Gene mutation dataset of ddg value. Type is QuerySet and obtained from database.

    figsize: 
        The figure size of heatmap. The default size is set as (12,10).

    title:
        The title of plot. The default title is set as 'Heatmap'.
    
    xlabel:
        The x-label of plot. The default label is set as 'mut_from'.

    ylabel:
        The y-label of plot. The default label is set as 'mut_to'.
    
    ===================================== [ Output ] =========================================

    img_url:
        The url of heatmap. It is used to pass the plot to the front-end django page.
    
    ===========================================================================================
    """
    
    df = pd.DataFrame.from_records(dataset)

    avg_ddg_df = df.groupby(['mut_from', 'mut_to']).ddg.mean().reset_index()

    heatmap_data = avg_ddg_df.pivot(index='mut_from', columns='mut_to', values='ddg')

    plt.figure(figsize=figsize)
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    img_url = save_encode_img_base64()

    return img_url

def save_encode_img_base64(fig_flag=False,fig=None):

    """ 
    ================================== [ Introduction ] ======================================

    This function is used to save the plot and encode it as the type base64.
    
    ===================================== [ Input ] ==========================================

    fig_flag:
        Set the flag as False when processing the plot of single variation. Set the flag as
        True when processing the plot of multiple variation.

    fig: 
        When the fig_flag is True. Add the graphic parameter for the subplot. For example, when
        the figure is built from 'fig1,ax = plt.subplot()' but not 'plt.figure()', set the
        'fig_flag' as True and 'fig' as fig1.
    
    ===================================== [ Output ] =========================================

    img_url:
        The url of figure. It is used to pass the plot to the front-end django page.
    
    ===========================================================================================
    """

    img = io.BytesIO()
    if not fig_flag:
        plt.savefig(img, format='png')
    else:
        fig.savefig(img, format='png')
    plt.close()
    img.seek(0)
    img = img.read()
    encoded_img = base64.b64encode(img)
    img_url = urllib.parse.quote(encoded_img.decode())

    return img_url

def cal_ratio_generate_pie(figsize,value):

    """ 
    ================================== [ Introduction ] ======================================

    This function is used to calculate and visualize the distribution of a dataset group with 
    multiple datasets. Distributions include reference predictive ddg, normal distribution
    and modal case.
    
    ===================================== [ Input ] ==========================================

    figsize: 
        The figure size of pie plot.

    value:
        The 'value' is a dictionary. The key of the dictionary is the specific information for
        each variation and the value is the corresponding ddg dataset.
    
    ===================================== [ Output ] =========================================

    pie_distribution_plot:
        The url of pie plot depicting the normal distribution of the dataset group. It is used
        to pass the plot to the front-end django page.
    
    pie_modal_plot:
        The url of pie plot depicting the modal case of the dataset group. It is used to pass
        the plot to the front-end django page.

    pie_threshold_plot:
        The url of pie plot depicting the predictive ddg of the dataset group. It is used to
        pass the plot to the front-end django page.

    ===========================================================================================
    """

    # initialize parameters
    threshold_over=0
    threshold_down=0
    distribution_normal=0
    distribution_nonnormal=0
    distribution_small=0
    nummodal_multi=0
    nummodal_uni=0
    nummodal_small=0

    # analysize dataset group
    for _, ddg_values in value.items():
        mean_ddg = np.mean(ddg_values)
        if mean_ddg > 2.5:
            # predictive ddg over the threshold
            threshold_over += 1
        else:
            # predictive ddg below the threshold
            threshold_down += 1

        if len(ddg_values)>3:

            _, p_normal = stats.shapiro(ddg_values)

            ddg_modal_array = np.array(ddg_values)
            _,p_multi = diptest(ddg_modal_array)

            # classify according to the type of data distribution
            if p_normal > 0.05:
                # normal distribution
                distribution_normal += 1
            else:
                # non-normal distribution
                distribution_nonnormal += 1

            # classify according to the type of modal
            if p_multi < 0.05:
                # multimodal
                nummodal_multi += 1
            else:
                # unimodal
                nummodal_uni += 1
        else:
            distribution_small += 1
            nummodal_small += 1

    pie_distribution_plot = generate_pieplot(figsize,[distribution_normal,distribution_nonnormal,distribution_small],['Normal', 'Non-normal', 'Small-sample'],title='Distribution')
    pie_modal_plot = generate_pieplot(figsize,[nummodal_uni,nummodal_multi,nummodal_small],['Unimodal', 'Multimodal', 'Small-sample'],title='Modal')
    pie_threshold_plot = generate_pieplot(figsize,[threshold_over,threshold_down],['Over_threshold', 'Down_threshold'],title='DDG Value')

    return pie_distribution_plot,pie_modal_plot,pie_threshold_plot