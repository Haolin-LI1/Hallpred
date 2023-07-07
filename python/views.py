from django.shortcuts import render
from django.forms import formset_factory
import numpy as np
from .forms import QueryForm,ShowForm
from . import view_algorithm,view_img
import matplotlib.pyplot as plt
import seaborn as sns
from myapp import models
from scipy import stats
import pandas as pd
from collections import defaultdict
import matplotlib.patches as mpatches
from diptest import diptest

def query(request,figsize_normal=(4,4),figsize_large=(5, 4),threshold = 2.5,n_iterations = 5000):
    
    """ 
    ============================== [ Superposition Analysis ] ================================

    This function is used to match the variation entered by the user with the database, and
    extract the corresponding ddg value to process. When multiple sets of variations are
    received, it will analyze the superposition effect. Once the analysis is complete, the
    variables will be passed to the front-end of the django page to generate a graphical page
    for the user to view the results.
    
    ===================================== [ Input ] ==========================================

    request: 
        The input format of user is "protein name" + "gene mutation from" + "position" + 
        "gene mutation to" (each item is separated by dot ".", each gene mutation is separated
        by comma ","). For example, 'pik3ca.PRO.3.CYS, abl1.GLY.1.ALA, abl1.LEU.6.VAL'. The type
        of request is 'POST'.

    figsize_normal:
        The normal figure size is set as (4,4).
    
    figsize_large:
        The large figure size is set as (5,4).
    
    threshold:
        This threshold determines whether a variation is at risk of causing a protein misfolding.
        The default threshold is set as 2.5 kcal/mol.
    
    n_iterations:
        The number of resampling iterations of superposition analysis. The default value is set
        as 5000.
    
    ===================================== [ Output ] =========================================

    superposition_result.html:
        The analysis results will be sent to the django front-end page.

    query.html:
        When the system detects the user's incorrect input, it will return to the variation
        input page.

    ===========================================================================================
    """

    QueryFormSet = formset_factory(QueryForm, extra=1)

    if request.method == 'POST':

        formset = QueryFormSet(request.POST)

        # initialize parameters
        single_gm_set = [] # save the analysis of each variation
        single_gm_count = 0 # initial the number of variation
        ddgs_set = []
        bar_ref_name = []
        bar_ref_value = []

        density_fig, density_ax = plt.subplots(figsize=figsize_large)
        box_fig, box_ax = plt.subplots(figsize=figsize_large)
        combined_box_set = {}
        color_pin = 0 # initial the color pin of the boxplot and kdeplot to distinguish different variations.
        
        if formset.is_valid():
            for form in formset:
                try:
                    query_inputs = form.cleaned_data['query_input'].split(',')
                except ValueError:
                    form.add_error('query_input', "The input format is incorrect. The correct format is 'Protein.mut_from.pdb_rid.mut_to' and use commas '，' to separate multiple queries.")
                    return render(request, 'query.html', {'formset': formset})
                except Exception as error:
                    print(error)
                    form.add_error('query_input', "The input format is incorrect. The correct format is 'Protein.mut_from.pdb_rid.mut_to' and use commas '，' to separate multiple queries.")
                    return render(request, 'query.html', {'formset': formset})

                for query_input in query_inputs:
                    try:
                        # extract variation information from user input
                        gene_mutation_name = query_input.strip()
                        protein, mut_from, pdb_rid, mut_to = gene_mutation_name.split('.')
                        pdb_rid = int(pdb_rid)
                    except ValueError:
                        form.add_error('query_input', "The input format is incorrect. The correct format is 'Protein.mut_from.pdb_rid.mut_to' and use commas '，' to separate multiple queries.")
                        return render(request, 'query.html', {'formset': formset})
                    except Exception as error:
                        print(error)
                        form.add_error('query_input', "The input format is incorrect. The correct format is 'Protein.mut_from.pdb_rid.mut_to' and use commas '，' to separate multiple queries.")
                        return render(request, 'query.html', {'formset': formset})
                    
                    protein = protein.lower()
                    mut_from = mut_from.upper()
                    mut_to = mut_to.upper()

                    try:
                        # match the variation with the database and extract ddg value
                        model_name = f'GeneList{protein[0].upper()}'
                        GeneList = getattr(models, model_name)

                        gene_list_data = GeneList.objects.filter(
                            source__exact=protein, 
                            mut_from__exact=mut_from, 
                            pdb_rid=pdb_rid, 
                            mut_to__exact=mut_to
                        ).values_list('ddg', flat=True)
                    except ValueError:
                        form.add_error('query_input', f"No query mutation {gene_mutation_name}，Please click on 'protein' on the home page to see the list of included proteins")
                        return render(request, 'query.html', {'formset': formset})
                    except Exception as error:
                        print(error)
                        form.add_error('query_input', f"No query mutation {gene_mutation_name}，Please click on 'protein' on the home page to see the list of included proteins")
                        return render(request, 'query.html', {'formset': formset})

                    if not gene_list_data:
                        form.add_error('query_input', f"No query mutation {gene_mutation_name}，Please click on 'protein' on the home page to see the list of included proteins")
                        return render(request, 'query.html', {'formset': formset})
                    elif gene_list_data:
                        single_gm = {}

                        single_gm_count += 1

                        gene_data_len = len(gene_list_data)

                        if gene_data_len > 3:
                            """
                            calculate the predictive value of ddg and 95% confidence interval.

                            when length of dataset > 3:
                            use the algorithm: view_algorithm.pred_result.

                            when length of dataset == 2 or 3:
                            use the mean value as the predictive value of ddg,
                            use the minimum value as the lower bound of confidence interval,
                            use the maximum value as the upper bound of confidence interval.

                            when length of dataset == 1:
                            use the unique data as the predictive value of ddg,
                            use the data ±5% as the confidence interval.

                            """
                            reference_value, lower_cl, upper_cl = view_algorithm.pred_result(gene_list_data)
                            _, sp_p = stats.shapiro(gene_list_data) # Shaprio-Wilk Test
                            skewness = stats.skew(gene_list_data) # skewness calculation
                            gene_list_array = np.array(gene_list_data)
                            _,hdt_p = diptest(gene_list_array) # Hartigans Dip Test
                            single_gm['sp_p'] = "{:.3e}".format(sp_p) # use scientific notation to represent floating-point numbers
                            single_gm['sp_p_data'] = sp_p
                            single_gm['skewness'] = "{:.3e}".format(skewness)
                            single_gm['hdt_p'] = "{:.3e}".format(hdt_p)
                            single_gm['sp_hdt_data'] = hdt_p
                        elif gene_data_len == 1:
                            reference_value = gene_list_data[0]
                            lower_cl = min(0.95*gene_list_data[0],1.05*gene_list_data[0])
                            upper_cl = max(0.95*gene_list_data[0],1.05*gene_list_data[0])
                        else:
                            lower_cl = np.min(gene_list_data)
                            upper_cl = np.max(gene_list_data)
                            reference_value = np.mean(gene_list_data)

                        bar_ref_name.append(gene_mutation_name)
                        if isinstance(reference_value, list):
                            bar_ref_value.append(reference_value[0])
                        else:
                            bar_ref_value.append(reference_value)

                        reference_std = np.std(gene_list_data)

                        # generate a kdeplot to show the ddg value of different variations
                        sns.kdeplot(gene_list_data,  fill=True, label=f'{protein}.{mut_from}.{pdb_rid}.{mut_to}', ax=density_ax, color=sns.color_palette()[color_pin % 8])
                        combined_box_set[f'{protein}.{mut_from}.{pdb_rid}.{mut_to}'] = gene_list_data

                        color_pin += 1
                        
                        # generate a kdeplot of single variation
                        single_gm_kde = view_img.generate_kdeplot(figsize_normal,gene_list_data,xlabel='ddg/(kcal/mol)')
                        single_gm['density_plot'] = single_gm_kde

                        # generate a boxplot of single variation
                        single_gm_box = view_img.generate_boxplot(figsize_normal,gene_list_data,ylabel='ddg/(kcal/mol)')
                        single_gm['box_plot'] = single_gm_box

                        # generate a cdfplot of single variation
                        single_gm_cdf,single_gm_probability = view_img.generate_cdfplot(figsize_normal,gene_list_data,2.5,xlabel='ddg/(kcal/mol)')
                        single_gm['cdf_plot'] = single_gm_cdf
                        single_gm['probability'] = round(single_gm_probability*100,2)

                        single_gm['protein'] = protein
                        single_gm['mut_from'] = mut_from
                        single_gm['pdb_rid'] = pdb_rid
                        single_gm['mut_to'] = mut_to
                        if isinstance(reference_value, list):
                            single_gm['reference_value_high'] = reference_value[0]
                            single_gm['reference_percentage_high'] = reference_value[1]
                            single_gm['reference_value_low'] = reference_value[2]
                            single_gm['reference_percentage_low'] = reference_value[3]
                        else:
                            single_gm['reference_value'] = reference_value
                            
                        single_gm['reference_std'] = round(reference_std,5)
                        single_gm['lower_cl'] = round(lower_cl,5)
                        single_gm['upper_cl'] = round(upper_cl,5)

                        if gene_data_len == 1:
                            gene_list_data = [0.95*gene_list_data[0], gene_list_data[0], 1.05*gene_list_data[0]]
                            ddgs_set.append(gene_list_data)
                        else:
                            ddgs_set.append(gene_list_data)

                        single_gm_set.append(single_gm)

            if len(single_gm_set) == 0:
                form.add_error('query_input', "The input format is incorrect. The correct format is 'Protein.mut_from.pdb_rid.mut_to' and use commas '，' to separate multiple queries.")
                return render(request, 'query.html', {'formset': formset})
            else:
                if single_gm_count > 1:
                    # User inputs multiple variations
                    combined = {}
                    combined_img = {}

                    combined_ddgs = []
                    # calculate the superposition effect of multiple variations
                    for _ in range(n_iterations):
                        samples = [np.random.choice(gene_list) for gene_list in ddgs_set]
                        superposition_num = np.mean(samples)
                        combined_ddgs.append(superposition_num)
                    
                    _, sp_p = stats.shapiro(combined_ddgs)
                    skewness = stats.skew(combined_ddgs)
                    
                    combined_list_array = np.array(combined_ddgs)
                    _,hdt_p = diptest(combined_list_array)
                    combined['combined_sp_p'] = "{:.3e}".format(sp_p)
                    combined['combined_sp_p_data'] = sp_p
                    combined['combined_skewness'] = "{:.3e}".format(skewness)
                    combined['combined_hdt_p'] = "{:.3e}".format(hdt_p)
                    combined['combined_hdt_p_data'] = hdt_p
                    
                    combined_mean = np.mean(combined_ddgs)
                    combined_std = np.std(combined_ddgs)
                    combined_se = stats.sem(combined_ddgs)
                    combined_lower_cl, combined_upper_cl = stats.t.interval(0.95, len(combined_ddgs)-1, loc=combined_mean, scale=combined_se)

                    density_ax.set_title('density plot')
                    density_ax.set_xlabel('ddg (kcal/mol)')
                    density_ax.set_ylabel('Density')
                    density_fig.legend()
                    density_plot_url = view_img.save_encode_img_base64(fig_flag=True,fig=density_fig)

                    # generate a boxplot to show the data distribution of different variations
                    combined_box_list = []
                    for variation_name, ddgs in combined_box_set.items():
                        combined_box_list.extend([{'Variation': variation_name, 'ddg': ddg} for ddg in ddgs])

                    df = pd.DataFrame(combined_box_list)

                    color_palette = sns.color_palette(n_colors=single_gm_count)
                    colors = [color_palette[i % single_gm_count] for i in range(single_gm_count)]
                    indices = range(single_gm_count)

                    sns.boxplot(x="Variation", y="ddg", data=df, ax=box_ax, width=0.3, palette=color_palette)
                    box_ax.set_xticks(indices)
                    box_ax.set_xticklabels([index + 1 for index in indices])
                    box_legend = [mpatches.Patch(color=colors[index], label=level) for index, level in enumerate(df['Variation'].unique())]
                    box_ax.set_ylabel('ddg (kcal/mol)')
                    box_ax.legend(handles=box_legend, title="Gene Mutation", loc="best")
                    box_fig = box_ax.get_figure()
                    box_fig.tight_layout()
                    box_plot_url = view_img.save_encode_img_base64(fig_flag=True,fig=box_fig)

                    # generate some plots of superposition effect
                    superposition_gm_kde = view_img.generate_kdeplot(figsize_normal,combined_ddgs,xlabel='ddg/(kcal/mol)',ylabel='density')
                    combined_img['kde_plot'] = superposition_gm_kde

                    superposition_gm_box = view_img.generate_boxplot(figsize_normal,combined_ddgs,title='Combined Box Plot',xlabel='index',ylabel='ddg/(kcal/mol)',stripflag=False)
                    combined_img['box_plot'] = superposition_gm_box

                    superposition_gm_cdf,probability = view_img.generate_cdfplot(figsize_normal,combined_ddgs,threshold,xlabel='ddg/(kcal/mol)')
                    combined_img['combined_cdf_plot'] = superposition_gm_cdf
                    combined['combined_probability'] = round(probability*100,2)

                    bar_ref_name.append('superposition')
                    bar_ref_value.append(combined_mean)
                    
                    combined_bar_ref_plot = view_img.bar_plot(bar_ref_name,bar_ref_value,2.5,title='Reference Value Analysis',xlabel='ddg/(kcal/mol)')

                    combined['combined_threshold'] = threshold
                    combined['combined_mean'] = round(combined_mean,5)
                    combined['combined_std'] = round(combined_std,5)
                    combined['combined_lower_cl'] = round(combined_lower_cl,5)
                    combined['combined_upper_cl'] = round(combined_upper_cl,5)
                    combined_img['combined_density_plot'] = density_plot_url
                    combined_img['combined_box_plot'] = box_plot_url
                    combined_img['combined_bar_ref_plot'] = combined_bar_ref_plot

                    return render(request, 'superposition_result.html', {'single_gm_set': single_gm_set,'combined':combined,'combined_img':combined_img})
                else:
                    # user inputs single variation
                    return render(request, 'superposition_result.html', {'single_gm_set': single_gm_set})
        else:
            # the error mechanism to process invaild formset
            form.add_error('query_input', "The input format is incorrect. The correct format is 'Protein.mut_from.pdb_rid.mut_to' and use commas '，' to separate multiple queries.")
            return render(request, 'query.html', {'formset': formset})
    else:
        # the error mechanism to process non-POST request
        formset = QueryFormSet()
    return render(request, 'query.html', {'formset': formset})

def show(request,figsize_normal = (4,3)):

    """ 
    ========================== ==== [ Systematic Analysis ] ==================================

    This function is used to process user input for systematic analysis of proteins (required)
    and mutation sources (optional). The system will match the data in the database according
    to the information entered by the user, and give a systematic analysis of the predictive
    ddg value, normal distribution case, and modal case. When the user enters only the protein
    type, the system will additionally analyze the relationship between all the mutated genes
    of that protein.
    
    ===================================== [ Input ] ==========================================

    request: 
        The input format of user is  "protein name"(required) + "gene mutation from"(optional)
        + "gene mutation to"(optional). For example, protein:pik3ca, mut from:PRO, mut to:CYS.

    figsize_normal:
        The normal figure size is set as (4,3).
    
    ===================================== [ Output ] =========================================

    systematic_result.html:
        The analysis results will be sent to the django front-end page.

    show.html:
        When the system detects the user's incorrect input, it will return to the variation
        input page.

    ===========================================================================================
    """

    if request.method == 'POST':
        set_form = ShowForm(request.POST)
        if set_form.is_valid():
            set_statistic = {}
            set_img = {}

            try:
                # extract the user input
                protein = set_form.cleaned_data.get('source').lower()
                mut_from = set_form.cleaned_data.get('mut_from').upper()
                mut_to = set_form.cleaned_data.get('mut_to').upper()

                model_name = f'GeneList{protein[0].upper()}'
                GeneList = getattr(models, model_name)
            except ValueError:
                set_form.add_error(None, f"The input is incorrect. Please check it.")
                return render(request, 'show.html', {'set_form': set_form})
            except Exception as error:
                print(error)
                set_form.add_error(None, f"The input is incorrect. Please check it.")
                return render(request, 'show.html', {'set_form': set_form})

            if protein and mut_from and mut_to:
                # when user inputs all the information of protein, mut_from and mut_to
                try:
                    # match the input information with database
                    setdata = GeneList.objects.filter(source__exact=protein,
                                                    mut_from__exact=mut_from,
                                                    mut_to__exact=mut_to).values('pdb_rid', 'ddg')
                    
                    # build a dictionary to match the value of pdb_rid and ddg
                    ddg_dict = defaultdict(list)

                    for item in setdata:
                        key = (item['pdb_rid'])
                        ddg_dict[key].append(item['ddg'])

                    # generate pie plots to show the distribution case
                    pie_distribution_plot,pie_modal_plot,pie_threshold_plot = view_img.cal_ratio_generate_pie(figsize_normal,ddg_dict)

                    set_statistic['protein'] = protein
                    set_statistic['mut_from'] = mut_from
                    set_statistic['mut_to'] = mut_to

                    set_img['pie_distribution_plot'] = pie_distribution_plot
                    set_img['pie_modal_plot'] = pie_modal_plot
                    set_img['pie_threshold_plot'] = pie_threshold_plot

                except ValueError:
                    set_form.add_error('mut_to', f"The input information is incorrect. Please check it.")
                    return render(request, 'show.html', {'set_form': set_form})
                except Exception as error:
                    print(error)
                    set_form.add_error('mut_to', f"The input information is incorrect. Please check it.")
                    return render(request, 'show.html', {'set_form': set_form})

            elif protein and mut_from and not mut_to:
                # when user inputs the information of protein and mut_from
                try:
                    # match the input information with database
                    setdata = GeneList.objects.filter(source__exact=protein,mut_from__exact=mut_from).values('mut_to','pdb_rid', 'ddg')

                    # build a dictionary to match the value of (pdb_rid,mut_to) and ddg
                    ddg_dict = defaultdict(list)

                    for item in setdata:
                        key = (item['mut_to'], item['pdb_rid'])
                        ddg_dict[key].append(item['ddg'])

                    # generate pie plots to show the distribution case
                    pie_distribution_plot,pie_modal_plot,pie_threshold_plot = view_img.cal_ratio_generate_pie(figsize_normal,ddg_dict)

                    set_statistic['protein'] = protein
                    set_statistic['mut_from'] = mut_from

                    set_img['pie_distribution_plot'] = pie_distribution_plot
                    set_img['pie_modal_plot'] = pie_modal_plot
                    set_img['pie_threshold_plot'] = pie_threshold_plot
                except ValueError:
                    set_form.add_error('mut_to', f"The input information is incorrect. Please check it.")
                    return render(request, 'show.html', {'set_form': set_form})
                except Exception as error:
                    print(error)
                    set_form.add_error('mut_to', f"The input information {mut_from} is incorrect. Please check it.")
                    return render(request, 'show.html', {'set_form': set_form})
                

            elif protein and mut_to and not mut_from:
                # when user inputs the information of protein and mut_to
                try:
                    # match the input information with database
                    setdata = GeneList.objects.filter(source__exact=protein,mut_to__exact=mut_to).values('mut_from','pdb_rid', 'ddg')
                    
                    # build a dictionary to match the value of (pdb_rid,mut_from) and ddg
                    ddg_dict = defaultdict(list)

                    for item in setdata:
                        key = (item['mut_from'], item['pdb_rid'])
                        ddg_dict[key].append(item['ddg'])
                    
                    # generate pie plots to show the distribution case
                    pie_distribution_plot,pie_modal_plot,pie_threshold_plot = view_img.cal_ratio_generate_pie(figsize_normal,ddg_dict)

                    set_statistic['protein'] = protein
                    set_statistic['mut_to'] = mut_to

                    set_img['pie_distribution_plot'] = pie_distribution_plot
                    set_img['pie_modal_plot'] = pie_modal_plot
                    set_img['pie_threshold_plot'] = pie_threshold_plot

                except ValueError:
                    set_form.add_error('mut_to', f"The input information is incorrect. Please check it.")
                    return render(request, 'show.html', {'set_form': set_form})
                except Exception as error:
                    print(error)
                    set_form.add_error('mut_to', f"The input information is incorrect. Please check it.")
                    return render(request, 'show.html', {'set_form': set_form})

            elif protein and not mut_to and not mut_from:
                # when user inputs only the information of protein
                try:
                    # match the input information with database
                    setdata = GeneList.objects.filter(source__exact=protein).values('mut_from', 'mut_to', 'pdb_rid', 'ddg')

                    # generate a heatmap to show the relationship of mut_from and mut_to
                    heatmap_plot = view_img.generate_heatmap(setdata)

                    # build a dictionary to match the value of (pdb_rid,mut_from,mut_to) and ddg
                    ddg_dict = defaultdict(list)

                    for item in setdata:
                        key = (item['mut_from'], item['pdb_rid'], item['mut_to'])
                        ddg_dict[key].append(item['ddg'])

                    # generate pie plots to show the distribution case
                    pie_distribution_plot,pie_modal_plot,pie_threshold_plot = view_img.cal_ratio_generate_pie(figsize_normal,ddg_dict)

                    set_statistic['protein'] = protein
                    set_img['pie_distribution_plot'] = pie_distribution_plot
                    set_img['pie_modal_plot'] = pie_modal_plot
                    set_img['pie_threshold_plot'] = pie_threshold_plot
                    set_img['heatmap_plot'] = heatmap_plot

                except ValueError:
                    set_form.add_error('source', f"The input information {protein} is incorrect. Please check it.")
                    return render(request, 'show.html', {'set_form': set_form})
                except Exception as error:
                    print(error)
                    set_form.add_error('source', f"The input information {protein} is incorrect. Please check it.")
                    return render(request, 'show.html', {'set_form': set_form})

            # generate pie plots to show the distribution case of the whole database
            pie_distribution_plot_ttl = view_img.generate_pieplot(figsize_normal,[2988305,1151681,7368257],['Normal', 'Non-normal', 'Small-sample'],title='Distribution-Total')
            pie_modal_plot_ttl = view_img.generate_pieplot(figsize_normal,[3927711,212275,7368257],['Unimodal', 'Multimodal', 'Small-sample'],title='Modal-Total')
            pie_threshold_plot_ttl = view_img.generate_pieplot(figsize_normal,[3000556,8507687],['Over_threshold', 'Down_threshold'],title='Threshold-Total')

            set_img['pie_distribution_total'] = pie_distribution_plot_ttl
            set_img['pie_modal_total'] = pie_modal_plot_ttl
            set_img['pie_threshold_total'] = pie_threshold_plot_ttl

            return render(request, 'systematic_result.html', {'set_statistic': set_statistic, 'set_img':set_img})
        else:
            # the error mechanism to process invaild form
            return render(request, 'show.html', {'set_form': set_form})
    else:
        # the error mechanism to process non-POST request
        set_form = ShowForm()
        return render(request, 'show.html', {'set_form': set_form})

def home(request):

    """ 
    ==================================== [ Introduction ] ========================================

    This function is used to generate the view of homepage. Links to home pages: www.hallpred.com

    ==============================================================================================
    """
    return render(request, 'home.html')
