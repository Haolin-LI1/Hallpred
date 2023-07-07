from scipy import stats
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from diptest import diptest

def pred_result(dataset,default_sp_threshold=0.05,hartigans_p_threshold=0.05):

    """ 
    ================================== [ Introduction ] ======================================

    The main algorithm to process variation dataset.This function analyzes the data of Gibbs 
    folding free energy (ddg) in the dataset and calculates the corresponding reference 
    prediction value and 95% confidence interval.
    
    ===================================== [ Input ] ==========================================

    dataset: 
        Gene mutation dataset of ddg value. Type is QuerySet and obtained from database.
    
    default_sp_threshold: 
        The threshold used to determine the distribution on the Shapiro-Wilk Test. The default
        value is set as 0.05.
    
    hartigans_p_threshold:
        The threshold used to determine the modal on the Hartigans Dip Test. The default value
        is set as 0.05.
    
    ===================================== [ Output ] =========================================

    reference_value:
        The predictive value of ddg in this mutation with unit of kcal/mol.
    
    lower_cl:
        The lower bound of 95% predicted confidence interval of ddg in this variation with unit
        of kcal/mol.
    
    upper_cl:
        The upper bound of 95% predicted confidence interval of ddg in this variation with unit
        of kcal/mol.

    ===========================================================================================
    """

    # classify according to the type of data distribution
    _, sp_p = stats.shapiro(dataset)
    
    if sp_p > default_sp_threshold:
        
        # normal distribution
        lower_cl, upper_cl = cl_algorithm(dataset)
        reference_value = ref_value(dataset)

    else: 

        # non-normal distribution
        bs_value = bs_sample(dataset)
        lower_cl, upper_cl = cl_algorithm(bs_value)
        dataset_array = np.array(dataset)
        _, hartigans_p = diptest(dataset_array)

        # classify according to the type of modal
        if hartigans_p > hartigans_p_threshold:
            # unimodal
            reference_value = ref_value(bs_value)
        else:
            # multimodal
            reference_value = kmeans_find_two_cluster(bs_value)

    return reference_value,lower_cl,upper_cl

def cl_algorithm(dataset,cl_range=0.95):
    
    """ 
    ================================== [ Introduction ] ======================================

    This function is used to compute the 95% confidence interval for the dataset.
    
    ===================================== [ Input ] ==========================================

    dataset: 
        Gene mutation dataset of ddg value. Type is QuerySet and obtained from database.
    
    cl_range: 
        The confidence level of calculation. The default value is set as 0.95.
    
    ===================================== [ Output ] =========================================

    lower_cl:
        The lower bound of 95% predicted confidence interval of ddg in this variation with unit
        of kcal/mol.
    
    upper_cl:
        The upper bound of 95% predicted confidence interval of ddg in this variation with unit
        of kcal/mol.

    ===========================================================================================
    """

    ddg_len = len(dataset)
    ddg_mean = np.mean(dataset)
    ddg_scale = stats.sem(dataset)

    lower_cl, upper_cl = stats.t.interval(cl_range, ddg_len-1, loc=ddg_mean, scale=ddg_scale)

    return lower_cl, upper_cl

def ref_value(dataset):
    
    """ 
    ================================== [ Introduction ] ======================================

    This function is Used to calculate the predictive ddg of the dataset. The two main criteria
    to consider are skwness and extreme value. 
    
    ===================================== [ Input ] ==========================================

    dataset: 
        Gene mutation dataset of ddg value. Type is QuerySet and obtained from database.
    
    ===================================== [ Output ] =========================================

    reference_value:
        The predictive value of ddg in this mutation with unit of kcal/mol.

    ===========================================================================================
    """

    # use the quartile method to determine if there are extreme values in the dataset
    dataset_25 = np.percentile(dataset, 25)
    dataset_75 = np.percentile(dataset, 75)

    lower_dataset = min(dataset)
    lower_threshold = dataset_25 * 2.5 - dataset_75 * 1.5

    upper_dataset = max(dataset)
    upper_threshold = dataset_75 * 2.5 - dataset_25 * 1.5

    # use the skwness to determine if the data is evenly distributed
    dataset_skewness = stats.skew(dataset)

    if lower_dataset < lower_threshold or upper_dataset > upper_threshold or dataset_skewness < -0.5 or dataset_skewness > 0.5:
        # uneven distribution
        reference_value = np.median(dataset)
    else:
        # even distribution
        reference_value = np.mean(dataset)
    
    return reference_value

def bs_sample(dataset,bs_size=5000):

    """ 
    ================================== [ Introduction ] ======================================

    This function is used to resample the dataset to extend its size.
    
    ===================================== [ Input ] ==========================================

    dataset: 
        Gene mutation dataset of ddg value. Type is QuerySet and obtained from database.

    bs_size:
        The number of resamples. The default value is set as 5000.
    
    ===================================== [ Output ] =========================================

    bs_ddgs:
        A resampled numpy array of ddg in this variation.

    ===========================================================================================
    """
    
    data_array = np.array(dataset)
    dataset_len = len(data_array)
    bs_ddgs = data_array[np.random.randint(0, dataset_len, size=bs_size)]

    return bs_ddgs

def kmeans_find_two_cluster(dataset, cluster_num=2):

    """ 
    ================================== [ Introduction ] ======================================

    This function is used to obtain the two clusters with the highest density as reference ddg
    when the data set is multimodal.
    
    ===================================== [ Input ] ==========================================

    dataset: 
        Gene mutation dataset of ddg value. Type is QuerySet and obtained from database.

    cluster_num:
        The number of cluster. The default value is set as 2.
    
    ===================================== [ Output ] =========================================

    ref_value_kmeans_list:
        The return list is consisted of the values of the two clusters and their corresponding
        dimension ratio. 

    ===========================================================================================
    """

    # use kmeans to find two clusters of the dataset
    bs_ddgs_reshaped_array = np.array(dataset).reshape(-1, 1)
    bs_ddgs_kmeans = KMeans(n_clusters=cluster_num, random_state=22, n_init=12)
    bs_ddgs_kmeans.fit(bs_ddgs_reshaped_array)

    # calculate the size of each cluster
    reference_values = np.around(bs_ddgs_kmeans.cluster_centers_, decimals=5)
    cluster_sizes = Counter(bs_ddgs_kmeans.labels_)

    cluster_ratios = dict()
    for cluster, amount in cluster_sizes.items():
        ratio = (amount / len(dataset)) * 100
        cluster_ratios[cluster] = "{:.2f}%".format(ratio)

    ref_values_ratios = []
    ref_values_ratios = list()
    for cluster in cluster_sizes.keys():
        ref_value = float(reference_values[cluster])
        ratio = cluster_ratios[cluster]
        ref_values_ratios.append((ref_value, ratio))
    
    ref_values_ratios.sort(key=lambda x: float(x[1].strip('%')), reverse=True)

    ref_value_high = ref_values_ratios[0][0]
    ref_value_high_percent = ref_values_ratios[0][1]
    ref_value_low = ref_values_ratios[1][0]
    ref_value_low_percent = ref_values_ratios[1][1]

    ref_value_kmeans_list = [ref_value_high, ref_value_high_percent, ref_value_low, ref_value_low_percent]

    return ref_value_kmeans_list



