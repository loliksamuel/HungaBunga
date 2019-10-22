
# import warnings
# from sklearn.utils.testing import ignore_warnings
# warnings.filterwarnings('ignore')
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.simplefilter  ("ignore", category=PendingDeprecationWarning)
# warnings.simplefilter  ("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=Warning)
# warnings.filterwarnings("ignore", category = RuntimeWarning)
# warnings.filterwarnings('ignore', message='Solver terminated early.*')
# warnings.filterwarnings('ignore', message='ConvergenceWarning: Maximum number of iteration reached.*')
# warnings.filterwarnings('ignore', message='UserWarning: Averaging for metrics other.*')
# warnings.filterwarnings("ignore")

from os import path
import pandas as pd
import random
from utils import data_prepare
from sklearn import datasets
from sklearn.linear_model import SGDClassifier, LogisticRegression, Perceptron, PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, Matern, StationaryKernelMixin, WhiteKernel
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterSampler

from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import is_classifier

from core import *
from params import *


# def warn(*args, **kwargs):
#     pass



models_n_params_linear = [
    (SGDClassifier,{
       'penalty': penalty_12none
      ,'loss'   : ['hinge', 'log', 'modified_huber', 'squared_hinge']
      ,'alpha'  : [0.00001, 0.001, 10],
      })
      ,

    (LogisticRegression,
     {'penalty': penalty_12, 'max_iter': max_iter, 'warm_start': warm_start, 'tol': tol, 'C':C, 'solver': ['liblinear']
      })
      ,

    (Perceptron,
     {'penalty': penalty_all, 'max_iter': n_iter, 'warm_start': warm_start, 'alpha': alpha, 'n_iter_no_change':[20], 'eta0': eta0
      })
      ,

    (PassiveAggressiveClassifier,
     {  'C'         : C
      , 'max_iter'  : n_iter
      , 'warm_start': warm_start
      ,'loss'       : ['hinge', 'squared_hinge']
      })
]

models_n_params_linear_small = models_n_params_linear

models_n_params_svm = [
    #error: very slow
    # (SVC,
    #  {'C':C, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0, 'shrinking': shrinking, 'tol': tol, 'max_iter': max_iter_inf2})
    #,
    #error: very slow
    # (NuSVC,
    #  {'nu': nu, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0, 'shrinking': shrinking, 'tol': tol
    #   })
    #,
    #error: inf
    # (LinearSVC,
    #  {      'C'         : C
    #      , 'penalty_12' : penalty_12
    #      , 'tol'        : tol
    #      , 'max_iter'   : max_iter
    #      , 'loss'       : ['hinge', 'squared_hinge'],
    #    })
]

models_n_params_svm_small = [
    #  (SVC,
    #   {   'C'       :C
    #   , 'tol'       : tol
    #   , 'max_iter'  : max_iter_inf2
    #   , 'kernel'    : kernel
    #   , 'degree'    : degree
    #   , 'gamma'     : gamma
    #   , 'coef0'     : coef0
    #   , 'shrinking' : shrinking
    #    })
    # ,
    # #error: very slow
    # # (NuSVC,
    # #  {'nu': nu, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0, 'shrinking': shrinking, 'tol': tol
    # #   }
    # # ,
    # (LinearSVC,
    #  { 'C'       : C
    #  , 'tol'     : tol
    #  , 'max_iter': max_iter
    #  , 'loss'    : ['hinge', 'squared_hinge']
    #  , 'penalty' : penalty_12
    #    })
]

models_n_params_neighbor = [

    (KMeans,
     {'algorithm'   : ['auto', 'full', 'elkan']
     ,'init'        : ['k-means++', 'random']
     ,'n_clusters'  : [2]
      })
    ,

    (KNeighborsClassifier,
       { 'metric'     : neighbor_metric
       , 'weights'    : ['uniform', 'distance']
       , 'algorithm'  : neighbor_algo
       , 'leaf_size'  : neighbor_leaf_size
       , 'p'          : [1, 4]
       , 'n_neighbors': n_neighbors
      })
    ,
    (NearestCentroid,
     {  'metric'          : neighbor_metric
      , 'shrink_threshold': [1e-4, 1e-2, 0.1, 0.5, 1.9, 20]
      })
    ,
    (RadiusNeighborsClassifier,
     {
           'metric'       : neighbor_metric
         , 'weights'      : ['uniform', 'distance']
         , 'algorithm'    : neighbor_algo
         , 'leaf_size'    : neighbor_leaf_size
         , 'p'            : [1, 2]
         , 'outlier_label': [-1]
         , 'radius'       : neighbor_radius
      })
]

models_n_params_gaussian = [
    (GaussianProcessClassifier,
     {'warm_start'          : warm_start,
      'kernel'              : [RBF(), ConstantKernel(), DotProduct(), WhiteKernel()],
      'max_iter_predict'    : [200],
      'n_restarts_optimizer': [6],
      })
]


models_n_params_bayes = [
    (GaussianNB, {})
]

models_n_params_nn = [
    #error: very slow
    # (MLPClassifier,
    #  { 'hidden_layer_sizes': [(16,), (64,), (100,), (32, 32)],
    #    'batch_size'    : ['auto', 50],
    #    'activation'    : ['identity', 'tanh', 'relu', 'logistic'],
    #    'max_iter'      : [1000],
    #    'early_stopping': [True, False],
    #    'learning_rate' : learning_rate,
    #    'alpha'         : alpha,
    #    'tol'           : tol,
    #    'warm_start'    : warm_start,
    #    'epsilon'       : [1e-8, 1e-5]
    #    })
]

models_n_params_nn_small = [
    (MLPClassifier,
     { 'hidden_layer_sizes' : [(64,), (32, 64)],
       'batch_size'         : ['auto', 50],
       'activation'         : ['identity', 'tanh', 'relu'],
       'max_iter'           : [500],
       'early_stopping'     : [True],
       'learning_rate'      : learning_rate_small
       })
]

models_n_params_tree = [
    #error: very slow
    # (RandomForestClassifier,
    #  {
    #     'max_features'        : max_features
    #   , 'max_depth'           : max_depth
    #   , 'min_samples_split'   : min_samples_split
    #   , 'min_samples_leaf': min_samples_leaf
    #  #, 'min_impurity_split'  : min_impurity_split
    #   , 'min_impurity_decrease'  : min_impurity_split
    #   , 'n_estimators'        : n_estimators
    #   , 'warm_start': warm_start
    #   , 'criterion'           : ['gini', 'entropy']
    #   })
    #   ,
    (DecisionTreeClassifier,
     {  'max_features'          : max_features
      , 'max_depth'             : max_depth
      , 'min_samples_split'     : min_samples_split
      , 'min_samples_leaf'      : min_samples_leaf
    # , 'min_impurity_split':min_impurity_split
      , 'min_impurity_decrease' :min_impurity_split
      , 'criterion'             : ['gini', 'entropy']
      })
    #,
    #error:very slow
    # (ExtraTreesClassifier,
    #  {     'n_estimators'           : n_estimators
    #      , 'max_features'           : max_features
    #      , 'max_depth'              : max_depth
    #      , 'min_samples_split'      : min_samples_split
    #      , 'min_samples_leaf'       : min_samples_leaf
    #      #, 'min_impurity_split'    : min_impurity_split
    #      , 'min_impurity_decrease'  : min_impurity_split
    #      , 'warm_start'             : warm_start
    #      , 'criterion'              : ['gini', 'entropy']})
]

models_n_params_tree_small = [

    (RandomForestClassifier,
     { 'max_features'      : max_features_small
     , 'max_depth'         : max_depth_small
     , 'min_samples_split' : min_samples_split
     , 'min_samples_leaf'  : min_samples_leaf
     , 'n_estimators'      : n_estimators_small
     })
    ,
    (DecisionTreeClassifier,
     {'max_features'     : max_features_small
    , 'max_depth'        : max_depth_small
    , 'min_samples_split': min_samples_split
    , 'min_samples_leaf' : min_samples_leaf
     })
    ,
    (ExtraTreesClassifier,
     { 'max_features'     : max_features_small
     , 'max_depth'        : max_depth_small
     , 'min_samples_split': min_samples_split
     , 'min_samples_leaf' : min_samples_leaf
     , 'n_estimators'     : n_estimators_small
      })
]



#@ignore_warnings
def run_all_classifiers(x, y, small = True, normalize_x = True, n_jobs=cpu_count()-1, brain=False, test_size=0.2, n_splits=5, upsample=True, scoring=None, verbose=False, grid_search=True):
    #all_params =   models_n_params_gaussian     + models_n_params_neighbor
    all_params =    (models_n_params_linear_small if small else models_n_params_linear) \
                  + (models_n_params_nn_small     if small else models_n_params_nn) \
                  + (models_n_params_gaussian     if small else []) \
                  + (models_n_params_neighbor     if small else []) \
                  + (models_n_params_svm_small    if small else models_n_params_svm) \
                  + (models_n_params_tree_small   if small else models_n_params_tree)
    return main_loop(  all_params
                     , StandardScaler().fit_transform(x) if normalize_x else x#MinMaxScaler().fit_transform(x)
                     , y
                     , isClassification=True, n_jobs=n_jobs, verbose=verbose, brain=brain, test_size=test_size, n_splits=n_splits, upsample=upsample, scoring=scoring, grid_search=grid_search)

def run_one_classifier(x, y, small = True, normalize_x = True, n_jobs=cpu_count()-1, brain=False, test_size=0.2, n_splits=5, upsample=True, scoring=None, verbose=False, grid_search=True):
    all_params = (models_n_params_linear_small if small else models_n_params_linear) + (models_n_params_nn_small if small else models_n_params_nn) + ([] if small else models_n_params_gaussian) + models_n_params_neighbor + (models_n_params_svm_small if small else models_n_params_svm) + (models_n_params_tree_small if small else models_n_params_tree)
    all_params = random.choice(all_params)
    return all_params[0](**(list(ParameterSampler(all_params[1], n_iter=1))[0]))


class HungaBungaClassifier(ClassifierMixin):
    def __init__(self
                 , grid_search=True
                 , test_size = 0.2
                 , n_splits = 5
                 , scoring=None
                 , normalize_x = True
                 , upsample=True
                 , verbose=False
                 , brain=False
                 , small = True
                 , n_jobs =cpu_count() - 1
                 , random_state=None
                 ):
        self.model = None
        self.brain = brain
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.upsample = upsample
        self.scoring = None#'accuracy'
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.normalize_x = normalize_x
        self.grid_search = grid_search
        self.small = small
        super(HungaBungaClassifier, self).__init__()

    def fit(self, x, y):
        self.model = run_all_classifiers(x, y, small = self.small, normalize_x=self.normalize_x, test_size=self.test_size, n_splits=self.n_splits, upsample=self.upsample, scoring=self.scoring, verbose=self.verbose, brain=self.brain, n_jobs=self.n_jobs, grid_search=self.grid_search)[0]
        #sleep(1)
        return self

    def predict(self, x):

        return self.model.predict(x)


class HungaBungaRandomClassifier(ClassifierMixin):
    def __init__(self, brain=False, test_size = 0.2, n_splits = 5, random_state=None, upsample=True, scoring=None, verbose=False, normalize_x = True, n_jobs =cpu_count() - 1, grid_search=True):
        self.model = None
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.upsample = upsample
        self.scoring = None
        self.n_jobs = n_jobs
        self.normalize_x = normalize_x
        self.grid_search = grid_search
        self.brain = brain
        self.verbose = verbose
        super(HungaBungaRandomClassifier, self).__init__()

    def fit(self, x, y):
        self.model = run_one_classifier(x, y, normalize_x=self.normalize_x, test_size=self.test_size, n_splits=self.n_splits, upsample=self.upsample, scoring=self.scoring, verbose=self.verbose, brain=self.brain, n_jobs=self.n_jobs, grid_search=self.grid_search)
        self.model.fit(x, y)
        return self

    def predict(self, x):
        return self.model.predict(x)




if __name__ == '__main__':
    ##warnings.warn = warn
    data_type       = 'spy283' #spy71  spy283   spyp71  spyp283  iris  random
    names_output    = ['Green bar', 'Red Bar']
    size_output     = len(names_output)
    use_raw_data    = True
    use_feature_tool= True

    data_path_all = path.join('files', 'input', f'{data_type}')
    if (use_raw_data):
        print(f'Loading from disc raw data   ')
        df_x, df_y = data_prepare(data_type=data_type, use_feature_tool=use_feature_tool)

        if isinstance(df_x,  pd.DataFrame):
            df_x.to_csv( f'{data_path_all}_x_{use_feature_tool}.csv', index=False, header=True)
            df_y.to_csv( f'{data_path_all}_y_{use_feature_tool}.csv', index=False, header=True)

    else:
        print(f'Loading from disc prepared data :{data_path_all + "_x.csv"} ')
        df_x = pd.read_csv(f'{data_path_all}_x_{use_feature_tool}.csv')
        df_y = pd.read_csv(f'{data_path_all}_y_{use_feature_tool}.csv')

    if isinstance(df_y,  pd.DataFrame):
        names_output = df_y['target'].unique()
    else:
        names_output = pd.Series(df_y, name='target').unique()#df_y.unique()#list(iris.target_names)
    names_input  = df_x.columns.tolist()
    names_input  = list(map(str, names_input))
    size_input   = len(names_input)
    size_output  = len(names_output)
    print(f'#features={size_input}, out={size_output}, names out={names_output }')
    print(f'df_y.describe()=\n{df_y.describe()}')
    print(f'\ndf_y[5]={df_y.shape}\n',df_y.head(5))
    print(f'\ndf_x[1]={df_x.shape}\n',df_x.head(1))
    X, y = df_x, df_y

    # iris = datasets.load_iris()
    # X, y = iris.data, iris.target

    test_size  = 0.3
    n_splits   =2
    small      =False
    brain      =True
    normalize_x=True
    upsample   =True
    scoring    =None
    verbose    =False#if True#sklearn.exceptions.NotFittedError: This Perceptron instance is not fitted yet

    clf = HungaBungaClassifier( brain      =brain
                               ,small      =small
                               ,normalize_x=normalize_x
                               ,upsample   =upsample
                               ,scoring    =scoring
                               ,verbose    =verbose
                               ,n_splits   =n_splits
                               ,test_size  =test_size

                                )

    clf.fit(X, y)
    print(clf.predict(X).shape)
    print(f'brain      ={brain}  \nsmall      ={small} \nnormalize_x={normalize_x}  \nupsample   ={upsample} \nscoring    ={scoring}  \nverbose    ={verbose} \nn_splits   ={n_splits} \ntest_size  ={test_size}')
    ''' 
    ========================================================================
    Model   large                  accuracy    Time/grid (s)    Time/clf (s)
    ---------------------------  ----------  ---------------  --------------
    SGDClassifier                     0.523              202           3.052
    LogisticRegression                0.487             1013           2.13
    Perceptron                        0.523              308           0.436
    PassiveAggressiveClassifier       0.518               20           0.182
    DecisionTreeClassifier            0.529             1154           0.027
    ========================================================================
    The winner is: DecisionTreeClassifier with score 0.529. 
    
    
     
    ========================================================================
    Model  small                   accuracy    Time/grid (s)    Time/clf (s)
    ---------------------------  ----------  ---------------  --------------
    SGDClassifier                     0.523              183           0.432
    LogisticRegression                0.465             1014           2.149
    Perceptron                        0.523              319           0.475
    PassiveAggressiveClassifier       0.514               20           0.182
    MLPClassifier                     0.498              100           2.264
    GaussianProcessClassifier         0.523            58071          651.76
    KMeans                            0.507               22           1.712
    KNeighborsClassifier              0.471            42346           0.648
    NearestCentroid                   0.515               11           0.182
    RadiusNeighborsClassifier         0.009            17416           0.887
    RandomForestClassifier            0.524             3054           0.175
    DecisionTreeClassifier            0.526               60           0.034
    ExtraTreesClassifier              0.523             1477           0.832
    ========================================================================
    The winner is: DecisionTreeClassifier with score 0.526.
    '''


