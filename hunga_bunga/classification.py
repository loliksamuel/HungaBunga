
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
from sklearn.preprocessing import StandardScaler
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



linear_models_n_params = [
    (SGDClassifier,
     {'loss'   : ['hinge', 'log', 'modified_huber', 'squared_hinge'],
      'alpha'  : [0.0001, 0.001, 0.1],
      'penalty': penalty_12none
      })
      ,

    (LogisticRegression,
     {'penalty': penalty_12, 'max_iter': max_iter, 'tol': tol,  'warm_start': warm_start, 'C':C, 'solver': ['liblinear']
      })
      ,

    (Perceptron,
     {'penalty': penalty_all, 'alpha': alpha, 'max_iter': n_iter, 'n_iter_no_change':[10], 'eta0': eta0, 'warm_start': warm_start
      })
      ,

    (PassiveAggressiveClassifier,
     {  'C'         : C
      , 'max_iter'  : n_iter
      , 'warm_start': warm_start
      ,'loss'       : ['hinge', 'squared_hinge']
      })
]

linear_models_n_params_small = linear_models_n_params

svm_models_n_params = [
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

svm_models_n_params_small = [
     (SVC,
      {'C':C, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0, 'shrinking': shrinking, 'tol': tol, 'max_iter': max_iter_inf2})
    ,
    #error: very slow
    # (NuSVC,
    #  {'nu': nu, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0, 'shrinking': shrinking, 'tol': tol
    #   }
    # ,
    (LinearSVC,
     { 'C'       : C
     , 'penalty' : penalty_12
     , 'tol'     : tol
     , 'max_iter': max_iter
     , 'loss'    : ['hinge', 'squared_hinge']
       })
]

neighbor_models_n_params = [

    (KMeans,
     {'algorithm'   : ['auto', 'full', 'elkan'],
      'init'        : ['k-means++', 'random']})
    ,

    (KNeighborsClassifier,
     {'n_neighbors' : n_neighbors
       , 'algorithm': neighbor_algo
       , 'leaf_size': neighbor_leaf_size
       , 'metric'   : neighbor_metric
       , 'weights'  : ['uniform', 'distance']
       ,'p'         : [1, 2]
      })
    ,
    (NearestCentroid,
     {'metric'          : neighbor_metric,
      'shrink_threshold': [1e-3, 1e-2, 0.1, 0.5, 0.9, 2]
      })
    ,

    (RadiusNeighborsClassifier,
     {'radius'            : neighbor_radius
         , 'algorithm'    : neighbor_algo
         , 'leaf_size'    : neighbor_leaf_size
         , 'metric'       : neighbor_metric
         , 'weights'      : ['uniform', 'distance']
         , 'p'            : [1, 2]
         , 'outlier_label': [-1]
      })
]

gaussianprocess_models_n_params = [
    (GaussianProcessClassifier,
     {'warm_start'          : warm_start,
      'kernel'              : [RBF(), ConstantKernel(), DotProduct(), WhiteKernel()],
      'max_iter_predict'    : [500],
      'n_restarts_optimizer': [3],
      })
]


bayes_models_n_params = [
    (GaussianNB, {})
]

nn_models_n_params = [
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

nn_models_n_params_small = [
    (MLPClassifier,
     { 'hidden_layer_sizes' : [(64,), (32, 64)],
       'batch_size'         : ['auto', 50],
       'activation'         : ['identity', 'tanh', 'relu'],
       'max_iter'           : [500],
       'early_stopping'     : [True],
       'learning_rate'      : learning_rate_small
       })
]

tree_models_n_params = [
    #error: very slow
    # (RandomForestClassifier,
    #  {'criterion'           : ['gini', 'entropy'],
    #   'max_features'        : max_features,
    #   'n_estimators'        : n_estimators,
    #   'max_depth'           : max_depth,
    #   'min_samples_split'   : min_samples_split,
    #   #'min_impurity_split'  : min_impurity_split
    #   'min_impurity_decrease'  : min_impurity_split
    #   , 'warm_start': warm_start
    #   , 'min_samples_leaf': min_samples_leaf,
    #   })
    #   ,

    (DecisionTreeClassifier,
     {    'criterion'           : ['gini', 'entropy']
         , 'max_features'       : max_features
         , 'max_depth'          : max_depth
         , 'min_samples_split'  : min_samples_split
    #  , 'min_impurity_split':min_impurity_split
      , 'min_impurity_decrease' :min_impurity_split
      , 'min_samples_leaf'      : min_samples_leaf
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

tree_models_n_params_small = [

    (RandomForestClassifier,
     {     'max_features'      : max_features_small
         , 'n_estimators'      : n_estimators_small
         , 'min_samples_split' : min_samples_split
         , 'max_depth'         : max_depth_small
         , 'min_samples_leaf'  : min_samples_leaf
      })
    ,


    (DecisionTreeClassifier,
     {'max_features': max_features_small, 'max_depth': max_depth_small, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf
      })
    ,

    (ExtraTreesClassifier,
     {'n_estimators'      : n_estimators_small
     , 'max_features'     : max_features_small
     , 'max_depth'        : max_depth_small
     , 'min_samples_split': min_samples_split
     , 'min_samples_leaf' : min_samples_leaf})
]



#@ignore_warnings
def run_all_classifiers(x, y, small = True, normalize_x = True, n_jobs=cpu_count()-1, brain=False, test_size=0.2, n_splits=5, upsample=True, scoring=None, verbose=False, grid_search=True):
    all_params = (linear_models_n_params_small if small else linear_models_n_params) +  (nn_models_n_params_small if small else nn_models_n_params) + (gaussianprocess_models_n_params if small else gaussianprocess_models_n_params) + neighbor_models_n_params + (svm_models_n_params_small if small else svm_models_n_params) + (tree_models_n_params_small if small else tree_models_n_params)
    return main_loop(all_params, StandardScaler().fit_transform(x) if normalize_x else x, y, isClassification=True, n_jobs=n_jobs, verbose=verbose, brain=brain, test_size=test_size, n_splits=n_splits, upsample=upsample, scoring=scoring, grid_search=grid_search)

def run_one_classifier(x, y, small = True, normalize_x = True, n_jobs=cpu_count()-1, brain=False, test_size=0.2, n_splits=5, upsample=True, scoring=None, verbose=False, grid_search=True):
    all_params = (linear_models_n_params_small if small else linear_models_n_params) +  (nn_models_n_params_small if small else nn_models_n_params) + ([] if small else gaussianprocess_models_n_params) + neighbor_models_n_params + (svm_models_n_params_small if small else svm_models_n_params) + (tree_models_n_params_small if small else tree_models_n_params)
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
    #warnings.warn = warn
    data_type       = 'spy283' #spy71  spy283   spyp71  spyp283  iris  random
    names_output    = ['Green bar', 'Red Bar']
    size_output     = len(names_output)
    use_raw_data    = True
    use_feature_tool= False
    test_size  = 0.2
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


    #iris = datasets.load_iris()
    X, y = df_x, df_y#iris.data, iris.target
    clf = HungaBungaClassifier( brain      =True
                               ,small      =True
                               ,normalize_x=True
                               ,upsample   =True
                               ,scoring    =None
                               ,verbose    =False#if True#sklearn.exceptions.NotFittedError: This Perceptron instance is not fitted yet
                               ,n_splits   =2
                               ,test_size  =test_size

                                )

    clf.fit(X, y)
    print(clf.predict(X).shape)
    ''' 
    ========================================================================
    Model  large                  accuracy    Time/grid (s)    Time/clf (s)
    ---------------------------  ----------  ---------------  --------------
    SGDClassifier                     0.983            3.099           3.099
    LogisticRegression                0.95             0.278           0.278
    Perceptron                        0.917            4.183           4.183
    PassiveAggressiveClassifier       0.967            0.329           0.329
    GaussianProcessClassifier         0.933            7.454           7.454
    KMeans                            0.533            0.344           0.344
    KNeighborsClassifier              1               13.726          13.726
    NearestCentroid                   0.95             0.103           0.103
    RadiusNeighborsClassifier         0.967           14.211          14.211
    DecisionTreeClassifier            1                1.441           1.441
    ========================================================================
    The winner is: KNeighborsClassifier with score 1.000.
    
    
     
    ========================================================================
    Model   small                  accuracy    Time/grid (s)    Time/clf (s)
    ---------------------------  ----------  ---------------  --------------
    Perceptron                        0.95             6.853           6.853
    PassiveAggressiveClassifier       0.967            0.307           0.307
    MLPClassifier                     0.95             1.358           1.358
    GaussianProcessClassifier         0.933            6.452           6.452
    KMeans                            0.783            0.344           0.344
    KNeighborsClassifier              1                7.601           7.601
    NearestCentroid                   0.933            0.102           0.102
    RadiusNeighborsClassifier         1               13.969          13.969
    SVC                               0.983           70.019          70.019
    LinearSVC                         0.967            0.271           0.271
    RandomForestClassifier            1               14.333          14.333
    DecisionTreeClassifier            0.95             0.104           0.104
    ExtraTreesClassifier              1               11              11
    ========================================================================
    The winner is: KNeighborsClassifier with score 1.000.
    '''


