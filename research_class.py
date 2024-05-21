# импорт модулей
import os
import re
# import shap
import random
# import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import psycopg2 as psycopg
import matplotlib.pyplot as plt

from typing import Dict, List
# from catboost import Pool, CatBoostRegressor
# from phik.report import plot_correlation_matrix
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import OrdinalEncoder, StandardScaler
# from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# # установка констант
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
sns.set_style('white')
sns.set_theme(style='whitegrid')
pd.options.display.max_rows = 20
pd.options.display.max_columns = 30

class DatasetExplorer:
    def __init__(self, data_path: str = None, connection: Dict = None, table_name: str = None):
        """
        Initialize the class with the path to the data or a DataFrame.

        Parameters:
        data_path (str): Path to the data in CSV format.
        data (pd.DataFrame): DataFrame containing the data.
        """
        try:
            # Load data from the specified path if provided
            self.data = pd.read_csv(data_path)
        except:
            # If loading data from path fails, use the table from DB
            assert all([var_value != "" for var_value in list(connection.values())])

            with psycopg.connect(**connection) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT * FROM {table_name}")
                    data = cur.fetchall()
                    columns = [col[0] for col in cur.description]

            self.data = pd.DataFrame(data, columns=columns)

        # Initialize attributes for data processing
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.table_name = table_name
        
    def explore_dataset(self, target: str = None, assets_dir: str = None):

        print('Общая информация по набору данных:')
        self.data.info()
        print('\nПервые пять строк набора данных:')
        display(self.data.head(5))
        print('\nКоличество полных дубликатов строк:')
        display(self.data.duplicated().sum())
        if self.data.duplicated().sum() > 0:
            sizes = [self.data.duplicated().sum(), self.data.shape[0]]
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=['duplicate', 'not a duplicate'], autopct='%1.0f%%')
            plt.title('Number of complete duplicates in the total number of rows', size=12)
            if assets_dir is not None:
                plt.savefig(os.path.join(assets_dir, 'Number of complete duplicates in\nthe total number of rows.png'))
            plt.show()

        print('\nКоличество пропущенных значений:')
        display(self.data.isnull().sum())
        if self.data.isnull().values.any():
            if self.data.shape[1] <= 20 or self.data.shape[0] < 1000000:
                sns.heatmap(self.data.isnull(), cmap=sns.color_palette(['#000099', '#ffff00']))
                plt.xticks(rotation=90)
                plt.title('Visualization of the number of missing values', size=12, y=1.02)
                if assets_dir is not None:
                    plt.savefig(os.path.join(assets_dir, f'Visualization of the number of missing values in {self.table_name}.png'))
                plt.show()

        print('\nПроцент пропущенных значений в признаках:')
        missing_values_ratios = {}
        for column in self.data.columns[self.data.isna().any()].tolist():
            missing_values_ratio = self.data[column].isna().sum() / self.data.shape[0]
            missing_values_ratios[column] = missing_values_ratio
        for column, ratio in missing_values_ratios.items():
            print(f"{column}: {ratio*100:.2f}%")

        # Исследование признаков, у которых в названии есть 'id'
        id_columns = [col for col in self.data.columns if 'id' in col]
        for col in id_columns:
            print(f"Количество уникальных значений в столбце '{col}': {self.data[col].nunique()}")
            print(f"Соотношение уникальных значений и общего количества записей в столбце '{col}': {self.data[col].nunique() / self.data.shape[0]:.4f}")

        if target is not None:
            self.y = self.data[target]
            print('\nОписательные статистики целевой переменной:')
            display(self.data[target].describe())
            print()
            sns.set_palette("husl")
            sns.histplot(data=self.data, x=target, bins=10, log_scale=(True, False))
            plt.xlabel('Sales in units')
            plt.ylabel('Sales count')
            plt.title('Target total distribution')
            if assets_dir is not None:
                plt.savefig(os.path.join(assets_dir, 'Target total distribution.png'))
            plt.show()

        return self.data

#     def data_preparing(self, dataset=None, target=None, test_size=None):
#         X_train, X_test, y_train, y_test = train_test_split(dataset.drop(target, axis=1),
# 															dataset[target],
# 															test_size=test_size,
# 															random_state=RANDOM_STATE)		
#         encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, dtype='float')
#         X_train_oe = pd.DataFrame(encoder.fit_transform(X_train), columns=X_train.columns.to_list(), index=X_train.index)
#         X_test_oe = pd.DataFrame(encoder.transform(X_test), columns=X_train.columns.to_list(), index=X_test.index)

#         scaler = StandardScaler()
#         X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_oe), columns=X_train_oe.columns.to_list(), index=X_train_oe.index)
#         X_test_sc = pd.DataFrame(scaler.transform(X_test_oe), columns=X_train_oe.columns.to_list(), index=X_test_oe.index)
        
#         print(f"Размер обучающего набора данных: {X_train_sc.shape}")
#         print(f"Размер тестового набора данных: {X_test_sc.shape}")

#         return X_train_sc, X_test_sc, y_train, y_test

#     def model_fitting(self, model_name=None, features=None, labels=None, params=None, cv=None):
#         if model_name == 'Baseline':
#             model = LinearRegression(**params)
#             model.fit(features, labels)
#             cv_strategy = KFold(n_splits=cv)
#             scoring = ['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error']
#             cv_res = cross_validate(model, features, labels, cv=cv_strategy, n_jobs=-1, scoring = scoring)
#             for key, value in cv_res.items():
#                 if 'neg_' in key:
#                     cv_res[key] = round(-value.mean(), 3)
#                 else:
#                     cv_res[key] = round(value.mean(), 3)

#             print(f"Результаты кросс-валидации: {cv_res}")

#         elif model_name == 'CatBoost':
#             model = CatBoostRegressor(random_state=RANDOM_STATE,
# 									  verbose=False,
# 									  loss_function='RMSE')
#             train_pool = Pool(features,
# 							  labels)
#             grid_search_result = model.grid_search(param_grid=params,
# 												   cv=cv,
# 												   X=train_pool,
# 												   y=None,
# 												   plot=False)
#             print("Лучший результат RMSE:", round(min(grid_search_result['cv_results']['test-RMSE-mean']), 3))
#             print("Лучшие гиперпараметры:", grid_search_result['params'])
            
#             best_params = grid_search_result['params']

#             cv_res = {}
#             y_pred = model.predict(features)
#             cv_res['test_neg_mean_squared_error'] = round(min(grid_search_result['cv_results']['test-RMSE-mean']), 3)
#             cv_res['test_neg_mean_absolute_error'] = round(mean_absolute_error(labels, y_pred), 3)
#             cv_res['test_r2'] = round(r2_score(labels, y_pred), 3)
#             cv_res['test_neg_mean_absolute_percentage_error'] = round(mean_absolute_percentage_error(labels, y_pred), 3)
                        
#         else:
#             if model_name == 'RandomForest':
#                 model = RandomForestRegressor()

#             elif model_name == 'LinearRegression':
#                 model = LinearRegression()
			
#             grid_search = GridSearchCV(model,
# 									   params,
#                                        cv=cv,
#                                        scoring=['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error'],
#                                        refit='neg_mean_squared_error')
#             grid_search.fit(features, labels)
#             cv_res = {'test_neg_mean_squared_error': round(-grid_search.cv_results_['mean_test_neg_mean_squared_error'].mean(), 3),
# 					  'test_r2': round(grid_search.cv_results_['mean_test_r2'].mean(), 3),
# 					  'test_neg_mean_absolute_error': round(-grid_search.cv_results_['mean_test_neg_mean_absolute_error'].mean(), 3),
# 					  'test_neg_mean_absolute_percentage_error': round(-grid_search.cv_results_['mean_test_neg_mean_absolute_percentage_error'].mean(), 3)}
#             print(f"Результаты кросс-валидации: {cv_res}")
#             print(f"\nЛучшие гиперпараметры: {grid_search.best_params_}")
#             model = grid_search.best_estimator_
#             model.fit(features, labels)
#             best_params = grid_search.best_params_
            
#         try:
#             return cv_res, model, best_params
#         except:
#             return cv_res, model

#     def model_logging(self,
# 					  experiment_name=None,
# 					  run_name=None,
# 					  registry_model=None,
# 					  params=None,
# 					  metrics=None,
# 					  model=None,
# 					  train_data=None,
# 					  train_label=None,
# 					  metadata=None,
# 					  code_paths=None,
# 					  tsh=None,
# 					  tsp=None,
# 					  assets_dir=None):

#         mlflow.set_tracking_uri(f"http://{tsh}:{tsp}")
#         mlflow.set_registry_uri(f"http://{tsh}:{tsp}")
#         experiment = mlflow.get_experiment_by_name(experiment_name)
#         experiment_id = mlflow.set_experiment(experiment_name).experiment_id
		
#         pip_requirements = "requirements.txt"
#         signature = mlflow.models.infer_signature(train_data, train_label.values)
#         input_example = (pd.DataFrame(train_data)).iloc[0].to_dict()

#         with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
#             run_id = run.info.run_id
#             mlflow.log_artifacts(assets_dir)
#             mlflow.log_params(params)
#             mlflow.log_metrics(metrics)
#             model_info = mlflow.catboost.log_model(  #sklearn
#                 cb_model=model,  #sk_model
#                 artifact_path="models",
#                 pip_requirements=pip_requirements,
#                 signature=signature,
#                 input_example=input_example,
#                 metadata=metadata,
#                 code_paths=code_paths,
#                 registered_model_name=registry_model,
#                 await_registration_for=60
# 			)

#     def model_testing(self, model=None, test_features=None, test_labels=None):
#         test_pool = Pool(test_features,
# 						 test_labels)
#         y_pred = model.predict(test_pool)
        
#         print(f"MSE лучшей модели на отложенной выборке: {round(mean_squared_error(test_labels, y_pred), 3)}")
#         print(f"MAE лучшей модели на отложенной выборке: {round(mean_absolute_error(test_labels, y_pred), 3)}")
#         print(f"R2 лучшей модели на отложенной выборке: {round(r2_score(test_labels, y_pred), 3)}")
#         print(f"MAPE лучшей модели на отложенной выборке: {round(mean_absolute_percentage_error(test_labels, y_pred), 3)}")

#     def feature_importance(self, model=None, features=None, assets_dir=None):
#         explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
#         shap_values = explainer.shap_values(features)
#         shap.summary_plot(shap_values, features, plot_size=(14, 5))
#         if assets_dir:
#             plt.savefig(os.path.join(assets_dir, 'Features importance.png'))