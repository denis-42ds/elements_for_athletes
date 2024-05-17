# # импорт модулей
import os
import re
# import shap
import random
# import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# from catboost import Pool, CatBoostRegressor
# from phik.report import plot_correlation_matrix
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import OrdinalEncoder, StandardScaler
# from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# # установка констант
# RANDOM_STATE = 42
# random.seed(RANDOM_STATE)
# np.random.seed(RANDOM_STATE)

class DatasetExplorer:
    pass
#     def __init__(self, DATA_PATH=None):
#         self.DATA_PATH = DATA_PATH
        
#     def explore_dataset(self, assets_dir=None):
#         with open (self.DATA_PATH, 'r') as f:
#             data = f.read()

#         wmi_pattern = r'(?P<wmi>[ABCDEFGHJKLMNPRSTUVWXYZ1234567890]{3})'  # pos. 1-3
#         brake_pattern = r'(?P<brake>[ABCDEFGHJKLMNPRSTUVWXYZ1234567890])' # pos. 4
#         body_pattern = r'(?P<body>[ABCDEFGHJKLMNPRSTUVWXYZ1234567890]{3})'  # pos. 5-7
#         engine_pattern = r'(?P<engine>[ABCDEFGHJKLMNPRSTUVWXYZ1234567890])'  # pos. 8
#         check_digit_pattern = r'(?P<check_digit>[ABCDEFGHJKLMNPRSTUVWXYZ1234567890])'  # pos. 9
#         year_pattern = r'(?P<year>[ABCDEFGHJKLMNPRSTUVWXYZ1234567890])'.replace('U', '').replace('Z', '').replace('0', '')  # pos. 10
#         plant_pattern = r'(?P<plant>[ABCDEFGHJKLMNPRSTUVWXYZ1234567890])'  # pos. 11
#         snum_pattern = r'(?P<snum>[ABCDEFGHJKLMNPRSTUVWXYZ1234567890]\d{5})'  # pos. 12-17
#         price_pattern = r'\:(?P<price>\d+)'
#         pattern = f'{wmi_pattern}{brake_pattern}{body_pattern}{engine_pattern}{check_digit_pattern}{year_pattern}{plant_pattern}{snum_pattern}{price_pattern}'
#         dataset = pd.DataFrame([x.groupdict() for x in re.compile(pattern).finditer(data)])		

#         # Преобразование типа данных целевой переменной
#         dataset['price'] = dataset['price'].astype(int)
#         print("Верхние пять строк датафрейма:\n")
#         display(dataset.head())
#         print("\nОбщая информация по датафрейму:\n")
#         dataset.info()
#         print(f"\nКоличество дубликатов: {dataset.duplicated().sum()}")
#         print("\nКоличество уникальных значений в каждом признаке:\n")
#         display(dataset.nunique())
#         print("\nАнализ целевой переменной:\n")
#         display(dataset['price'].describe())
#         plt.figure(figsize=(14, 6))
#         sns.set_palette("husl")
#         sns.histplot(data=dataset, x='price', bins=100)
#         plt.xlabel('Price of auto')
#         plt.ylabel('Number of cars')
#         plt.title('Car price distribution')
#         if assets_dir:
#             plt.savefig(os.path.join(assets_dir, 'Car price distribution.png'))
#         plt.show()
        
#         plt.figure(figsize=(14, 6))
#         sns.boxplot(data=dataset['price'], orient='h')
#         plt.xlim(0, 40000)
#         plt.title('Target boxplot', size=12)
#         plt.xlabel('Price')
#         if assets_dir:
#             plt.savefig(os.path.join(assets_dir, 'Target boxplot.png'))
#         plt.show()

#         phik_overview = dataset.drop('snum', axis=1).phik_matrix(interval_cols=['price'])
#         sns.set()
#         plot_correlation_matrix(phik_overview.values,
# 		                        x_labels=phik_overview.columns,
# 								y_labels=phik_overview.index,
# 								fontsize_factor=1.0,
# 								figsize=(10, 10))
#         plt.xticks(rotation=0)
#         plt.title(f'Correlations between features', fontsize=12, y=1.02)
#         if assets_dir:
#             plt.savefig(os.path.join(assets_dir, 'Features correlations.png'))
#         plt.tight_layout()

#         return dataset


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