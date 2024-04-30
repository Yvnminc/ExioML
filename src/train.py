import time
import pandas as pd
from copy import deepcopy
from data import get_train_vail_test
from model import init_deep_model, init_shallow_model
import numpy as np
from pytorch_tabular import TabularModel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid

from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
from pytorch_tabular.tabular_model_tuner import TabularModelTuner
import random
    
class ShallowModel:
    def __init__(self, type = "pxp", data = "LOOE") -> None:
        self.type = type
        self.data = data
        self.models = init_shallow_model()
        
        self.shaffle_data()
        print(f"Train Shape: {self.train_data.shape} | Val Shape: {self.val_data.shape} | Test Shape: {self.test_data.shape}")

    def train(self, mode = "val", iter = 5):
        results = {}

        for name, model in self.models.items():
            
            print(f"Training {name} in {mode} mode for {iter} times...")
            
            mse_scores = []
            runtime_scores = []
            
            for i in range(iter):
                self.shaffle_data()
                
                X_train = self.X_train
                y_train = self.y_train
                
                if mode == "val":
                    X_test = self.X_val
                    y_test = self.y_val   
                    
                elif mode == "test":
                    X_test = self.X_test
                    y_test = self.y_test
            
                
                start_time = time.time()  # Start time
                # Train the model
                model.fit(X_train, y_train)
                end_time = time.time()  # End time

                # Make predictions
                predictions = model.predict(X_test)

                # Calculate runtime
                runtime = end_time - start_time
                
                # Evaluate the model
                mse = mean_squared_error(y_test, predictions)
            
                mse_scores.append(mse)
                runtime_scores.append(runtime)
            
            mse_mean = np.mean(mse_scores)
            mse_std = np.std(mse_scores)
            
            time_mean = np.mean(runtime_scores)
            time_std = np.std(runtime_scores)
            
            # Store results
            results[name] = {"MSE mean": mse_mean, "MSE std": mse_std, "Time mean (s)": time_mean, "Time std (s)": time_std}

        results_df = pd.DataFrame(results).transpose()
        results_df
        
        return results_df
    
    def tune(self, model_name = "KNN", params = {}, n_iters = 10):
        
        # 初始化最佳分数和最佳参数
        best_score = float('inf')
        best_param = None
        model = self.models[model_name]
        unique_list = []
        
        # # 使用 ParameterGrid 生成所有参数组合
        # param_grid = ParameterGrid(params)

        # 迭代网格搜索
        for _ in range(n_iters):
            param = {k: np.random.choice(v) for k, v in params.items()}
            
            if param in unique_list:
                continue
            
            print(f"Training {model_name} for {n_iters} iter with {param}...")
            
            model.set_params(**param)
            model.fit(self.X_train, self.y_train)
            
            unique_list.append(param)
            
            predictions = model.predict(self.X_val)
            mse = mean_squared_error(self.y_val, predictions)

            if mse < best_score:
                best_score = mse
                best_param = param
        
        print(f"Best parameters: {best_param}, MSE: {best_score}")
        
        self.set_model_param(model_name, best_param)
        
        return best_param
    
    def shaffle_data(self):
        num_name = ["Value Added [M.EUR]", "Employment [1000 p.]", "Energy Carrier Net Total [TJ]", "Year", "region", "sector"]
        lable_name = ["GHG emissions [kg CO2 eq.]"]
        
        self.train_data, self.val_data, self.test_data = get_train_vail_test(self.type, self.data)
        
        self.X_train, self.y_train,  self.X_val, self.y_val, self.X_test, self.y_test = self.train_data[num_name], self.train_data[lable_name].values.ravel(), self.val_data[num_name], self.val_data[lable_name].values.ravel(), self.test_data[num_name], self.test_data[lable_name].values.ravel()
        
        self.data_set = {"X_train": self.X_train, "y_train": self.y_train, "X_val": self.X_val, "y_val": self.y_val, "X_test": self.X_test, "y_test": self.y_test}
        
    def set_model_param(self, model_name = "KNN", params = {}):
        model = self.models[model_name]
        model.set_params(**params)
        
        
class DeepModel:
    def __init__(self, type, data, batch_size = 8192, max_epochs = 30):
        self.type = type
        self.data = data
        
        cat_names = ["region", "sector"]
        num_name = ["Value Added [M.EUR]", "Employment [1000 p.]", "Energy Carrier Net Total [TJ]", "Year"]
        lable_name = ["GHG emissions [kg CO2 eq.]"]
        
        # Data config
        self.data_config = DataConfig(
            target=lable_name, # Label Column
            continuous_cols=num_name, # Numeric columns
            categorical_cols=cat_names,  # Category columns
            continuous_feature_transform=None, # Feature engineering transformations
            normalize_continuous_features=True,  # Normalize continuous features by mean/std when using batch
            # num_workers=5,  # For GPU, change this to number of GPUs available
        )
        
        # Optimastion config
        self.optimizer_config = OptimizerConfig(
            optimizer="torch_optimizer.QHAdam",
            optimizer_params={ 'nus':(0.7, 1.0), 'betas':(0.95, 0.998) },
            lr_scheduler="ReduceLROnPlateau",
            lr_scheduler_params={"mode": "min", "patience": 3, "eps": 1e-5},
        )
    
        # Head config
        self.head_config = LinearHeadConfig(
            layers="",  # No additional layer in head, just a mapping layer to output_dim
            dropout=0.5,
            initialization="kaiming",
            use_batch_norm=True,
        ).__dict__  # Convert to dict to pass to the model config (OmegaConf doesn't accept objects)
                
        # Trainer config
        self.trainer_config = TrainerConfig(
            auto_lr_find=False,  # Automatically infers the learning rate
            batch_size= batch_size,  # Batch size for training
            max_epochs=max_epochs,  # Maximum number of epochs
            accelerator="auto",
            devices=-1,
            checkpoints_every_n_epochs = 100,
            early_stopping="valid_loss",  # Monitor valid_loss for early stopping
            early_stopping_mode="min",  # Set the mode as min because for val_loss, lower is better
            early_stopping_patience=max_epochs,  # No. of epochs of degradation training will wait before terminating
        )
                
        self.shaffle_data()
        print(f"Train Shape: {self.train_data.shape} | Val Shape: {self.val_data.shape} | Test Shape: {self.test_data.shape}")
        
        self.models, self.model_config = init_deep_model(self.data_config, self.optimizer_config, self.trainer_config, self.head_config)
        
        
    def shaffle_data(self):
        self.train_data, self.val_data, self.test_data = get_train_vail_test(self.type, self.data)


        
    def test(self, iter = 5):

        results = {}

        for name, model in self.models.items():
            
            mse_scores = []
            runtime_scores = []
            
            for i in range(iter):
                print(f"Training {name} for {i} iters")
                
                train_data, _, test_data = get_train_vail_test(self.type, self.data)
                
                X_y_train = train_data
                X_y_test = test_data
                    
                start_time = time.time()
                
                model.fit(train= X_y_train, validation= X_y_test)
                
                end_time = time.time()  # End time
                
                runtime = end_time - start_time
                
                result = model.evaluate(X_y_test)
                mse = result[0]["test_loss"]

                mse_scores.append(mse)
                runtime_scores.append(runtime)
            
            mse_mean = np.mean(mse_scores)
            mse_std = np.std(mse_scores)
            
            time_mean = np.mean(runtime_scores)
            time_std = np.std(runtime_scores)

            
            # Store results
            results[name] = {"MSE mean": mse_mean, "MSE std": mse_std, "Time mean (s)": time_mean, "Time std (s)": time_std}

        results_df = pd.DataFrame(results).transpose().round(4)
        
        return results_df
    
    
    def set_model_param(self, model_name, param):
        model_config = self.model_config[model_name]
        model_config.update(param)
            
        model = TabularModel(
            data_config=self.data_config,
            model_config= model_config,
            optimizer_config= self.optimizer_config,
            trainer_config= self.trainer_config
        )
        
        self.models[model_name] = model
        
    def tune(self, model_name, search_space, n_trials = 10):
        
        results = []
        unique_params = set()
        
        # Compute total number of combinations
        total_combinations = 1
        for values in search_space.values():
            total_combinations *= len(values)
            
        for i in range(min(n_trials, total_combinations)):
            while True:
                # Generate random parameters
                
                param = {k: random.choice(v) for k, v in search_space.items()}
                # Convert to tuple and sort
                param_tuple = tuple(sorted(param.items()))
                
                # Check if the parameters are unique
                if param_tuple not in unique_params:
                    unique_params.add(param_tuple)
                    break
                
                # Break if all possible combinations have been tried
                if len(unique_params) == total_combinations:
                    print("Tried all possible parameter combinations. Stopping tuning")
                    break
            
            print(f"Training {model_name} for {i} iter with {param}")
            
            train_data, val_data, _ = get_train_vail_test(self.type, self.data)
            
            model_config = self.model_config[model_name]
            model_config.update(param)
            
            model = TabularModel(
                data_config=self.data_config,
                model_config= model_config,
                optimizer_config= self.optimizer_config,
                trainer_config= self.trainer_config
            )
            
            start_time = time.time()
                
            model.fit(train= train_data, validation= val_data)
            
            end_time = time.time()  # End time
            
            runtime = end_time - start_time
            
            result = model.evaluate(val_data)
            mse = result[0]["test_loss"]
 
            # Store results
            results.append({**param, "MSE": mse, "Time (s)": runtime})

        results_df = pd.DataFrame(results).round(4).sort_values(by= "MSE", ascending= True)
        
        return results_df