from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig, TabNetModelConfig, GANDALFConfig, FTTransformerConfig, DANetConfig, NodeConfig


import time

def init_shallow_model():
    shallow_model = {
        "KNN": KNeighborsRegressor(),
        "Ridge": Ridge(),
        "DT": DecisionTreeRegressor(),
        "RF": RandomForestRegressor(n_estimators=50),
        "GBDT": GradientBoostingRegressor()
    }
    return shallow_model

def init_deep_model(data_config, optimizer_config, trainer_config, head_config):
    
    TabNet_config = TabNetModelConfig(
        task="regression",
        n_d= 8,
        head="LinearHead",  # Linear Head
        head_config=head_config,  # Linear Head Config
    )
    
    NODE_config = NodeConfig(
        task="regression",
        num_layers= 1,
        num_trees= 128,
        depth= 6,
    )

    # 配置模型
    CEM_config = CategoryEmbeddingModelConfig(
        task="regression",  # 选择任务类型
        layers="256-128-64-32",  # 网络层配置
        activation="LeakyReLU",  # 激活函数
        head="LinearHead",  # Linear Head
        head_config=head_config,  # Linear Head Config
        dropout= 0,
        initialization="kaiming",
        use_batch_norm=True,
    ).__dict__

    DANet_config = DANetConfig(
        task="regression",
        abstlay_dim_1 = 8,
        abstlay_dim_2 = 16,
        n_layers = 8,
        k = 5,
        head="LinearHead",  # Linear Head
        head_config=head_config,  # Linear Head Config
    ).__dict__

    FTT_config = FTTransformerConfig(
        task="regression",
        input_embed_dim= 32,
        ff_dropout = 0,
        add_norm_dropout = 0,
        attn_dropout = 0,
        ff_hidden_multiplier = 4,
        num_attn_blocks = 2,
        num_heads= 8,
        learning_rate= 0.001,
        head="LinearHead",  # Linear Head
        head_config=head_config,  # Linear Head Config
    ).__dict__
    
    GANDALF_config = GANDALFConfig(
        task="regression",
        gflu_stages=10,  # Number of stages in the GFLU block
        gflu_dropout=0.5,  # Dropout in each of the GFLU block
        gflu_feature_init_sparsity=0.4,  # Sparsity of the initial feature selection
        head= "LinearHead",
        head_config=head_config,  # Linear Head Config
        learning_rate=0.01,
    ).__dict__

    # 创建模型
    CEM_model = TabularModel(
        data_config=data_config,
        model_config=CEM_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config
    )

    GANDALF_model = TabularModel(
        data_config=data_config,
        model_config=GANDALF_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config
    )

    FTT_model = TabularModel(
        data_config=data_config,
        model_config=FTT_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config
    )

    DANet_model = TabularModel(
        data_config=data_config,
        model_config=DANet_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config
    )

    TabNet_model = TabularModel(
        data_config=data_config,
        model_config=TabNet_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config
    )
    
    NODE_model = TabularModel(
        data_config=data_config,
        model_config=NODE_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config
    )

    deep_model_config = { 
        "CEM": CEM_config,
        "GANDALF": GANDALF_config,
        "FTT": FTT_config,
        # "NODE": NODE_config,
        # "TabNet": TabNet_config,
    }
    
    deep_model = {
        "CEM": CEM_model,
        "GANDALF": GANDALF_model,
        "FTT": FTT_model,
        # "NODE": NODE_model,
        # "TabNet": TabNet_model
    }
    return deep_model, deep_model_config
