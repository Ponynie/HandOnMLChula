

from sklearn.metrics import mean_absolute_error, mean_squared_error


def model_training():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    data_big = pd.read_csv("Kaggle/df_train.csv")
    data_big_target = data_big['SalePrice']
    data_big_features = data_big.drop(['SalePrice'], axis=1)
    data_big_features_train, data_big_features_test, data_big_target_train, data_big_target_test = train_test_split(data_big_features, data_big_target, test_size=0.2, random_state=42)

    #*LINEAR REGRESSION MODEL---------------------------------------------
    from sklearn.linear_model import LinearRegression
    linear_model = LinearRegression()
    linear_model.fit(data_big_features_train, data_big_target_train)
    print(f'{linear_model.score(data_big_features_test, data_big_target_test):.4f} Linear R-squared')
    
    #*TREE REGRESSION MODEL-----------------------------------------------
    from sklearn.tree import DecisionTreeRegressor
    tree_model = DecisionTreeRegressor(
        max_depth=None,   # Maximum depth of the tree
        min_samples_split=2,  # Minimum number of samples required to split an internal node
        min_samples_leaf=1,   # Minimum number of samples required to be at a leaf node
        max_features=None,   # Number of features to consider when looking for the best split
        random_state=42     # Seed for reproducibility
    )

    tree_model.fit(data_big_features_train, data_big_target_train)
    print(f'{tree_model.score(data_big_features_test, data_big_target_test):.4f} Tree R-squared')

    #*FOREST REGRESSION MODEL----------------------------------------------
    from sklearn.ensemble import RandomForestRegressor
    forest_model = RandomForestRegressor(
        n_estimators=100,  # Number of trees in the forest
        max_depth=None,    # Maximum depth of the tree
        min_samples_split=2,  # Minimum number of samples required to split an internal node
        min_samples_leaf=1,   # Minimum number of samples required to be at a leaf node
        max_features=10,  # Number of features to consider when looking for the best split
        random_state=42     # Seed for reproducibility
    )
    forest_model.fit(data_big_features_train, data_big_target_train)
    print(f'{forest_model.score(data_big_features_test, data_big_target_test):.4f} Forest R-squared')

    #*BOOST REGRESSION MODEL-----------------------------------------------
    from sklearn.ensemble import GradientBoostingRegressor
    boost_model = GradientBoostingRegressor(
        n_estimators=100,  # Number of boosting stages to be run
        learning_rate=0.1,  # Step size shrinkage to prevent overfitting
        max_depth=3,       # Maximum depth of the individual trees
        min_samples_split=2,  # Minimum number of samples required to split an internal node
        min_samples_leaf=1,   # Minimum number of samples required to be at a leaf node
        max_features=None,   # Number of features to consider when looking for the best split
        random_state=42     # Seed for reproducibility
    )
    boost_model.fit(data_big_features_train, data_big_target_train)
    print(f'{boost_model.score(data_big_features_test, data_big_target_test):.4f} Gradient Boost R-squared')

    #*ALL MODEL REGRESSION PREDICTION
    linear_model_prediction = linear_model.predict(data_big_features_train)
    tree_model_prediction = tree_model.predict(data_big_features_train)
    forest_model_prediction = forest_model.predict(data_big_features_train)
    boost_model_prediction = boost_model.predict(data_big_features_train)

    #*ENSEMBLE MODEL TRAINING DATA
    ensemble_data = pd.DataFrame()
    ensemble_data['linear'] = linear_model_prediction
    ensemble_data['tree'] = tree_model_prediction
    ensemble_data['forest'] = forest_model_prediction
    ensemble_data['boost'] = boost_model_prediction
    ensemble_target = pd.Series(data_big_target_train)
    ensemble_target = ensemble_target.reset_index(drop=True)

    #*ENSEMBLE MODEL TRAINING
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn
    import torch.optim as optim
    ensemble_features_tensor = torch.tensor(ensemble_data.values, dtype=torch.float32)
    ensemble_target_tensor = torch.tensor(ensemble_target.values, dtype=torch.float32)
    print(ensemble_features_tensor.shape, ensemble_target_tensor.shape)
    
    class Neural_Regressor(nn.Module):

        def __init__(self, input_size, layers, p=0.4):
            super(Neural_Regressor, self).__init__()

            all_layers = []

            for i in layers:
                all_layers.append(nn.Linear(input_size, i))
                all_layers.append(nn.ReLU(inplace=True))
                all_layers.append(nn.Dropout(p))
                input_size = i

            all_layers.append(nn.Linear(layers[-1], 1))
            all_layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*all_layers)

        def forward(self, x_tensor):
            logits = self.layers(x_tensor)
            return logits
                
    #*HYPERPARAMETER 2
    lr = 0.001
    epochs = 20
    hidden_layers = [2]
    dropout_p = 0.2
    batch_size = 2
    input_size = ensemble_features_tensor.shape[1]
    ensemble_model = Neural_Regressor(input_size, hidden_layers, p=dropout_p)
    print(ensemble_model)
    train_dataset = TensorDataset(ensemble_features_tensor, ensemble_target_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ensemble_model.parameters(), lr=lr)
    
    #*TRAINING LOOP
    for epoch in range(epochs):
        ensemble_model.train()
        for X, y in train_loader:
            y_pred = ensemble_model(X)
            loss = loss_fn(y_pred, y.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % 4 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')
    
    #*ENSEMBLE MODEL TESTING DATA
    linear_test_prediction = linear_model.predict(data_big_features_test)
    tree_test_prediction = tree_model.predict(data_big_features_test)
    forest_test_prediction = forest_model.predict(data_big_features_test)
    boost_test_prediction = boost_model.predict(data_big_features_test)
    
    #*ENSEMBLE MODEL TESTING
    ensemble_test_data = pd.DataFrame()
    ensemble_test_data['linear'] = linear_test_prediction
    ensemble_test_data['tree'] = tree_test_prediction
    ensemble_test_data['forest'] = forest_test_prediction
    ensemble_test_data['boost'] = boost_test_prediction
    ensemble_test_target = pd.Series(data_big_target_test)
    ensemble_test_target = ensemble_test_target.reset_index(drop=True)
    
    ensemble_test_features_tensor = torch.tensor(ensemble_test_data.values, dtype=torch.float32)
    ensemble_test_target_tensor = torch.tensor(ensemble_test_target.values, dtype=torch.float32)
    
    #*TESTING
    ensemble_model.eval()
    with torch.no_grad():
        y_pred = ensemble_model(ensemble_test_features_tensor)
        loss = loss_fn(y_pred, ensemble_test_target_tensor.unsqueeze(1))
        print(f'Test MSE (loss): {loss.item():.4f}')
        print(f'Test MAE: {mean_absolute_error(ensemble_test_target, y_pred):.4f}')
        varr = torch.var(ensemble_test_target_tensor, unbiased=False)
        r_squared = 1 - (loss / varr)
        print(f'Test R^2: {r_squared.item():.4f}')
        
    print("Done!")   

model_training()