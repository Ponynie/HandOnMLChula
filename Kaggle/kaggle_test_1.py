

from sklearn.metrics import mean_absolute_error, mean_squared_error


def model_training():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    data_big = pd.read_csv("Kaggle/df_train.csv")
    # data_big = data_big.sample(frac=1).reset_index(drop=True)
    data_big_target = data_big['SalePrice']
    data_big_features = data_big.drop(['SalePrice'], axis=1)
    data_big_features_train, data_big_features_test, data_big_target_train, data_big_target_test = train_test_split(data_big_features, data_big_target, test_size=0.2, random_state=42)
    #train80% -> test 20%
    # total_records = data_big_features_train.shape[0]
    #total records is 80% remaining

    #*HYPERPARAMETER 1
    t = 0.2
    
    # #*LINEAR REGRESSION SPLIT
    # data_features_linear = data_big_features_train[:int(total_records*0.25)]
    # data_target_linear = data_big_target_train[:int(total_records*0.25)]   
    # data_features_linear_train, data_features_linear_test, data_target_linear_train, data_target_linear_test = train_test_split(data_features_linear, data_target_linear, test_size=t, random_state=42) 

    # #*TREE REGRESSION SPLIT
    # data_features_tree = data_big_features_train[int(total_records*0.25):int(total_records*0.5)]
    # data_target_tree = data_big_target_train[int(total_records*0.25):int(total_records*0.5)]
    # data_features_tree_train, data_features_tree_test, data_target_tree_train, data_target_tree_test = train_test_split(data_features_tree, data_target_tree, test_size=t, random_state=42)

    # #*FOREST REGRESSION SPLIT
    # data_features_forest = data_big_features_train[int(total_records*0.5):int(total_records*0.75)]
    # data_target_forest = data_big_target_train[int(total_records*0.5):int(total_records*0.75)]
    # data_features_forest_train, data_features_forest_test, data_target_forest_train, data_target_forest_test = train_test_split(data_features_forest, data_target_forest, test_size=t, random_state=42)

    # #*BOOST REGRESSION SPLIT
    # data_features_boost = data_big_features_train[int(total_records*0.75):]
    # data_target_boost = data_big_target_train[int(total_records*0.75):]
    # data_features_boost_train, data_features_boost_test, data_target_boost_train, data_target_boost_test = train_test_split(data_features_boost, data_target_boost, test_size=t, random_state=42)

    #*LINEAR REGRESSION MODEL
    from sklearn.linear_model import LinearRegression
    linear_model = LinearRegression()
    linear_model.fit(data_big_features_train, data_big_target_train)
    print(linear_model.score(data_big_features_test, data_big_target_test))
    
    #*TREE REGRESSION MODEL
    from sklearn.tree import DecisionTreeRegressor
    tree_model = DecisionTreeRegressor()
    tree_model.fit(data_big_features_train, data_big_target_train)
    print(tree_model.score(data_big_features_test, data_big_target_test))
    
    #*FOREST REGRESSION MODEL
    from sklearn.ensemble import RandomForestRegressor
    forest_model = RandomForestRegressor()
    forest_model.fit(data_big_features_train, data_big_target_train)
    print(forest_model.score(data_big_features_test, data_big_target_test))

    #*BOOST REGRESSION MODEL
    from sklearn.ensemble import GradientBoostingRegressor
    boost_model = GradientBoostingRegressor()
    boost_model.fit(data_big_features_train, data_big_target_train)
    print(boost_model.score(data_big_features_test, data_big_target_test))

    #*ALL MODEL REGRESSION PREDICTION
    linear_model_prediction = linear_model.predict(data_big_features_train)
    # linear_loss = mean_squared_error(data_big_target_train, linear_model_prediction)
    # print(linear_loss)
    
    tree_model_prediction = tree_model.predict(data_big_features_train)
    forest_model_prediction = forest_model.predict(data_big_features_train)
    boost_model_prediction = boost_model.predict(data_big_features_train)
    

    #*ENSEMBLE MODEL TRAINING DATA
    ensemble_data = pd.DataFrame()
    ensemble_data['linear'] = linear_model_prediction
    ensemble_data['tree'] = tree_model_prediction
    ensemble_data['forest'] = forest_model_prediction
    ensemble_data['boost'] = boost_model_prediction
    ensemble_target = data_big_target_train

    #*ENSEMBLE MODEL TRAINING
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn
    import torch.optim as optim
    ensemble_features_tensor = torch.tensor(ensemble_data.values, dtype=torch.float32)
    ensemble_target_tensor = torch.tensor(ensemble_target.values, dtype=torch.float32)
    
    class Neural_Regressor(nn.Module):

        def __init__(self, input_size, layers, p=0.4):
            super().__init__()

            all_layers = []

            for i in layers:
                all_layers.append(nn.Linear(input_size, i))
                all_layers.append(nn.ReLU(inplace=True))
                all_layers.append(nn.Dropout(p))
                input_size = i

            all_layers.append(nn.Linear(layers[-1], 1))

            self.layers = nn.Sequential(*all_layers)

        def forward(self, x_tensor):
            logits = self.layers(x_tensor)
            return logits
        
    class CustomDataset(TensorDataset):
        def __init__(self, x_tensor, y_tensor):
            super().__init__(x_tensor, y_tensor)
            self.x_tensor = x_tensor
            self.y_tensor = y_tensor
            
        def __getitem__(self, index):
            return self.x_tensor[index], self.y_tensor[index]

        def __len__(self):
            return len(self.y_tensor)
                
    #*HYPERPARAMETER 2
    lr = 0.01
    epochs = 20
    hidden_layers = [3,2]
    dropout_p = 0.2
    batch_size = 10
    ensemble_model = Neural_Regressor(4, hidden_layers, p=dropout_p)
    print(ensemble_model)
    train_dataset = CustomDataset(ensemble_features_tensor, ensemble_target_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ensemble_model.parameters(), lr=lr)
    
    def train_loop(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            pred = pred.squeeze()
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 10 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                
    def test_loop(dataloader, model, loss_fn):
        model.eval()
        num_batches = len(dataloader)
        avg_r_squared = 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                pred = pred.squeeze()
                loss = loss_fn(pred, y)
                varr = torch.var(y, unbiased=False)
                r_squared = 1 - (loss / varr)
                avg_r_squared += r_squared.item()

        avg_r_squared /= num_batches
        print(f"Avg r^2: {avg_r_squared:>8f} \n")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(train_loader, ensemble_model, loss_fn, optimizer)
    
    
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
    ensemble_test_target = data_big_target_test
    
    ensemble_test_features_tensor = torch.tensor(ensemble_test_data.values, dtype=torch.float32)
    ensemble_test_target_tensor = torch.tensor(ensemble_test_target.values, dtype=torch.float32)
    
    test_dataset = CustomDataset(ensemble_test_features_tensor, ensemble_test_target_tensor)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    
    test_loop(test_loader, ensemble_model, loss_fn)
    print("Done!")   


model_training()