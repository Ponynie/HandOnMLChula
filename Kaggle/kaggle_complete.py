

#from sklearn.metrics import mean_absolute_error, mean_squared_error


def model_training():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    data_big = pd.read_csv("Kaggle/df_train.csv")
    data_big_target = data_big['SalePrice']
    data_big_features = data_big.drop(['SalePrice'], axis=1)
    data_big_features_train, data_big_features_test, data_big_target_train, data_big_target_test = train_test_split(data_big_features, data_big_target, test_size=0.2, random_state=42)
    r_squareds = []

    #*LINEAR REGRESSION MODEL---------------------------------------------
    from sklearn.linear_model import LinearRegression
    data_features_train, data_features_test, data_target_train, data_target_test = train_test_split(data_big_features_train, data_big_target_train, test_size=0.1, random_state=1)
    linear_model = LinearRegression()
    linear_model.fit(data_features_train, data_target_train)
    r_squareds.append(linear_model.score(data_features_test, data_target_test))
    print(f'{linear_model.score(data_features_test, data_target_test):.4f} Linear R-squared')
    
    #*TREE REGRESSION MODEL-----------------------------------------------
    from sklearn.tree import DecisionTreeRegressor
    data_features_train, data_features_test, data_target_train, data_target_test = train_test_split(data_big_features_train, data_big_target_train, test_size=0.1, random_state=2)
    tree_model = DecisionTreeRegressor(
        max_depth=None,   # Maximum depth of the tree
        min_samples_split=2,  # Minimum number of samples required to split an internal node
        min_samples_leaf=1,   # Minimum number of samples required to be at a leaf node
        max_features=None,   # Number of features to consider when looking for the best split
        random_state=42     # Seed for reproducibility
    )
    tree_model.fit(data_features_train, data_target_train)
    r_squareds.append(tree_model.score(data_features_test, data_target_test))
    print(f'{tree_model.score(data_features_test, data_target_test):.4f} Tree R-squared')

    #*FOREST REGRESSION MODEL----------------------------------------------
    from sklearn.ensemble import RandomForestRegressor
    data_features_train, data_features_test, data_target_train, data_target_test = train_test_split(data_big_features_train, data_big_target_train, test_size=0.1, random_state=3)
    forest_model = RandomForestRegressor(
        n_estimators=100,  # Number of trees in the forest
        max_depth=None,    # Maximum depth of the tree
        min_samples_split=2,  # Minimum number of samples required to split an internal node
        min_samples_leaf=1,   # Minimum number of samples required to be at a leaf node
        max_features=10,  # Number of features to consider when looking for the best split
        random_state=42     # Seed for reproducibility
    )
    forest_model.fit(data_features_train, data_target_train)
    r_squareds.append(forest_model.score(data_features_test, data_target_test))
    print(f'{forest_model.score(data_features_test, data_target_test):.4f} Forest R-squared')

    #*BOOST REGRESSION MODEL-----------------------------------------------
    from sklearn.ensemble import GradientBoostingRegressor
    data_features_train, data_features_test, data_target_train, data_target_test = train_test_split(data_big_features_train, data_big_target_train, test_size=0.1, random_state=4)
    boost_model = GradientBoostingRegressor(
        n_estimators=100,  # Number of boosting stages to be run
        learning_rate=0.1,  # Step size shrinkage to prevent overfitting
        max_depth=3,       # Maximum depth of the individual trees
        min_samples_split=2,  # Minimum number of samples required to split an internal node
        min_samples_leaf=1,   # Minimum number of samples required to be at a leaf node
        max_features=None,   # Number of features to consider when looking for the best split
        random_state=42     # Seed for reproducibility
    )
    boost_model.fit(data_features_train, data_target_train)
    r_squareds.append(boost_model.score(data_features_test, data_target_test))
    print(f'{boost_model.score(data_features_test, data_target_test):.4f} Gradient Boost R-squared')

    #*NEURAL REGRESSION MODEL--------------------------------------------------
    import torch 
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    data_features_train, data_features_test, data_target_train, data_target_test = train_test_split(data_big_features_train, data_big_target_train, test_size=0.1, random_state=5)
    
    class neural_model(nn.Module):

        def __init__(self, input_size, layers, p=0.4):
            super(neural_model, self).__init__()

            all_layers = []

            for i in layers:
                all_layers.append(nn.Linear(input_size, i))
                all_layers.append(nn.ReLU(inplace=True))
                all_layers.append(nn.Dropout(p))
                input_size = i

            all_layers.append(nn.Linear(layers[-1], 1))
            all_layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*all_layers)

        def forward(self, x):
            logits = self.layers(x)
            return logits
    
    #*NEURAL HYPERPARAMETERS
    input_size = data_features_train.shape[1]
    layers = [100, 100]
    dropout_p = 0.2
    lr = 0.001
    net = neural_model(input_size=input_size, layers=layers, p=dropout_p)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    trainset = TensorDataset(torch.tensor(data_features_train.values, dtype=torch.float32), torch.tensor(data_target_train.values, dtype=torch.float32))
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
    
    #*NEURAL TRAINING LOOP
    for epoch in range(100):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, target = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if i % 1000 == 999:
            #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
            #     running_loss = 0.0
    # print('Finished Training')
    
    #*NEURAL TESTING DATA
    net.eval()
    with torch.no_grad():
        outputs = net(torch.tensor(data_features_test.values, dtype=torch.float32))
        loss = criterion(outputs, torch.tensor(data_target_test.values, dtype=torch.float32).unsqueeze(1))
        # print(loss.item())
        print(f'{1 - (loss.item() / torch.var(torch.tensor(data_target_test.values, dtype=torch.float32), unbiased=False)):.4f} Neural Network R-squared')
        r_squareds.append((1 - (loss.item() / torch.var(torch.tensor(data_target_test.values, dtype=torch.float32), unbiased=False))).item())
    

    #*ALL MODEL REGRESSION PREDICTION--------------------------------------------------
    linear_model_prediction = linear_model.predict(data_big_features_test)
    tree_model_prediction = tree_model.predict(data_big_features_test)
    forest_model_prediction = forest_model.predict(data_big_features_test)
    boost_model_prediction = boost_model.predict(data_big_features_test)
    with torch.no_grad():
        neural_model_prediction = net(torch.tensor(data_big_features_test.values, dtype=torch.float32)).squeeze().detach().numpy()

    
    #*ENSEMBLE MODEL PREDICTION--------------------------------------------------------
    weights = [r2 / sum(r_squareds) for r2 in r_squareds] # Adjust weights based on model performance
    print(weights)
    ensemble_prediction = (weights[0] * linear_model_prediction +
                        weights[1] * tree_model_prediction +
                        weights[2] * forest_model_prediction +
                        weights[3] * boost_model_prediction +
                        weights[4] * neural_model_prediction)
    
    #*ENSEMBLE MODEL EVALUATE-----------------------------------------------------------
    from sklearn.metrics import r2_score

    r2_value = r2_score(data_big_target_test, ensemble_prediction)
    formatted_r2_value = "{:.4f}".format(r2_value)
    print(f'Ensemble R-squared value: {formatted_r2_value}')

    
    


model_training()