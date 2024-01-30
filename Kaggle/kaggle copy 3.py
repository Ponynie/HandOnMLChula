

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

    #*NEURAL NETWORK MODEL--------------------------------------------------
    import torch 
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset

    class Net(nn.Module):
        def __init__(self, input_size):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    input_size = data_big_features_train.shape[1]
    net = Net(input_size=input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    trainset = TensorDataset(torch.tensor(data_big_features_train.values, dtype=torch.float32), torch.tensor(data_big_target_train.values, dtype=torch.float32))
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
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
    
    net.eval()
    with torch.no_grad():
        outputs = net(torch.tensor(data_big_features_test.values, dtype=torch.float32))
        loss = criterion(outputs, torch.tensor(data_big_target_test.values, dtype=torch.float32).unsqueeze(1))
        # print(loss.item())
        print(f'{1 - (loss.item() / torch.var(torch.tensor(data_big_target_test.values, dtype=torch.float32), unbiased=False)):.4f} Neural Network R-squared')
    

    #*ALL MODEL REGRESSION PREDICTION
    linear_model_prediction = linear_model.predict(data_big_features_train)
    tree_model_prediction = tree_model.predict(data_big_features_train)
    forest_model_prediction = forest_model.predict(data_big_features_train)
    boost_model_prediction = boost_model.predict(data_big_features_train)


model_training()