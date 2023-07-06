import pandas as pd
import numpy as np
import timeit
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# Import solar training data from csv file
solar_train = pd.read_csv("solar_training.csv")

def format_data(solar_data):
    """Create additional data features for date, time, and labels"""
    # Break timestamp into numeric components
    solar_data["YEAR"] = pd.to_numeric(solar_train["TIMESTAMP"].str[:4])
    solar_data["MONTH"] = pd.to_numeric(solar_train["TIMESTAMP"].str[4:6])
    solar_data["DAY"] = pd.to_numeric(solar_train["TIMESTAMP"].str[6:8])
    solar_data["HOUR"] = pd.to_numeric(solar_train["TIMESTAMP"].str[9:11])

    # Add power 24 hours ahead as labels
    Y = solar_data["POWER"][24:].to_numpy()
    solar_data["Y"] = np.concatenate((Y, np.repeat(np.nan, 24)))
    
    return solar_data.dropna()






def group_data_tree(solar_data, zone):
    """Groups data to form samples and labels for night time classifier"""
    # Split data by zone
    solar_data = solar_data[solar_data["ZONEID"] == zone]
    
    # Select datetime columns as training samples
    X = solar_data[["MONTH", "DAY", "HOUR"]]
    
    # Find labels where power is below threshold
    power_threshold = 0.01
    Y = solar_data["Y"] < power_threshold
    
    return X, Y
    
def train_tree(solar_data, param_grid, n_folds, tree):
    """Fit a decision tree classifier to separate day vs. night"""
    trees = []
    
    for zone in range(1, 4):
        # Group data by zone and format for decision tree
        X_train, Y_train = group_data_tree(solar_data, zone)

        # Perform grid search to optimize hyperparameters
        grid_search = GridSearchCV(tree, param_grid, cv=n_folds, scoring='accuracy')
        grid_search.fit(X_train, Y_train)

        # Store and evaluate best tree
        best_tree = grid_search.best_estimator_
        print(f"Best parameters for Zone {zone}:", grid_search.best_params_)
        trees.append(best_tree)
        
    return trees
    
def eval_tree(solar_data, zone, tree):
    """Compute accuracy of decision tree and visualize output"""
    # Make predictions using classifier
    X_tree, Y_tree = group_data_tree(solar_data, zone)
    Y_pred = tree.predict(X_tree)

    # Compute accuracy of classifier
    acc = tree.score(X_tree, Y_tree)
    print(f"Best accuracy for Zone {zone}:", acc)
    
    # Visualize night predictor vs. power output
    n_hours_plot = 500
    power = solar_data[solar_data["ZONEID"] == zone]["POWER"]
    
    plt.figure(figsize=(15, 3))
    plt.plot(range(n_hours_plot), power[:n_hours_plot], "-b", label="Power")
    plt.plot(range(n_hours_plot), np.logical_not(Y_pred[:n_hours_plot]), "-r", label="Predictor")
    plt.title(f"Power Output Vs. Night Predictor (Zone {zone})")
    plt.xlabel("Hour")
    plt.ylabel("Power Output")
    plt.legend(loc="upper right")
    plt.show()
def gg2():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV

    n_folds = 3

    # Construct decision tree and optimize hyperparameters for each zone
    tree = DecisionTreeClassifier()
    param_grid = {
        "criterion": ["gini", "entropy"],
        "min_samples_leaf": range(1, 5)
    }
    trees = train_tree(solar_train, param_grid, n_folds, tree)
        
    # Compute accuracy of tree and visualize output for Zone 1
    eval_tree(solar_train, 1, trees[0])

    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error

def eval_reg(solar_data, zone, tree, cols, n_hours, step, reg):
    """Compute error metrics and plot predictions for a regression model"""
    # Make predictions on solar data
    X_test, Y_test = group_data_reg(solar_data, zone, cols, n_hours, step)
    Y_pred = reg.predict(X_test)
    
    # Set night samples to 0
    X_tree, Y_tree = group_data_tree(solar_data, zone)
    night = tree.predict(X_tree)
    Y_pred[night[n_hours:]] = 0
    
    # Remove negative predictions
    Y_pred[Y_pred < 0] = 0

    # Compute mean absolute error and root mean squared error
    print(f"Mean absolute error for Zone {zone}:", mean_absolute_error(Y_test, Y_pred))
    print(f"Root mean squared error for Zone {zone}", mean_squared_error(Y_test, Y_pred, squared=False))

    # Visualize true vs. predicted power output
    n_hours_plot = 500
    
    plt.figure(figsize=(15, 3))
    plt.plot(range(n_hours_plot), Y_test[:n_hours_plot], "-b", label="True Power")
    plt.plot(range(n_hours_plot), Y_pred[:n_hours_plot], "-r", label="Predicted Power")
    plt.title(f"True Power Output Vs. Predicted Power Output (Zone {zone})")
    plt.legend(loc="upper right")
    plt.xlabel("Hour")
    plt.show()


from sklearn.preprocessing import StandardScaler

def group_data_reg(solar_data, zone, cols, n_hours, step, train=False):
    """Groups data to form samples and labels for regression model."""
    # Split data by zone
    solar_data = solar_data[solar_data["ZONEID"] == zone]
    
    # Group records to form samples
    X = solar_data[cols]
    n = X.shape[0]
    idx = range(0, n_hours, step)
    X = np.concatenate([X[n_hours-i:n-i] for i in idx], axis=1)
  
    # Use power as labels
    Y = solar_data["POWER"][n_hours:]
    
    if train:
        # Remove night time samples
        power_threshold = 0.01
        day = Y >= power_threshold
        X, Y = X[day], Y[day]
    
    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, Y
    
def train_reg(solar_data, param_grid, n_folds, cols, n_hours, step, reg):
    """Train regression model using grid search"""
    regs = []
    
    for zone in range(1, 4):
        # Group data by zone
        X_train, Y_train = group_data_reg(solar_data, zone, cols, n_hours, step, train=True)

        # Perform grid search to optimize hyperparameters
        grid_search = GridSearchCV(reg, param_grid, cv=n_folds, scoring="neg_mean_absolute_error")
        grid_search.fit(X_train, Y_train)

         # Store and evaluate best regression model
        best_reg = grid_search.best_estimator_
        regs.append(best_reg)
        print(f"Best parameters for Zone {zone}:", grid_search.best_params_)
        
    return regs
    
def gg3():
    from sklearn.neural_network import MLPRegressor

    n_hours_mlp = 20
    step_mlp = 2
    cols_mlp = cols_best

    # Train neural network regressor
    mlp = MLPRegressor()
    param_grid = {
     'hidden_layer_sizes' : [(10,10), (20,20)],
     'activation' : ['tanh', 'relu'],
     'alpha' : [0.0001, 0.001, 0.01],
    }
    mlps = train_reg(solar_train, param_grid, n_folds, cols_mlp, n_hours_mlp, step_mlp, mlp)

    print("")

    # Evaluate neural network regressor for Zone 1
    eval_reg(solar_train, 1, trees[0], cols_mlp, n_hours_mlp, step_mlp, mlps[0])
    
    
def main():
    s1 = timeit.default_timer()  
    # Create numeric datetime and label features for training data
    
    solar_train = pd.read_csv("solar_training.csv")
    solar_train = format_data(solar_train)
    print(solar_train[["TIMESTAMP", "YEAR", "MONTH", "DAY", "HOUR", "POWER", "Y"]].head())
    from sklearn.feature_selection import SelectKBest, f_regression
    # def gg():
    # Select all data columns
    rm_cols = ["ZONEID", "TIMESTAMP", "Y"]
    cols_knn = [col for col in solar_train.columns if col not in rm_cols]

    import seaborn as sns

    # Plot correlation between all variables
    corr_data = solar_train.drop(["ZONEID", "TIMESTAMP"], axis=1)
    plt.figure(figsize=(15, 7))
    dataplot = sns.heatmap(corr_data.corr(), cmap="YlGnBu", annot=True)
    plt.show()
    # Select features with best 7 F-values for regression
    X_best = solar_train[cols_knn]
    k = 8
    kbest = SelectKBest(f_regression, k=k)
    kbest.fit(X_best, solar_train["Y"])

    # Find best 7 features
    cols_best = kbest.get_feature_names_out()
    print(f"Best {k} features:", cols_best)


    n_folds = 3

    # Construct decision tree and optimize hyperparameters for each zone
    tree = DecisionTreeClassifier()
    param_grid = {
        "criterion": ["gini", "entropy"],
        "min_samples_leaf": range(1, 5)
    }
    trees = train_tree(solar_train, param_grid, n_folds, tree)
        
    # Compute accuracy of tree and visualize output for Zone 1
    eval_tree(solar_train, 1, trees[0])
    
    from sklearn.neural_network import MLPRegressor

    n_hours_mlp = 20
    step_mlp = 2
    cols_mlp = cols_best

    # Train neural network regressor
    mlp = MLPRegressor()
    param_grid = {
     'hidden_layer_sizes' : [(10,10), (20,20)],
     'activation' : ['tanh', 'relu'],
     'alpha' : [0.0001, 0.001, 0.01],
    }
    mlps = train_reg(solar_train, param_grid, n_folds, cols_mlp, n_hours_mlp, step_mlp, mlp)

    print("")

    # Evaluate neural network regressor for Zone 1
    eval_reg(solar_train, 1, trees[0], cols_mlp, n_hours_mlp, step_mlp, mlps[0])    
    
    
    s2 = timeit.default_timer()  
    print ('Runing time is Hour:',round((s2 -s1)/3600,2))
    
""" 
"""

if __name__ == "__main__":
    main()