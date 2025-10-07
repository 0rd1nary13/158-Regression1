# %%
from collections import defaultdict
from sklearn import linear_model
import numpy
import math

# %%
### Question 1

# %%
def getMaxLen(dataset):
    # Find the longest review (number of characters)
    text_fields = ['review/text', 'text', 'review']
    lengths = [len(datum.get(field, '')) for datum in dataset for field in text_fields if field in datum]
    return max(lengths) if lengths else 0

# %%
def featureQ1(datum, maxLen):
    # Feature vector for one data point
    # Returns [1, scaled_length] where scaled_length is between 0 and 1
    text_fields = ['review/text', 'text', 'review']
    text_len = next((len(datum[field]) for field in text_fields if field in datum), 0)
    scaled_length = text_len / maxLen if maxLen > 0 else 0
    return [1, scaled_length]

# %%
def Q1(dataset):
    # Implement...
    maxLen = getMaxLen(dataset)
    rating_fields = ['rating', 'overall', 'stars', 'review/overall']
    
    # Create feature matrix X and label vector Y using list comprehensions
    X = [featureQ1(datum, maxLen) for datum in dataset]
    Y = [next((datum[field] for field in rating_fields if field in datum), 0) for datum in dataset]
    
    # Convert to numpy arrays
    X, Y = numpy.array(X), numpy.array(Y)
    
    # Train linear regression model
    model = linear_model.LinearRegression()
    model.fit(X, Y)
    
    # Get theta parameters and calculate MSE
    theta = [model.intercept_, model.coef_[1]]
    MSE = numpy.mean((Y - model.predict(X)) ** 2)
    
    return theta, MSE

# %%
### Question 2

# %%
def featureQ2(datum, maxLen):
    # Implement (should be 1, length feature, day feature, month feature)
    # Returns [1, scaled_length, day_features(7), month_features(12)]
    
    # Get text length and scale it
    text_fields = ['review/text', 'text', 'review']
    text_len = next((len(datum[field]) for field in text_fields if field in datum), 0)
    scaled_length = text_len / maxLen if maxLen > 0 else 0
    
    # Get parsed date for weekday and month
    time_struct = datum.get('review/timeStruct')
    if time_struct:
        weekday = time_struct['wday']  # 0=Sunday, 6=Saturday
        month = time_struct['mon']  # 1-12
    else:
        weekday, month = 1, 1  # Default values (Monday, January)
    
    # One-hot encoding for weekday (6 features, dropping Sunday=0)
    day_features = [1 if weekday == i else 0 for i in range(1, 7)]
    
    # One-hot encoding for month (11 features, dropping December=12)
    month_features = [1 if month == i else 0 for i in range(1, 12)]
    
    return [1, scaled_length] + day_features + month_features

# %%
def Q2(dataset):
    # Implement (note MSE should be a *number*, not e.g. an array of length 1)
    maxLen = getMaxLen(dataset)
    rating_fields = ['rating', 'overall', 'stars', 'review/overall']
    
    # Create feature matrix X and label vector Y
    X2 = [featureQ2(datum, maxLen) for datum in dataset]
    Y2 = [next((datum[field] for field in rating_fields if field in datum), 0) for datum in dataset]
    
    # Convert to numpy arrays
    X2, Y2 = numpy.array(X2), numpy.array(Y2)
    
    # Train linear regression model
    model = linear_model.LinearRegression()
    model.fit(X2, Y2)
    
    # Calculate MSE
    MSE2 = numpy.mean((Y2 - model.predict(X2)) ** 2)
    
    return X2, Y2, MSE2

# %%
### Question 3

# %%
def featureQ3(datum, maxLen):
    # Implement
    # Returns [1, scaled_length, weekday, month] - direct numerical features
    
    # Get text length and scale it
    text_fields = ['review/text', 'text', 'review']
    text_len = next((len(datum[field]) for field in text_fields if field in datum), 0)
    scaled_length = text_len / maxLen if maxLen > 0 else 0
    
    # Get parsed date for weekday and month
    time_struct = datum.get('review/timeStruct')
    if time_struct:
        weekday = time_struct['wday']  # 0=Sunday, 6=Saturday
        month = time_struct['mon']  # 1-12
    else:
        weekday, month = 0, 1  # Default values
    
    return [1, scaled_length, weekday, month]

# %%
def Q3(dataset):
    # Implement
    maxLen = getMaxLen(dataset)
    rating_fields = ['rating', 'overall', 'stars', 'review/overall']
    
    # Create feature matrix X and label vector Y
    X3 = [featureQ3(datum, maxLen) for datum in dataset]
    Y3 = [next((datum[field] for field in rating_fields if field in datum), 0) for datum in dataset]
    
    # Convert to numpy arrays
    X3, Y3 = numpy.array(X3), numpy.array(Y3)
    
    # Train linear regression model
    model = linear_model.LinearRegression()
    model.fit(X3, Y3)
    
    # Calculate MSE
    MSE3 = numpy.mean((Y3 - model.predict(X3)) ** 2)
    
    return X3, Y3, MSE3

# %%
### Question 4

# %%
def Q4(dataset):
    # Implement
    # Split data into 50%/50% train/test (first half / second half)
    n = len(dataset)
    train_data = dataset[:n//2]
    test_data = dataset[n//2:]
    
    # Train models on training data
    maxLen = getMaxLen(train_data)
    rating_fields = ['rating', 'overall', 'stars', 'review/overall']
    
    # Prepare training data
    X_train = [featureQ2(datum, maxLen) for datum in train_data]
    Y_train = [next((datum[field] for field in rating_fields if field in datum), 0) for datum in train_data]
    X_train, Y_train = numpy.array(X_train), numpy.array(Y_train)
    
    # Train Q2 model (one-hot encoding)
    model2 = linear_model.LinearRegression()
    model2.fit(X_train, Y_train)
    
    # Train Q3 model (direct numerical)
    X_train3 = [featureQ3(datum, maxLen) for datum in train_data]
    Y_train3 = [next((datum[field] for field in rating_fields if field in datum), 0) for datum in train_data]
    X_train3, Y_train3 = numpy.array(X_train3), numpy.array(Y_train3)
    
    model3 = linear_model.LinearRegression()
    model3.fit(X_train3, Y_train3)
    
    # Test on test data
    X_test = [featureQ2(datum, maxLen) for datum in test_data]
    Y_test = [next((datum[field] for field in rating_fields if field in datum), 0) for datum in test_data]
    X_test, Y_test = numpy.array(X_test), numpy.array(Y_test)
    
    X_test3 = [featureQ3(datum, maxLen) for datum in test_data]
    Y_test3 = [next((datum[field] for field in rating_fields if field in datum), 0) for datum in test_data]
    X_test3, Y_test3 = numpy.array(X_test3), numpy.array(Y_test3)
    
    # Calculate test MSE
    test_mse2 = numpy.mean((Y_test - model2.predict(X_test)) ** 2)
    test_mse3 = numpy.mean((Y_test3 - model3.predict(X_test3)) ** 2)
    
    return test_mse2, test_mse3

# %%
### Question 5

# %%
def featureQ5(datum, maxLen):
    # Implement
    # Returns [1, scaled_length] for beer review classification
    text_fields = ['review/text', 'text', 'review']
    text_len = next((len(datum[field]) for field in text_fields if field in datum), 0)
    scaled_length = text_len / maxLen if maxLen > 0 else 0
    return [1, scaled_length]

# %%
def Q5(dataset, feat_func):
    # Implement
    # Create binary labels: 1 if rating >= 4, 0 otherwise
    rating_fields = ['rating', 'overall', 'stars', 'review/overall']
    y = [1 if next((datum[field] for field in rating_fields if field in datum), 0) >= 4 else 0 for datum in dataset]
    
    # Get max length for scaling
    maxLen = getMaxLen(dataset)
    
    # Create feature matrix
    X = [feat_func(datum, maxLen) for datum in dataset]
    X, y = numpy.array(X), numpy.array(y)
    
    # Train logistic regression with balanced class weights
    model = linear_model.LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate confusion matrix components
    TP = numpy.sum((y == 1) & (y_pred == 1))
    TN = numpy.sum((y == 0) & (y_pred == 0))
    FP = numpy.sum((y == 0) & (y_pred == 1))
    FN = numpy.sum((y == 1) & (y_pred == 0))
    
    # Calculate Balanced Error Rate
    BER = 0.5 * (FP / (FP + TN) + FN / (FN + TP)) if (FP + TN) > 0 and (FN + TP) > 0 else 0.5
    
    return TP, TN, FP, FN, BER

# %%
### Question 6

# %%
def Q6(dataset):
    # Implement
    # Create binary labels and features
    rating_fields = ['rating', 'overall', 'stars', 'review/overall']
    y = [1 if next((datum[field] for field in rating_fields if field in datum), 0) >= 4 else 0 for datum in dataset]
    
    # Get max length for scaling
    maxLen = getMaxLen(dataset)
    
    X = [featureQ5(datum, maxLen) for datum in dataset]
    X, y = numpy.array(X), numpy.array(y)
    
    # Train logistic regression
    model = linear_model.LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X, y)
    
    # Get prediction probabilities
    y_proba = model.predict_proba(X)[:, 1]  # Probability of positive class
    
    # Calculate Precision@K for K in {1, 100, 1000, 10000}
    K_values = [1, 100, 1000, 10000]
    precs = []
    
    for K in K_values:
        # Get top K predictions
        top_k_indices = numpy.argsort(y_proba)[-K:]
        top_k_labels = y[top_k_indices]
        
        # Calculate precision@K
        precision_k = numpy.sum(top_k_labels) / K if K > 0 else 0
        precs.append(precision_k)
    
    return precs

# %%
### Question 7

# %%
def featureQ7(datum, maxLen):
    # Implement (any feature vector which improves performance over Q5)
    # Enhanced features: scaled_length, beer style, ABV, appearance, aroma, taste, palate
    text_fields = ['review/text', 'text', 'review']
    text_len = next((len(datum[field]) for field in text_fields if field in datum), 0)
    scaled_length = text_len / maxLen if maxLen > 0 else 0
    
    # Beer style (one-hot encoding for common styles)
    style = datum.get('beer/style', 'Unknown')
    common_styles = ['IPA', 'Stout', 'Porter', 'Lager', 'Ale', 'Wheat', 'Pilsner', 'Saison']
    style_features = [1 if style in common_style else 0 for common_style in common_styles]
    
    # ABV (alcohol by volume)
    abv = datum.get('beer/ABV', 0)
    
    # Review scores (appearance, aroma, taste, palate)
    appearance = datum.get('review/appearance', 0)
    aroma = datum.get('review/aroma', 0)
    taste = datum.get('review/taste', 0)
    palate = datum.get('review/palate', 0)
    
    # Text length features (log, sqrt of scaled length)
    log_length = numpy.log(scaled_length + 1)
    sqrt_length = numpy.sqrt(scaled_length)
    
    return [1, scaled_length, abv, appearance, aroma, taste, palate, log_length, sqrt_length] + style_features

# %%



