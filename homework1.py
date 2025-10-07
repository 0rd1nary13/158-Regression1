# %%
from collections import defaultdict
from sklearn import linear_model
import numpy
import math

# %%
### Question 1 - 问题1：基于评论长度的线性回归

# %%
def getMaxLen(dataset):
    """
    Calculates the maximum length of a review text in the entire dataset.
    
    Args:
        dataset: The full list of data dictionaries.
        
    Returns:
        int: The length of the longest review.
    """
    # These are the same fields your featureQ1 function checks
    text_fields = ['review/text', 'text', 'review']
    
    # Initialize max_len to 0
    max_len = 0
    
    # Check every single item in the dataset
    for datum in dataset:
        # Find the length of the text field for the current item.
        # Defaults to 0 if no text field is found.
        current_len = next((len(datum[field]) for field in text_fields if field in datum), 0)
        
        # Update max_len if the current one is longer
        if current_len > max_len:
            max_len = current_len
            
    return max_len

# %%
def featureQ1(datum, maxLen):
    """
    为单个数据点创建特征向量
    特征向量只包含scaled_length（缩放后的评论长度，范围在0-1之间）
    sklearn会自动添加偏置项（intercept）
    
    Args:
        datum: 单个数据点（字典）
        maxLen: 数据集中最长评论的长度，用于缩放
        
    Returns:
        float: 缩放后的长度特征
    """
    # 支持多种数据集格式的文本字段名称
    text_fields = ['review/text', 'text', 'review']
    
    # 获取当前数据点的文本长度
    # next()函数找到第一个存在的文本字段并返回其长度，如果都不存在则返回0
    text_len = next((len(datum[field]) for field in text_fields if field in datum), 0)
    
    # 将文本长度缩放到0-1之间
    # 如果maxLen为0（没有文本），则scaled_length为0
    scaled_length = text_len / maxLen if maxLen > 0 else 0
    
    return scaled_length

# %%
def Q1(dataset):
    """
    问题1的主要函数：使用评论长度预测评分（已修正数据过滤逻辑）
    """
    # 步骤 1: 仍然从完整数据集中计算 maxLen，这能确保缩放标准一致
    maxLen = getMaxLen(dataset)
    
    # 步骤 2: 创建两个空列表，用于存储有效的数据
    X_filtered = []
    Y_filtered = []
    
    # 定义可能的字段名
    text_fields = ['review/text', 'text', 'review']
    rating_fields = ['rating', 'overall', 'stars', 'review/overall']
    
    # 步骤 3: 遍历数据集，筛选出有效数据点
    for datum in dataset:
        # 尝试获取评论文本和评分
        text = next((datum[field] for field in text_fields if field in datum), None)
        rating = next((datum[field] for field in rating_fields if field in datum), None)
        
        # **关键：只有当文本和评分都存在时，才处理该数据点**
        if text is not None and rating is not None:
            # 计算缩放后的长度并添加到 X_filtered
            scaled_length = len(text) / maxLen if maxLen > 0 else 0
            X_filtered.append(scaled_length)
            
            # 将评分添加到 Y_filtered
            Y_filtered.append(rating)

    # 步骤 4: 使用筛选后的数据进行回归分析
    # 转换为numpy数组并reshape X为2D数组 (n_samples, 1)
    X = numpy.array(X_filtered).reshape(-1, 1)
    Y = numpy.array(Y_filtered)
    
    # 如果没有有效数据，返回一个默认值避免出错
    if len(X) == 0:
        return ([0, 0], 0)
    
    # 使用scikit-learn的LinearRegression拟合线性回归模型
    model = linear_model.LinearRegression()
    model.fit(X, Y)
    
    # 获取回归系数: [theta_0, theta_1]
    theta = [model.intercept_, model.coef_[0]]
    
    # 计算均方误差（MSE）
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
    parsed_date = datum.get('parsed_date')
    weekday = parsed_date.weekday() if parsed_date else 0  # 0=Monday, 6=Sunday
    month = parsed_date.month if parsed_date else 1  # 1-12
    
    # One-hot encoding for weekday (6 features, dropping Monday=0)
    # Keep Tuesday(1) through Sunday(6)
    day_features = [1 if weekday == i else 0 for i in range(1, 7)]
    
    # One-hot encoding for month (11 features, dropping January=1)
    # Keep February(2) through December(12)
    month_features = [1 if month == i else 0 for i in range(2, 13)]
    
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
    # fit_intercept=False because features already include bias term [1, ...]
    model = linear_model.LinearRegression(fit_intercept=False)
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
    parsed_date = datum.get('parsed_date')
    weekday = parsed_date.weekday() if parsed_date else 0  # 0=Monday, 6=Sunday
    month = parsed_date.month if parsed_date else 1  # 1-12
    
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
    # fit_intercept=False because features already include bias term [1, ...]
    model = linear_model.LinearRegression(fit_intercept=False)
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



