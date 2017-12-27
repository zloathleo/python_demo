import tensorflow as tf
# NumPy 经常用于加载、操作和预处理数据。
import numpy as np

# 声明特征列表。 我们只有一个数字特征。 有许多
# 其它类型的列，它们更复杂而且更有用。
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). 有许多预定义类型，如线性回归、
# 线性分类以及许多神经网络分类器和回归器。
# 下面的代码给出一个estimator，它实现线性回归。
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# TensorFlow 提供许多辅助方法来读取和建立数据集。
# 这里我们使用两个数据集：一个用于训练，一个用于评估。
# 我们必须告诉函数
# 我们想要数据的多少个batch(num_epochs)以及每个batch的大小。
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# We can invoke 1000 training steps by invoking the  method and passing the
# training data set.
estimator.train(input_fn=input_fn, steps=1000)

# 这里我们评估我们的模型表现如何
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)