"""
pip install tsfresh
pip install lightgbm
说明文档
https://tsfresh.readthedocs.io/en/latest/

有一个叫做 hctsa 的 matlab 包，可以自动用于 从时间序列中提取特征。 也可以通过 pyopy 包从 Python 中使用 hctsa。 其他可用的打包程序包括 featuretools、FATS 和 cesium。
"""

import pandas as pd
from tsfresh import extract_features
df = pd.DataFrame({
    "id": ["A"]*5 + ["B"]*5,
    "time": list(range(5))*2,
    "value": [1, 2, 3, 2, 1, 5, 4, 4, 3, 5]
})
features = extract_features(df, column_id="id", column_sort="time")

print(features)


from tsfresh import select_features
# 示例目标变量（回归或分类皆可）
y = pd.Series([0, 1], index=["A", "B"])
# 筛选有效特征
filtered = select_features(features, y)
print(filtered)


from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(filtered, y, test_size=0.5)
clf = LGBMClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print("准确率：", accuracy_score(y_test, y_test))