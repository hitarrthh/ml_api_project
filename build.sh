
set -o errexit
pip install -r requirements.txt
python -c "from sklearn.datasets import fetch_california_housing; \
import pandas as pd; \
housing = fetch_california_housing(); \
df = pd.DataFrame(data=housing.data, columns=housing.feature_names); \
df['MedHouseVal'] = housing.target; \
df.to_csv('data/california_housing.csv', index=False)"
python train.py