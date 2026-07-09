import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("dataset.csv")
df.drop("Unnamed: 83", axis=1, inplace=True)

imputer = SimpleImputer(strategy="mean")

for x in df.columns:
    if df[x].dtype == "object":
        df[x], _ = pd.factorize(df[x])

for x in df.columns:
    df[x] = imputer.fit_transform(df[[x]])

x = df.drop("hospital_death", axis=1)
y = df["hospital_death"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression(C=0.1,solver="saga",penalty="l2")

model.fit(x_train,y_train)
print(model.score(x_test,y_test))
# accuracy: 92.2%
