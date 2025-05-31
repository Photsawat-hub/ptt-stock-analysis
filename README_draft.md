Import Modules


```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
```

Read CSV


```python
df = pd.read_csv("ptt_daily_price.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price</th>
      <th>Adj Close</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ticker</td>
      <td>PTT.BK</td>
      <td>PTT.BK</td>
      <td>PTT.BK</td>
      <td>PTT.BK</td>
      <td>PTT.BK</td>
      <td>PTT.BK</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Date</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-05-02</td>
      <td>31.116962432861328</td>
      <td>33.25</td>
      <td>33.75</td>
      <td>33.25</td>
      <td>33.5</td>
      <td>39656500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-05-03</td>
      <td>31.584888458251953</td>
      <td>33.75</td>
      <td>33.75</td>
      <td>33.25</td>
      <td>33.25</td>
      <td>32885000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-05-07</td>
      <td>31.584888458251953</td>
      <td>33.75</td>
      <td>33.75</td>
      <td>33.75</td>
      <td>33.75</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Delete not interested row


```python
df_cleaned = df.iloc[2:].copy()
df_cleaned.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price</th>
      <th>Adj Close</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2024-05-02</td>
      <td>31.116962432861328</td>
      <td>33.25</td>
      <td>33.75</td>
      <td>33.25</td>
      <td>33.5</td>
      <td>39656500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-05-03</td>
      <td>31.584888458251953</td>
      <td>33.75</td>
      <td>33.75</td>
      <td>33.25</td>
      <td>33.25</td>
      <td>32885000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-05-07</td>
      <td>31.584888458251953</td>
      <td>33.75</td>
      <td>33.75</td>
      <td>33.75</td>
      <td>33.75</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2024-05-08</td>
      <td>31.584888458251953</td>
      <td>33.75</td>
      <td>34.0</td>
      <td>33.5</td>
      <td>33.75</td>
      <td>63054700</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2024-05-09</td>
      <td>31.584888458251953</td>
      <td>33.75</td>
      <td>34.0</td>
      <td>33.5</td>
      <td>33.75</td>
      <td>20770900</td>
    </tr>
  </tbody>
</table>
</div>



Change type of Columns to Numeric


```python
df_cleaned.rename(columns={"Price":"Date"}, inplace=True)
df_cleaned["Date"] = pd.to_datetime(df_cleaned["Date"])

cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
df_cleaned[cols] = df_cleaned[cols].apply(pd.to_numeric)
df_cleaned.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Adj Close</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2024-05-02</td>
      <td>31.116962</td>
      <td>33.25</td>
      <td>33.75</td>
      <td>33.25</td>
      <td>33.50</td>
      <td>39656500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-05-03</td>
      <td>31.584888</td>
      <td>33.75</td>
      <td>33.75</td>
      <td>33.25</td>
      <td>33.25</td>
      <td>32885000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-05-07</td>
      <td>31.584888</td>
      <td>33.75</td>
      <td>33.75</td>
      <td>33.75</td>
      <td>33.75</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2024-05-08</td>
      <td>31.584888</td>
      <td>33.75</td>
      <td>34.00</td>
      <td>33.50</td>
      <td>33.75</td>
      <td>63054700</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2024-05-09</td>
      <td>31.584888</td>
      <td>33.75</td>
      <td>34.00</td>
      <td>33.50</td>
      <td>33.75</td>
      <td>20770900</td>
    </tr>
  </tbody>
</table>
</div>



Exploratory Data Analysis


```python
mean_close = df_cleaned["Close"].mean()
median_close = df_cleaned["Close"].median()
mode_close = df_cleaned["Close"].mode().iloc[0]

print(f"Mean : {mean_close}")
print(f"Median : {median_close}")
print(f"Mode : {mode_close}")
```

    Mean : 32.17110266159696
    Median : 32.25
    Mode : 33.5


Histogram (Data Distribution)


```python
plt.figure(figsize=(10,6))
sns.histplot(df_cleaned["Close"], kde=True, bins=30)
plt.axvline(mean_close, color='red', linestyle='--', label=f'Mean: {mean_close:.2f}')
plt.axvline(median_close, color='green', linestyle='--', label=f'Median: {median_close:.2f}')
plt.axvline(mode_close, color='blue', linestyle='--', label=f'Mode: {mode_close}')
plt.title("Distribution of Closing Prices (PTT.BK)")
plt.xlabel("Close Price")
plt.ylabel("Frequency")
plt.legend()
plt.show()
```


    
![png](output_11_0.png)
    


Finding Outliers


```python
close_prices = df_cleaned["Close"]

Q1 = close_prices.quantile(0.25)
Q3 = close_prices.quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = close_prices[(close_prices < lower_bound)|(close_prices > upper_bound)]

print(f"จำนวน outliers : {len(outliers)}")
print(outliers)
```

    จำนวน outliers : 2
    213    27.25
    214    27.25
    Name: Close, dtype: float64


Boxplot Diagram (Show the Outliers)


```python
sns.boxplot(data=close_prices)
plt.title("Boxplot of Close Prices")
plt.show()
```


    
![png](output_15_0.png)
    


Covarience analysis


```python
covarience = df_cleaned[["Open", "High", "Low", "Close", "Volume"]].cov()

plt.figure(figsize=(8,6))
sns.heatmap(covarience, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title("Covariance Matrix of PTT.BK Price Data")
plt.show()
```


    
![png](output_17_0.png)
    


Correlation analysis


```python
correlation = df_cleaned[['Open', 'High', 'Low', 'Close', 'Volume']].corr()

print(correlation)

plt.figure(figsize=(8,6))
sns.heatmap(correlation, annot=True, cmap='Blues')
plt.title('Correlation Matrix of Stock Features')
plt.show()
```

                Open      High       Low     Close    Volume
    Open    1.000000  0.981659  0.985374  0.968853 -0.333568
    High    0.981659  1.000000  0.974576  0.982557 -0.243713
    Low     0.985374  0.974576  1.000000  0.981152 -0.419309
    Close   0.968853  0.982557  0.981152  1.000000 -0.325033
    Volume -0.333568 -0.243713 -0.419309 -0.325033  1.000000



    
![png](output_19_1.png)
    


Hypothesis Testing (Paired t-test)


```python
before = df_cleaned[(df_cleaned["Date"] >= "2025-04-01") & (df_cleaned["Date"] <= "2025-04-11")]["Close"]
after = df_cleaned[(df_cleaned["Date"] >= "2025-04-17") & (df_cleaned["Date"] <= "2025-04-28")]["Close"]

min_len = min(len(before), len(after))
before = before[:min_len]
after = after[:min_len]

t_stat, p_value = ttest_rel(before, after)

print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.3f}")
```

    T-statistic: 3.464
    P-value: 0.010


Model training (Linear regression to predict Close price of stock market)


```python
#Prepare data
X = df_cleaned[['Open', 'High', 'Low', 'Volume']]
y = df_cleaned['Close']

#Fit Linear Regression
model = LinearRegression()
model.fit(X, y)

#Predict Close price by Open price
df_cleaned['Predicted_Close'] = model.predict(X)
df_cleaned.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Adj Close</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
      <th>Predicted_Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2024-05-02</td>
      <td>31.116962</td>
      <td>33.25</td>
      <td>33.75</td>
      <td>33.25</td>
      <td>33.50</td>
      <td>39656500</td>
      <td>33.474682</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-05-03</td>
      <td>31.584888</td>
      <td>33.75</td>
      <td>33.75</td>
      <td>33.25</td>
      <td>33.25</td>
      <td>32885000</td>
      <td>33.585136</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-05-07</td>
      <td>31.584888</td>
      <td>33.75</td>
      <td>33.75</td>
      <td>33.75</td>
      <td>33.75</td>
      <td>0</td>
      <td>33.698105</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2024-05-08</td>
      <td>31.584888</td>
      <td>33.75</td>
      <td>34.00</td>
      <td>33.50</td>
      <td>33.75</td>
      <td>63054700</td>
      <td>33.740920</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2024-05-09</td>
      <td>31.584888</td>
      <td>33.75</td>
      <td>34.00</td>
      <td>33.50</td>
      <td>33.75</td>
      <td>20770900</td>
      <td>33.706393</td>
    </tr>
  </tbody>
</table>
</div>



RMSE Calculation


```python
rmse = np.sqrt(mean_squared_error(df_cleaned['Close'], df_cleaned['Predicted_Close']))
print(f"RMSE: {rmse:.3f} Baht")
```

    RMSE: 0.188 Baht


Comparison Graph


```python
plt.figure(figsize=(10,5))
plt.plot(df_cleaned['Date'], df_cleaned['Close'], label='Actual Close', color='blue')
plt.plot(df_cleaned['Date'], df_cleaned['Predicted_Close'], label='Predicted Close', color='orange')
plt.xlabel('Date')
plt.ylabel('Price (Baht)')
plt.title('Actual vs Predicted Close Price (Linear Regression)')
plt.legend()
plt.tight_layout()
plt.show()
```


    
![png](output_27_0.png)
    

