<h2><b><center>SMARTPHONE MARKET ANALYSIS:<br> A Data-Driven Approach to Popularity, Pricing, and Sales Trends</center></b></h2>


<h2>1.1. BUSINESS UNDERSTANDING</h2>
<P>The company (or marketplace) wants to understand the mobile phone market:
    <ul>
        <li>Which brands/models are most popular</li>
        <li>How price, features (battery, screen size, memory), and release date influence popularity and sales</li>
        <li>Which operating systems dominate (Android, iOS, KaiOS, etc.)</li>
        <li>How many sellers are offering each phone model</li>
    </ul>
</P>

<h2>1.2 BUSINESS OBJECTIVES</h2>
<ul>
    <li>Identifying leading brands and products</li>
    <li>Analyzing pricing strategies</li>
     <li>Evaluating operating system performance</li>
     <li>Monitoring consumer demand trends </li>
     <li>Spotting opportunities in niche brands </li>
</ul>


```python
#importing of the libraries 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

```


```python
phone=pd.read_csv("phones_data.csv")
phone.head()
```


```python
#identifying the shape of the data frame
phone.shape
```


```python
#identifying the datatypes information
phone.info()
```


```python
#identifying the null values
phone.isna().idxmax()
```


```python
#filling in the float with meadian
for col in phone.select_dtypes(include='float64').columns:
    if phone[col].isnull().any():
        median_val = phone[col].median()
        phone[col].fillna(median_val, inplace=True)
```


```python
#using the for loop we will fill all the string datatypes with mode 
for col in phone.select_dtypes(include='object').columns:
    if phone[col].isnull().any():
        mode_val = phone[col].mode()[0]
        phone[col].fillna(mode_val, inplace=True)
```


```python
phone.isna().idxmax()
```


```python
phone['release_date'] = pd.to_datetime(phone['release_date'], errors='coerce')

# Then convert to integer in YYYYMMDD format
phone['release_date'] = phone['release_date'].dt.strftime("%Y%m%d").astype(int)
```


```python
phone['year'] = phone['release_date'].astype(str).str[:4].astype(int)
```


```python
phone.head()
```

<h2>EXPLANATORY DATA ANALYSIS</h2>

<h4>PRICE DISTRIBUTION</h4>


```python
# 1. Price distribution
sns.histplot(phone['best_price'], bins=30, kde=True)
plt.title("Distribution of Best Prices")
plt.show()
```

<H4>BEST MARKET PRICE PER PHONE BRAND </H4>


```python
#Average price per brand_name
avg_prc_brnd = phone.groupby('brand_name')["best_price"].sum().sort_values(ascending=False)

# (Optional) limit to top 10 for clearer plot
avg_prc_brnd_10= avg_prc_brnd.head(20)

# Plot
plt.figure(figsize=(12, 6))
avg_prc_brnd_10.plot(kind='barh', color='skyblue')

plt.title('Average Number of phone make per best price')
plt.ylabel('phone make')
plt.xlabel('average best price ')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

<H4>CORRELATIONSHIP BETWEEN BATTERY SIZE AND THE BEST PRICE</H4>

<h4>conc:Microsoft phones are the best in average prices in market</h4>


```python
# 3. Battery vs Price
sns.scatterplot(x='battery_size', y='best_price', data=phone)
plt.title("Battery Size vs Best Price")
plt.show()
```

<h4>conc:There is a negative relationship between the battery size and the price of the phone</h4>




```python
# Screen Size vs Price
sns.scatterplot(x='screen_size', y='best_price', hue='os', data=phone)
plt.title("Screen Size vs Best Price")
plt.show()
```

<H4>THE ANNUAL PRICES TREND LINE FROM 2013 TO 2021</H4>

<h4>CONC:There is a positive relationship between the phone screen and the prices </h4>


```python
# 5. Price trend by release year
sns.lineplot(x='year', y='best_price', data=phone, estimator='mean')
plt.title("Average Phone Price Over Years")
plt.show()
```

<h4>OPERATION SYSTEM vs POPULARITY</h4>

<h4>CONC:There is an yearly increase with phone increases</h4>


```python
# Create a boxplot of OS vs Popularity
plt.figure(figsize=(10,6))
phone.boxplot(column="popularity", by="os", grid=False)
plt.title("OS vs Popularity")
plt.suptitle("")  # Remove automatic title
plt.xlabel("Operating System")
plt.ylabel("Popularity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

```

<h3>The OS with the highest number of popularity is iphone </h3>


```python

```

<H4>BRAND NAME WITH THE HIGHEST PRICES</H4>


```python
make_pop = phone.groupby('brand_name')["highest_price"].sum().sort_values(ascending=False)

# (Optional) limit to top 10 for clearer plot
make_pop_10 = make_pop.head(10)

# Plot
plt.figure(figsize=(12, 6))
make_pop_10.plot(kind='barh', color='skyblue')

plt.title('Average Number of phone make per best price')
plt.ylabel('brand_name')
plt.xlabel( "brand popula")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

<h3>CONC:From the grapgh, phone with the highest number of popularity is blackbery</h3>


```python
# Aggregate sellers_amount by brand
brand_sellers = phone.groupby("brand_name")["sellers_amount"].sum().sort_values(ascending=False)

# Plot Brand vs Sellers Amount
plt.figure(figsize=(12,6))
brand_sellers.plot(kind="bar", color="seagreen")
plt.title("Brand Name vs Total Sellers Amount", fontsize=14)
plt.xlabel("Brand Name")
plt.ylabel("Total Sellers Amount")
plt.xticks(rotation=75)
plt.tight_layout()
plt.show()

brand_sellers.head(10)

```

<h3>From the graph the brand with the highest number of the seeler counts is Nokia</h3>


```python
# Screen Size vs Price
sns.scatterplot(x='popularity', y='best_price', hue='os', data=phone)
plt.title("popularity vs Best Price")
plt.show()
```


```python
plt.figure(figsize=(10,6))

sns.regplot(x="popularity", y="best_price", data=phone,
            scatter_kws={"alpha":0.6},  # transparency for points
            line_kws={"color":"red"})   # color of best-fit line

plt.title("Popularity vs Best Price with Line of Best Fit")
plt.xlabel("Popularity")
plt.ylabel("Best Price")
plt.show()
```

<h3>THis graph shows that there is a apositie relationship between popularity and the best prices in the markets </h3>


```python
#phone.to_csv('cleaned_phone_data.csv', index=False)

```


```python
#converting the Jupyter file to a README file 
import nbformat
from nbconvert import MarkdownExporter

# Load notebook
with open("PHONES_ANAL.ipynb") as f:
    nb = nbformat.read(f, as_version=4)

# Convert to markdown
exporter = MarkdownExporter()
(body, resources) = exporter.from_notebook_node(nb)

# Save as README.md
with open("README.md", "w", encoding="utf-8") as f:
    f.write(body)

```


```python

```

<h1><b>MODELS CREATION AND PRICE PREDICTIONS WITH MACHINE LEARNING</b> </h1>


```python
#display data
phone.head()
```
