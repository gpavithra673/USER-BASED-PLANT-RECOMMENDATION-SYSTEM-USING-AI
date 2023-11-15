# USER-BASED PLANT RECOMMENDATION SYSTEM FOR DECORATION USING AI
## PROJECT OUTLINE:
"Botanical Harmony: A Content-Based Plant Decor Recommendation System" is an essential task to serve the purpose of recommending stuffs.Combining computer science and linguistics, NER studies various theories and methods for effective communication between humans and computers using natural language, aiming to extract specific entities from unstructured text relationships.
For example, names of people, places, organizations etc.
Here we are using CountVectorizor to organize a relation between words of the given input dataset.The second part of the model is that, we will be using Kmean Clustering to understand the complex relationship between the given matrix of CountVectorizor. In the third part of model, we will predict the output using sample test data and obtain the result.
In the realm of interior decor, individuals harbor diverse preferences and aspirations for enhancing the aesthetic appeal of their surroundings, aspiring for a contemporary ambiance. Indoor plants, including numerous hybrid species adaptable to diverse environments, emerge as a versatile option for embellishing living spaces. By incorporating these plants, one can create an elegant home environment or cultivate a stylish garden featuring a variety of captivating species. To facilitate the selection of the most fitting plant based on user preferences, a recommendation system proves invaluable. This system efficiently caters to user requests, streamlining the process of suggesting suitable plants for an enhanced and personalized decor experience.

## METHODOLOGY:
The objective of this project is to offer users personalized plant recommendations based on their preferences. We will gather relevant information, including the user's preferred location for placing the plants and specific criteria such as leaf color or shape. By processing this input data, we can generate output suggesting plants that are well-suited to the specified conditions. Currently, our project provides plant names as the output; however, we have the potential to enhance the user experience by expanding the output scope. This expansion could include detailed information and images of the recommended plants, providing users with a comprehensive understanding and visual representation. This not only aids in better decision-making for users but also enriches their overall engagement with the recommended plant selections.


## REQUIREMENTS:
A suitable python environment

Python packages:

pandas
train_test_split
CountVectorizer
KMeans

## PROGRAM:
### IMPORTING AND SPLITING DATASET INTO TRAINING SET AND TESTING SET
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
```
### loading the dataset to a pandas Dataframe
```
data=pd.read_csv("Plant_DS.csv")
data=pd.DataFrame(data)
data.head()
```
### preprocessing of data
```
word_to_append='water consumption'
data['water consumption'] = [word +' '+ word_to_append for word in data['water consumption']]
column_name=['Type','less care','more care','Max- Size','bears flower/not']
data=data.drop(columns=column_name,axis=1)
df={'number': [i for i in range(1,90)],'data': data['Design']+" "+data['Sunlight requirement']+" "+data['water consumption']+" "+data['color']+" "+data['Direction- North,south,east,west']+" "+data['Plant_Names']}
df=pd.DataFrame(df)
df.head()
```
### applying CountVectorizer
```
vectorizer= CountVectorizer()
x=vectorizer.fit_transform(df['data'])
print(x)
```
### applying kmean clusttering's elbow method
```
kmean=KMeans(n_clusters=5, init='k-means++')
kmean.fit(x)
kmean.inertia_
SSE = []
for cluster in range(1,10):
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(x)
    SSE.append(kmeans.inertia_)
```
### converting the results into a dataframe and plotting them
```
frame = pd.DataFrame({'Cluster':range(1,10), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
```
### Applying kmean clustering
```
kmean=KMeans(n_clusters=9, init='k-means++')
kmean.fit(x)
from pandas.core.algorithms import value_counts
predict=kmean.predict(x)
frame=pd.DataFrame(x)
frame['Cluster']=predict
frame['Cluster'].value_counts()
data["cluster"]=kmean.predict(x)
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
df5 = data[data["cluster"]==5]
plt.scatter(df0["Plant_Names"],df0["Design"],c="red",label="cluster-0")
plt.scatter(df1["Plant_Names"],df1["Design"],c="green",label="cluster-1")
plt.scatter(df2["Plant_Names"],df2["Design"],c="blue",label="cluster-2")
plt.scatter(df3["Plant_Names"],df3["Design"],c="pink",label="cluster-3")
plt.scatter(df4["Plant_Names"],df4["Design"],c="yellow",label="cluster-4")
plt.scatter(df5["Plant_Names"],df5["Design"],c="purple",label="cluster-4")
plt.legend()
```
### prediction 
```
sample_input =["bright spot,no need of direct sunlight,deep green,Garden,Common desk"]
sample_input_bow = vectorizer.transform(sample_input)
predicted_cluster = kmeans.predict(sample_input_bow)

print("\nPredicted Cluster for Sample Input:")
print(sample_input[0])
print(f"Predicted Cluster: {predicted_cluster[0] + 1}")
output=[]
for i in range(89):
  if(data.at[i,'cluster']==predicted_cluster):
    output.append(data.at[i,'Plant_Names'])
print("\n")
print("Recommended plants: ")
for i in range(3):
  print(output[i])
```
## FLOW OF THE PROJECT:
### 1)Load Dataset and Display Dataset
### 2)Data Preprocessing 
### 3)Elbow method
### 4)KMean clustering
### 5)Prediction using new data

![Business process flow example](https://github.com/gpavithra673/USER-BASED-PLANT-RECOMMENDATION-SYSTEM-USING-AI/assets/93427264/a746e1b4-d449-4f77-98a3-52c305834864)

## OUTPUT:
### KMean Cluster:
![image](https://github.com/gpavithra673/USER-BASED-PLANT-RECOMMENDATION-SYSTEM-USING-AI/assets/93427264/54645bed-e436-4914-8b72-b0b588f35f81)
### Predicted output:
![image](https://github.com/gpavithra673/USER-BASED-PLANT-RECOMMENDATION-SYSTEM-USING-AI/assets/93427264/249bb904-0030-477c-8426-1723677cf2c2)

## RESULT:
The ultimate goal of the experiment is to create a machine learning model that allows for bettter recommendation based on user request and to provide a proper direction for the user to maintain these plants.
