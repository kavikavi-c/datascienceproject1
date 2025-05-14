from google.colab import files
import pandas as pd

df = pd.read_csv("/content/Fake.csv")  # Use uploaded file name
df.head()
df.info()
df.describe()
df.columns
print("Missing values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,4))
sns.countplot(x='subject', data=df)
plt.title("Distribution of Subjects")
plt.xticks(rotation=45)
plt.show()
df['label'] = 0  
df['text'] = df['title'].astype('category').cat.codes
import pandas as pd
df = pd.read_csv("/content/Fake.csv")  


df = pd.get_dummies(df, columns=['subject'], drop_first=True)
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
df = pd.read_csv("/content/Fake.csv")

# Create the 'label' column and assign 0 to all rows
df['label'] = 0  

# Now, 'subject' column should be present.
# Perform one-hot encoding on the 'subject' column
df = pd.get_dummies(df, columns=['subject'], drop_first=True)

# Convert 'title' column to numerical representation using category codes
df['text'] = df['title'].astype('category').cat.codes

# Define features (X) and target (y)
X = df['text']
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
!pip install --upgrade scikit-learn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the CSV file for fake news
df_fake = pd.read_csv("/content/Fake.csv")

# Create the 'label' column and assign 0 to all rows for fake news
df_fake['label'] = 0  

# Load or create a dataset for real news (replace with your real news data)
# Here's an example if you have a CSV file for real news:
# df_real = pd.read_csv("/content/real_news.csv") 
# Or, if you want to generate some dummy real news data:
df_real = pd.DataFrame({'title': ['Real News 1', 'Real News 2', 'Real News 3'], 
                         'text': ['This is a real news article.', 'Another real news story.', 'Breaking real news.'],
                         'subject': ['News', 'Politics', 'World']})
df_real['label'] = 1  # Assign 1 to real news

# Concatenate the fake and real news dataframes
df = pd.concat([df_fake, df_real], ignore_index=True)

# Perform one-hot encoding on the 'subject' column
df = pd.get_dummies(df, columns=['subject'], drop_first=True)

# Keep the original 'title' column for text processing
# df['text'] = df['title'].astype('category').cat.codes # Comment out or remove this line

# Define features (X) and target (y)
X = df[['title']]  # Use 'title' column for text features
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train['title']) # Use 'title' column here
X_test_tfidf = vectorizer.transform(X_test['title']) # Use 'title' column here

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
sample_text = ["The government has declared a national emergency amid new reports."]
sample_vec = vectorizer.transform(sample_text)
prediction = model.predict(sample_vec)

print("Prediction:", "Fake" if prediction[0] == 0 else "Real")
user_df = pd.DataFrame({'text': sample_text})
user_df['text_vector'] = vectorizer.transform(user_df['text']).toarray().tolist()
!pip install gradio
import gradio as gr
def predict_fake_news(text):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    return "Fake News ❌" if prediction == 0 else "Real News ✅"
iface = gr.Interface(fn=predict_fake_news,
                     inputs="text",
                     outputs="text",
                     title="📰 Fake News Detector",
                     description="Enter a news article to determine if it's real or fake.")
iface.launch()

