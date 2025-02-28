import pandas as pd

# Load the dataset
df = pd.read_csv('SalaryData.csv')

# Display the first few rows of the dataset
print("Original Data:")
print(df.head())

# Display the shape of the dataset
print("Dataset shape:", df.shape)

# Drop NaN values
df = df.dropna()
print("Shape after dropping NaN values:", df.shape)

# Show unique values in the specified columns
print("Unique values in 'Education Level':", df['Education Level'].unique())
print("Unique values in 'Job Title':", df['Job Title'].unique())
print("Unique values in 'Gender':", df['Gender'].unique())

# === Create Decoding Mappings ===
# To preserve the original mapping, we create the categorical objects before conversion.
gender_cat = pd.Categorical(df['Gender'])
education_cat = pd.Categorical(df['Education Level'])
job_title_cat = pd.Categorical(df['Job Title'])

# Generate mapping dictionaries (number -> original category)
gender_mapping = dict(enumerate(gender_cat.categories))
education_mapping = dict(enumerate(education_cat.categories))
job_title_mapping = dict(enumerate(job_title_cat.categories))

# Create a new DataFrame that consolidates the decoding information
decoding_df = pd.DataFrame(
    [{'Column': 'Gender', 'Code': code, 'Category': category} 
     for code, category in gender_mapping.items()] +
    [{'Column': 'Education Level', 'Code': code, 'Category': category} 
     for code, category in education_mapping.items()] +
    [{'Column': 'Job Title', 'Code': code, 'Category': category} 
     for code, category in job_title_mapping.items()]
)

# === Convert Categorical Variables to Numeric Codes ===
df['Gender'] = df['Gender'].astype('category').cat.codes
df['Education Level'] = df['Education Level'].astype('category').cat.codes
df['Job Title'] = df['Job Title'].astype('category').cat.codes

# Display the modified DataFrame with numeric codes
print("Modified DataFrame with Numeric Codes:")
print(df.head())
df.to_excel('ConvertedSalaryData.xlsx')

# Display the decoding DataFrame to show what each number represents
print("Decoding DataFrame:")
print(decoding_df)
