#/// script
#requires-python = ">=3.11"
#dependencies = [
#    "sys",
#    "os",
#    "shutil",
#    "pandas",
#    "seaborn",
#    "matplotlib",
#    "sklearn",
#    "scipy"
#]
#///


import sys
import os
import shutil
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from scipy.stats import skew


API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = os.getenv("AIRPROXY_TOKEN")
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

def load_data(file_path, encodings=['utf-8', 'latin1', 'windows-1252', 'utf-16', 'iso-8859-2', 'cp1250', 'mac-roman']):
    """
    Load the CSV data into a pandas DataFrame with specified encodings.
    Tries each encoding in the list until it successfully loads the data.
    """
    for encoding in encodings:
        try:
            data = pd.read_csv(file_path, encoding=encoding)
            print(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns using {encoding} encoding.")
            return data
        except UnicodeDecodeError:
            print(f"Error with {encoding} encoding. Trying next encoding...")
        except Exception as e:
            print(f"Error loading data with {encoding} encoding: {e}")
            return None
    
    # If none of the encodings worked, return None
    print("Failed to load data with all attempted encodings.")
    return None

def perform_generic_analysis(df):
    """Performs basic statistical and data analysis on the dataset."""
    analysis = {}

    # Ensure dataframe is not empty
    if df.empty:
        return {"error": "The dataframe is empty"}

    # Data structure and summary stats
    analysis['data_structure'] = {
        'columns': df.columns.tolist(),
        'types': df.dtypes.apply(str).tolist(),
    }
    analysis['summary_stats'] = df.describe(include='all')

    # Handle missing values
    analysis['missing_values'] = df.isnull().sum()

    # Skewness of numerical features
    numerical_features = df.select_dtypes(include=np.number)
    analysis['skewness'] = numerical_features.apply(lambda x: skew(x.dropna()))

    # Detecting categorical vs numerical features
    analysis['categorical_features'] = df.select_dtypes(include=['object', 'category']).columns.tolist()
    analysis['numerical_features'] = numerical_features.columns.tolist()

    # Handle missing numerical data before PCA, Imputation using the mean strategy
    numerical_features_for_imputation = df[analysis['numerical_features']]
    imputer = SimpleImputer(strategy='mean')  # You can also use 'median' instead of 'mean'
    df[analysis['numerical_features']] = imputer.fit_transform(numerical_features_for_imputation)

    # Correlation heatmap (512x512 px)
    plt.figure(figsize=(5, 5))

    # Ensure that we only compute correlation on numerical features
    numerical_df = df[analysis['numerical_features']].select_dtypes(include=np.number)
    
    # Compute correlation matrix on only numerical columns
    correlation_matrix = numerical_df.corr()

    # Plot the heatmap if there are numerical columns
    if not correlation_matrix.empty:
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        heatmap_path = "correlation_heatmap.png"
        plt.savefig(heatmap_path, dpi=100)
        plt.close()
    else:
        heatmap_path = None

    # PCA analysis (512x512 px)
    pca_path = None
    if len(analysis['numerical_features']) > 1:
        plt.figure(figsize=(5, 5))
        # Standardize the numerical features for PCA
        df_scaled = StandardScaler().fit_transform(df[analysis['numerical_features']])
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df_scaled)
        
        # Scatter plot of PCA results
        plt.scatter(pca_result[:, 0], pca_result[:, 1])
        plt.title("PCA Analysis")
        pca_path = "pca_analysis.png"
        plt.savefig(pca_path, dpi=100)
        plt.close()

    # Outlier detection using Isolation Forest (512x512 px)
    outlier_path = None
    if len(analysis['numerical_features']) > 1:
        isolation_forest = IsolationForest(contamination=0.1)
        outliers = isolation_forest.fit_predict(df[analysis['numerical_features']])
        df['outliers'] = outliers
        
        # Scatter plot of the outliers
        plt.figure(figsize=(5, 5))
        sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=df['outliers'], palette='coolwarm')
        outlier_path = "outliers.png"
        plt.savefig(outlier_path, dpi=100)
        plt.close()

    # Collecting the analysis results
    analysis['plots'] = [heatmap_path, pca_path, outlier_path]
    return analysis
 

# Generate the LLM prompt based on the analysis
def generate_llm_prompt(analysis, include_images=True):
    """Generate prompt for LLM integration based on analysis."""
 
    image_paths = f"Images: {', '.join(analysis['plots'])}"


    prompt = f"""
    Dataset Description:
    - Columns: {', '.join(analysis['data_structure']['columns'])}
    - Data Types: {', '.join(analysis['data_structure']['types'])}
    - Summary Statistics: {analysis['summary_stats'].to_dict()}

    Missing Values: {analysis['missing_values'].to_dict()}
    Skewness of Features: {analysis['skewness'].to_dict()}

    Categorical Features: {', '.join(analysis['categorical_features'])}
    Numerical Features: {', '.join(analysis['numerical_features'])}

    {image_paths}

    Please provide a detailed narrative:
    1. Provide a brief description of the dataset's structure and key characteristics.
    2. Summarize the main insights from the analysis (e.g., missing data, statistical trends, skewness).
    3. Identify any significant relationships, patterns, or anomalies observed in the data.
    4. Discuss how these findings could inform decision-making or guide further analysis.
    """
    return prompt


# Save the LLM narrative to a README file
def save_narrative_to_readme(narrative, folder):
    """Save the generated narrative into a README.md file inside the specified folder."""
    readme_path = os.path.join(folder, "README.md")  # Save the README in the specified folder
    with open(readme_path, "w") as readme_file:
        readme_file.write(narrative)

def query_llm(prompt):
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
        }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role":"system", "content":"You are a data analyst."},{"role":"user", "content": prompt}],
        "max_tokens": 1000,
        "temperature":0.7
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()

def main(dataset_path, include_images=True):
    """Main function to load dataset, perform analysis, and generate LLM narrative."""

    # Load data
    df = load_data(dataset_path)

    # 1. **Create a unique folder based on the CSV filename**  
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]  # Highlighted: Extract base name of CSV
    output_folder = f"analysis_{dataset_name}"  # Highlighted: Name the folder based on the CSV file name

    # 2. **Create the directory if it doesn't exist**  
    if not os.path.exists(output_folder):  # Highlighted: Check if folder exists and create it if it doesn't
        os.makedirs(output_folder)

    # Perform generic analysis
    analysis = perform_generic_analysis(df)

    # Generate prompt for LLM
    prompt = generate_llm_prompt(analysis, include_images)

    # Get LLM's narrative
    narrative = query_llm(prompt)

    # Save the narrative to README.md inside the folder
    save_narrative_to_readme(narrative, output_folder)  # Highlighted: Save README inside the new folder

    # 3. **Move the plots to the folder**  
    for plot_path in analysis['plots']:  # Highlighted: Iterate over the plots
        if plot_path:  # If the plot exists
            shutil.move(plot_path, os.path.join(output_folder, os.path.basename(plot_path)))  # Highlighted: Move the plot into the folder

    # Print analysis and narrative paths
    print(f"Analysis complete. Plots saved in: {output_folder}")
    print("Narrative saved in README.md")

if len(sys.argv) < 2:
    print("Usage: python autolysis.py <dataset.csv>")
    sys.exit(1)

# Retrieve the CSV file from the command-line argument
csv_file = sys.argv[1]

# Run the data loading and analysis
main(csv_file)
