#/// script
#requires-python = ">=3.11"
#dependencies = [
#    "sys",
#    "os",
#    "shutil",
#    "requests",
#    "pandas",
#    "seaborn",
#    "matplotlib",
#    "sklearn",
#    "scipy",
#    "logging"
#    "numpy"
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
import logging


# Constants for configurable values
MAX_TOKENS = 1000
FIGURE_SIZE = (5, 5)
OUTLIER_CONTAMINATION = 0.1
ENCODINGS = ['utf-8', 'latin1', 'windows-1252', 'utf-16', 'iso-8859-2', 'cp1250', 'mac-roman']

# API settings
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

if not AIPROXY_TOKEN:
    logging.error("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)


def load_data(file_path, encodings=ENCODINGS):
    """
    Load the CSV data into a pandas DataFrame with specified encodings.
    Tries each encoding in the list until it successfully loads the data.
    """
    for encoding in encodings:
        try:
            data = pd.read_csv(file_path, encoding=encoding)
            logging.info(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns using {encoding} encoding.")
            return data
        except UnicodeDecodeError:
            logging.warning(f"Error with {encoding} encoding. Trying next encoding...")
        except Exception as e:
            logging.error(f"Error loading data with {encoding} encoding: {e}")
            return None
    
    logging.error("Failed to load data with all attempted encodings.")
    return None


def perform_generic_analysis(df, include_outliers=True):
    """Performs basic statistical and data analysis on the dataset."""
    analysis = {}

    # Ensure dataframe is not empty
    if df.empty:
        logging.warning("The dataframe is empty.")
        return {"error": "The dataframe is empty"}

    # Data structure and summary stats
    analysis['data_structure'] = {
        'columns': df.columns.tolist(),
        'types': df.dtypes.apply(str).tolist(),
    }
    analysis['summary_stats'] = df.describe(include='all')

    # Count missing values in each column
    analysis['missing_values'] = df.isnull().sum()

    # Skewness of numerical features
    numerical_features = df.select_dtypes(include=np.number)
    analysis['skewness'] = numerical_features.apply(lambda x: skew(x.dropna()))

    # Detecting categorical vs numerical features
    analysis['categorical_features'] = df.select_dtypes(include=['object', 'category']).columns.tolist()
    analysis['numerical_features'] = numerical_features.columns.tolist()

    # Handle missing numerical data before PCA, Imputation using the mean strategy
    numerical_features_for_imputation = df[analysis['numerical_features']]
    imputer = SimpleImputer(strategy='mean')  # Impute missing values with mean
    df[analysis['numerical_features']] = imputer.fit_transform(numerical_features_for_imputation)

    # Store numerical features for later use in plots
    numerical_features_to_plot = analysis['numerical_features'][:2]  # Take the first two numerical features for plotting

    # Correlation heatmap (512x512 px)
    plt.figure(figsize=FIGURE_SIZE)
    numerical_df = df[analysis['numerical_features']].select_dtypes(include=np.number)
    correlation_matrix = numerical_df.corr()

    # Plot the heatmap if there are numerical columns
    heatmap_path = None
    if not correlation_matrix.empty:
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={'label': 'Correlation coefficient'})
        
        # **Added Titles, Axis Labels, and Colorbar**
        plt.title("Correlation Heatmap of Numerical Features", fontsize=14)
        plt.xlabel("Features", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)  # Rotate feature names for better visibility
        plt.yticks(rotation=45, ha="right", fontsize=10)

        heatmap_path = "correlation_heatmap.png"
        plt.savefig(heatmap_path, dpi=100)
        plt.close()

    # PCA Analysis (512x512 px)
    pca_path = None
    if len(numerical_features_to_plot) > 1:
        plt.figure(figsize=FIGURE_SIZE)
        df_scaled = StandardScaler().fit_transform(df[numerical_features_to_plot])
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df_scaled)

        # **Added Variance Explained, Grid, Titles, and Axis Labels for PCA plot**
        explained_variance = pca.explained_variance_ratio_ * 100  # Percentage of variance explained
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)  # Scatter plot of PCA components
        plt.title(f"PCA Analysis: {numerical_features_to_plot[0]} vs {numerical_features_to_plot[1]}", fontsize=14)
        plt.xlabel(f"Principal Component 1 ({numerical_features_to_plot[0]}) - {explained_variance[0]:.2f}% Variance", fontsize=12)
        plt.ylabel(f"Principal Component 2 ({numerical_features_to_plot[1]}) - {explained_variance[1]:.2f}% Variance", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)  # Add grid for readability

        pca_path = "pca_analysis.png"
        plt.savefig(pca_path, dpi=100)
        plt.close()

    # Outlier detection using Isolation Forest (512x512 px)
    outlier_path = None
    if include_outliers and len(numerical_features_to_plot) > 1:
        isolation_forest = IsolationForest(contamination=OUTLIER_CONTAMINATION)
        outliers = isolation_forest.fit_predict(df[numerical_features_to_plot])
        df['outliers'] = outliers

        # **Highlighted Outliers with Different Colors, Annotations, and Data Density Visualization**
        plt.figure(figsize=FIGURE_SIZE)
        sns.scatterplot(x=df[numerical_features_to_plot[0]], y=df[numerical_features_to_plot[1]], 
                        hue=df['outliers'], palette='coolwarm', style=df['outliers'], markers=["o", "X"], s=100)
        
        # Annotating outliers if needed (Example: Labeling the first 3 outliers)
        for i in range(3):
            plt.text(df[numerical_features_to_plot[0]].iloc[i], df[numerical_features_to_plot[1]].iloc[i], 
                     f"Outlier {i+1}", fontsize=9, ha='right', color='black')

        plt.title(f"Outlier Detection: {numerical_features_to_plot[0]} vs {numerical_features_to_plot[1]}", fontsize=14)
        plt.xlabel(numerical_features_to_plot[0], fontsize=12)
        plt.ylabel(numerical_features_to_plot[1], fontsize=12)
        plt.legend(title="Outliers", loc='upper right')

        outlier_path = "outliers.png"
        plt.savefig(outlier_path, dpi=100)
        plt.close()

    # Collecting the analysis results
    analysis['plots'] = [heatmap_path, pca_path, outlier_path]
    return analysis

def generate_llm_prompt(analysis):
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


def save_narrative_to_readme(narrative, folder):
    """Save the generated narrative into a README.md file inside the specified folder."""
    readme_path = os.path.join(folder, "README.md")
    with open(readme_path, "w") as readme_file:
        readme_file.write(narrative)


def query_llm(prompt):
    """Query the LLM API to generate a narrative based on the analysis prompt."""
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": "You are a data analyst."}, {"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}") 
        return "Error: Could not generate response from LLM."


def main(dataset_path, include_outliers=True):
    """Main function to load dataset, perform analysis, and generate LLM narrative."""
    # Load data
    df = load_data(dataset_path)
    
    if df is None:
        logging.error("Data loading failed. Exiting")
        sys.exit(1)
    
    # Create a unique folder based on the CSV filename 
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0] 
    output_folder = f"analysis_{dataset_name}"

    # Create the directory if it doesn't exist
    try:
        os.makedirs(output_folder, exist_ok=True)
    except OSError as e:
        logging.error(f"Error creating folder: {e}")
        sys.exit(1)

    # Perform generic analysis
    analysis = perform_generic_analysis(df, include_outliers)

    # Generate prompt for LLM
    prompt = generate_llm_prompt(analysis)

    # Get LLM's narrative
    narrative = query_llm(prompt)

    # Save the narrative to README.md inside the folder
    save_narrative_to_readme(narrative, output_folder)

    # Move the plots to the folder
    for plot_path in analysis['plots']: 
        if plot_path: 
            shutil.move(plot_path, os.path.join(output_folder, os.path.basename(plot_path)))

    # Print analysis and narrative paths
    logging.info(f"Analysis complete. Plots saved in: {output_folder}")
    logging.info("Narrative saved in README.md")


# Command-line execution check
if len(sys.argv) < 2:
    logging.error("Usage: uv autolysis.py <dataset.csv>")
    sys.exit(1)

# Retrieve the CSV file from the command-line argument
csv_file = sys.argv[1]

# Run the analysis
main(csv_file)
