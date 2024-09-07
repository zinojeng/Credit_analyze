import nltk
import os
import ssl

# Disable SSL verification (only if necessary)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
print(f"NLTK data path: {nltk_data_path}")

if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
    print(f"Created NLTK data directory: {nltk_data_path}")

# Download required NLTK data
print("Downloading required NLTK data...")
nltk.download('punkt', quiet=True, download_dir=nltk_data_path)
nltk.download('stopwords', quiet=True, download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger', quiet=True, download_dir=nltk_data_path)
print("Download completed.")

# Check if required NLTK data is available
print("punkt available:", nltk.data.find('tokenizers/punkt') is not None)
print("stopwords available:", nltk.data.find('corpora/stopwords') is not None)
print("averaged_perceptron_tagger available:", nltk.data.find('taggers/averaged_perceptron_tagger') is not None)

if os.path.exists(nltk_data_path):
    print(f"NLTK data files: {os.listdir(nltk_data_path)}")
else:
    print("NLTK data directory does not exist.")

# Set NLTK_DATA environment variable
os.environ['NLTK_DATA'] = nltk_data_path
print(f"NLTK_DATA environment variable set to: {os.environ['NLTK_DATA']}")

# Add this line to update nltk.data.path
nltk.data.path.append(nltk_data_path)
print(f"Updated nltk.data.path: {nltk.data.path}")
