import io
import nbformat
import requests

# Get the contents of a text file as a string
def read_txtfile(file_obj):
    # Read the contents of a text file as a string
    stringio = io.StringIO(file_obj.getvalue().decode("utf-8"))
    text = stringio.read()
    return text

# A function to get the contents (nbformat) of a file with the extension ipynb
def read_ipynb(file_obj):
    # Read the contents from the object directly into memory
    file_content = file_obj.read().decode("utf-8")
    # Read the file contents with nbformat
    notebook = nbformat.reads(file_content, as_version=4)
    return notebook

# A function that separates the contents of a file with the extension ipynb into comments and code and obtains them as strings.
def extract_code_and_comments(notebook):
    text = ""
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            # Processing code cells
            text += "\n# Code start\n"
            text += "".join(cell['source'])  # Add code as text
            text += "\n# Code end\n"
        
        elif cell['cell_type'] == 'markdown':
            # Comment (Markdown cell) processing
            text += "\n# Markdown start\n"
            text += "".join(cell['source'])  # Add comment as is
            text += "\n# Markdown end\n"
    
    return text

#Function to retrieve comments and code strings from a github source file
def get_file_from_github(github_url,token):

    try:
        parts = github_url.split("/")
        if "blob" not in parts:
            raise ValueError("Include 'blob' in the URL. Specify the correct GitHub file URL.")

        repo_owner = parts[3]
        repo_name = parts[4]
        branch = parts[6]
        file_path = "/".join(parts[7:])
        raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{file_path}"
        
        headers = {'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3+json'}
        response = requests.get(raw_url, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Failed to fetch file: {response.status_code}")

        file_data = response.text  

        # For `.ipynb`, extract code and comments
        if file_path.endswith('.ipynb'):
            return extract_code_and_comments_from_ipynb(file_data)
        elif file_path.endswith('.R') or file_path.endswith('.py'):
            return file_data
        else:
            raise Exception("Unsupported file type")

    except Exception as e:
        raise Exception(f"Error processing GitHub URL: {e}")

#Extract comments and codes from ipynb files as strings
def extract_code_and_comments_from_ipynb(ipynb_content):
    try:
        # Convert `.ipynb` to `nbformat` format.
        notebook = nbformat.reads(ipynb_content, as_version=4)

        # Extract source code strings
        return extract_code_and_comments(notebook)
    
    except Exception as e:
        raise Exception(f"Error processing .ipynb file: {e}")