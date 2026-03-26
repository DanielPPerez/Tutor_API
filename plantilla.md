# Build a Small Machine Learning Email Search and Filtering Demo

## Description

I have multiple emails stored as text files in a local folder. I need a machine learning-based system that can filter and search through these emails for me. The system should let me look for emails by keywords or by describing what I am looking for in natural language - like chatting with ChatGPT - and then suggest the most relevant emails based on the information I provide. Build a small, functional demo from scratch in Python that illustrates this capability. The solution should be a self-contained repository with source code organized in a `src/` directory, a sample email dataset in `data/emails/`, a `requirements.txt` with pinned dependency versions, and a `README.md` with setup instructions, usage examples, and a brief explanation of the ML approach used.

## Tech Stack

- Python 3.10+
- scikit-learn for TF-IDF vectorization and cosine similarity
- NLTK for text preprocessing (tokenization, stopword removal, stemming)
- pandas for email metadata management
- No external APIs, databases, or internet connectivity required at runtime; all data is file-based

## Key Requirements

### 1. Email Loading and Parsing

- Load all `.txt` email files from a configurable directory path.
- Each email file follows a simple structured format: header lines (`From:`, `To:`, `Subject:`, `Date:`) followed by a blank line and the body text.
- Parse and extract: sender, recipient(s), subject, date, and body content.
- Return a list of dictionaries with keys: `id`, `sender`, `recipients`, `subject`, `date`, `body`, `filename`.
- Raise a `ValueError` if the directory is missing, contains no `.txt` files, or if a file is missing required headers.

### 2. Text Preprocessing

- Lowercase all text.
- Remove URLs, email addresses, punctuation, and special characters, retaining only alphanumeric tokens.
- Tokenize into words.
- Remove English stopwords using NLTK's stopword list.
- Apply Porter Stemmer to normalize word forms.
- Return cleaned text as a single space-separated string.
- Return an empty string when given empty or None input.

### 3. Email Indexing

- Combine each email's subject and body into a single document string.
- Build a TF-IDF matrix using scikit-learn's `TfidfVectorizer` with `max_features=5000` by default.
- Store email metadata and body content in a pandas DataFrame (columns: `id`, `sender`, `recipients`, `subject`, `date`, `body`, `filename`) alongside the TF-IDF matrix.
- Return the DataFrame, TF-IDF matrix, and fitted vectorizer as a tuple.

### 4. Search Engine

- **Keyword search:** Find emails where specified keywords appear in subject, body, or sender fields. Rank results by the number of keyword occurrences.
- **Semantic search:** Transform a natural language query with the fitted TF-IDF vectorizer, compute cosine similarity against all indexed emails, and return results ranked by descending similarity score.
- **Combined search:** Filter by metadata (sender as case-insensitive partial match, date as exact match) first, then rank the filtered subset by semantic similarity.
- All search functions return a list of result dictionaries with keys: `id`, `subject`, `sender`, `date`, `score`, and `snippet` (first 200 characters of body).
- Support a `top_k` parameter (default 5) to limit the number of results returned.
- Return an empty list when no results match.

### 5. Interactive Conversational Interface

- Provide a CLI-based interactive loop that accepts user input until the user exits.
- Natural language input is automatically treated as a semantic search query.
- Supported commands:
  - `search <query>` - semantic search.
  - `keyword <terms>` - keyword search.
  - `filter sender:<name>` - filter by sender (case-insensitive partial match).
  - `filter date:<YYYY-MM-DD>` - filter by date.
  - `show <email_id>` - display the full content of an email by its ID.
  - `help` - display available commands and usage.
  - `quit` or `exit` - terminate the session.
- Display results as a formatted table with columns: Rank, ID, Subject (truncated to 50 chars), Sender, Date, Score.
- Handle empty results with a "No results found." message and invalid commands with a helpful error suggesting the `help` command.

### 6. Sample Email Dataset

- Include a `data/emails/` directory with at least 20 `.txt` email files.
- Cover diverse topics: meetings, budget reports, project updates, technical discussions, personal messages, newsletters, event announcements.
- Each file follows the header format:









Title: CLI Password Manager with AES-256 Encryption
Description: Build a Python-based command-line interface (CLI) application that allows users to securely store and manage their credentials locally
. The application must utilize the AES-256 encryption algorithm to ensure data stored on the disk is unreadable without the correct authorization
. All retrieval operations must be gated by a master password
.
Key Requirements:
Encryption Standard: Implement local storage where all credential data is encrypted using AES-256
.
CRUD Operations: Support adding new credentials, retrieving passwords for a specific service, deleting records, and listing all stored service names
.
Authentication: Require a master password input for all decryption and retrieval actions
.
Persistence: Data must be saved to a local file (e.g., passwords.db) that persists between different application executions
.
User Feedback: The CLI must provide clear success or error messages for every user operation (e.g., "Credential added successfully" or "Invalid master password")
.

--------------------------------------------------------------------------------
## 2. Complete Expected Interface Section
Path
Name
Type
Input
Output
Description
pm.py
PasswordManager
Class
master_pw: str, db_path: str
None
Initializes the manager, sets the master password, and defines the database location
.
pm.py
PasswordManager.add
Method
service: str, user: str, pw: str
None
Encrypts and saves a new set of credentials to the local database
.
pm.py
PasswordManager.get
Method
service: str
str
Decrypts and returns the password for the requested service
.
pm.py
PasswordManager.list
Method
None
List[str]
Returns a list of all service names currently stored in the encrypted database
.
pm.py
PasswordManager.delete
Method
service: str
bool
Removes the service entry; returns True if the entry existed and was deleted
.

--------------------------------------------------------------------------------
3. Unit Test Examples
Well-Written Test:
Reasoning: This test focuses on functional correctness by ensuring the interface behaves as expected according to the "Technical Contract" without dictating internal logic
.
Intentionally Overly Specific Test:
Violation Explanation: This test violates project guidelines because it checks how the code achieved the result (the specific internal library state or initialization vector) rather than what it achieved (correct encryption)
. It "accidentally punishes creativity" because another valid implementation using a random IV (which is safer) would fail this test even if it fulfills all prompt requirements
.

--------------------------------------------------------------------------------
4. Rubric Criterion
Criterion: The implementation uses a strong Key Derivation Function (KDF), such as PBKDF2 or Argon2, to derive the AES key from the master password rather than using the raw password string.
Dimension: Code Quality
.
Weight: 5 (Mandatory)
.
Reasoning: Automated unit tests can check if a password is recovered correctly, but they often cannot qualitatively assess the security strength of the key derivation process, which is critical for a password manager's integrity
.

--------------------------------------------------------------------------------
5. Docker Execution Commands
Baseline Execution (Negative Verification):
# Run the test runner against the empty environment
./run_tests > stdout.txt 2> stderr.txt

# Parse the results into before.json
parse_results stdout.txt stderr.txt before.json
Golden Patch Verification (Positive Verification):
# Run the test runner after the solution is injected into /app
./run_tests > stdout.txt 2> stderr.txt

# Parse the results into after.json
parse_results stdout.txt stderr.txt after.json
(Note: run_tests and parse_results are the standardized symlinks created within the /eval_assets directory during the container setup







4. TTA (Test Time Augmentation) en la App
Esta es una solución que no requiere re-entrenar, se hace en el código de tu API/App.
Cómo funciona: Cuando el usuario envía una imagen, la API crea 5 versiones (original, un poco rotada, un poco más grande, etc.). El modelo predice las 5 y tú haces un promedio de los resultados (Logits).
Beneficio: Elimina errores por "mala suerte" en el ángulo de la foto. Suele subir un 2-3% de precisión de forma gratuita.