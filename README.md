# Multi-Task Sentence Transformer

This project implements a multi-task learning model based on the Sentence Transformer architecture. It tackles two tasks:

* **Task A: Sentence Classification:** Classifying sentences into predefined categories.
* **Task B: Named Entity Recognition (NER):** Identifying and classifying named entities within sentences.

## Project Structure

* `main.py`: Contains the main execution logic, including task implementations and training.
* `sentTransformer.py`: Defines the model architectures, including `SentenceTransformer` and `MultiTaskTransformer`.
* `utils.py`: Contains utility functions such as `preprocess_sentences` and `train_model`.
* `requirements.txt`: Lists the project's dependencies.
* `Dockerfile`: Defines the Docker image for running the application.
* `ref/`: Folder which contains the .ipynb notebook.

## Tasks

* **Task 1: Sentence Transformer Implementation:** Demonstrates the use of the `SentenceTransformer` model to generate sentence embeddings.
* **Task 2: Multi-Task Learning Expansion:** Shows how the `MultiTaskTransformer` model performs forward passes for both sentence classification and NER.
* **Task 4: Training Loop Implementation (BONUS):** Implements and demonstrates a training loop for the `MultiTaskTransformer` model.

## Setup and Installation

1.  **Clone the Repository:**

    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On macOS and Linux
    venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    You can add these lines to the top of your `if __name__ == '__main__':` block in `main.py`.

## Running the Code

Execute the `main.py` script:

```bash
python main.py
```

This will run the demonstrations for each task and print the results to the console.

## Docker

A Dockerfile is included to containerize the application.

1.  **Build the Docker Image:**

    ```bash
    docker build -t <your-dockerhub-username>/multi-task-transformer .
    ```

2.  **Run the Docker Container:**

    ```bash
    docker run <your-dockerhub-username>/multi-task-transformer
    ```

## Docker Hub Image

You can find the Docker image on Docker Hub at:

[Docker Hub Link](<your-docker-hub-image-link>)

To pull the image:

```bash
docker pull <your-dockerhub-username>/multi-task-transformer:latest