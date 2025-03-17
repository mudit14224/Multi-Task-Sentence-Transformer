# Multi-Task Sentence Transformer

This project implements a multi-task learning model based on the Sentence Transformer architecture. It tackles two tasks:

* **Task A: Sentence Classification:** Classifying sentences into predefined categories.
* **Task B: Named Entity Recognition (NER):** Identifying and classifying named entities within sentences.

These Tasks have a Sentence Transformer model as a backbone. 

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
    git clone https://github.com/mudit14224/Multi-Task-Sentence-Transformer.git
    cd Multi-Task-Sentence-Transformer
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

[Docker Hub Link](https://hub.docker.com/r/mj14224/multi-task-transformer)

To pull the image:

```bash
docker pull mj14224/multi-task-transformer
```

### **Note on Using `.to(device)`**  

The `.to(device)` function is used to ensure that the code can **run on a GPU (CUDA or MPS) if available**; otherwise, it falls back to the **CPU**. This is particularly beneficial when dealing with **large models and datasets**, as running computations on a GPU can significantly speed up training and inference.  

However, in our **hypothetical case**, the dataset is **very small**, and the model itself is relatively lightweight. Because of this, the **time taken to transfer tensors to the GPU** is actually **greater** than the time it takes to **run the entire code on the CPU**. As a result, the code **runs faster without sending data to the GPU**.  

Despite this, I have still included `.to(device)` in **all functions** to maintain **good coding practices**. This ensures that if we scale up the model or dataset in the future, the code will be **ready to utilize GPU acceleration efficiently**. 🚀  
