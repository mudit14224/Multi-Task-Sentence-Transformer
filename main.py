# Imports
from sentTransformer import SentenceTransformer, MultiTaskTransformer
from utils import preprocess_sentences, train_model, count_parameters
import torch
import torch.nn as nn
import nltk

def sentence_transformer_t1(sentences, device):
    """
    Generates sentence embeddings using the SentenceTransformer model

    Args:
        sentences (list of str): List of sentences to embed

    Returns:
        torch.Tensor: Tensor containing sentence embeddings
    """
    # Get the vocab and the input tensor to the model
    vocab, padded_tensor = preprocess_sentences(sentences)

    # Model Initialization parameters
    VOCAB_SIZE = len(vocab) # length of the vocab
    D_MODEL = 128 # Dimension of the fixed size sentence embedding
    NUM_HEADS = 8 # Number of heads for multi head attention 
    NUM_LAYERS = 4 # Number of repeated multiheadattention blocks
    D_FF = 512 # intermediate dimension for feedforward network 
    # Initialize the model with the hyperparameters
    model = SentenceTransformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, max_len=len(padded_tensor[0])).to(device)

    # To get an idea about the model size
    print(f"Total Trainable Parameters: {count_parameters(model)}")

    # Move input to device
    padded_tensor = padded_tensor.to(device)  

    # Generate Sentence Embeddings
    embeddings = model(padded_tensor)

    return embeddings

def multi_task_transformer_t2(sentences, device): 
    """
    Performs a forward pass through the MultiTaskTransformer model

    Args:
        sentences (list of str): List of sentences to process

    Returns:
        tuple: A tuple containing output tensors for sentence classification and NER
    """
    # Get the vocab and the input tensor to the model
    vocab, padded_tensor = preprocess_sentences(sentences)

    VOCAB_SIZE = len(vocab) # length of the vocab
    D_MODEL = 128 # Dimension of the fixed size sentence embedding
    NUM_HEADS = 8 # Number of heads for multi head attention
    NUM_LAYERS = 4 # Number of repeated multiheadattention blocks
    D_FF = 512 # intermediate dimension for feedforward network
    MAX_LEN = len(padded_tensor[0]) # Max len of sentence in the data
    NUM_CLASSES_A = 3 # Number of classes for task A: Sentence classification
    NUM_CLASSES_B = 4 # Number of classes for task B: NER 
    # Initialize the model with the hyperparameters
    multi_task_model = MultiTaskTransformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, NUM_CLASSES_A, NUM_CLASSES_B, MAX_LEN).to(device)

    # To get an idea about the model size
    print(f"Total Trainable Parameters: {count_parameters(multi_task_model)}")

    # Move input to device
    padded_tensor = padded_tensor.to(device) 
    
    # Get the output from the model
    out_task_a, out_task_b = multi_task_model(padded_tensor)

    return out_task_a, out_task_b

def train_multi_task_transformer_t4(sentences, device):
    """
    Trains the MultiTaskTransformer model and returns the trained model

    Args:
        sentences (list of str): List of sentences for training

    Returns:
        MultiTaskTransformer: Trained MultiTaskTransformer model
    """
    # Get the vocab and the input tensor to the model
    vocab, padded_tensor = preprocess_sentences(sentences)

    VOCAB_SIZE = len(vocab) # length of the vocab
    D_MODEL = 128 # Dimension of the fixed size sentence embedding
    NUM_HEADS = 8 # Number of heads for multi head attention
    NUM_LAYERS = 4 # Number of repeated multiheadattention blocks
    D_FF = 512 # intermediate dimension for feedforward network
    MAX_LEN = padded_tensor.shape[1] # Max len of sentence in the data
    NUM_CLASSES_A = 3 # Number of classes for task A: Sentence classification
    NUM_CLASSES_B = 4 # Number of classes for task B: NER 

    # Get random labels for both the tasks
    labels_a = torch.randint(0, 3, (len(sentences),)).to(device) 
    labels_b = torch.randint(0, 4, (len(sentences), MAX_LEN)).to(device)

    # Loss functions
    lf_a = nn.CrossEntropyLoss().to(device)
    lf_b = nn.CrossEntropyLoss().to(device)

    # Initialize the model
    multi_task_model = MultiTaskTransformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, NUM_CLASSES_A, NUM_CLASSES_B, MAX_LEN).to(device)

    # To get an idea about the model size
    print(f"Total Trainable Parameters: {count_parameters(multi_task_model)}")

    # Optimizer
    optimizer = torch.optim.Adam(multi_task_model.parameters(), lr=0.001)

    # Move input to device
    padded_tensor = padded_tensor.to(device)

    # call the train model function to train the model
    train_model(multi_task_model, optimizer, lf_a, lf_b, padded_tensor, labels_a, labels_b, NUM_CLASSES_B, epochs=10)

    return multi_task_model


if __name__ == '__main__':
    # Download NLTK data
    nltk.download("punkt_tab")
    nltk.download("punkt")

    # Select device (CUDA, MPS, or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using Device: {device}")

    # Task 1: Sentence Transformer Implementation
    print("=" * 50)
    print("TASK 1: Sentence Transformer Implementation")
    print("=" * 50)
    sentences = [
        "The cat sat on the mat.",
        "A quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world."
    ]

    embeddings = sentence_transformer_t1(sentences=sentences, device=device)
    print("Embeddings shape:", embeddings.shape)  # Expected shape: (batch_size, D_MODEL)
    print(f"Sentence Embeddings: {embeddings}")

    # Task 2: Multi-Task Learning Expansion
    print("=" * 50)
    print("TASK 2: Multi-Task Learning Expansion")
    print("=" * 50)
    sentences = [
        "Barack Obama was the 44th President of the United States.",
        "Apple Inc. is based in Cupertino, California."
    ]

    out_task_a, out_task_b = multi_task_transformer_t2(sentences, device=device)
    print("Sentence Classification Prediction:", torch.argmax(out_task_a, dim=1).tolist())
    print("NER Prediction:", torch.argmax(out_task_b, dim=2).tolist())


    # Task 4: Training Loop Implementation (BONUS)
    print("=" * 50)
    print("TASK 4: Training Loop Implementation (BONUS)")
    print("=" * 50)
    sentences = [
        "Barack Obama was the 44th President of the United States.",
        "Apple Inc. is based in Cupertino, California.",
        "The Eiffel Tower is located in Paris.",
        "Elon Musk is the CEO of Tesla.",
        "Microsoft Corporation is headquartered in Redmond.",
        "Google was founded by Larry Page and Sergey Brin.",
        "The Great Wall of China is a famous landmark."
    ]

    trained_model = train_multi_task_transformer_t4(sentences, device=device)
    