# Imports
from collections import defaultdict
from nltk.tokenize import word_tokenize
import torch

## Function to preprocess sentences. Gets vocab and the padded sentence
def preprocess_sentences(sentences):
    """
    Preprocesses a list of sentences by tokenizing, creating a vocabulary,
    converting tokens to indices, and padding sequences to a uniform length

    Args:
        sentences (list of str): A list of sentences to preprocess

    Returns:
        tuple: A tuple containing:
            - vocab (defaultdict): A vocabulary mapping words to indices
            - padded_tensor (torch.Tensor): A tensor of padded sentence indices 
                                            which will be the input to the model
    """
    # Create a vocab
    vocab = defaultdict(lambda: len(vocab))
    vocab['<PAD>'] = 0

    # Tokenize and convert to IDs
    tokenized = [word_tokenize(s.lower()) for s in sentences]
    # print(tokenized)
    indexed = [[vocab[w] for w in s] for s in tokenized]
    # print(indexed)

    # Padding
    max_len = max(len(s) for s in indexed)
    # Add padding to shorter sentences
    padded = [s + [0] * (max_len - len(s)) for s in indexed] 

    # Convert to tensor
    padded_tensor = torch.tensor(padded)

    return vocab, padded_tensor

## Training loop function
def train_model(model, optimizer, lf_a, lf_b, inputs, labels_a, labels_b, num_classes_b, epochs=3):
    """
    Trains a multi-task model

    Args:
        model (nn.Module): The multi-task model to train
        optimizer (torch.optim.Optimizer): The optimizer to use
        lf_a (callable): Loss function for Task A
        lf_b (callable): Loss function for Task B
        inputs (torch.Tensor): Input tensor
        labels_a (torch.Tensor): Labels for Task A: Sentence classification 
        labels_b (torch.Tensor): Labels for Task B: NER
        num_classes_b (int): Number of classes for Task B (To reshape out_task_b tensor to calculate loss)
        epochs (int): Number of training epochs
    """
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Get the outputs from the model
        out_task_a, out_task_b = model(inputs)

        # Calculate the loss for both the tasks
        loss_a = lf_a(out_task_a, labels_a)
        loss_b = lf_b(out_task_b.view(-1, num_classes_b), labels_b.view(-1))
        
        # Get the total loss
        total_loss = loss_a + loss_b
        total_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}: Loss Task A: {loss_a.item():.4f}, Loss Task B: {loss_b.item():.4f}, Total Loss: {total_loss.item():.4f}")