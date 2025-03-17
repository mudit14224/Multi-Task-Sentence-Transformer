# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Define MultiHead Self Attention Block
class SelfMultiHeadAttention(nn.Module):
    """
    Multi head self attention module

    Args: 
        d_model (int): Dimension of input embeddings
        num_heads (int): Number of attention heads

    Attributes:
        d_model (int): Dimension of input embeddings
        num_heads (int): Number of attention heads
        d_k (int): Dimension of each attention head
        q_linear (nn.Linear): Linear layer for Q
        k_linear (nn.Linear): Linear layer for K
        v_linear (nn.Linear): Linear layer for V
        out_linear (nn.Linear): Linear layer for output

    Methods: 
        forward(x): Performs multi head self attention
    """
    def __init__(self, d_model, num_heads): 
        super(SelfMultiHeadAttention, self).__init__()
        # Check if num_heads divides d_model 
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads

        # Size per attention head
        self.d_k = d_model // num_heads

        # Initialize the Q, K and V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Forward pass of the multi-head self-attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Get the batch size
        batch_size = x.shape[0]
        # Pass in the x to Q, K and V and reshape
        # Q, K, V: (batch_size, num_heads, seq_length, d_k)
        Q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # Shape: (batch_size, num_heads, seq_len, seq_len)
        atn_weights = F.softmax(scores, dim=-1) 
        atn_out = torch.matmul(atn_weights, V) # Shape: (batch_size, num_heads, seq_len, d_k)
        atn_out = atn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # Shape: (batch_size, seq_len, d_model)
        return self.out_linear(atn_out) # Shape: (batch_size, seq_len, d_model)
    

# FeedForward Class
class FeedForward(nn.Module):
    """
    FeedForward network module

    Args: 
        d_model (int): Dimension of input embeddings
        d_ff (int): Dimension of hidden layer

    Attributes: 
        linear1 (nn.Linear): First Linear layer
        linear2 (nn.Linear): Second Linear layer

    Methods:
        forward(x): Performs feedforward operation
    """
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff) 
        self.linear2 = nn.Linear(d_ff, d_model) 
    
    def forward(self, x):
        """
        Forward pass of the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        return self.linear2(F.relu(self.linear1(x))) # Shape: (batch_size, seq_len, d_ff) => (batch_size, seq_len, d_model)
    

# Positional Encoding (sin & cosine curves)
class PositionalEncode(nn.Module):
    """
    Positional encoding module.

    Args:
        d_model (int): Dimension of input embeddings
        max_len (int): Maximum sequence length

    Attributes:
        pe (torch.Tensor): Positional encoding tensor

    Methods:
        forward(x): Adds positional encoding to the input
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncode, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Encode even indices (sin curve)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Encode the odd indices (cos curve)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model) (to handle with batch_size)
        # Register 'pe' as a persistent buffer so it is saved with the model 
        # but not considered as a trainable parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass of the positional encoding module

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]
    

# Transformer encoder class
class TransformerEncoder(nn.Module):
    """
    Transformer encoder layer

    Args: 
        d_model (int): Dimension of input embeddings
        num_heads (int): Number of attention heads
        d_ff (int): Dimension of the feed-forward network's hidden layer
        dropout (float): Dropout probability
    
    Attributes:
        self_atn (SelfMultiHeadAttention): Multi-head self-attention module
        ff (FeedForward): Feed-forward network module
        norm1 (nn.LayerNorm): First layer normalization
        norm2 (nn.LayerNorm): Second layer normalization
        dropout (nn.Dropout): Dropout layer

    Methods:
        forward(x): Performs forward pass of the transformer encoder layer
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.2):
        super(TransformerEncoder, self).__init__()
        self.self_atn = SelfMultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the transformer encoder layer

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        atn_out = self.self_atn(x) # Shape: (batch_size, seq_len, d_model)
        x = self.norm1(x + self.dropout(atn_out)) # Shape: (batch_size, seq_len, d_model)
        ff_out = self.ff(x) # Shape: (batch_size, seq_len, d_model)
        x = self.norm2(x + self.dropout(ff_out)) # Shape: (batch_size, seq_len, d_model)
        return x
    

# Sentence Transformer class
class SentenceTransformer(nn.Module):
    """
    Sentence Transformer model

    Args:
        vocab_size (int): Size of the vocabulary
        d_model (int): Dimension of input embeddings
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer encoder layers
        d_ff (int): Dimension of the feed-forward network's hidden layer
        max_len (int): Maximum sequence length

    Attributes:
        embedding (nn.Embedding): Embedding layer
        pos_encode (PositionalEncode): Positional encoding module
        encoder_layers (nn.ModuleList): List of transformer encoder layers
        pooling (nn.AdaptiveAvgPool1d): Adaptive average pooling layer

    Methods:
        forward(x): Performs forward pass of the sentence transformer
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=100):
        super(SentenceTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encode = PositionalEncode(d_model, max_len)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Forward pass of the sentence transformer

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, d_model), representing sentence embeddings
        """
        x = self.embedding(x) # Shape: (batch_size, seq_len, d_model)
        x = self.pos_encode(x) # Shape: (batch_size, seq_len, d_model)
        for layer in self.encoder_layers:
            x = layer(x) # Shape: (batch_size, seq_len, d_model)
        x = x.permute(0, 2, 1) # Shape: (batch_size, d_model, seq_len)
        x = self.pooling(x).squeeze(-1) # Shape: (batch_size, d_model, 1) => (batch_size, d_model)
        return x
    

### Task 2: Multi-Task Learning Expansion
class MultiTaskTransformer(nn.Module):
    """
    Multi-Task Transformer model for sentence classification and NER

    Args:
        vocab_size (int): Size of the vocabulary
        d_model (int): Dimension of input embeddings
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer encoder layers
        d_ff (int): Dimension of the feed-forward network's hidden layer
        num_classes_a (int): Number of classes for sentence classification (Task A)
        num_classes_b (int): Number of classes for NER (Task B)
        max_len (int): Maximum sequence length

    Attributes:
        embedding (nn.Embedding): Embedding layer
        pos_encode (PositionalEncode): Positional encoding module
        encoder_layers (nn.ModuleList): List of transformer encoder layers
        pooling (nn.AdaptiveAvgPool1d): Adaptive average pooling layer
        classifier_a (nn.Linear): Linear layer for sentence classification (Task A)
        ner_classifier_b (nn.Linear): Linear layer for NER (Task B)

    Methods:
        forward(x): Performs forward pass of the multi-task transformer
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, num_classes_a, num_classes_b, max_len=100):
        super(MultiTaskTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encode = PositionalEncode(d_model, max_len)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # Task A: Sentence Classification Head
        self.classifier_a = nn.Linear(d_model, num_classes_a)

        # Task B: Named Entity Recognition (NER) - per token basis Head
        self.ner_classifier_b = nn.Linear(d_model, num_classes_b)

    def forward(self, x):
        """
        Forward pass of the multi-task transformer

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len)

        Returns:
            tuple: A tuple containing the output tensors for Task A and Task B.
                - out_task_a (torch.Tensor): Output tensor for sentence classification (Task A).
                - out_task_b (torch.Tensor): Output tensor for NER (Task B)
        """
        # x -> Shape: (batch_size, seq_len)
        x = self.embedding(x) # Shape: (batch_size, seq_len, d_model)
        x = self.pos_encode(x) # Shape: (batch_size, seq_len, d_model)
        for layer in self.encoder_layers:
            x = layer(x) # Shape: (batch_size, seq_len, d_model)
        
        # Task A: Sentence Classification
        # we want to pool over the seq_len 
        x_pooled = x.permute(0, 2, 1) # Shape: (batch_size, d_model, seq_len)
        x_pooled = self.pooling(x_pooled).squeeze(-1) # Shape: (batch_size, d_model)
        out_task_a = self.classifier_a(x_pooled) # Shape: (batch_size, num_classes_a)
        # Task B: NER 
        # We pass in x and not x_pooled as we want per token label in NER
        out_task_b = self.ner_classifier_b(x) # Shape: (batch_size, seq_len, num_classes_b)

        return out_task_a, out_task_b