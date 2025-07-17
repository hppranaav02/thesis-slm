import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Tuple
import ast
import re
from collections import Counter

class CodeChunker:
    """Handles chunking of large code files into manageable pieces"""
    
    def __init__(self, tokenizer, max_length=512, overlap=50):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overlap = overlap
        
    def chunk_by_functions(self, code):
        """Split code into function-level chunks"""
        # Simple regex to find Go functions
        function_pattern = r'func\s+(?:\([^)]*\)\s+)?(\w+)\s*\([^)]*\)\s*(?:\([^)]*\))?\s*{'
        functions = []
        
        lines = code.split('\n')
        current_func = []
        brace_count = 0
        in_function = False
        
        for line in lines:
            if re.search(function_pattern, line):
                if current_func:  # Save previous function
                    functions.append('\n'.join(current_func))
                current_func = [line]
                in_function = True
                brace_count = line.count('{') - line.count('}')
            elif in_function:
                current_func.append(line)
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0:
                    functions.append('\n'.join(current_func))
                    current_func = []
                    in_function = False
            else:
                current_func.append(line)
        
        # Add remaining code
        if current_func:
            functions.append('\n'.join(current_func))
        
        return functions
    
    def chunk_by_tokens(self, code):
        """Split code into token-based chunks with overlap"""
        tokens = self.tokenizer.encode(code, add_special_tokens=False)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + self.max_length - 2, len(tokens))  # -2 for [CLS] and [SEP]
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if end == len(tokens):
                break
            start = end - self.overlap
            
        return chunks
    
    def smart_chunk(self, code):
        """Combine function-level and token-level chunking"""
        # First try function-level chunking
        function_chunks = self.chunk_by_functions(code)
        
        final_chunks = []
        for func_chunk in function_chunks:
            # Check if function chunk is too long
            tokens = self.tokenizer.encode(func_chunk, add_special_tokens=False)
            if len(tokens) <= self.max_length - 2:
                final_chunks.append(func_chunk)
            else:
                # Split large functions using token chunking
                token_chunks = self.chunk_by_tokens(func_chunk)
                final_chunks.extend(token_chunks)
        
        return final_chunks

class GoCodeCWEDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, mlb=None, chunking_strategy='smart'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlb = mlb
        self.chunking_strategy = chunking_strategy
        self.chunker = CodeChunker(tokenizer, max_length)
        
        # Pre-process data to handle large files
        self.processed_data = self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess data to handle large code files"""
        processed = []
        
        for idx, item in self.data.iterrows():
            code = item['input']
            cwe_labels = item['output']
            
            # Process CWE labels
            if isinstance(cwe_labels, str):
                try:
                    cwe_labels = ast.literal_eval(cwe_labels)
                except:
                    cwe_labels = []
            
            # Check if code is too long
            tokens = self.tokenizer.encode(code, add_special_tokens=False)
            
            if len(tokens) <= self.max_length - 2:
                # Code fits in one chunk
                processed.append({
                    'code': code,
                    'labels': cwe_labels,
                    'is_chunk': False,
                    'chunk_index': 0,
                    'total_chunks': 1
                })
            else:
                # Need to chunk the code
                if self.chunking_strategy == 'smart':
                    chunks = self.chunker.smart_chunk(code)
                elif self.chunking_strategy == 'function':
                    chunks = self.chunker.chunk_by_functions(code)
                else:
                    chunks = self.chunker.chunk_by_tokens(code)
                
                # Each chunk inherits the same labels
                for i, chunk in enumerate(chunks):
                    processed.append({
                        'code': chunk,
                        'labels': cwe_labels,
                        'is_chunk': True,
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    })
        
        return processed
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        item = self.processed_data[idx]
        
        # Tokenize the Go code
        encoding = self.tokenizer(
            item['code'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert to binary labels
        if self.mlb is not None:
            labels = self.mlb.transform([item['labels']])[0]
        else:
            labels = np.zeros(1)  # Placeholder for training setup
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(labels),
            'is_chunk': item['is_chunk'],
            'chunk_index': item['chunk_index'],
            'total_chunks': item['total_chunks']
        }

class BERTCWEClassifier(nn.Module):
    def __init__(self, model_name='microsoft/codebert-base', num_labels=100, dropout=0.1):
        super(BERTCWEClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        return self.sigmoid(logits)

class ChunkAggregator:
    """Aggregates predictions from multiple chunks of the same file"""
    
    def __init__(self, strategy='max'):
        self.strategy = strategy
    
    def aggregate_predictions(self, chunk_predictions):
        """Aggregate predictions from multiple chunks"""
        if len(chunk_predictions) == 1:
            return chunk_predictions[0]
        
        chunk_array = np.array(chunk_predictions)
        
        if self.strategy == 'max':
            # Take maximum probability across chunks
            return np.max(chunk_array, axis=0)
        elif self.strategy == 'mean':
            # Take mean probability across chunks
            return np.mean(chunk_array, axis=0)
        elif self.strategy == 'voting':
            # Binary voting: if majority of chunks predict positive
            binary_preds = (chunk_array > 0.5).astype(int)
            return (np.sum(binary_preds, axis=0) > len(chunk_predictions) // 2).astype(float)
        else:
            return np.max(chunk_array, axis=0)

class GoCodeCWETrainer:
    def __init__(self, model_name='microsoft/codebert-base', max_length=512, batch_size=16, 
                 learning_rate=2e-5, chunking_strategy='smart', aggregation_strategy='max'):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.chunking_strategy = chunking_strategy
        self.aggregation_strategy = aggregation_strategy
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mlb = MultiLabelBinarizer()
        self.aggregator = ChunkAggregator(aggregation_strategy)
        
    def prepare_data(self, df):
        """Prepare and split the dataset"""
        # Extract all unique CWE IDs from the dataset
        all_cwes = []
        for output in df['output']:
            if isinstance(output, str):
                try:
                    cwe_list = ast.literal_eval(output)
                    all_cwes.extend(cwe_list)
                except:
                    pass
            elif isinstance(output, list):
                all_cwes.extend(output)
        
        # Fit the MultiLabelBinarizer
        unique_cwes = list(set(all_cwes))
        self.mlb.fit([unique_cwes])
        print(f"Found {len(unique_cwes)} unique CWE IDs: {unique_cwes}")
        
        # Split data (80% train, 20% validation)
        train_size = int(0.8 * len(df))
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:]
        
        # Create datasets with chunking
        train_dataset = GoCodeCWEDataset(
            train_df, self.tokenizer, self.max_length, self.mlb, self.chunking_strategy
        )
        val_dataset = GoCodeCWEDataset(
            val_df, self.tokenizer, self.max_length, self.mlb, self.chunking_strategy
        )
        
        print(f"Training samples after chunking: {len(train_dataset)}")
        print(f"Validation samples after chunking: {len(val_dataset)}")
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, len(unique_cwes)
    
    def train_model(self, train_loader, val_loader, num_labels, num_epochs=3):
        """Train the BERT model"""
        # Initialize model
        model = BERTCWEClassifier(self.model_name, num_labels).to(self.device)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Loss function for multi-label classification
        criterion = nn.BCELoss()
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
            
            # Validation
            val_accuracy = self.evaluate_model(model, val_loader)
            print(f'Validation Accuracy: {val_accuracy:.4f}')
        
        return model
    
    def evaluate_model(self, model, val_loader):
        """Evaluate the model on validation set"""
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids, attention_mask)
                
                # Convert probabilities to binary predictions (threshold = 0.5)
                predictions = (outputs > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Subset accuracy (exact match)
        subset_accuracy = accuracy_score(all_labels, all_predictions)
        
        # Per-label metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='micro'
        )
        
        print(f"Subset Accuracy: {subset_accuracy:.4f}")
        print(f"Micro Precision: {precision:.4f}")
        print(f"Micro Recall: {recall:.4f}")
        print(f"Micro F1: {f1:.4f}")
        
        return subset_accuracy
    
    def predict(self, model, code_text):
        """Predict CWE IDs for a given Go code (handles large files)"""
        model.eval()
        
        # Check if code needs chunking
        tokens = self.tokenizer.encode(code_text, add_special_tokens=False)
        
        if len(tokens) <= self.max_length - 2:
            # Code fits in one chunk
            encoding = self.tokenizer(
                code_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                predictions = outputs.cpu().numpy()[0]
        else:
            # Code needs chunking
            chunker = CodeChunker(self.tokenizer, self.max_length)
            
            if self.chunking_strategy == 'smart':
                chunks = chunker.smart_chunk(code_text)
            elif self.chunking_strategy == 'function':
                chunks = chunker.chunk_by_functions(code_text)
            else:
                chunks = chunker.chunk_by_tokens(code_text)
            
            chunk_predictions = []
            
            for chunk in chunks:
                encoding = self.tokenizer(
                    chunk,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask)
                    chunk_predictions.append(outputs.cpu().numpy()[0])
            
            # Aggregate predictions from all chunks
            predictions = self.aggregator.aggregate_predictions(chunk_predictions)
        
        # Convert probabilities to binary predictions and then to CWE IDs
        binary_predictions = (predictions > 0.5).astype(int)
        predicted_cwes = self.mlb.inverse_transform([binary_predictions])
        
        return predicted_cwes[0] if predicted_cwes[0] else []
    
    def predict_with_probabilities(self, model, code_text):
        """Predict with probability scores for each CWE"""
        model.eval()
        
        # Similar to predict but returns probabilities
        tokens = self.tokenizer.encode(code_text, add_special_tokens=False)
        
        if len(tokens) <= self.max_length - 2:
            encoding = self.tokenizer(
                code_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                predictions = outputs.cpu().numpy()[0]
        else:
            chunker = CodeChunker(self.tokenizer, self.max_length)
            
            if self.chunking_strategy == 'smart':
                chunks = chunker.smart_chunk(code_text)
            elif self.chunking_strategy == 'function':
                chunks = chunker.chunk_by_functions(code_text)
            else:
                chunks = chunker.chunk_by_tokens(code_text)
            
            chunk_predictions = []
            
            for chunk in chunks:
                encoding = self.tokenizer(
                    chunk,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask)
                    chunk_predictions.append(outputs.cpu().numpy()[0])
            
            predictions = self.aggregator.aggregate_predictions(chunk_predictions)
        
        # Return CWE IDs with their probabilities
        cwe_probs = {}
        for i, prob in enumerate(predictions):
            if prob > 0.1:  # Only show probabilities > 0.1
                cwe_id = self.mlb.classes_[i]
                cwe_probs[cwe_id] = prob
        
        return cwe_probs

# Example usage
def main():
    # Load the dataset csv
    # df = pd.read_csv('your_dataset.csv')  # Should have 'input' and 'output' columns

    # Load the dataset json
    df = pd.read_json('/local/s3905020/code/dataset-creation/train.jsonl', lines=True)  # Should have 'input' and 'output' columns
    
    # Test dataset with large code
    sample_data = {
        'input': [
            'package main\nimport "fmt"\nfunc main() {\n    fmt.Println("Hello, World!")\n}',
            '''package main
import (
    "fmt"
    "os"
    "bufio"
    "strings"
)

func vulnerableFunction() {
    var buffer [10]int
    buffer[20] = 1  // Buffer overflow - CWE-119
}

func anotherVulnerableFunction() {
    filename := os.Args[1]  // No input validation - CWE-20
    file, err := os.Open(filename)
    if err != nil {
        return
    }
    defer file.Close()
    
    scanner := bufio.NewScanner(file)
    for scanner.Scan() {
        line := scanner.Text()
        if strings.Contains(line, "password") {
            fmt.Println(line)  // Information disclosure - CWE-200
        }
    }
}

func main() {
    vulnerableFunction()
    anotherVulnerableFunction()
}''',
            '''package main
import (
    "crypto/rand"
    "crypto/rsa"
    "crypto/sha256"
    "fmt"
)

func secureFunction() {
    // Generate RSA key pair
    privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
    if err != nil {
        fmt.Println("Error generating key:", err)
        return
    }
    
    publicKey := &privateKey.PublicKey
    
    // Encrypt data
    message := []byte("Hello, World!")
    ciphertext, err := rsa.EncryptOAEP(sha256.New(), rand.Reader, publicKey, message, nil)
    if err != nil {
        fmt.Println("Error encrypting:", err)
        return
    }
    
    // Decrypt data
    plaintext, err := rsa.DecryptOAEP(sha256.New(), rand.Reader, privateKey, ciphertext, nil)
    if err != nil {
        fmt.Println("Error decrypting:", err)
        return
    }
    
    fmt.Println("Decrypted:", string(plaintext))
}

func main() {
    secureFunction()
}'''
        ],
        'output': [
            [],  # Secure code
            ['CWE-119', 'CWE-20', 'CWE-200'],  # Multiple vulnerabilities
            []   # Secure code
        ]
    }
    
    # Comment when loading from file
    # df = pd.DataFrame(sample_data)
    
    # Initialize trainer with chunking options
    trainer = GoCodeCWETrainer(
        model_name='microsoft/codebert-base',
        max_length=512,
        batch_size=8,
        learning_rate=2e-5,
        chunking_strategy='smart',  # Options: 'smart', 'function', 'token'
        aggregation_strategy='max'  # Options: 'max', 'mean', 'voting'
    )
    
    # Prepare data
    train_loader, val_loader, num_labels = trainer.prepare_data(df)
    
    # Train model
    model = trainer.train_model(train_loader, val_loader, num_labels, num_epochs=3)
    
    # Save the model
    torch.save(model.state_dict(), 'bert_go_cwe_classifier.pth')
    
    # Example prediction on large code
    large_test_code = '''package main
import (
    "fmt"
    "os"
    "unsafe"
)

func bufferOverflowExample() {
    var buffer [10]int
    for i := 0; i < 20; i++ {  // Buffer overflow
        buffer[i] = i
    }
    fmt.Println(buffer)
}

func unsafePointerExample() {
    var x int = 42
    ptr := unsafe.Pointer(&x)
    // Unsafe operations
    y := (*int)(ptr)
    *y = 100
}

func main() {
    if len(os.Args) < 2 {
        fmt.Println("Usage: program <input>")
        return
    }
    
    input := os.Args[1]  // No input validation
    fmt.Println("Input:", input)
    
    bufferOverflowExample()
    unsafePointerExample()
}'''
    
    # Standard prediction
    predicted_cwes = trainer.predict(model, large_test_code)
    print(f"Predicted CWEs: {predicted_cwes}")
    
    # Prediction with probabilities
    cwe_probabilities = trainer.predict_with_probabilities(model, large_test_code)
    print(f"CWE Probabilities: {cwe_probabilities}")

if __name__ == "__main__":
    main()