import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

# 1. Rebuild the Brain's Architecture
class LegalExpertModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return {"logits": logits}

class IPCClassifier:
    def __init__(self, model_dir="./expert_ipc_brain"):
        print("Loading Expert AI Engine...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- THE AUTO-TRANSLATOR FIX ---
        print("Fetching real IPC names...")
        dataset = load_dataset("Exploration-Lab/IL-TUR", "lsi", split='train[:1]', trust_remote_code=True)
        real_ipc_names = dataset.features['labels'].feature.names
        
        # Load the internal numbers and translate them to real laws (e.g., "13" -> "IPC 453")
        internal_classes = np.load(f"{model_dir}/ipc_classes.npy", allow_pickle=True)
        self.classes = [real_ipc_names[int(c)].replace("Section ", "IPC ") if c.isdigit() else c for c in internal_classes]
        num_labels = len(self.classes)
        # -------------------------------
        
        # Load the Optimized Math Thresholds
        self.thresholds = np.load(f"{model_dir}/optimal_thresholds.npy")
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load the Custom Brain Weights
        self.model = LegalExpertModel("nlpaueb/legal-bert-base-uncased", num_labels)
        self.model.load_state_dict(torch.load(f"{model_dir}/custom_model_weights.bin", map_location=self.device))
        self.model.to(self.device)
        self.model.eval() 
        print("System Ready.")

    def predict(self, text_scenario):
        """Takes a raw legal string and returns the predicted IPC sections."""
        
        encodings = self.tokenizer(text_scenario, truncation=True, padding='max_length', max_length=512, return_tensors="pt")
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"].squeeze()
            
        probabilities = torch.sigmoid(logits).cpu().numpy()
        
        predicted_laws = []
        for i, prob in enumerate(probabilities):
            if prob > self.thresholds[i]:
                predicted_laws.append({
                    "ipc_section": self.classes[i],
                    "confidence": round(float(prob * 100), 2)
                })
                
        if not predicted_laws:
            return [{"ipc_section": "No crime detected or insufficient details", "confidence": 0.0}]
            
        return sorted(predicted_laws, key=lambda x: x['confidence'], reverse=True)

# ==========================================
# HOW TO USE IT (Student 1's Web Backend)
# ==========================================
if __name__ == "__main__":
    ai_engine = IPCClassifier(model_dir="./expert_ipc_brain")
    
    fake_case = """
    On the night of October 12th, the accused forcefully broke the lock of the 
    complainant's house and entered without permission. The accused then took 
    gold jewelry worth Rs. 50,000 from the locker and fled the scene. When 
    confronted by the guard, the accused brandished a knife and threatened him.
    """
    
    results = ai_engine.predict(fake_case)
    
    print("\n--- AI PREDICTIONS ---")
    for res in results:
        print(f"Law: {res['ipc_section']} | Confidence: {res['confidence']}%")