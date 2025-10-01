#!/usr/bin/env python3
"""
Combined inference script for ADL HW1
Step 1: Use multiple choice model to select the best paragraph for each question
Step 2: Use question answering model to find the answer in the selected paragraph
"""

import json
import csv
import argparse
import torch
import subprocess
import sys
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForMultipleChoice, 
    AutoModelForQuestionAnswering,
    default_data_collator
)
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import datasets
from utils_qa import postprocess_qa_predictions


class MultipleChoiceDataset(Dataset):
    """Dataset for multiple choice inference"""
    def __init__(self, test_data, contexts, tokenizer, max_length=512):
        self.test_data = test_data
        self.contexts = contexts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.test_data)
    
    def __getitem__(self, idx):
        item = self.test_data[idx]
        question = item["question"]
        paragraphs = [self.contexts[idx] for idx in item["paragraphs"]]
        
        # Prepare 4 choices as question + paragraph pairs
        choices = []
        for paragraph in paragraphs:
            encoding = self.tokenizer(
                question,
                paragraph,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            choices.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            })
        
        return {
            'id': item['id'],
            'choices': choices
        }


def prepare_qa_features(examples, tokenizer, max_seq_length, doc_stride):
    """Prepare features for QA model - based on model_inference.py"""
    pad_on_right = tokenizer.padding_side == "right"
    examples["question"] = [q.lstrip() for q in examples["question"]]
    
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def collate_fn_mc(batch):
    """Collate function for multiple choice"""
    ids = [item['id'] for item in batch]
    input_ids = []
    attention_masks = []
    
    for item in batch:
        choice_input_ids = torch.stack([choice['input_ids'] for choice in item['choices']])
        choice_attention_masks = torch.stack([choice['attention_mask'] for choice in item['choices']])
        input_ids.append(choice_input_ids)
        attention_masks.append(choice_attention_masks)
    
    return {
        'ids': ids,
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks)
    }


def collate_fn_qa(batch):
    """Collate function for question answering"""
    ids = [item['id'] for item in batch]
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    contexts = [item['context'] for item in batch]
    questions = [item['question'] for item in batch]
    
    return {
        'ids': ids,
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'contexts': contexts,
        'questions': questions
    }


def predict_multiple_choice(model, dataloader, device):
    """Generate predictions using multiple choice model"""
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="MC Inference"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_choices = torch.argmax(logits, dim=1)
            
            for i, pred in enumerate(predicted_choices):
                predictions[batch['ids'][i]] = pred.item()
    
    return predictions


def run_question_answering_model(args, mq_output, contexts, qa_model, qa_tokenizer):
    """Generate predictions using question answering model - based on model_inference.py"""
    
    # Prepare the data in the correct format
    for item in mq_output:
        item["context"] = contexts[item["predicted_relevant"]]

    raw_datasets = datasets.Dataset.from_list(mq_output)
    
    tokenized_datasets = raw_datasets.map(
        lambda examples: prepare_qa_features(examples, qa_tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=raw_datasets.column_names,
    )

    data_collator = default_data_collator
    eval_dataset = tokenized_datasets.remove_columns(["example_id", "offset_mapping"])
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.batch_size)

    qa_model.eval()

    all_start_logits = []
    all_end_logits = []

    device = next(qa_model.parameters()).device
    
    for batch in tqdm(eval_dataloader, desc="Question Answering"):
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = qa_model(**batch)
            start_logits = outputs.start_logits.cpu().numpy()
            end_logits = outputs.end_logits.cpu().numpy()
            all_start_logits.append(start_logits)
            all_end_logits.append(end_logits)

    max_len = max([x.shape[1] for x in all_start_logits])
    start_logits_concat = np.concatenate(all_start_logits, axis=0)
    end_logits_concat = np.concatenate(all_end_logits, axis=0)
    
    outputs = (start_logits_concat, end_logits_concat)
    
    theoretical_answers = postprocess_qa_predictions(
        raw_datasets,
        tokenized_datasets,
        outputs,
        args.version_2_with_negative,
        20,
        30,
        0.0,
    )
    
    return theoretical_answers


def load_data(test_file, context_file):
    """Load test data and contexts"""
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    with open(context_file, 'r', encoding='utf-8') as f:
        contexts = json.load(f)
    
    return test_data, contexts


def save_predictions(predictions, output_file):
    """Save predictions to CSV file"""
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'answer'])
        
        for example_id, answer in predictions.items():
            writer.writerow([example_id, answer])

def save_mc_predictions(predictions, output_file):
    """Save MC predictions (paragraph indices) to CSV file"""
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'paragraph_idx'])
        
        for example_id, paragraph_idx in predictions.items():
            writer.writerow([example_id, paragraph_idx])


def test_inference():
    """Test the inference script with default parameters"""
    
    # Check if required model directories exist
    mc_model_path = "/tmp2/young91319/ADL/hw1/ckpt/mq/chinese_lert"
    qa_model_path = "/tmp2/young91319/ADL/hw1/ckpt/qa/chinese_lert"
    
    if not os.path.exists(mc_model_path):
        print(f"Error: MC model path {mc_model_path} does not exist")
        return False
        
    if not os.path.exists(qa_model_path):
        print(f"Error: QA model path {qa_model_path} does not exist")
        return False
    
    # Test command arguments
    test_args = argparse.Namespace(
        test_file="/tmp2/young91319/ADL/hw1/data/test.json",
        context_file="/tmp2/young91319/ADL/hw1/data/context.json",
        mc_model_path=mc_model_path,
        qa_model_path=qa_model_path,
        output_file="/tmp2/young91319/ADL/hw1/output/chinese_lert/test_predictions.csv",
        batch_size=4,
        max_length=512,
        doc_stride=128,
        version_2_with_negative=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("🔧 Running inference test with default parameters...")
    print(f"📁 Test file: {test_args.test_file}")
    print(f"📁 Context file: {test_args.context_file}")
    print(f"🤖 MC model: {test_args.mc_model_path}")
    print(f"🤖 QA model: {test_args.qa_model_path}")
    print(f"💾 Output file: {test_args.output_file}")
    
    try:
        # Run the main inference function with test arguments
        run_inference(test_args)
        print("✅ Inference test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error running inference test: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_inference(args):
    """Run the inference pipeline with given arguments"""
    print(f"Loading data from {args.test_file} and {args.context_file}")
    test_data, contexts = load_data(args.test_file, args.context_file)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Step 1: Multiple Choice Model Prediction (Always needed)
    print("Step 1: Running Multiple Choice inference...")
    print("Loading Multiple Choice model...")
    mc_tokenizer = AutoTokenizer.from_pretrained(args.mc_model_path)
    mc_model = AutoModelForMultipleChoice.from_pretrained(args.mc_model_path)
    mc_model.to(device)
    
    mc_dataset = MultipleChoiceDataset(test_data, contexts, mc_tokenizer, args.max_length)
    mc_dataloader = DataLoader(mc_dataset, batch_size=args.batch_size, collate_fn=collate_fn_mc)
    
    mc_predictions = predict_multiple_choice(mc_model, mc_dataloader, device)
    
    # # Save MC predictions to CSV file
    # mc_output_file = "output/mc_predictions.csv"
    # save_mc_predictions(mc_predictions, mc_output_file)
    # print(f"Multiple choice predictions saved to {mc_output_file}")
    
    # Step 2: Question Answering Model Prediction using MC results
    print("\nStep 2: Running Question Answering inference...")
    print("Loading Question Answering model...")
    qa_tokenizer = AutoTokenizer.from_pretrained(args.qa_model_path)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(args.qa_model_path)
    qa_model.to(device)
    
    # Prepare test data with MC predictions
    mq_output = []
    for item in test_data:
        new_item = item.copy()
        if item["id"] in mc_predictions:
            new_item["predicted_relevant"] = item["paragraphs"][mc_predictions[item["id"]]]
        else:
            new_item["predicted_relevant"] = item["paragraphs"][0]  # fallback
        mq_output.append(new_item)
    
    qa_predictions = run_question_answering_model(args, mq_output, contexts, qa_model, qa_tokenizer)
    
    # Final predictions are from QA model
    final_predictions = qa_predictions
    
    print(f"Saving predictions to {args.output_file}")
    save_predictions(final_predictions, args.output_file)
    
    print(f"Inference completed! Generated {len(final_predictions)} predictions.")
    print(f"Sample predictions:")
    for i, (example_id, answer) in enumerate(list(final_predictions.items())[:5]):
        print(f"  {example_id}: {answer}")


def main():
    parser = argparse.ArgumentParser(description="ADL HW1 Combined Inference")
    parser.add_argument("--test", action="store_true",
                      help="Run in test mode with default parameters")
    parser.add_argument("--test_file", type=str, default="/tmp2/young91319/ADL/hw1/data/test.json",
                      help="Path to test data file")
    parser.add_argument("--context_file", type=str, default="/tmp2/young91319/ADL/hw1/data/context.json",
                      help="Path to context file")
    parser.add_argument("--mc_model_path", type=str, 
                      default="/tmp2/young91319/ADL/hw1/ckpt/mq/chinese_lert",
                      help="Path to trained multiple choice model")
    parser.add_argument("--qa_model_path", type=str,
                      default="/tmp2/young91319/ADL/hw1/ckpt/qa/chinese_lert", 
                      help="Path to trained question answering model")
    parser.add_argument("--output_file", type=str, default="predictions.csv",
                      help="Output CSV file path")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=512,
                      help="Maximum sequence length")
    parser.add_argument("--doc_stride", type=int, default=128,
                      help="Document stride for QA model")
    parser.add_argument("--version_2_with_negative", action="store_true",
                      help="If true, some of the examples do not have an answer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to use for inference")
    
    args = parser.parse_args()
    
    # If test mode, run the test function
    if args.test:
        success = test_inference()
        sys.exit(0 if success else 1)
    
    # Otherwise, run normal inference
    try:
        run_inference(args)
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()