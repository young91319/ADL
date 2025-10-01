## ADL HW1 
### r13922191 呂廷洋

### Environment Setup 

    conda env create -f environment.yml -n ADL
    conda activate ADL

### Data Prepatation
under main directory (hw1/)

    chmod +x download.sh
    bash download.sh

check if your folder looks like this

    r13922191  
        ├── ckpt/  
        ├── data/           
        ├── download.sh  
        ├── environment.yml  
        ├── inference.py     
        ├── multiple_choice.py  
        ├── question_answering.py      
        ├── README.md  
        ├── run.sh   
        └── utils_qa.py    


### Inference
    chmod +x run.sh
    bash run.sh ./data/context.json ./data/test.json ./output/prediction.csv


### Training(Option)
    # multiple_choice:

    python multiple_choice.py \
    --model_name_or_path hfl/chinese-lert-base  \
    --max_seq_length 512   \
    --per_device_train_batch_size 8   \
    --learning_rate 3e-5   \
    --num_train_epochs 1   \
    --output_dir ckpt/mq/chinese_lert

    
    # question_answering:
    
    python question_answering.py \
    --model_name_or_path hfl/chinese-lert-base \
    --max_seq_length 512 \
    --doc_stride 128 \
    --output_dir ckpt/qa/chinese_lert
