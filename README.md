# Self-PT: Adaptive Self-Prompt Tuning for Low-Resource Visual Question Answering (ACM MM 2023)
This repository contains the implementation of Self-PT described in the paper.
Code is based on [FewVLM](https://github.com/woojeongjin/FewVLM) and [VL-adapter](https://github.com/ylsung/VL_adapter)

## Updates
I am currently **in the process of updating my GitHub repository**, but it is not yet complete. I wanted to let everyone know to expect the new changes soon. Thank you for your patience and stay tuned for updates!

## Installation

```bash
# Create python environment (optional)
conda create -n SelfPT
conda activate SelfPT

# Install python dependencies
pip install -r requirements.txt
```

## Datasets

- The VQA and GQA datasets can be downloaded from [VQA&GQA](https://nlp.cs.unc.edu/data/lxmert_data/) for image features and annotations.

- The OK-VQA dataset can be downloaded from [OK-VQA_a](https://drive.google.com/drive/folders/1T8x5O3sZp83_x9XX2Yg0HgCP8P_Bh8KD?usp=sharing) for annotations and [OK-VQA_f](https://nlp.cs.unc.edu/data/lxmert_data/) for image features.

## Code structure
```bash
./Self_PT/
    datasets/                                 <= Store images features and annotations
        VQA/
            train.json
            nominival.json
            minival.json
            v2_mscoco_val2014_annotations.json
            v2_mscoco_train2014_annotations.json
            trainval_ans2label.json
            trainval_label2ans.json
            test2015_obj36.h5
            train2014_obj36.h5
            val2014_obj36.h5
        GQA/
            train.json
            testdev.json
            trainval_ans2label.json
            trainval_label2ans.json
            gqa_testdev_obj36.h5
            vg_gqa_obj36.h5
        okvqa/
            train.json
            val.json
            mscoco_train2014_annotations.json
            mscoco_val2014_annotations.json
            trainval_label2ans.json
            trainval_ans2label.json
            (okvqa shares the same .h5 files of image features with VQA)
    src/                                                      <= Train Self-PT
        adapters/                                             <= adapter tuning methods
        lora/                                                 <= lora method
        prompt/                                               <= prompt tuning methods
        my_transformers/                                      <= baseline module modeling
        modeling_t5.py                                        <= baseline modeling
        vqa.py, vqa_data.py vqa_model.py                      <= Self-PT on VQA
        gqa.py, gqa_data.py gqa_model.py                      <= Self-PT on GQA
        okvqa.py, okvqa_data.py okvqa_model.py                <= Self-PT on OK-VQA
        param.py                                              <= (argparse) configuration
        tokenization.py                                       <= custom tokenizer
        utils.py, dist_utils.py                               <= utility functions
    scripts/                                                  <= bash scripts 
```

## Pre-trained checkpoints

- We use the pre-trained checkpoints provided by FewVLM: [VL-T5 w/o vqa pretraining](https://drive.google.com/file/d/17B-3TcXJ1tumNPYAQpg2MisSi9-7Dz-Y/view?usp=sharing)


## Low-Resource Visual Question Answering

All commands are runnable on a single GPU. We provide the examples to use Self-PT for low-resource VQA when the number of training samples is 16.

### VQA

```bash
bash scripts/VQA.sh 
```

### OKVQA

```bash
bash scripts/OKVQA.sh 
```

### GQA

```bash
bash scripts/GQA.sh 
```


Some important command line arguments are listed as follows:

| Args                             | Values                                                     | Descriptions                      | Notes                                                        |
| ------------------------------- | ---------------------------------------------------------- | -------------------------------- | ------------------------------------------------------------ |
| `--load`                        | path for trained checkpoints                               | load a checkpoint                |                                                              |                                            |
| `--subsample`                   | store_true                                                 | Subsample train and val sets for low-resource setting  |                                           |
| `--num_data`                    | {16, 32, 64, 100, 500, 1000}                                             | Number of subsamples for train and val sets | default=16                                       |
|`--pre_seq_len`| 5 | prompt length  | default=5 |
| `--prompt_index_dim` | 2 | the width of weight bank ||
| `--prompt_reduction_factor` | 6 | the feature dimension / the bottleneck dimension | default=768/128 | |
| `--prompt_phm_rank` | 8 | the rank of parameter factorization | |
| `--prompt_hypercomplex_division` | 4 | the number of summations of Kronecker product | |
| `--prompt_input_type` | 'cls' | choose the conditions for Self-PT: 'cls' for [cls] token, 'mean' for mean pooling, 'max' for max pooling | |
| `--prompt_type` | 'hyper_phm_new' | choose the prompt tuning methods: 'orip' for general prompt tuning, 'hyper_phm_new' for Self-PT | |
|`--prompt_cross` | False | set prompt tuning methods in cross-attention | default=False |
|`--add_adapter_cross_attn`| False | set adapters in self-attention | default=False  |
|`--add_adapter_cross_attn`| False | set adapters in cross-attention | default=False  |
| `--prompt_encoder` | True | set prompt tuning methods in encoder | default=True |
| `--prompt_decoder` | True | set prompt tuning methods in decoder | default=True |
