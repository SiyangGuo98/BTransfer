import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import copy
from transformers import AutoTokenizer, TrainingArguments
import wandb

from constants import CACHE_DIR
from utils import set_random_seed, load_and_merge_hparams, standardize_metrics
from baselines import create_transfer_from_hparams
from src.args import parse_data_arguments
from src.dataset import FewShotDataset
from src.utils import load_models_from_cache, build_compute_metrics_fn, evaluate_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning transfer experiments for prompt-based classification.")
    
    # Model arguments
    parser.add_argument("--model_pt_old", type=str, default="roberta-base",
                        help="Name of the old pre-trained model")
    parser.add_argument("--model_pt_new", type=str, default="legal",
                        help="Name of the new pre-trained model")
    parser.add_argument("--fine_tuning_dataset", type=str, default="SST-2",
                        help="Dataset for fine-tuning and evaluation")
    parser.add_argument("--transfer_method", type=str, default="paramD",
                        help="Transfer method to use: paramD, BTransfer, TransFusion")
    parser.add_argument("--evaluate_all", action="store_true", default=True,
                        help="Whether to evaluate all models")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory for saving outputs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Method-specific arguments
    parser.add_argument("--lambda_", type=float, default=None,
                        help="[paramD] Lambda parameter")
    parser.add_argument("--fisher_sample_size", type=int, default=None,
                        help="[BTransfer] Number of samples for Fisher information estimation")
    parser.add_argument("--fisher_batch_size", type=int, default=None,
                        help="[BTransfer] Batch size for Fisher information estimation")
    
    args = parser.parse_args()
    
    set_random_seed(seed=args.seed)
    hparams = load_and_merge_hparams(args)

    run_name = f"{args.model_pt_old}_{args.model_pt_new}_{args.fine_tuning_dataset}_{args.transfer_method}"
    run_config = {
        "benchmark": "prompt-based classification",
        **hparams.to_dict(),
        "seed": args.seed,
    }

    wandb.init(
        project='btransfer',
        name=run_name,
        config=run_config,
    )
    
    print(f"\n{'='*60}")
    print(f"Experiment Configuration:")
    print(f"  Old Pre-trained Model: {args.model_pt_old}")
    print(f"  New Pre-trained Model: {args.model_pt_new}")
    print(f"  Dataset: {args.fine_tuning_dataset}")
    print(f"  Transfer Method: {args.transfer_method}")
    print(f"  Random Seed: {args.seed}")
    print(f"{'='*60}")

    # =========================================================================
    # Step 1: Load tokenizer and prepare datasets
    # =========================================================================
    print("\n[Step 1] Loading tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=os.path.join(CACHE_DIR, "models/lm-bff", args.model_pt_old)
    )

    data_args = parse_data_arguments(args.fine_tuning_dataset)
    train_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="train", use_demo=False)
    )
    test_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="test", use_demo=False)
    )
    print(f"  Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # =========================================================================
    # Step 2: Load pre-trained and fine-tuned models
    # =========================================================================
    print("\n[Step 2] Loading models...")
    model_pt_old, model_ft_old, model_pt_new = load_models_from_cache(
        model_pt_old_name=args.model_pt_old,
        model_pt_new_name=args.model_pt_new,
        dataset_name=args.fine_tuning_dataset,
        cache_dir=CACHE_DIR
    )
    print(f"  Models loaded successfully.")

    # =========================================================================
    # Step 3: Initialize transfer method
    # =========================================================================
    print(f"\n[Step 3] Initializing transfer method: {args.transfer_method}")
    common_keys = {"model_pt_old_name", "model_pt_new_name", "fine_tuning_dataset_name", "transfer_method"}
    for key, value in hparams.to_dict().items():
        if key not in common_keys:
            print(f"  {key}: {value}")
    transfer_method = create_transfer_from_hparams(args.transfer_method, hparams)

    # =========================================================================
    # Step 4: Perform fine-tuning transfer
    # =========================================================================
    print("\n[Step 4] Performing fine-tuning transfer...")
    transfer_kwargs = {
        "fine_tuning_dataset": train_dataset,
        "tokenizer": tokenizer,
        "benchmark": "prompt-based classification",
    }
    
    transferred_model = transfer_method.get_transferred_model(
        model_pt_old=model_pt_old,
        model_ft_old=model_ft_old,
        model_pt_new=model_pt_new,
        **transfer_kwargs
    )
    print("  Transfer completed.")

    # =========================================================================
    # Step 5: Evaluate transferred model and baselines
    # =========================================================================
    print("\n[Step 5] Evaluating models...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.batch_size,
        do_eval=True,
        dataloader_num_workers=4,
        report_to="none",
    )
    
    compute_metrics_fn = build_compute_metrics_fn(args.fine_tuning_dataset.lower(), test_dataset)

    # Evaluate transferred model
    print("\n  Evaluating Transferred Model...")
    transferred_metrics = evaluate_model(
        model=transferred_model,
        test_dataset=test_dataset,
        training_args=training_args,
        compute_metrics_fn=compute_metrics_fn,
    )


    std_transferred = standardize_metrics(transferred_metrics, args.fine_tuning_dataset, "prompt-based classification")
    wandb_log = {
        "primary_metric_name": std_transferred["primary_metric_name"],
        "transferred_primary_metric": std_transferred["primary_metric"],
        **{f"transferred_{k}": v for k, v in transferred_metrics.items() if not k.endswith("loss") and not k.endswith(std_transferred["primary_metric_name"])}
    }

    if args.evaluate_all:
        model_pt_new_prompt = copy.deepcopy(model_ft_old)
        model_pt_new_state = model_pt_new.state_dict()
        model_ft_old_state = model_ft_old.state_dict()
        for key in model_ft_old_state:
            if 'pooler' in key:
                model_pt_new_state[key] = model_ft_old_state[key]
        model_pt_new_prompt.load_state_dict(model_pt_new_state, strict=True)
        
        # Evaluate the new pre-trained model
        print("\n  Evaluating The New Pre-trained Model...")
        pt_new_metrics = evaluate_model(
            model=model_pt_new_prompt,
            test_dataset=test_dataset,
            training_args=training_args,
            compute_metrics_fn=compute_metrics_fn,
        )

        # Evaluate the old fine-tuned model
        print("\n  Evaluating The Old Fine-tuned Model...")
        ft_old_metrics = evaluate_model(
            model=model_ft_old,
            test_dataset=test_dataset,
            training_args=training_args,
            compute_metrics_fn=compute_metrics_fn,
        )

        std_pt_new = standardize_metrics(pt_new_metrics, args.fine_tuning_dataset, "prompt-based classification")
        std_ft_old = standardize_metrics(ft_old_metrics, args.fine_tuning_dataset, "prompt-based classification")
        
        wandb_log.update({
            "pre_training_new_primary_metric": std_pt_new["primary_metric"],
            "fine_tuned_primary_metric": std_ft_old["primary_metric"],
        })

        wandb_log.update({
            **{f"pre_training_new_{k}": v for k, v in pt_new_metrics.items() if not k.endswith("loss") and not k.endswith(std_pt_new["primary_metric_name"])},
            **{f"fine_tuned_{k}": v for k, v in ft_old_metrics.items() if not k.endswith("loss") and not k.endswith(std_ft_old["primary_metric_name"])},
        })


    wandb.log(wandb_log)
    wandb.finish()
    
    print(f"\n{'='*60}")
    print("Experiment completed successfully.")
    print(f"{'='*60}")