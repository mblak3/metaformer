from mmfreelm.models import HGRNBitConfig
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer, DataCollatorForLanguageModeling, get_scheduler
from tqdm.notebook import tqdm 
from torch.nn import CrossEntropyLoss
import torch
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator
from torch.optim import AdamW

#CUDA
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")

# Config for the 370M model
# Reference: https://huggingface.co/ridger/MMfreeLM-370M/blob/main/config.json
config_params = {
    "attn_mode": "fused_recurrent",
    "bos_token_id": 1,
    "conv_size": 4,
    "eos_token_id": 2,
    "expand_ratio": 1,
    "fuse_cross_entropy": True,
    "hidden_act": "swish",
    "hidden_ratio": 4,
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": None,
    "max_position_embeddings": 2048,
    "model_type": "hgrn_bit",
    "num_heads": 1,
    "num_hidden_layers": 24,
    "rms_norm_eps": 1e-06,
    "share_conv_kernel": True,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.40.2",
    "use_cache": True,
    "use_lower_bound": True,
    "use_short_conv": False,
    "vocab_size": 32000,
}

########
# TODO #
########
#change data set

#Load datasets
ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")
#dataset = load_dataset('range3/wikipedia-ja-20230101', split='train[:1%]')

raw_datasets = DatasetDict(
    {
        "train": ds_train,  # .shuffle().select(range(50000)),
        "valid": ds_valid,  # .shuffle().select(range(500))
    }
)
raw_datasets

#Prepare datasets

context_length = 2048
# tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
tokenizer = AutoTokenizer.from_pretrained("ridger/MMfreeLM-370M")

outputs = tokenizer(
    raw_datasets["train"][:2]["content"],
    truncation=True,
    max_length=context_length,
    return_overflowing_tokens=True,
    return_length=True,
)

print(f"Input IDs length: {len(outputs['input_ids'])}")
print(f"Input chunk lengths: {(outputs['length'])}")
print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")

#Tokenize datasets 

def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)
tokenized_datasets

#Initialize model 

config = HGRNBitConfig(**config_params)
model = AutoModelForCausalLM.from_config(config)

model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
for key in out:
    print(f"{key} shape: {out[key].shape}")

#Training arguments 

args = TrainingArguments(
    output_dir="codeparrot-ds",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)

trainer.train()

trainer.push_to_hub()


#####################################################################

#####################################################################

# keytoken_ids = []
# for keyword in [
#     "plt",
#     "pd",
#     "sk",
#     "fit",
#     "predict",
#     " plt",
#     " pd",
#     " sk",
#     " fit",
#     " predict",
#     "testtest",
# ]:
#     ids = tokenizer([keyword]).input_ids[0]
#     if len(ids) == 1:
#         keytoken_ids.append(ids[0])
#     else:
#         print(f"Keyword has not single token: {keyword}")

# def keytoken_weighted_loss(inputs, logits, keytoken_ids, alpha=1.0):
#     # Shift so that tokens < n predict n
#     shift_labels = inputs[..., 1:].contiguous()
#     shift_logits = logits[..., :-1, :].contiguous()
#     # Calculate per-token loss
#     loss_fct = CrossEntropyLoss(reduce=False)
#     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#     # Resize and average loss per sample
#     loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
#     # Calculate and scale weighting
#     weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(
#         axis=[0, 2]
#     )
#     weights = alpha * (1.0 + weights)
#     # Calculate weighted average
#     weighted_loss = (loss_per_sample * weights).mean()
#     return weighted_loss

# tokenized_datasets.set_format("torch")
# train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=32, shuffle=True)
# eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=32)

# weight_decay = 0.1

# def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
#     params_with_wd, params_without_wd = [], []
#     for n, p in model.named_parameters():
#         if any(nd in n for nd in no_decay):
#             params_without_wd.append(p)
#         else:
#             params_with_wd.append(p)
#     return [
#         {"params": params_with_wd, "weight_decay": weight_decay},
#         {"params": params_without_wd, "weight_decay": 0.0},
#     ]

# def evaluate():
#     model.eval()
#     losses = []
#     for step, batch in enumerate(eval_dataloader):
#         with torch.no_grad():
#             outputs = model(batch["input_ids"], labels=batch["input_ids"])

#         losses.append(accelerator.gather(outputs.loss))
#     loss = torch.mean(torch.cat(losses))
#     try:
#         perplexity = torch.exp(loss)
#     except OverflowError:
#         perplexity = float("inf")
#     return loss.item(), perplexity.item()


# optimizer = AdamW(get_grouped_params(model), lr=5e-4)

# accelerator = Accelerator(mixed_precision="fp16")

# model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
#     model, optimizer, train_dataloader, eval_dataloader
# )

# num_train_epochs = 1
# num_update_steps_per_epoch = len(train_dataloader)
# num_training_steps = num_train_epochs * num_update_steps_per_epoch

# lr_scheduler = get_scheduler(
#     name="linear",
#     optimizer=optimizer,
#     num_warmup_steps=1_000,
#     num_training_steps=num_training_steps,
# )

# output_dir = 'test'

# gradient_accumulation_steps = 8
# eval_steps = 5_000

# model.to('cuda')


# model.train()
# completed_steps = 0
# for epoch in range(num_train_epochs):
#     for step, batch in tqdm(
#         enumerate(train_dataloader, start=1), total=num_training_steps
#     ):
#         logits = model(batch["input_ids"]).logits
#         loss = keytoken_weighted_loss(batch["input_ids"], logits, keytoken_ids)
#         if step % 100 == 0:
#             accelerator.print(
#                 {
#                     "samples": step * samples_per_step,
#                     "steps": completed_steps,
#                     "loss/train": loss.item() * gradient_accumulation_steps,
#                 }
#             )
#         loss = loss / gradient_accumulation_steps
#         accelerator.backward(loss)
#         if step % gradient_accumulation_steps == 0:
#             accelerator.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             lr_scheduler.step()
#             optimizer.zero_grad()
#             completed_steps += 1
#         if (step % (eval_steps * gradient_accumulation_steps)) == 0:
#             eval_loss, perplexity = evaluate()
#             accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
#             model.train()
#             accelerator.wait_for_everyone()
#             unwrapped_model = accelerator.unwrap_model(model)
#             unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
#             if accelerator.is_main_process:
#                 tokenizer.save_pretrained(output_dir)
