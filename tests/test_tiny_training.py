"""Real Trainer + PEFT API integration on a tiny random CPU model, no downloads."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

HAS_ML = all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "peft", "accelerate"))

@unittest.skipUnless(HAS_ML, "Install ML requirements for tiny CPU training integration")
class TinyTrainingTest(unittest.TestCase):
    def test_train_save_reload(self):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments
        from finance_qlora.modeling import CompletionCollator
        torch.manual_seed(42)
        torch.set_num_threads(1)
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=64, bos_token_id=1, eos_token_id=2, pad_token_id=2)
        base = LlamaForCausalLM(config)
        initial = {k:v.detach().clone() for k,v in base.state_dict().items()}
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4,
            target_modules="all-linear", task_type="CAUSAL_LM", lora_dropout=0))
        rows = [{"input_ids":[1,4,5,8,2],"attention_mask":[1]*5,"labels":[-100,-100,-100,8,2]},
                {"input_ids":[1,6,9,2],"attention_mask":[1]*4,"labels":[-100,-100,9,2]}]
        with tempfile.TemporaryDirectory() as d:
            args = TrainingArguments(output_dir=d, max_steps=2, per_device_train_batch_size=1,
                gradient_accumulation_steps=2, learning_rate=0.01, optim="adamw_torch",
                use_cpu=True, report_to="none", save_strategy="no", disable_tqdm=True, label_names=["labels"], prediction_loss_only=True)
            trainer = Trainer(model=model,args=args,train_dataset=rows,eval_dataset=rows,
                data_collator=CompletionCollator(2))
            result = trainer.train()
            self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))
            evaluation = trainer.evaluate()
            self.assertTrue(torch.isfinite(torch.tensor(evaluation["eval_loss"])))
            self.assertTrue(any(torch.count_nonzero(p).item() for n,p in model.named_parameters() if "lora_B" in n))
            adapter = Path(d)/"adapter"
            trainer.save_model(str(adapter))
            self.assertTrue((adapter/"adapter_model.safetensors").exists())
            model.eval()
            source = torch.tensor([[1,4,5]])
            with torch.no_grad(): expected = model(source).logits
            fresh = LlamaForCausalLM(config); fresh.load_state_dict(initial)
            reloaded = PeftModel.from_pretrained(fresh, adapter); reloaded.eval()
            with torch.no_grad(): actual = reloaded(source).logits
            torch.testing.assert_close(actual, expected)
