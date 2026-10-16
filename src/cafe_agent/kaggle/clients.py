"""GPU clients with identical rendering and explicit context bounds; lazy imports."""

from typing import Any

from cafe_agent.models.base import Message, ModelResponse, TokenUsage
from cafe_agent.prompting.prompts import render_chat


class ContextLimitError(ValueError):
    pass


def context_failure() -> ModelResponse:
    return ModelResponse(content="CONTEXT_LIMIT_EXCEEDED", finish_reason="length")


class PromptClient:
    def __init__(
        self, tokenizer: Any, context: int, max_tokens: int, examples: list[Message] | None = None
    ) -> None:
        self.tokenizer, self.context, self.max_tokens = tokenizer, context, max_tokens
        self.examples = examples or []

    def prompt(self, messages: list[Message], tools: list[dict]) -> tuple[str, list[int]]:
        history = [messages[0], *self.examples, *messages[1:]] if self.examples else messages
        text = render_chat(self.tokenizer, history, tools)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) + self.max_tokens > self.context:
            raise ContextLimitError("Inference context exceeded; refusing to truncate history")
        return text, ids


class VLLMClient(PromptClient):
    def __init__(self, *args: Any, engine: Any, sampling: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.engine, self.sampling = engine, sampling

    def generate(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        try:
            _, ids = self.prompt(messages, tools)
        except ContextLimitError:
            return context_failure()
        request = self.engine.generate([{"prompt_token_ids": ids}], self.sampling, use_tqdm=False)[
            0
        ]
        result = request.outputs[0]
        return ModelResponse(
            content=result.text,
            finish_reason=result.finish_reason,
            usage=TokenUsage(
                prompt_tokens=len(ids),
                completion_tokens=len(result.token_ids),
                total_tokens=len(ids) + len(result.token_ids),
            ),
        )


class HFClient(PromptClient):
    def __init__(self, *args: Any, model: Any, generation: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model, self.generation = model, generation

    def generate(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        import torch

        try:
            _, ids = self.prompt(messages, tools)
        except ContextLimitError:
            return context_failure()
        inputs = torch.tensor([ids], dtype=torch.long, device=self.model.device)
        with torch.inference_mode():
            result = self.model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                generation_config=self.generation,
            )
        torch.cuda.synchronize()
        generated = result[0, len(ids) :].tolist()
        ended = bool(generated) and generated[-1] == self.tokenizer.eos_token_id
        text_ids = generated[:-1] if ended else generated
        return ModelResponse(
            content=self.tokenizer.decode(text_ids, skip_special_tokens=False),
            finish_reason="stop" if ended or len(generated) < self.max_tokens else "length",
            usage=TokenUsage(
                prompt_tokens=len(ids),
                completion_tokens=len(generated),
                total_tokens=len(ids) + len(generated),
            ),
        )


def load_client(
    backend: str, model: str, revision: str, tokenizer: Any, settings: dict, examples: list[Message]
) -> PromptClient:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle CUDA GPU is required")
    if backend == "vllm":
        from vllm import LLM, SamplingParams

        engine = LLM(
            model=model,
            revision=revision,
            tokenizer=settings["model_id"],
            tokenizer_revision=settings["revision"],
            dtype="float16",
            tensor_parallel_size=1,
            max_model_len=settings["context"],
            gpu_memory_utilization=settings["gpu_memory_utilization"],
            max_num_seqs=1,
            enforce_eager=True,
            seed=settings["seed"],
            trust_remote_code=False,
            generation_config="vllm",
        )
        sampling = SamplingParams(
            temperature=0,
            max_tokens=settings["max_tokens"],
            seed=settings["seed"],
            stop_token_ids=[tokenizer.eos_token_id],
        )
        return VLLMClient(
            tokenizer,
            settings["context"],
            settings["max_tokens"],
            examples,
            engine=engine,
            sampling=sampling,
        )
    from transformers import AutoModelForCausalLM, GenerationConfig

    model_object = AutoModelForCausalLM.from_pretrained(
        model,
        revision=revision,
        torch_dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="sdpa",
        trust_remote_code=False,
    ).eval()
    generation = GenerationConfig(
        do_sample=False,
        max_new_tokens=settings["max_tokens"],
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    return HFClient(
        tokenizer,
        settings["context"],
        settings["max_tokens"],
        examples,
        model=model_object,
        generation=generation,
    )
