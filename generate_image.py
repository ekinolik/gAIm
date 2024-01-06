import torch
from diffusers import StableDiffusionPipeline

class GenImage:
    def __init__(self, model, device='mps', dtype=torch.float16):
        self.prompt_embeds = None
        self.negative_prompt_embeds = None

        self.device = device

        self.pipeline = StableDiffusionPipeline.from_single_file(
            pretrained_model_link_or_path = model,
            torch_dtype                   = torch.float16,
            load_safety_checker           = None
        )
        self.pipeline.to('mps')

    def loadPrompt(self, prompt):
        self.prompt = prompt

    def loadNegPrompt(self, neg_prompt):
        self.neg_prompt = neg_prompt

    def splitLargePrompt(self):
        max_length = self.pipeline.tokenizer.model_max_length

        input_ids = self.pipeline.tokenizer(self.prompt, return_tensors='pt').input_ids
        input_ids = input_ids.to(self.device)

        negative_ids = self.pipeline.tokenizer(self.neg_prompt, truncation=False, padding="max_length", max_length=input_ids.shape[-1], return_tensors='pt').input_ids
        negative_ids = negative_ids.to(self.device)

        concat_embeds = []
        neg_embeds = []
        for i in range(0, input_ids.shape[-1], max_length):
            concat_embeds.append(self.pipeline.text_encoder(input_ids[:, i: i + max_length])[0])
            neg_embeds.append(self.pipeline.text_encoder(negative_ids[:, i: i + max_length])[0])

        self.prompt_embeds = torch.cat(concat_embeds, dim=1)
        self.negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

    def generateImage(self):
        if self.prompt_embeds is None or self.negative_prompt_embeds is None:
            return False

        self.image = self.pipeline(prompt_embeds=self.prompt_embeds, negative_prompt_embeds=self.negative_prompt_embeds).images[0]
        return True