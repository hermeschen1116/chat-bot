from transformers import TextGenerationPipeline
from transformers.pipelines.text_generation import Chat, ReturnType


class ResponseGeneratorPipeline(TextGenerationPipeline):
	def postprocess(self, model_outputs, return_type=ReturnType.FULL_TEXT, clean_up_tokenization_spaces=True):
		generated_sequence = model_outputs["generated_sequence"][0]
		input_ids = model_outputs["input_ids"]
		prompt_text = model_outputs["prompt_text"]
		generated_sequence = generated_sequence.numpy().tolist()
		records = []
		for sequence in generated_sequence:
			if return_type == ReturnType.TENSORS:
				record = {"generated_token_ids": sequence}
			elif return_type in {ReturnType.NEW_TEXT, ReturnType.FULL_TEXT}:
				# Decode text
				text = self.tokenizer.decode(
					sequence, skip_special_tokens=True, clean_up_tokenization_spaces=clean_up_tokenization_spaces
				)

				# Remove PADDING prompt of the sequence if XLNet or Transfo-XL model is used
				if input_ids is None:
					prompt_length = 0
				else:
					prompt_length = len(
						self.tokenizer.decode(
							input_ids[0],
							skip_special_tokens=True,
							clean_up_tokenization_spaces=clean_up_tokenization_spaces,
						)
					)

				all_text = text[prompt_length:].strip()
				if return_type == ReturnType.FULL_TEXT:
					if isinstance(prompt_text, str):
						all_text = prompt_text + all_text
					elif isinstance(prompt_text, Chat):
						# Explicit list parsing is necessary for parsing chat datasets
						prompt_text.messages[-1]["content"]["dialog"] = all_text
						all_text = list(prompt_text.messages)
				record = all_text
			records.append(record)

		return records
