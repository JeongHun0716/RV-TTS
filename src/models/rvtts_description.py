import torch
import torch.nn as nn
from src.models.modeling_rvtts_description import ParlerTTSForConditionalGeneration

class RVTTS(nn.Module):
    def __init__(self, spk_embed_dim, use_aud_mapper=False, bf16=False, dropout_input=0) -> None:
        super().__init__()

        if bf16:
            self.model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        else:
            self.model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1")

        self.config = self.model.config
        self.config.decoder.dropout_input = dropout_input
        self.config.decoder.mask_token_id = self.config.pad_token_id

        self.face_mapper = nn.Linear(spk_embed_dim, self.config.decoder.hidden_size)
        self.use_aud_mapper = use_aud_mapper
        if self.use_aud_mapper:
            self.audio_mapper = nn.Linear(192, self.config.decoder.hidden_size)

    def forward(self, decoder_input_ids, decoder_attention_mask, spk_description_embedding, description_embedding, spk_embedding, description_mask, prompt_input_ids, prompt_attention_mask, labels, aud_embedding=None, use_auds=None, parallel=False):
        
        spk_embedding = self.face_mapper(spk_embedding)
        if self.use_aud_mapper and aud_embedding is not None:
            aud_embedding = self.audio_mapper(aud_embedding)
            spk_embeds = torch.zeros_like(spk_embedding)
            spk_embeds[~use_auds] = spk_embedding[~use_auds]
            spk_embeds[use_auds] = aud_embedding[use_auds]
            spk_embedding = spk_embeds

        if (
            self.config.text_encoder.hidden_size != self.config.decoder.hidden_size
                and self.config.decoder.cross_attention_hidden_size is None
            ):
            spk_description_embedding = self.model.enc_to_dec_proj(spk_description_embedding)
            description_embedding = self.model.enc_to_dec_proj(description_embedding)

        encoder_outputs = torch.cat([spk_description_embedding.repeat(spk_embedding.size(0), 1, 1), spk_embedding, description_embedding], 1)
        encoder_mask = torch.cat([torch.ones([description_mask.size(0), encoder_outputs.size(1) - description_mask.size(1)], device=description_mask.device), description_mask], 1)
        encoder_outputs = (encoder_outputs, )
                
        outputs = self.model(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            labels=labels,
            prompt_spk_embedding=None,
            parallel=parallel,
        )

        return outputs

    def generate(self, encoder_outputs, encoder_attention_mask, prompt_input_ids, prompt_attention_mask=None, input_values=None, max_new_tokens=430+9):
        if input_values is not None:
            outputs = self.model.generate(encoder_outputs=encoder_outputs, attention_mask=encoder_attention_mask, prompt_input_ids=prompt_input_ids, prompt_attention_mask=prompt_attention_mask, max_new_tokens=max_new_tokens, input_values=input_values, prompt_spk_embedding=None)
        else:
            outputs = self.model.generate(encoder_outputs=encoder_outputs, attention_mask=encoder_attention_mask, prompt_input_ids=prompt_input_ids, prompt_attention_mask=prompt_attention_mask, max_new_tokens=max_new_tokens, prompt_spk_embedding=None)
        return outputs
    