import torch
from torch import nn
from transformers import CLIPProcessor, CLIPModel
import transformers
from transformers import CLIPConfig, CLIPModel
from transformers.models.clip.modeling_clip import clip_loss, CLIPOutput
import torch_dct as dct
from typing import Any, Optional, Tuple, Union
import torch_dct as dct
from collections import OrderedDict

from loguru import logger

class FreqApdapterModel(CLIPModel):
    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        self.image_size = self.vision_model.config.image_size

        self.vision_dim = self.vision_model.config.hidden_size
        self.text_dim = self.text_model.config.hidden_size
        self.patch_size = self.vision_model.config.patch_size

        self.ln = nn.LayerNorm(self.vision_dim, elementwise_affine=False)
        # print(self.config)
        self.freq_adapter = FrequencyAdapter(image_dim=self.vision_dim, text_dim=self.text_dim, image_size=self.image_size, patch_size=self.patch_size)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # pixel_values = pixel_values.half()

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        text_embedding_to_fuse = text_outputs['hidden_states'][-2]

        # reforward vision model using the last two layers
        vision_embedding_to_fuse = vision_outputs['hidden_states'][-2]
        # [bs, seq_len, dim] -> [bs, dim]
        text_embedding_to_fuse = text_embedding_to_fuse * attention_mask.unsqueeze(-1)
        text_embedding_to_fuse = torch.sum(text_embedding_to_fuse, dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
        vision_embedding_to_fuse = self.freq_adapter(vision_embedding_to_fuse, text_embedding_to_fuse)

        last_vision_layer_output = self.vision_model.encoder.layers[-1](vision_embedding_to_fuse, None, None, output_attentions=False)
        last_vision_hidden_state = last_vision_layer_output[0]
        # for cam
        # last_vision_hidden_state = self.freq_ln_for_cam(last_vision_hidden_state)
        
        adapted_vision_pooler_output = last_vision_hidden_state[:, 0, :]
        adapted_vision_output = self.vision_model.post_layernorm(adapted_vision_pooler_output)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(adapted_vision_output)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # generate frequency mask
        # freq_mask = self.freq_adapter(vision_outputs['last_hidden_state'][:,1:,:], text_outputs['pooler_output'])

        # image_embeds = 0.9 * image_embeds + 0.1 * image_embeds * freq_mask

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

    def get_all_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)
        return [text_outputs['hidden_states'][-2], text_features]

    def get_all_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return [vision_outputs['hidden_states'][-2], image_features]


    def get_adapted_image_features(self, visual_features, textual_features):
        # visual_embeds [bs, seq_len, visual_dim]
        # text_embeds [bs, text_dim]
        # visual_features = self.freq_adapter(visual_features, textual_features)
        # visual_features = self.visual_projection(visual_features)
        vision_embedding_to_fuse = self.freq_adapter(visual_features, textual_features)
        # print(f'vision_embedding_to_fuse: {vision_embedding_to_fuse}')
        last_vision_layer_output = self.vision_model.encoder.layers[-1](vision_embedding_to_fuse, None, None, output_attentions=False)
        last_vision_hidden_state = last_vision_layer_output[0]
        adapted_vision_pooler_output = last_vision_hidden_state[:, 0, :]
        adapted_vision_output = self.vision_model.post_layernorm(adapted_vision_pooler_output)
        image_embeds = self.visual_projection(adapted_vision_output)
        return image_embeds

    def get_llava_adapted_image_features(        
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # pixel_values = pixel_values.half()

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        text_embedding_to_fuse = text_outputs['hidden_states'][-2]

        # reforward vision model using the last two layers
        vision_embedding_to_fuse = vision_outputs['hidden_states'][-2]
        # fuse
        vision_embedding_to_fuse = self.freq_adapter(vision_embedding_to_fuse, text_embedding_to_fuse)
        return vision_embedding_to_fuse

    def set_adapter(self, enable_mcfa, enable_mgfa, use_freq_adapter):
        self.freq_adapter.use_mcfa = enable_mcfa
        self.freq_adapter.use_mgfa = enable_mgfa
        self.freq_adapter.use_freq_adapter = use_freq_adapter
        logger.info(f'Set freq adapter: use_mcfa: {enable_mcfa}, use_mgfa: {enable_mgfa}, use_freq_adapter: {use_freq_adapter}, share: {self.freq_adapter.share}')

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

# multi-scale global frequency adapter
class MGFA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.bottle_dim = 32
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)

        self.adapter = nn.Sequential(
            nn.Linear(dim, self.bottle_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.bottle_dim, dim),
        )
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [bs, seq_len, dim]
        return self.adapter(self.ln(x))

# multi-scale cross-modal frequency adapter
class MCFA(nn.Module):
    def __init__(self, vision_dim, text_dim):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.inter_dim = 32
        self.visual_ln = nn.LayerNorm(vision_dim, elementwise_affine=False) 
        self.textual_ln = nn.LayerNorm(text_dim, elementwise_affine=False)

        self.modulator = nn.Sequential(
            nn.Linear(text_dim, self.inter_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.inter_dim, vision_dim*2),
        )

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, visual_features, textual_features):
        # visual features: [bs, seq_len, vision_dim]
        # textual features: [bs, text_dim]
        global_text_feature = self.textual_ln(textual_features)
        normal_visual_features = self.visual_ln(visual_features)

        params = self.modulator(global_text_feature) # [bs, vision_dim*2]
        gamma, beta = params.chunk(2, dim=-1) # [bs, vision_dim], [bs, vision_dim]
        modulated_visual_features = normal_visual_features * gamma.unsqueeze(1) + beta.unsqueeze(1)
        # transformed_visual_features = self.visual_transformer(modulated_visual_features)
        return modulated_visual_features

class FrequencyAdapter(nn.Module):
    def __init__(self, image_dim, text_dim, image_size, patch_size):
        super().__init__()
        self.dim = text_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.patch_num = int(image_size / self.patch_size)
        self.patch_num_sq = self.patch_num * self.patch_num
        self.share = False
        self.use_mcfa = True
        self.use_mgfa = True
        self.use_freq_adapter = True

        self.down_number = 3
        self.visual_ln = nn.LayerNorm(image_dim, elementwise_affine=False)
        self.avg_poolers = nn.ModuleList([
            nn.AvgPool2d(kernel_size=2**i, stride=2**i) for i in range(self.down_number)
        ])
        if self.share:
            self.mcfa = MCFA(vision_dim=image_dim, text_dim=text_dim)
            self.mgfa = MGFA(dim=image_dim)
        else:
            self.mcfa = nn.ModuleList([
                MCFA(vision_dim=image_dim, text_dim=text_dim) for i in range(self.down_number)
            ])
            self.mgfa = nn.ModuleList([
                MGFA(dim=image_dim) for i in range(self.down_number)
            ])

    def init_weights(self):
        # self.mcfa.init_weights()
        # self.mgfa.init_weights()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # 在 FrequencyAdapter 类里，替换 visual_freq_adaptation 为这个版本
    def visual_freq_adaptation(self, visual_freq_features, textual_freq_features, return_per_scale: bool=False):
        bs = visual_freq_features.shape[0]
        multi_scale_visual_freq_features = []
        for i in range(self.down_number):
            tmp_visual_freq_features = visual_freq_features.permute(0,2,1)\
                .reshape(bs, self.image_dim, self.patch_num, self.patch_num)
            tmp_visual_freq_features = self.avg_poolers[i](tmp_visual_freq_features)\
                .reshape(bs, self.image_dim, -1).permute(0,2,1)
            multi_scale_visual_freq_features.append(self.visual_ln(tmp_visual_freq_features))

        adapted_multi_scale_global_visual_freq_features = []
        per_scale_upsampled = []  # <--- 新增：用于返回每个scale上采样回原尺度后的结果
        for i in range(self.down_number):
            tmp_visual_freq_features = multi_scale_visual_freq_features[i]
            cur_visual_global_freq_features = 0
            cur_visual_cross_modal_freq_features = 0
            if self.use_mgfa:
                cur_visual_global_freq_features = (self.mgfa if self.share else self.mgfa[i])(tmp_visual_freq_features)
            if self.use_mcfa:
                cur_visual_cross_modal_freq_features = (self.mcfa if self.share else self.mcfa[i])(tmp_visual_freq_features, textual_freq_features)
            cur_visual_freq_features = cur_visual_global_freq_features + cur_visual_cross_modal_freq_features
            if i != 0:
                scale_factor = 2 ** i
                cur_visual_freq_features = cur_visual_freq_features.permute(0,2,1)\
                    .reshape(bs, self.image_dim, self.patch_num//scale_factor, self.patch_num//scale_factor)
                cur_visual_freq_features = cur_visual_freq_features.repeat_interleave(scale_factor, dim=2)\
                    .repeat_interleave(scale_factor, dim=3).reshape(bs, self.image_dim, -1).permute(0,2,1)
            adapted_multi_scale_global_visual_freq_features.append(cur_visual_freq_features)
            per_scale_upsampled.append(cur_visual_freq_features)  # <---

        fused = torch.mean(torch.stack(adapted_multi_scale_global_visual_freq_features, dim=0), dim=0)
        out = 0.1 * fused + 0.9 * visual_freq_features
        if return_per_scale:
            # 以 dict 形式返回每个scale（已上采样到原始patch数目）的贡献，均在“空域融合前”
            return out, per_scale_upsampled
        return out



    def forward(self, visual_features, text_features):
        data_type = visual_features.dtype
        # visul feature shape: [bs, 577, 1024], textual feature shape: [bs, 77, 768]
        bs = visual_features.shape[0]
        bs_text = text_features.shape[0]
        # check whether visual features and text features have the same batch size
        if bs != bs_text and bs == 1:
            visual_features = visual_features.expand(bs_text, -1, -1)

        if self.use_freq_adapter:
            print(f'use freq adapter')
            visual_freq_features = dct.dct(visual_features[:, 1:, :].float(), norm='ortho') # [bs, 576, 1024]
            textual_freq_features = dct.dct(text_features.float(), norm='ortho') # [bs, 768]

            visual_freq_features = visual_freq_features.to(data_type)
            textual_freq_features = textual_freq_features.to(data_type)

            tmp_visual_features, per_scale_ups = self.visual_freq_adaptation(
                visual_freq_features, textual_freq_features, return_per_scale=True
            )
            self._last_per_scale_ups = per_scale_ups  # 缓存起来，供可视化调用

            tmp_visual_features = dct.idct(tmp_visual_features.float(), norm='ortho').to(data_type) # [bs, 576, 1024]
        else:
            tmp_visual_features = self.visual_freq_adaptation(visual_features[:, 1:, :], text_features)
        # print(f'tmp_visual_features: {tmp_visual_features}')
        visual_features_new = visual_features.clone()
        visual_features_new[:, 1:, :] = tmp_visual_features
        return visual_features_new

if __name__ == '__main__':
    visual_features = torch.randn(2, 576, 1024).cuda()
    textual_features = torch.randn(2, 77, 768).cuda()
    freq_adapter = FrequencyAdapter(image_dim=1024, text_dim=768, image_size=336, patch_size=14).cuda()
    output = freq_adapter(visual_features, textual_features)
    print(f'output: {output.shape}')