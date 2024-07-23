from . import story_attention
from comfy.ldm.modules.attention import SpatialTransformer
import comfy.model_management as model_management
import torch
import comfy.samplers
import nodes
import copy

class SeaArtApplyStory:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "id_length": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
            "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 1}),
            "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 1}),
            "same": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step":0.01, "round": 0.01}),
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    CATEGORY = "SeaArt"

    def apply(self, model, id_length, width, height, same):
        import gc
        import comfy.model_management as model_management
        gc.collect()
        model_management.soft_empty_cache(True)
        model = model.clone()
        model.model = copy.deepcopy(model.model)
        story_attention.width = width
        story_attention.height = height
        story_attention.total_count = 0
        story_attention.total_length = id_length + 1
        story_attention.id_length = id_length
        story_attention.mask1024, story_attention.mask4096 = story_attention.cal_attn_mask_xl(
        story_attention.total_length,story_attention.id_length,story_attention.sa32,story_attention.sa64,story_attention.height,story_attention.width,device=model_management.get_torch_device(),dtype= torch.float16)
        story_attention.write = True
        story_attention.sa32 = same
        story_attention.sa64 = same
        for time_step in model.model.diffusion_model.output_blocks:
            #block是否包含上采样
            for block in time_step:
                if isinstance(block, SpatialTransformer):
                    for transformer_block in block.transformer_blocks:
                        if transformer_block.attn1 is not None:
                            transformer_block.attn1 = story_attention.StoryCrossAttention(transformer_block.attn1,torch.float16)
                            story_attention.total_count += 1
        print("total patch:",story_attention.total_count)
        return (model,)

class SeaArtCharactorPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                        "prompt": ("STRING", {"multiline": True,"default": ""})},
                    }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "input"

    CATEGORY = "SeaArt"

    def input(self,prompt):
        return (prompt,)

class SeaArtAppendPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                        "charactor_prompt": ("STRING", {"multiline": True,"default": ""}),
                        "prompt": ("STRING", {"multiline": True,"default": ""})
                        },
                    }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "input"

    CATEGORY = "SeaArt"

    def input(self,charactor_prompt,prompt):
        return (charactor_prompt + ", " + prompt,)

class SeaArtMergeStoryCondition:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "conditioning_1": ("CONDITIONING", ), 
                 },
                 "optional":{
            "conditioning_2": ("CONDITIONING", ), 
            "conditioning_3": ("CONDITIONING", ), 
            "conditioning_4": ("CONDITIONING", ), 
            "conditioning_5": ("CONDITIONING", )
                 }}
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "merge"

    CATEGORY = "SeaArt"

    def clip_tensor_clean(self,clip,list):
        ##寻找尺寸最大的张量
        max_token = 0
        for item in list:
            max_token = max(max_token,item.shape[1])
        tokens = clip.tokenize("")
        empty_cond, empty_pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        for index,item in enumerate(list):
            if item.shape[1] < max_token:
                r = (max_token - item.shape[1]) // 77
                for i in range(r):
                    list[index] = torch.cat((list[index],empty_cond[0][0]),dim=1)
        return list  

    def merge(self,clip,conditioning_1,conditioning_2=None,conditioning_3=None,conditioning_4=None,conditioning_5=None):
        filter_conds = []
        for x in [conditioning_1,conditioning_2,conditioning_3,conditioning_4,conditioning_5]:
            if isinstance(x,list):
                for c in x:
                    filter_conds.append(c)
        positive_conds_list = []
        positive_conds_pool_list = []
        for condition in filter_conds:
            positive_conds_list.append(condition[0])
            if len(condition) > 1 and torch.is_tensor(condition[1]["pooled_output"]):
                positive_conds_pool_list.append(condition[1]["pooled_output"])
        positive_conds_list = self.clip_tensor_clean(clip,positive_conds_list)
        positive_conds = [[torch.cat(positive_conds_list), {"pooled_output": torch.cat(positive_conds_pool_list)}]]
        return (positive_conds,)
        
class SeaArtStoryKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("LATENT","MODEL",)
    FUNCTION = "sample"

    CATEGORY = "SeaArt"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        story_attention.write = True
        story_attention.cur_step = 0
        latent = nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)[0]
        return (latent,model,)

class SeaArtStoryKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                     }
                }

    RETURN_TYPES = ("LATENT","MODEL",)
    FUNCTION = "sample"

    CATEGORY = "SeaArt"

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        story_attention.write = True
        story_attention.cur_step = 0
        latent = nodes.KSamplerAdvanced().sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise)[0]  
        return (latent,model,)

class SeaArtStoryInfKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("LATENT","MODEL",)
    FUNCTION = "sample"

    CATEGORY = "SeaArt"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        story_attention.write = False
        story_attention.cur_step = 0
        latent = nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)[0]
        return (latent,model,)
    
class SeaArtFirstImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"image": ("IMAGE", ),},
                }
    

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "do"

    CATEGORY = "SeaArt"

    def do(self,image):
        image = image[0]
        return (image.unsqueeze(0),)

class SeaArtStoryInfKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                     }
                }

    RETURN_TYPES = ("LATENT","MODEL",)
    FUNCTION = "sample"

    CATEGORY = "SeaArt"

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        story_attention.write = False
        story_attention.cur_step = 0
        latent = nodes.KSamplerAdvanced().sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise)[0]  
        return (latent,model,)
