from . import nodes as nodes

NODE_CLASS_MAPPINGS = {
    "SeaArtApplyStory": nodes.SeaArtApplyStory,
    "SeaArtMergeStoryCondition": nodes.SeaArtMergeStoryCondition,
    "SeaArtAppendPrompt": nodes.SeaArtAppendPrompt,
    "SeaArtCharactorPrompt": nodes.SeaArtCharactorPrompt,
    "SeaArtStoryKSampler": nodes.SeaArtStoryKSampler,
    "SeaArtStoryKSamplerAdvanced": nodes.SeaArtStoryKSamplerAdvanced,
    "SeaArtStoryInfKSampler": nodes.SeaArtStoryInfKSampler,
    "SeaArtStoryInfKSamplerAdvanced": nodes.SeaArtStoryInfKSamplerAdvanced,
    "SeaArtFirstImage": nodes.SeaArtFirstImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeaArtApplyStory": "SeaArtApplyStory",
    "SeaArtMergeStoryCondition": "SeaArtMergeStoryCondition",
    "SeaArtAppendPrompt": "SeaArtAppendPrompt",
    "SeaArtCharactorPrompt": "SeaArtCharactorPrompt",
    "SeaArtStoryKSampler": "SeaArtStoryKSampler",
    "SeaArtStoryKSamplerAdvanced": "SeaArtStoryKSamplerAdvanced",
    "SeaArtStoryInfKSampler": "SeaArtStoryInfKSampler",
    "SeaArtStoryInfKSamplerAdvanced": "SeaArtStoryInfKSamplerAdvanced",
    "SeaArtFirstImage": "SeaArtFirstImage"
}
