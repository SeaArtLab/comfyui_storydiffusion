from torch import nn

import comfy
import comfy.ops as ops
from comfy.ldm.modules.attention import default,CrossAttention,optimized_attention,optimized_attention_masked
import comfy.ops
ops = comfy.ops.disable_weight_init
import torch.nn.functional as F
import torch
import random

total_count = 0
attn_count = 0
cur_step = 0
mask1024 = None
mask4096 = None
indices1024 = None
indices4096 = None
sa32 = 0.5
sa64 = 0.5
write = False
height = 0
width = 0
id_length = 0
total_length = 0

def cal_attn_indice_xl_effcient_memory(total_length,id_length,sa32,sa64,height,width,device="cpu",dtype= torch.float16):
    nums_1024 = (height // 32) * (width // 32)
    nums_4096 = (height // 16) * (width // 16)
    bool_matrix1024 = torch.rand((total_length,nums_1024),device = device,dtype = dtype) < sa32
    bool_matrix4096 = torch.rand((total_length,nums_4096),device = device,dtype = dtype) < sa64
    # 用nonzero()函数获取所有为True的值的索引
    indices1024 = [torch.nonzero(bool_matrix1024[i], as_tuple=True)[0] for i in range(total_length)]
    indices4096 = [torch.nonzero(bool_matrix4096[i], as_tuple=True)[0] for i in range(total_length)]

    return indices1024,indices4096

def cal_attn_mask_xl(total_length,id_length,sa32,sa64,height,width,device="cuda",dtype= torch.float16):
    nums_1024 = (height // 32) * (width // 32)
    nums_4096 = (height // 16) * (width // 16)
    bool_matrix1024 = torch.rand((1, total_length * nums_1024),device = device,dtype = dtype) < sa32
    bool_matrix4096 = torch.rand((1, total_length * nums_4096),device = device,dtype = dtype) < sa64
    bool_matrix1024 = bool_matrix1024.repeat(total_length,1)
    bool_matrix4096 = bool_matrix4096.repeat(total_length,1)
    for i in range(total_length):
        bool_matrix1024[i:i+1,id_length*nums_1024:] = False
        bool_matrix4096[i:i+1,id_length*nums_4096:] = False
        bool_matrix1024[i:i+1,i*nums_1024:(i+1)*nums_1024] = True
        bool_matrix4096[i:i+1,i*nums_4096:(i+1)*nums_4096] = True
    mask1024 = bool_matrix1024.unsqueeze(1).repeat(1,nums_1024,1).reshape(-1,total_length * nums_1024)
    mask4096 = bool_matrix4096.unsqueeze(1).repeat(1,nums_4096,1).reshape(-1,total_length * nums_4096)
    return mask1024,mask4096

def __call1__(
    self,
    hidden_states,
    encoder_hidden_states=None,
    attention_mask=None,
    value=None,
):
    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        total_batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(total_batch_size, channel, height * width).transpose(1, 2)
    total_batch_size,nums_token,channel = hidden_states.shape
    img_nums = total_batch_size//2
    hidden_states = hidden_states.view(-1,img_nums,nums_token,channel).reshape(-1,img_nums * nums_token,channel)
    
    query = self.to_q(hidden_states)
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states  # B, N, C
    else:
        encoder_hidden_states = encoder_hidden_states.view(-1,id_length+1,nums_token,channel).reshape(-1,(id_length+1) * nums_token,channel)
    key = self.to_k(encoder_hidden_states)
    value = self.to_v(encoder_hidden_states)
    batch_size, sequence_length, _ = hidden_states.shape

    query = query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)

    key = key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
    value = value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
    hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

    hidden_states = hidden_states.transpose(1, 2).reshape(total_batch_size, -1, self.heads * self.dim_head)
    hidden_states = hidden_states.to(query.dtype)
    return self.to_out(hidden_states)

def __call2__(
    self,
    hidden_states,
    encoder_hidden_states=None,
    attention_mask=None,
    value=None):
    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
    batch_size, sequence_length, channel = (
        hidden_states.shape
    )
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states  # B, N, C
    else:
        encoder_hidden_states = encoder_hidden_states.view(-1,id_length+1,sequence_length,channel).reshape(-1,(id_length+1) * sequence_length,channel)
    query = self.to_q(hidden_states)
    key = self.to_k(encoder_hidden_states)
    value = self.to_v(encoder_hidden_states)

    query = query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)

    key = key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
    value = value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)

    hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * self.dim_head)
    hidden_states = hidden_states.to(query.dtype)
    return self.to_out(hidden_states)



class StoryCrossAttention(nn.Module):
    def __init__(self, cross_attention, dtype):
        import comfy.model_management as model_management
        super().__init__()

        self.heads = cross_attention.heads
        self.dim_head = cross_attention.dim_head

        self.to_q = cross_attention.to_q
        self.to_k = cross_attention.to_k
        self.to_v = cross_attention.to_v

        self.to_out = cross_attention.to_out
        self.device = model_management.get_torch_device()
        self.dtype = dtype
        self.id_bank = {}
        self.origin_attn = cross_attention

    def forward(self, x, context=None, value=None, mask=None):
        return self.new_forward(self,hidden_states=x,encoder_hidden_states=context,temb=value,attention_mask=mask)

    def old_forward(self, x, context=None, value=None, mask=None):
        global total_count,attn_count,cur_step,mask1024,mask4096
        global sa32, sa64
        global write
        global height,width
        if write:
            # print(f"white:{cur_step}")
            self.id_bank[cur_step] = [x[:id_length], x[id_length:]]
        else:
            context = torch.cat((self.id_bank[cur_step][0].to(self.device),x[:1],self.id_bank[cur_step][1].to(self.device),x[1:]))
        # skip in early step
        if cur_step < 5:
            x = __call2__(self,x,context,mask,value)
        else:   # 256 1024 4096
            random_number = random.random()
            if cur_step <20:
                rand_num = 0.3
            else:
                rand_num = 0.1
            if random_number > rand_num:
                if not write:
                    if x.shape[1] == (height//32) * (width//32):
                        attention_mask = mask1024[mask1024.shape[0] // total_length * id_length:]
                    else:
                        attention_mask = mask4096[mask4096.shape[0] // total_length * id_length:]
                else:
                    if x.shape[1] == (height//32) * (width//32):
                        attention_mask = mask1024[:mask1024.shape[0] // total_length * id_length,:mask1024.shape[0] // total_length * id_length]
                    else:
                        attention_mask = mask4096[:mask4096.shape[0] // total_length * id_length,:mask4096.shape[0] // total_length * id_length]
                attention_mask = attention_mask.to(self.device)
                x = __call1__(self,x,context,attention_mask,value)
            else:
                x = __call2__(self,x,None,mask,value)
        attn_count +=1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            mask1024,mask4096 = cal_attn_mask_xl(total_length,id_length,sa32,sa64,height,width, device=self.device, dtype= self.dtype)
        return x


    def new_forward(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None):
        # un_cond_hidden_states, cond_hidden_states = hidden_states.chunk(2)
        # un_cond_hidden_states = self.__call2__(attn, un_cond_hidden_states,encoder_hidden_states,attention_mask,temb)
        # 生成一个0到1之间的随机数
        global total_count,attn_count,cur_step, indices1024,indices4096
        global sa32, sa64
        global write
        global height,width
        global total_length,id_length
        if attn_count == 0 and cur_step == 0:
            indices1024,indices4096 = cal_attn_indice_xl_effcient_memory(total_length,id_length,sa32,sa64,height,width, dtype= self.dtype)
        if write:
            if hidden_states.shape[1] == (height//32) * (width//32):
                indices = indices1024
            else:
                indices = indices4096
            # print(f"white:{cur_step}")
            total_batch_size,nums_token,channel = hidden_states.shape
            img_nums = total_batch_size // 2
            hidden_states = hidden_states.reshape(-1,img_nums,nums_token,channel)
            cache_hidden_states = hidden_states.to("cpu")
            self.id_bank[cur_step] = [cache_hidden_states[:,img_ind,indices[img_ind],:].reshape(2,-1,channel).clone() for img_ind in range(img_nums)]
            hidden_states = hidden_states.reshape(-1,nums_token,channel)
            #self.id_bank[cur_step] = [hidden_states[:self.id_length].clone(), hidden_states[self.id_length:].clone()]
        else:
            #encoder_hidden_states = torch.cat((self.id_bank[cur_step][0].to(self.device),self.id_bank[cur_step][1].to(self.device)))
            encoder_arr = [tensor.to(self.device)  for tensor in self.id_bank[cur_step]]
        # 判断随机数是否大于0.5
        if cur_step <1:
            hidden_states = self.__call2__(attn, hidden_states,None,attention_mask,temb)
        else:   # 256 1024 4096
            random_number = random.random()
            if cur_step <20:
                rand_num = 0.3
            else:
                rand_num = 0.1
            # print(f"hidden state shape {hidden_states.shape[1]}")
            if random_number > rand_num:
                if hidden_states.shape[1] == (height//32) * (width//32):
                    indices = indices1024
                else:
                    indices = indices4096
                # print("before attention",hidden_states.shape,attention_mask.shape,encoder_hidden_states.shape if encoder_hidden_states is not None else "None")
                if write:
                    total_batch_size,nums_token,channel = hidden_states.shape
                    img_nums = total_batch_size // 2
                    hidden_states = hidden_states.reshape(-1,img_nums,nums_token,channel)
                    encoder_arr = [hidden_states[:,img_ind,indices[img_ind],:].reshape(2,-1,channel) for img_ind in range(img_nums)]
                    for img_ind in range(img_nums):
                        encoder_hidden_states_tmp = torch.cat(encoder_arr[0:img_ind] + encoder_arr[img_ind+1:] + [hidden_states[:,img_ind,:,:]],dim=1)
                        hidden_states[:,img_ind,:,:] = self.__call2__(attn, hidden_states[:,img_ind,:,:],encoder_hidden_states_tmp,None,temb)
                else:
                    _,nums_token,channel = hidden_states.shape
                    # img_nums = total_batch_size // 2
                    # encoder_hidden_states = encoder_hidden_states.reshape(-1,img_nums,nums_token,channel)
                    hidden_states = hidden_states.reshape(2,-1,nums_token,channel)
                    # print(len(indices))
                    # encoder_arr = [encoder_hidden_states[:,img_ind,indices[img_ind],:].reshape(2,-1,channel) for img_ind in range(img_nums)]
                    encoder_hidden_states_tmp = torch.cat(encoder_arr+[hidden_states[:,0,:,:]],dim=1)
                    hidden_states[:,0,:,:] = self.__call2__(attn, hidden_states[:,0,:,:],encoder_hidden_states_tmp,None,temb)
                hidden_states = hidden_states.reshape(-1,nums_token,channel)
            else:
                hidden_states = self.__call2__(attn, hidden_states,None,attention_mask,temb)
        attn_count +=1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            indices1024,indices4096 = cal_attn_indice_xl_effcient_memory(total_length,id_length,sa32,sa64,height,width, dtype= self.dtype)

        return hidden_states
    def __call1__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        attn_indices = None,
    ):
        # print("hidden state shape",hidden_states.shape,self.id_length)
        residual = hidden_states
        # if encoder_hidden_states is not None:
        #     raise Exception("not implement")
        # if attn.spatial_norm is not None:
        #     hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total_batch_size, channel, height * width).transpose(1, 2)
        total_batch_size,nums_token,channel = hidden_states.shape
        img_nums = total_batch_size//2
        hidden_states = hidden_states.view(-1,img_nums,nums_token,channel).reshape(-1,img_nums * nums_token,channel)
        batch_size, sequence_length, _ = hidden_states.shape

        # if attn.group_norm is not None:
        #     hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states   # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1,id_length+1,nums_token,channel).reshape(-1,(id_length+1) * nums_token,channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)


        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # print(key.shape,value.shape,query.shape,attention_mask.shape)
        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        #print(query.shape,key.shape,value.shape,attention_mask.shape)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(total_batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)



        # # linear proj
        # hidden_states = attn.to_out[0](hidden_states)
        # # dropout
        # hidden_states = attn.to_out[1](hidden_states)

        hidden_states = attn.to_out(hidden_states)

        # if input_ndim == 4:
        #     tile_hidden_states = tile_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # if attn.residual_connection:
        #     tile_hidden_states = tile_hidden_states + residual

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(total_batch_size, channel, height, width)
        # if attn.residual_connection:
        #     hidden_states = hidden_states + residual
        # hidden_states = hidden_states / attn.rescale_output_factor
        # print(hidden_states.shape)
        return hidden_states
    
    def __call2__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None):
        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, channel = (
            hidden_states.shape
        )
        # print(hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        # else:
        #     encoder_hidden_states = encoder_hidden_states.view(-1,self.id_length+1,sequence_length,channel).reshape(-1,(self.id_length+1) * sequence_length,channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # # linear proj
        # hidden_states = attn.to_out[0](hidden_states)
        # # dropout
        # hidden_states = attn.to_out[1](hidden_states)
        hidden_states = attn.to_out(hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # if attn.residual_connection:
        #     hidden_states = hidden_states + residual

        # hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states