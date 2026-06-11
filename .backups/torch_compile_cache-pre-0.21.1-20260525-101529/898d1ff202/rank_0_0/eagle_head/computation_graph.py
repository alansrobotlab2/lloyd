from __future__ import annotations
import torch
class GraphModule(torch.nn.Module):
    def forward(self, s59: "Sym(s18)", L_inputs_embeds_: "bf16[s18, 3072]", L_hidden_states_: "bf16[s18, 3072]", L_self_modules_model_modules_pre_fc_norm_embedding_parameters_weight_: "bf16[3072]", L_self_modules_model_modules_pre_fc_norm_hidden_parameters_weight_: "bf16[3072]", L_self_modules_model_modules_fc_parameters_weight_: "bf16[3072, 6144]", L_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_: "bf16[3072]", L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_weight_: "bf16[17408, 3072]", L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_: "bf16[256]", L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_: "bf16[256]", s18: "Sym(s18)", s7: "Sym(s7)", L_positions_: "i64[3, s18]", L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_: "bf16[1048576, 64]", SYNTHETIC_LOCAL_tmp_0_ : vllm_utils_torch_utils_LayerName, L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_: "bf16[3072, 8192]", L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_: "bf16[3072]", SYNTHETIC_LOCAL_tmp_2_ : vllm_utils_torch_utils_LayerName, L_self_modules_model_modules_norm_parameters_weight_: "bf16[3072]"):
        l_inputs_embeds_ = L_inputs_embeds_
        l_hidden_states_ = L_hidden_states_
        l_self_modules_model_modules_pre_fc_norm_embedding_parameters_weight_ = L_self_modules_model_modules_pre_fc_norm_embedding_parameters_weight_
        l_self_modules_model_modules_pre_fc_norm_hidden_parameters_weight_ = L_self_modules_model_modules_pre_fc_norm_hidden_parameters_weight_
        l_self_modules_model_modules_fc_parameters_weight_ = L_self_modules_model_modules_fc_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_
        l_positions_ = L_positions_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_
        synthetic_local_tmp_0_ = SYNTHETIC_LOCAL_tmp_0_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
        synthetic_local_tmp_2_ = SYNTHETIC_LOCAL_tmp_2_
        l_self_modules_model_modules_norm_parameters_weight_ = L_self_modules_model_modules_norm_parameters_weight_

        # No stacktrace found for following nodes
        submod_0 = self.submod_0(l_self_modules_model_modules_pre_fc_norm_embedding_parameters_weight_, l_inputs_embeds_, s59, l_self_modules_model_modules_pre_fc_norm_hidden_parameters_weight_, l_hidden_states_, l_self_modules_model_modules_fc_parameters_weight_, l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_, l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_weight_, s18, l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_, l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_, l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_, l_positions_, s7);  l_self_modules_model_modules_pre_fc_norm_embedding_parameters_weight_ = l_inputs_embeds_ = l_self_modules_model_modules_pre_fc_norm_hidden_parameters_weight_ = l_hidden_states_ = l_self_modules_model_modules_fc_parameters_weight_ = l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_weight_ = s18 = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_ = l_positions_ = s7 = None
        getitem = submod_0[0]
        getitem_1 = submod_0[1]
        getitem_2 = submod_0[2]
        getitem_3 = submod_0[3]
        getitem_4 = submod_0[4]
        getitem_5 = submod_0[5]
        getitem_6 = submod_0[6];  submod_0 = None
        submod_1 = self.submod_1(getitem, s59, getitem_1, synthetic_local_tmp_0_, getitem_2, getitem_3);  getitem = getitem_1 = synthetic_local_tmp_0_ = getitem_2 = submod_1 = None
        submod_2 = self.submod_2(getitem_3, s59, getitem_4, l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_, getitem_5, l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_, getitem_6, synthetic_local_tmp_2_, l_self_modules_model_modules_norm_parameters_weight_);  getitem_3 = s59 = getitem_4 = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = getitem_5 = l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = getitem_6 = synthetic_local_tmp_2_ = l_self_modules_model_modules_norm_parameters_weight_ = None
        return (submod_2,)

    class submod_0(torch.nn.Module):
        def forward(self, l_self_modules_model_modules_pre_fc_norm_embedding_parameters_weight_: "bf16[3072]", l_inputs_embeds_: "bf16[s18, 3072]", s59: "Sym(s18)", l_self_modules_model_modules_pre_fc_norm_hidden_parameters_weight_: "bf16[3072]", l_hidden_states_: "bf16[s18, 3072]", l_self_modules_model_modules_fc_parameters_weight_: "bf16[3072, 6144]", l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_: "bf16[3072]", l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_weight_: "bf16[17408, 3072]", s18: "Sym(s18)", l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_: "bf16[256]", l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_: "bf16[256]", l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_: "bf16[1048576, 64]", l_positions_: "i64[3, s18]", s7: "Sym(s7)"):
            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py:159 in forward_native, code: weight = self.weight.data.float() + 1.0
            _get_data_attr: "bf16[3072]" = torch._C._autograd._get_data_attr(l_self_modules_model_modules_pre_fc_norm_embedding_parameters_weight_);  l_self_modules_model_modules_pre_fc_norm_embedding_parameters_weight_ = None
            float_1: "f32[3072]" = _get_data_attr.float();  _get_data_attr = None
            add: "f32[3072]" = float_1 + 1.0;  float_1 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/ir/op.py:324 in __call__, code: return self.torch_op(*args, **kwargs)
            rms_norm_default: "bf16[s18, 3072]" = torch.ops.vllm_ir.rms_norm.default(l_inputs_embeds_, add, 1e-06);  l_inputs_embeds_ = add = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py:170 in forward_native, code: out.to(orig_dtype) if residual is None else (out.to(orig_dtype), residual)
            to: "bf16[s18, 3072]" = rms_norm_default.to(torch.bfloat16);  rms_norm_default = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py:159 in forward_native, code: weight = self.weight.data.float() + 1.0
            _get_data_attr_1: "bf16[3072]" = torch._C._autograd._get_data_attr(l_self_modules_model_modules_pre_fc_norm_hidden_parameters_weight_);  l_self_modules_model_modules_pre_fc_norm_hidden_parameters_weight_ = None
            float_2: "f32[3072]" = _get_data_attr_1.float();  _get_data_attr_1 = None
            add_1: "f32[3072]" = float_2 + 1.0;  float_2 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/ir/op.py:324 in __call__, code: return self.torch_op(*args, **kwargs)
            rms_norm_default_1: "bf16[s18, 3072]" = torch.ops.vllm_ir.rms_norm.default(l_hidden_states_, add_1, 1e-06);  l_hidden_states_ = add_1 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py:170 in forward_native, code: out.to(orig_dtype) if residual is None else (out.to(orig_dtype), residual)
            to_1: "bf16[s18, 3072]" = rms_norm_default_1.to(torch.bfloat16);  rms_norm_default_1 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_5_mtp.py:138 in forward, code: hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
            cat: "bf16[s18, 6144]" = torch.cat([to, to_1], dim = -1);  to = to_1 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/parameter.py:126 in __torch_function__, code: return super().__torch_function__(func, types, args, kwargs)
            linear: "bf16[s18, 3072]" = torch._C._nn.linear(cat, l_self_modules_model_modules_fc_parameters_weight_, None);  cat = l_self_modules_model_modules_fc_parameters_weight_ = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py:159 in forward_native, code: weight = self.weight.data.float() + 1.0
            _get_data_attr_2: "bf16[3072]" = torch._C._autograd._get_data_attr(l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_);  l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = None
            float_3: "f32[3072]" = _get_data_attr_2.float();  _get_data_attr_2 = None
            add_2: "f32[3072]" = float_3 + 1.0;  float_3 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/ir/op.py:324 in __call__, code: return self.torch_op(*args, **kwargs)
            rms_norm_default_2: "bf16[s18, 3072]" = torch.ops.vllm_ir.rms_norm.default(linear, add_2, 1e-06);  add_2 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py:170 in forward_native, code: out.to(orig_dtype) if residual is None else (out.to(orig_dtype), residual)
            to_2: "bf16[s18, 3072]" = rms_norm_default_2.to(torch.bfloat16);  rms_norm_default_2 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py:406 in forward, code: self_attention_output = torch.empty_like(hidden_states)
            empty_like: "bf16[s18, 3072]" = torch.empty_like(to_2)

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/parameter.py:126 in __torch_function__, code: return super().__torch_function__(func, types, args, kwargs)
            linear_1: "bf16[s18, 17408]" = torch._C._nn.linear(to_2, l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_weight_, None);  to_2 = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_weight_ = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py:285 in forward, code: q_gate, k, v = qkv.split(
            split = linear_1.split([16384, 512, 512], dim = -1);  linear_1 = None
            getitem: "bf16[s18, 16384]" = split[0]
            getitem_1: "bf16[s18, 512]" = split[1]
            getitem_2: "bf16[s18, 512]" = split[2];  split = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py:289 in forward, code: q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            view: "bf16[s18, 32, 512]" = getitem.view(s18, 32, -1);  getitem = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py:290 in forward, code: q, gate = torch.chunk(q_gate, 2, dim=-1)
            chunk = torch.chunk(view, 2, dim = -1);  view = None
            getitem_3: "bf16[s18, 32, 256]" = chunk[0]
            getitem_4: "bf16[s18, 32, 256]" = chunk[1];  chunk = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py:291 in forward, code: q = q.reshape(*orig_shape, -1)
            reshape: "bf16[s18, 8192]" = getitem_3.reshape(s18, -1);  getitem_3 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py:292 in forward, code: gate = gate.reshape(*orig_shape, -1)
            reshape_1: "bf16[s18, 8192]" = getitem_4.reshape(s18, -1);  getitem_4 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py:296 in forward, code: q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            view_1: "bf16[s18, 32, 256]" = reshape.view(-1, 32, 256);  reshape = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py:159 in forward_native, code: weight = self.weight.data.float() + 1.0
            _get_data_attr_3: "bf16[256]" = torch._C._autograd._get_data_attr(l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_);  l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_ = None
            float_4: "f32[256]" = _get_data_attr_3.float();  _get_data_attr_3 = None
            add_3: "f32[256]" = float_4 + 1.0;  float_4 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/ir/op.py:324 in __call__, code: return self.torch_op(*args, **kwargs)
            rms_norm_default_3: "bf16[s18, 32, 256]" = torch.ops.vllm_ir.rms_norm.default(view_1, add_3, 1e-06);  view_1 = add_3 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py:170 in forward_native, code: out.to(orig_dtype) if residual is None else (out.to(orig_dtype), residual)
            to_3: "bf16[s18, 32, 256]" = rms_norm_default_3.to(torch.bfloat16);  rms_norm_default_3 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py:296 in forward, code: q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            view_2: "bf16[s18, 8192]" = to_3.view(-1, 8192);  to_3 = None

            # No stacktrace found for following nodes
            sym_size_int: "Sym(s18)" = torch.ops.aten.sym_size.int(view_2, 0)

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py:299 in forward, code: k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            view_3: "bf16[s18, 2, 256]" = getitem_1.view(-1, 2, 256);  getitem_1 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py:159 in forward_native, code: weight = self.weight.data.float() + 1.0
            _get_data_attr_4: "bf16[256]" = torch._C._autograd._get_data_attr(l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_);  l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_ = None
            float_5: "f32[256]" = _get_data_attr_4.float();  _get_data_attr_4 = None
            add_4: "f32[256]" = float_5 + 1.0;  float_5 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/ir/op.py:324 in __call__, code: return self.torch_op(*args, **kwargs)
            rms_norm_default_4: "bf16[s18, 2, 256]" = torch.ops.vllm_ir.rms_norm.default(view_3, add_4, 1e-06);  view_3 = add_4 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py:170 in forward_native, code: out.to(orig_dtype) if residual is None else (out.to(orig_dtype), residual)
            to_4: "bf16[s18, 2, 256]" = rms_norm_default_4.to(torch.bfloat16);  rms_norm_default_4 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py:299 in forward, code: k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            view_4: "bf16[s18, 512]" = to_4.view(-1, 512);  to_4 = None

            # No stacktrace found for following nodes
            sym_size_int_1: "Sym(s18)" = torch.ops.aten.sym_size.int(view_4, 0)

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/mrope.py:284 in forward_native, code: cos_sin = cos_sin_cache[positions]
            getitem_5: "bf16[3, s18, 64]" = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_[l_positions_];  l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_ = l_positions_ = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/mrope.py:285 in forward_native, code: cos, sin = cos_sin.chunk(2, dim=-1)
            chunk_1 = getitem_5.chunk(2, dim = -1);  getitem_5 = None
            getitem_6: "bf16[3, s18, 32]" = chunk_1[0]
            getitem_7: "bf16[3, s18, 32]" = chunk_1[1];  chunk_1 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/mrope.py:195 in apply_interleaved_rope, code: x_t = x[0].clone()
            getitem_8: "bf16[s18, 32]" = getitem_6[0]
            clone: "bf16[s18, 32]" = getitem_8.clone();  getitem_8 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/mrope.py:196 in apply_interleaved_rope, code: x_t[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]
            getitem_9: "bf16[s18, 11]" = getitem_6[(1, Ellipsis, slice(1, 33, 3))]
            clone[(Ellipsis, slice(1, 33, 3))] = getitem_9;  setitem = clone;  getitem_9 = setitem = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/mrope.py:197 in apply_interleaved_rope, code: x_t[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]
            getitem_10: "bf16[s18, 10]" = getitem_6[(2, Ellipsis, slice(2, 30, 3))];  getitem_6 = None
            clone[(Ellipsis, slice(2, 30, 3))] = getitem_10;  setitem_1 = clone;  getitem_10 = setitem_1 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/mrope.py:195 in apply_interleaved_rope, code: x_t = x[0].clone()
            getitem_11: "bf16[s18, 32]" = getitem_7[0]
            clone_1: "bf16[s18, 32]" = getitem_11.clone();  getitem_11 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/mrope.py:196 in apply_interleaved_rope, code: x_t[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]
            getitem_12: "bf16[s18, 11]" = getitem_7[(1, Ellipsis, slice(1, 33, 3))]
            clone_1[(Ellipsis, slice(1, 33, 3))] = getitem_12;  setitem_2 = clone_1;  getitem_12 = setitem_2 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/mrope.py:197 in apply_interleaved_rope, code: x_t[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]
            getitem_13: "bf16[s18, 10]" = getitem_7[(2, Ellipsis, slice(2, 30, 3))];  getitem_7 = None
            clone_1[(Ellipsis, slice(2, 30, 3))] = getitem_13;  setitem_3 = clone_1;  getitem_13 = setitem_3 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/mrope.py:302 in forward_native, code: query = query.view(num_tokens, -1, self.head_size)
            view_5: "bf16[s18, 32, 256]" = view_2.view(s18, -1, 256);  view_2 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/mrope.py:303 in forward_native, code: query_rot = query[..., : self.rotary_dim]
            getitem_14: "bf16[s18, 32, 64]" = view_5[(Ellipsis, slice(None, 64, None))]

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/mrope.py:304 in forward_native, code: query_pass = query[..., self.rotary_dim :]
            getitem_15: "bf16[s18, 32, 192]" = view_5[(Ellipsis, slice(64, None, None))];  view_5 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:163 in forward_static, code: cos = cos.unsqueeze(-2).to(x.dtype)
            unsqueeze: "bf16[s18, 1, 32]" = clone.unsqueeze(-2)
            to_5: "bf16[s18, 1, 32]" = unsqueeze.to(torch.bfloat16);  unsqueeze = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:164 in forward_static, code: sin = sin.unsqueeze(-2).to(x.dtype)
            unsqueeze_1: "bf16[s18, 1, 32]" = clone_1.unsqueeze(-2)
            to_6: "bf16[s18, 1, 32]" = unsqueeze_1.to(torch.bfloat16);  unsqueeze_1 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:167 in forward_static, code: x1, x2 = torch.chunk(x, 2, dim=-1)
            chunk_2 = torch.chunk(getitem_14, 2, dim = -1);  getitem_14 = None
            getitem_16: "bf16[s18, 32, 32]" = chunk_2[0]
            getitem_17: "bf16[s18, 32, 32]" = chunk_2[1];  chunk_2 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:172 in forward_static, code: o1 = x1 * cos - x2 * sin
            mul: "bf16[s18, 32, 32]" = getitem_16 * to_5
            mul_1: "bf16[s18, 32, 32]" = getitem_17 * to_6
            sub: "bf16[s18, 32, 32]" = mul - mul_1;  mul = mul_1 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:173 in forward_static, code: o2 = x2 * cos + x1 * sin
            mul_2: "bf16[s18, 32, 32]" = getitem_17 * to_5;  getitem_17 = to_5 = None
            mul_3: "bf16[s18, 32, 32]" = getitem_16 * to_6;  getitem_16 = to_6 = None
            add_5: "bf16[s18, 32, 32]" = mul_2 + mul_3;  mul_2 = mul_3 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:176 in forward_static, code: output = torch.cat((o1, o2), dim=-1)
            cat_1: "bf16[s18, 32, 64]" = torch.cat((sub, add_5), dim = -1);  sub = add_5 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/mrope.py:310 in forward_native, code: query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
            cat_2: "bf16[s18, 32, 256]" = torch.cat((cat_1, getitem_15), dim = -1);  cat_1 = getitem_15 = None
            reshape_2: "bf16[s18, 8192]" = cat_2.reshape(sym_size_int, 8192);  cat_2 = sym_size_int = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/mrope.py:313 in forward_native, code: key = key.view(num_tokens, -1, self.head_size)
            view_6: "bf16[s18, 2, 256]" = view_4.view(s18, -1, 256);  view_4 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/mrope.py:314 in forward_native, code: key_rot = key[..., : self.rotary_dim]
            getitem_18: "bf16[s18, 2, 64]" = view_6[(Ellipsis, slice(None, 64, None))]

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/mrope.py:315 in forward_native, code: key_pass = key[..., self.rotary_dim :]
            getitem_19: "bf16[s18, 2, 192]" = view_6[(Ellipsis, slice(64, None, None))];  view_6 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:163 in forward_static, code: cos = cos.unsqueeze(-2).to(x.dtype)
            unsqueeze_2: "bf16[s18, 1, 32]" = clone.unsqueeze(-2);  clone = None
            to_7: "bf16[s18, 1, 32]" = unsqueeze_2.to(torch.bfloat16);  unsqueeze_2 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:164 in forward_static, code: sin = sin.unsqueeze(-2).to(x.dtype)
            unsqueeze_3: "bf16[s18, 1, 32]" = clone_1.unsqueeze(-2);  clone_1 = None
            to_8: "bf16[s18, 1, 32]" = unsqueeze_3.to(torch.bfloat16);  unsqueeze_3 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:167 in forward_static, code: x1, x2 = torch.chunk(x, 2, dim=-1)
            chunk_3 = torch.chunk(getitem_18, 2, dim = -1);  getitem_18 = None
            getitem_20: "bf16[s18, 2, 32]" = chunk_3[0]
            getitem_21: "bf16[s18, 2, 32]" = chunk_3[1];  chunk_3 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:172 in forward_static, code: o1 = x1 * cos - x2 * sin
            mul_4: "bf16[s18, 2, 32]" = getitem_20 * to_7
            mul_5: "bf16[s18, 2, 32]" = getitem_21 * to_8
            sub_1: "bf16[s18, 2, 32]" = mul_4 - mul_5;  mul_4 = mul_5 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:173 in forward_static, code: o2 = x2 * cos + x1 * sin
            mul_6: "bf16[s18, 2, 32]" = getitem_21 * to_7;  getitem_21 = to_7 = None
            mul_7: "bf16[s18, 2, 32]" = getitem_20 * to_8;  getitem_20 = to_8 = None
            add_6: "bf16[s18, 2, 32]" = mul_6 + mul_7;  mul_6 = mul_7 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:176 in forward_static, code: output = torch.cat((o1, o2), dim=-1)
            cat_3: "bf16[s18, 2, 64]" = torch.cat((sub_1, add_6), dim = -1);  sub_1 = add_6 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/mrope.py:321 in forward_native, code: key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
            cat_4: "bf16[s18, 2, 256]" = torch.cat((cat_3, getitem_19), dim = -1);  cat_3 = getitem_19 = None
            reshape_3: "bf16[s18, 512]" = cat_4.reshape(sym_size_int_1, 512);  cat_4 = sym_size_int_1 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/attention/attention.py:450 in forward, code: output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
            size = torch.Size([s18, 8192]);  s18 = None
            empty: "bf16[s18, 8192]" = torch.empty(size, dtype = torch.bfloat16, device = device(type='cuda', index=0));  size = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/attention/attention.py:455 in forward, code: query = query.view(-1, self.num_heads, self.head_size)
            view_7: "bf16[s18, 32, 256]" = reshape_2.view(-1, 32, 256);  reshape_2 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/attention/attention.py:456 in forward, code: output = output.view(-1, self.num_heads, self.head_size_v)
            view_8: "bf16[s18, 32, 256]" = empty.view(-1, 32, 256);  empty = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/attention/attention.py:458 in forward, code: key = key.view(-1, self.num_kv_heads, self.head_size)
            view_9: "bf16[s18, 2, 256]" = reshape_3.view(-1, 2, 256);  reshape_3 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/attention/attention.py:460 in forward, code: value = value.view(-1, self.num_kv_heads, self.head_size_v)
            view_10: "bf16[s18, 2, 256]" = getitem_2.view(-1, 2, 256);  getitem_2 = None
            return (view_9, view_10, view_7, view_8, reshape_1, empty_like, linear)

    class submod_1(torch.nn.Module):
        def forward(self, key_2: "bf16[s18, 2, 256]", s59: "Sym(s18)", value: "bf16[s18, 2, 256]", synthetic_local_tmp_0_ : vllm_utils_torch_utils_LayerName, query_2: "bf16[s18, 32, 256]", output_3: "bf16[s18, 32, 256]"):
            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/attention/attention.py:490 in forward, code: kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
            unified_kv_cache_update: "bf16[0]" = torch.ops.vllm.unified_kv_cache_update(key_2, value, synthetic_local_tmp_0_)

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/attention/attention.py:493 in forward, code: torch.ops.vllm.unified_attention_with_output(
            unified_attention_with_output = torch.ops.vllm.unified_attention_with_output(query_2, key_2, value, output_3, synthetic_local_tmp_0_, kv_cache_dummy_dep = unified_kv_cache_update);  query_2 = key_2 = value = output_3 = synthetic_local_tmp_0_ = unified_kv_cache_update = unified_attention_with_output = None
            return ()

    class submod_2(torch.nn.Module):
        def forward(self, output_3: "bf16[s18, 32, 256]", s59: "Sym(s18)", gate_1: "bf16[s18, 8192]", l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_: "bf16[3072, 8192]", self_attention_output: "bf16[s18, 3072]", l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_: "bf16[3072]", output_parallel: "bf16[s18, 3072]", synthetic_local_tmp_2_ : vllm_utils_torch_utils_LayerName, l_self_modules_model_modules_norm_parameters_weight_: "bf16[3072]"):
            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/attention/attention.py:501 in forward, code: return output.view(-1, hidden_size)
            view: "bf16[s18, 8192]" = output_3.view(-1, 8192);  output_3 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py:308 in forward, code: gate = torch.sigmoid(gate)
            sigmoid: "bf16[s18, 8192]" = torch.sigmoid(gate_1);  gate_1 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py:309 in forward, code: attn_output = attn_output * gate
            mul: "bf16[s18, 8192]" = view * sigmoid;  view = sigmoid = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/parameter.py:126 in __torch_function__, code: return super().__torch_function__(func, types, args, kwargs)
            linear: "bf16[s18, 3072]" = torch._C._nn.linear(mul, l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_, None);  mul = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py:311 in forward, code: output[:], _ = self.o_proj(attn_output)
            self_attention_output[slice(None, None, None)] = linear;  setitem = self_attention_output;  linear = setitem = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py:159 in forward_native, code: weight = self.weight.data.float() + 1.0
            _get_data_attr: "bf16[3072]" = torch._C._autograd._get_data_attr(l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_);  l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = None
            float_1: "f32[3072]" = _get_data_attr.float();  _get_data_attr = None
            add: "f32[3072]" = float_1 + 1.0;  float_1 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py:164 in forward_native, code: else x + residual
            add_1: "bf16[s18, 3072]" = self_attention_output + output_parallel;  self_attention_output = output_parallel = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/ir/op.py:324 in __call__, code: return self.torch_op(*args, **kwargs)
            rms_norm_default: "bf16[s18, 3072]" = torch.ops.vllm_ir.rms_norm.default(add_1, add, 1e-06);  add = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py:170 in forward_native, code: out.to(orig_dtype) if residual is None else (out.to(orig_dtype), residual)
            to: "bf16[s18, 3072]" = rms_norm_default.to(torch.bfloat16);  rms_norm_default = None

            # No stacktrace found for following nodes
            sym_size_int: "Sym(s18)" = torch.ops.aten.sym_size.int(to, 0)

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py:171 in forward, code: hidden_states = hidden_states.view(-1, hidden_dim)
            view_1: "bf16[s18, 3072]" = to.view(-1, 3072);  to = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:574 in forward, code: result = self._forward_entry(
            moe_forward_shared = torch.ops.vllm.moe_forward_shared(view_1, view_1, view_1, None, synthetic_local_tmp_2_);  view_1 = synthetic_local_tmp_2_ = None
            getitem: "bf16[s18, 3072]" = moe_forward_shared[0]
            getitem_1: "bf16[s18, 3072]" = moe_forward_shared[1];  moe_forward_shared = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:610 in forward, code: result = shared_output + fused_output
            add_2: "bf16[s18, 3072]" = getitem + getitem_1;  getitem = getitem_1 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:379 in _maybe_reduce_final_output, code: return states[..., :trunc_size]
            getitem_2: "bf16[s18, 3072]" = add_2[(Ellipsis, slice(None, 3072, None))];  add_2 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py:194 in forward, code: return final_hidden_states.view(orig_shape)
            view_2: "bf16[s18, 3072]" = getitem_2.view(sym_size_int, 3072);  getitem_2 = sym_size_int = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py:159 in forward_native, code: weight = self.weight.data.float() + 1.0
            _get_data_attr_1: "bf16[3072]" = torch._C._autograd._get_data_attr(l_self_modules_model_modules_norm_parameters_weight_);  l_self_modules_model_modules_norm_parameters_weight_ = None
            float_2: "f32[3072]" = _get_data_attr_1.float();  _get_data_attr_1 = None
            add_3: "f32[3072]" = float_2 + 1.0;  float_2 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py:164 in forward_native, code: else x + residual
            add_4: "bf16[s18, 3072]" = view_2 + add_1;  view_2 = add_1 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/ir/op.py:324 in __call__, code: return self.torch_op(*args, **kwargs)
            rms_norm_default_1: "bf16[s18, 3072]" = torch.ops.vllm_ir.rms_norm.default(add_4, add_3, 1e-06);  add_4 = add_3 = None

            # File: /home/alansrobotlab/lloyd/.venvs/vllm-experimental/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py:170 in forward_native, code: out.to(orig_dtype) if residual is None else (out.to(orig_dtype), residual)
            to_1: "bf16[s18, 3072]" = rms_norm_default_1.to(torch.bfloat16);  rms_norm_default_1 = None
            return to_1
