import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class NewGELU(nn.Module):
	def forward(self, x):
		return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):

	def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, block_size):
		super().__init__()
		assert n_embd % n_head == 0

		self.c_attn = nn.Linear(n_embd, 3 * n_embd)

		self.c_proj = nn.Linear(n_embd, n_embd)

		self.attn_dropout = nn.Dropout(attn_pdrop)
		self.resid_dropout = nn.Dropout(resid_pdrop)

		self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
		                     .view(1, 1, block_size, block_size))
		self.n_head = n_head
		self.n_embd = n_embd

	def forward(self, x):
		B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

		# calculate query, key, values for all heads in batch and move head forward to be the batch dim
		q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
		k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
		q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
		v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

		# causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
		att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

		att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
		# print(self.bias[:, :, :T, :T] == 0)

		att = F.softmax(att, dim=-1)

		att = self.attn_dropout(att)

		y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

		y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

		# output projection
		y = self.resid_dropout(self.c_proj(y))
		return y


class Block(nn.Module):

	def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, block_size):
		super().__init__()
		self.ln_1 = nn.LayerNorm(n_embd)

		self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, block_size)

		self.ln_2 = nn.LayerNorm(n_embd)

		self.mlp = nn.ModuleDict(dict(
			c_fc=nn.Linear(n_embd, 4 * n_embd),
			c_proj=nn.Linear(4 * n_embd, n_embd),
			act=NewGELU(),
			dropout=nn.Dropout(resid_pdrop),
		))

		m = self.mlp

		self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

	def forward(self, x):
		x = x + self.attn(self.ln_1(x))

		x = x + self.mlpf(self.ln_2(x))

		return x


class Transformer(nn.Module):
	def __init__(
			self,
			input_size,
			n_embd,
			output_size,
			n_head,
			n_layer,
			attn_pdrop,
			resid_pdrop,
			embd_pdrop,
			block_size
	):
		super(Transformer, self).__init__()

		self.input_size = input_size
		self.n_embd = n_embd
		self.block_size = block_size

		self.transformer = nn.ModuleDict(dict(
			ite=nn.Linear(input_size, n_embd),
			drop=nn.Dropout(embd_pdrop),
			h=nn.ModuleList([Block(n_embd, n_head, attn_pdrop, resid_pdrop, block_size) for _ in range(n_layer)]),
			ln_f=nn.LayerNorm(n_embd),
		))
		self.lm_head = nn.Linear(n_embd, output_size, bias=False)

		self.apply(self._init_weights)
		for pn, p in self.named_parameters():
			if pn.endswith('c_proj.weight'):
				torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

	def _init_weights(self, module):
		# if isinstance(module, nn.Linear):
		# 	torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
		# 	if module.bias is not None:
		# 		torch.nn.init.zeros_(module.bias)
		# elif isinstance(module, nn.Embedding):
		# 	torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
		# elif isinstance(module, nn.LayerNorm):
		# 	torch.nn.init.zeros_(module.bias)
		# 	torch.nn.init.ones_(module.weight)
		if isinstance(module, nn.Linear):
			torch.nn.init.orthogonal_(module.weight, gain=1)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.orthogonal_(module.weight, gain=1)
		elif isinstance(module, nn.LayerNorm):
			torch.nn.init.zeros_(module.bias)
			torch.nn.init.ones_(module.weight)

	def configure_optimizers(self, weight_decay, learning_rate, betas):
		decay = set()
		no_decay = set()
		whitelist_weight_modules = (torch.nn.Linear,)
		blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

		for mn, m in self.named_modules():
			for pn, p in m.named_parameters():
				fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

				if pn.endswith('bias'):
					# all biases will not be decayed
					no_decay.add(fpn)

				elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
					# weights of whitelist modules will be weight decayed
					decay.add(fpn)

				elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
					# weights of blacklist modules will NOT be weight decayed
					no_decay.add(fpn)

		# validate that we considered every parameter
		param_dict = {pn: p for pn, p in self.named_parameters()}
		inter_params = decay & no_decay
		union_params = decay | no_decay
		assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
		assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
		                                                   % (str(param_dict.keys() - union_params),)

		# create the pytorch optimizer object
		optim_groups = [
			{"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
			{"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
		]
		optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
		return optimizer

	def forward(self, x):
		B, T, C = x.shape

		assert C == self.input_size

		x = self.transformer.ite(x)

		x = self.transformer.drop(x)

		for block in self.transformer.h:
			x = block(x)

		x = self.transformer.ln_f(x)

		outputs = self.lm_head(x)

		return outputs

	def generate(self, x, length):
		B, T, C = x.shape

		assert B == 1
		assert length > T

		with torch.no_grad():
			seq = x.detach()

			for _ in tqdm(range(length - seq.size(1) + 1)):
				seq_cond = seq if seq.size(1) <= self.block_size else seq[:, -self.block_size:]

				seq_next = self(seq_cond)
				seq_next = torch.atleast_3d(seq_next[:, -1, :])

				seq = torch.cat((seq, seq_next), dim=1)

		return seq
