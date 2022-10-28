import torch
from torch import nn

import matplotlib.pyplot as plt

from mingpt import Transformer

if __name__ == '__main__':

	seq_len = 100
	n_minibatch = 64

	transformer = Transformer(
		input_size=1,
		n_embd=10,
		output_size=1,
		n_head=10,
		n_layer=3,
		attn_pdrop=0.1,
		resid_pdrop=0.1,
		embd_pdrop=0.1,
		block_size=seq_len,
	)

	x = torch.randn(5, 20, 1)

	with torch.no_grad():
		y = transformer(x)

	print(x.shape)
	print(y.shape)

	optimizer = transformer.configure_optimizers(
		weight_decay=0.0001,
		learning_rate=0.001,
		betas=(0.9, 0.999),
	)

	t_ = torch.arange(0, seq_len * n_minibatch + 1) / 20
	print(t_.size())
	long_sequence = torch.sin(t_)

	data_x = long_sequence[:seq_len * n_minibatch].detach().reshape(n_minibatch, seq_len, 1)
	data_y = long_sequence[1:].detach().reshape(n_minibatch, seq_len, 1)

	loss_mse = nn.MSELoss()

	plt.figure()

	plt.pause(0.1)

	for n in range(1000):
		data_x_ = data_x + 0.03 * torch.randn_like(data_x)

		y_pred = transformer(data_x_)

		loss_ = loss_mse(y_pred, data_y)

		transformer.zero_grad(set_to_none=True)
		loss_.backward()
		torch.nn.utils.clip_grad_norm_(transformer.parameters(), 0.5)
		optimizer.step()

		print(f"{n}: {loss_.item()}")

		plt.clf()
		x_plot = data_x_.detach().flatten()[:2000]
		y_plot = y_pred.detach().flatten()[:2000]
		if n % 100 == 0:
			gen_y = transformer.generate(data_x[0][None], 2000).flatten()

		plt.plot(x_plot, "r", alpha=0.5)
		plt.plot(y_plot, "b", alpha=0.5)
		plt.plot(gen_y, "g", alpha=0.5)
		plt.legend(["target", "one-step prediction", "multi-step prediction"])
		plt.tight_layout()
		plt.pause(0.0001)

	plt.plot(x_plot, "r", alpha=0.5)
	plt.plot(y_plot, "b", alpha=0.5)
	plt.plot(gen_y, "g", alpha=0.5)
	plt.legend(["target", "one-step prediction", "multi-step prediction"])
	plt.tight_layout()
	plt.show()
