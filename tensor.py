import torch
# print(torch.__version__)

# Tensors - Smart box of numbers which can be trained and works on gpu

# x = torch.tensor(2.0)
# w = torch.tensor(3.0, requires_grad = True)

# y_true = torch.tensor(10.0)
# lr = 0.1

# y_pred = w*x
# loss = (y_pred-y_true) ** 2

# loss.backward()

# print("Before update")
# print("w:" , w.item())
# print("loss:" , loss.item())
# print("gradient:", w.grad.item())

# with torch.no_grad(): # no gradients will be calculated in this block
#     w -= lr * w.grad

# w.grad.zero_() # reset the gradient to zero after updating the weights

# print("After update")
# print("w:" , w.item())

