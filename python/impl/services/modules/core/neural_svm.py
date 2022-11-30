from .model_resources import *

def train_SVM(X, Y, ins, outs, epochs):

    model = nn.Sequential(nn.Linear(ins, outs), nn.Linear(outs, outs), nn.Sigmoid())
    model.to(device)

    N = len(Y)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)
 
    model.train()
    for epoch in range(epochs):
        sum_loss = 0

        for xi, y in zip(X, Y):
            empty_mem_cache()
            with torch.no_grad():
                x = xi * 1
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)#.squeeze()

            loss = F.cross_entropy(output, y)
            # loss = torch.mean(torch.clamp(1 - y * output, min=0))
            loss.backward()
            optimizer.step()

            sum_loss += float(loss)

    return model

def eval_SVM(X, Y, model):

    model.eval()

    pred = model(X)
    Y = Y.to(device)
    loss = F.cross_entropy(pred, Y, reduction='mean')

    return loss

