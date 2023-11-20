from emg2qwerty.modules import TDSConvEncoder
import torch

def test_TDSConvEncoder():
    num_features = 48
    net = TDSConvEncoder(num_features=num_features)
    batch = torch.randn(8000, 16, num_features)
    feats = net(batch)
    loss = feats.sum()
    loss.backward()
