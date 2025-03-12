import torch.nn as nn
import torch.quantization as tq


class QATModel(nn.Module):
    def __init__(self, model_fp32):
        super(QATModel, self).__init__()
        self.model = model_fp32
        
        # Convert model to quantization aware training mode
        self.model.train()
        self.model.fuse_model()  # Fuse conv, bn, relu layers where possible
        self.qconfig = tq.get_default_qat_qconfig("fbgemm")
        self.model.qconfig = self.qconfig
        tq.prepare_qat(self.model, inplace=True)
    
    def forward(self, x):
        return self.model(x)

    @staticmethod
    def convert_fully_quantized(model_qat):
        model_qat.cpu()
        tq.convert(model_qat, inplace=True)

