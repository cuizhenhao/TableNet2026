import numpy as np
import torch
import torch.nn.functional as F

from modules.cct.white_cat2d import WhiteCat2D
from modules.cct.white_conv_embedding import WhiteConvEmbedding
from modules.cct.white_layer_norm import WhiteLayerNorm
from modules.cct.white_mat_mul import WhiteMatMul
from modules.cct.white_mat_mul2 import WhiteMatMulPro
from modules.cct.white_nonzero import NoneZeroActivationFunction
from modules.cct.white_parameter_add import WhiteParameterAdd
from modules.cct.white_softmax import WhiteSoftmax
from modules.cct.white_transpose import WhiteTranspose
from modules.white_max_pooling import WhiteMaxPool
from modules.white_res_block import WhiteAdd
from modules.white_linear import WhiteLinear
from modules.white_net import WhiteNet
from modules.white_sequential import WhiteSequential

class Tokenizer(WhiteNet):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False):
        super(Tokenizer, self).__init__()

        self.emb_conv = WhiteConvEmbedding(
            10001, 500, 50,
            1, 64,
            kernel_size=(1, 50),
            stride=stride,
            padding=0, bias=False,
            activate_function=F.relu
        )
        # self.conv = WhiteConv2(1, 64,
        #           kernel_size=1,
        #           stride=stride,
        #           padding=0, bias=False,
        #            kernel_size_2d=(1, 50)
        #            )
        # self.act = WhiteActivation(64,activate_function=activation)
        self.pool = WhiteMaxPool(64,
                                 pool_size=(10, 1),
                                 stride=(10, 1),
                                 padding=0)
        self.transpose = WhiteTranspose(64, 50, -2, -1)
        # self.flattener = nn.Flatten(2, 3)

    def forward(self, x, delta=None, mode=None):
        # x = x.unsqueeze(1)
        # x = self.conv(x)
        if mode != 'skip':
            x = self.emb_conv(x)
        if delta is not None:
            x = x + delta[0]
        # x = self.act(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=2, end_dim=3)
        x = self.transpose(x)
        # x = x.transpose(-2, -1)
        return x


class WhiteAttention(WhiteNet):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self):
        super().__init__()
        self.scale = 0.125
        self.nonzero = NoneZeroActivationFunction(64, 0, -0)
        self.q = WhiteLinear(64, 32, bias=False, merge_last_codebook_flag=True)
        self.k = WhiteLinear(64, 32, bias=False, merge_last_codebook_flag=True)
        self.v = WhiteLinear(64, 32, bias=False, merge_last_codebook_flag=True)
        self.transpose = WhiteTranspose(32, 32, -2, -1)
        self.matmul = WhiteMatMul(50, 50, 32, self.scale)
        self.softmax = WhiteSoftmax(50, 50, dim=-1)
        # self.softmax = WhiteActivation(64, activate_function=F.softmax, dim=-1)

        self.matmul2 = WhiteMatMulPro(50, 32, 50, 1)

        # self.proj = WhiteLinear(dim, dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        # q = self.nonzero(q)
        # k = self.nonzero(k)
        # v = self.nonzero(v)
        # k = self.transpose(k)
        k = k.transpose(-2, -1)
        attn = self.matmul((q, k))
        # attn = (q @ k) * self.scale
        attn = self.softmax(attn)
        # attn = self.nonzero(attn)
        # attn = attn.softmax(dim=-1)

        # x = (attn @ v
        attn = self.matmul2((attn, v))
        return attn

    def init_info(self, layer_name, previous):
        network = []
        network.extend(self.q.init_info(layer_name + ".q", previous=previous))
        network.extend(self.k.init_info(layer_name + ".k", previous=previous))
        network.extend(self.v.init_info(layer_name + ".v", previous=previous))
        # network.extend(self.transpose.init_info(layer_name + ".transpose",previous = self.k))
        network.extend(self.matmul.init_info(layer_name + ".matmul", previous=(self.q, self.k)))

        network.extend(self.softmax.init_info(layer_name + ".softmax", previous=self.matmul))
        network.extend(self.matmul2.init_info(layer_name + ".matmul2", previous=(self.softmax, self.v)))

        return network


class WhiteMultiHeadAttention(WhiteNet):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim):
        super().__init__()
        self.head1 = WhiteAttention()
        self.head2 = WhiteAttention()
        self.concat = WhiteCat2D(32, 32)
        self.proj = WhiteLinear(dim, dim, merge_last_codebook_flag=True)

    def forward(self, x):
        x1 = self.head1(x)
        x2 = self.head2(x)
        x = self.concat((x1, x2))
        x = self.proj(x)
        return x

    def init_info(self, layer_name, previous):
        network = []
        sub1 = self.head1.init_info(layer_name + ".head1", previous=previous)
        network.extend(sub1)

        sub2 = self.head2.init_info(layer_name + ".head2", previous=previous)
        network.extend(sub2)
        network.extend(self.concat.init_info(layer_name + ".concat", previous=(sub1[-1], sub2[-1])))
        network.extend(self.proj.init_info(layer_name + ".proj", previous=network[-1]))
        return network


class Attention(WhiteNet):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = WhiteLinear(dim, dim * 3, bias=False)

        self.proj = WhiteLinear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k = k.transpose(-2, -1)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerEncoderLayer(WhiteNet):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = WhiteLayerNorm(64, d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead)
        self.self_attention = WhiteMultiHeadAttention(dim=d_model)

        self.add1 = WhiteAdd(50, merge_last_codebook=True)
        self.norm1 = WhiteLayerNorm(64, d_model)
        self.linear1 = WhiteLinear(d_model, dim_feedforward, merge_last_codebook_flag=True, activate_function=F.gelu)
        # self.activation = WhiteActivation(50, activate_function=F.gelu)
        self.linear2 = WhiteLinear(dim_feedforward, d_model, merge_last_codebook_flag=True)
        self.add2 = WhiteAdd(50, merge_last_codebook=True)

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # x = self.pre_norm(src)
        # x = self.self_attn(x)
        x = self.self_attention(src)
        src = self.add1((src, x))
        # src = self.norm1(src)
        src2 = self.linear1(src)
        # src2 = self.activation(src2)
        src2 = self.linear2(src2)
        src = self.add2((src, src2))
        return src

    def init_info(self, layer_name, previous):
        network = []
        # network.extend(self.pre_norm.init_info(layer_name + ".pre_norm", previous=previous))
        network.extend(self.self_attention.init_info(layer_name + ".self_attention", previous=previous))
        self.add1.init_info(layer_name + ".add1", previous=(previous, network[-1]))
        # self.norm1.init_info(layer_name + ".norm1",previous= self.add1)
        self.linear1.init_info(layer_name + ".linear1", previous=self.add1)
        # self.activation.init_info(layer_name + ".activation",previous = self.linear1)

        self.linear2.init_info(layer_name + ".linear2", previous=self.linear1)
        self.add2.init_info(layer_name + ".add2", previous=(self.add1, self.linear2))

        network.extend([self.add1, self.linear1, self.linear2, self.add2])
        return network


class TransformerClassifier(WhiteNet):
    def __init__(self,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 sequence_length=None):
        super().__init__()
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.positional_emb = WhiteParameterAdd(64, 50)
        self.embedding_dim = embedding_dim
        self.num_tokens = 0
        self.blocks = WhiteSequential(*[
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward)
            for i in range(num_layers)])
        self.norm = WhiteLayerNorm(64, embedding_dim)

        self.attention_pool = WhiteLinear(self.embedding_dim, 1, merge_last_codebook_flag=True)
        self.softmax = WhiteSoftmax(1, 50, dim=-1)
        self.matmul = WhiteMatMulPro(1, 64, 50, 1)
        self.fc = WhiteLinear(embedding_dim, num_classes, is_last_layer=True)

    def forward(self, x):
        x = self.positional_emb(x)
        for blk in self.blocks:
            x = blk(x)
        # x = self.norm(x)
        x1 = self.attention_pool(x)
        x1 = x1.transpose(-1, -2)
        # x1 = F.softmax(x1, dim=-1)
        x1 = self.softmax(x1)
        x = self.matmul((x1, x)).squeeze(-2)

        x = self.fc(x)
        return x

    def init_info(self, layer_name, previous):
        network = []
        network.extend(self.positional_emb.init_info(layer_name + ".positional_emb", previous=previous))
        network.extend(self.blocks.init_info(layer_name + ".blocks", previous=network[-1]))
        last_blk = network[-1]
        network.extend(self.attention_pool.init_info(layer_name + ".attention_pool", previous=network[-1]))
        network.extend(self.softmax.init_info(layer_name + ".softmax", previous=network[-1]))
        network.extend(self.matmul.init_info(layer_name + ".matmul", previous=(self.softmax, last_blk)))

        network.extend(self.fc.init_info(layer_name + ".fc", previous=network[-1]))
        return network


class CCT_IMDB(WhiteNet):
    def __init__(self,
                 seq_len=64,
                 word_embedding_dim=50,
                 embedding_dim=64,
                 kernel_size=2,
                 stride=1,
                 padding=1,
                 pooling_kernel_size=2,
                 pooling_stride=2,
                 pooling_padding=1,
                 *args, **kwargs):
        super(CCT_IMDB, self).__init__()
        self.last_codebook = np.zeros(500, dtype=object)
        for i in range(500):
            self.last_codebook[i] = (10001, np.arange(0, 10001))
        # self.embedder = WhiteEmbedding(10001,500, word_embedding_dim)

        self.tokenizer = Tokenizer(n_input_channels=word_embedding_dim,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=F.relu,
                                   n_conv_layers=1,
                                   conv_bias=False)

        self.classifier = TransformerClassifier(
            sequence_length=34,
            embedding_dim=embedding_dim,
            num_layers=1,
            num_heads=2,
            mlp_ratio=0.5,
            num_classes=1,
        )

    def forward(self, x, delta=None, mode=None):
        # x = self.embedder(x)
        x = self.tokenizer(x, delta=delta, mode=mode)
        return self.classifier(x)

def cct_imdb(*args, **kwargs):
    model = CCT_IMDB(*args, **kwargs)

    return model
