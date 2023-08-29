import torch

from os.path import join, realpath
from sys import path
root = realpath(join(__file__, '..', '..'))
path.append(root)
print(root)
from attn.attention import Attention, MultiheadAttention, MultiQueryAttention, GroupedQueryAttention
del join, realpath, root


def test_forward_Attention():
    attention = Attention(word_size=512, embed_dim=64)

    # Create embedding of 3 word
    word1 = torch.randn(1, 512)  # Embedding of 1st word
    word2 = torch.randn(1, 512)  # Embedding of 2nd word
    word3 = torch.randn(1, 512)  # Embedding of 3rt word

    # Concat embeddings into one input tensor
    input_tensor = torch.cat([word1, word2, word3], dim=0)

    # Forward pass to caculate output
    output = attention(input_tensor)

    query = attention.query(input_tensor)
    key = attention.key(input_tensor)
    value = attention.value(input_tensor)

    out2 = torch.nn.functional.scaled_dot_product_attention(
        query, key, value)

    sub = out2 - output
    print(sub)
    print(sub.shape)
    assert not sub.round(decimals=6).sum()

    # print(output)
    # print(output.shape) #torch.Size([3, 64])


def _test_forward_MultilHeadAttention(word_size=512, embed_dim=64, n_head=8,
                                      device=torch.device('cuda:0')):

    mha = MultiheadAttention(word_size=word_size,
                              embed_dim=embed_dim,
                              n_head=n_head).to(device=device)

    # Tạo các embedding của 3 từ
    word1 = torch.randn(1, word_size)  # Embedding của từ thứ nhất
    word2 = torch.randn(1, word_size)  # Embedding của từ thứ hai
    word3 = torch.randn(1, word_size)  # Embedding của từ thứ ba

    # Gộp các embedding thành một tensor đầu vào
    input_tensor = torch.cat([word1, word2, word3], dim=0).to(device=device)

    # Forward pass để tính toán đầu ra
    output = mha(input_tensor)

    # In ra kết quả đầu ra
    # print(output)
    # print(output.shape) #torch.Size([3, 64])

def test_forward_MultilHeadAttention(n_test:int=10):
    from random import randint, choice, seed
    from time import time
    seed(0)
    start = time()
    for i in range(n_test):

        kwargs = {'word_size':choice([512, 768, 1024]),
                'embed_dim':choice([64, 512, 1024]),
                'n_head':randint(2, 12),
                'device':torch.device('cpu')
                }
        _test_forward_MultilHeadAttention(**kwargs)
    end = time()
    print('runtime = ', end-start)


def test_forward_MultiQueryAttention():
    word_size =512
    device = torch.device('cuda', 0)
    mqa = MultiQueryAttention(word_size=word_size, embed_dim=64, n_query=8).to(device=device)

    # Tạo các embedding của 3 từ
    word1 = torch.randn(1, word_size)  # Embedding của từ thứ nhất
    word2 = torch.randn(1, word_size)  # Embedding của từ thứ hai
    word3 = torch.randn(1, word_size)  # Embedding của từ thứ ba
        # Gộp các embedding thành một tensor đầu vào
    input_tensor = torch.cat([word1, word2, word3], dim=0).to(device=device)
    output = mqa(input_tensor)
    print(output)
    print(output.shape)


def test_forward_GroupedQueryAttention():
    word_size =512
    device = torch.device('cuda', 0)
    mqa = GroupedQueryAttention(word_size=word_size, embed_dim=64,
                                n_grouped=4, n_query_each_group=2).to(device=device)

    # Tạo các embedding của 3 từ
    word1 = torch.randn(1, word_size)  # Embedding của từ thứ nhất
    word2 = torch.randn(1, word_size)  # Embedding của từ thứ hai
    word3 = torch.randn(1, word_size)  # Embedding của từ thứ ba
        # Gộp các embedding thành một tensor đầu vào
    input_tensor = torch.cat([word1, word2, word3], dim=0).to(device=device)
    output = mqa(input_tensor)
    print(output)
    print(output.shape)


def test_all():
    test_forward_Attention()
    # test_forward_MultilHeadAttention(n_test=10)
    # test_forward_MultiQueryAttention()
    # test_forward_GroupedQueryAttention()

if __name__ == '__main__':
    test_all()
