class NetProperties:
    def __init__(self, word_embed_dim, pos_embed_dim, label_embed_dim, hidden_one_dim, hidden_two_dim, minibatch_size):
        self.word_embed_dim = word_embed_dim
        self.pos_embed_dim = pos_embed_dim
        self.label_embed_dim = label_embed_dim
        self.hidden_one_dim = hidden_one_dim
        self.hidden_two_dim = hidden_two_dim
        self.minibatch_size = minibatch_size