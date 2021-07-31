from typing import no_type_check


docs = [[doc1],[doc2]]
indexs = [[word2vec_model.wv.vocab[word].index for word in doc] for doc in docs]
word_embdeing = [[list(embed(torch.LongTensor([word2vec_model.wv.vocab
[word].index])).squeeze(0).numpy()) for word in doc]for doc in docs]
word_embdeing[1].extend([list(np.zeros(200)) for _ in range(3)]
