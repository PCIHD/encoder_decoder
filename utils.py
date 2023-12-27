import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def get_weights(word_embedding, vocabulary: list):
    data = {}
    for width in range(word_embedding.network_width):
        data[width] = []
        for vocab_item_weight in range(word_embedding.vocab_size):
            data[width].append(
                getattr(
                    word_embedding, f"input_weight_{vocab_item_weight}_{width}"
                ).item()
            )
    data["token"] = vocabulary
    df = pd.DataFrame(data)
    return df


def plot_token_relations(word_embedding, vocabulary: list):
    df = get_weights(word_embedding, vocabulary)
    sns.scatterplot(data=df, x=0, y=1, legend=True)
    for i in range(df.shape[0]):
        plt.text(df[0][i], df[1][i], df["token"][i])
    plt.show()


def vectorize_text(eos_symbol: str, text: str, vocabulary: list):
    encoding = []
    label = []
    words = text.split(" ")
    for i, word in enumerate(words):
        word_encoding = [0.0 for _ in range(len(vocabulary))]
        label_encoding = [0.0 for _ in range(len(vocabulary))]
        word_encoding[vocabulary.index(word)] = 1.0
        if word != eos_symbol:
            # get the encoding for the next word if it is not eos
            word_encoding[vocabulary.index(words[i + 1])] = 1.0
            label_encoding[vocabulary.index(words[i + 1])] = 1.0

        else:
            word_encoding[vocabulary.index(eos_symbol)] = 1.0
            label_encoding[vocabulary.index(eos_symbol)] = 1.0
        encoding.append(word_encoding)
        label.append(label_encoding)

    return encoding, label
