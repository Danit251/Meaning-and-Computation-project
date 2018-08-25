from nltk.corpus import brown
import statistics, viterbi_algo

corpus_tagged_sentences = brown.tagged_sents(categories='news')
corpus_sentences = brown.sents(categories='news')

training_size = round(len(corpus_sentences) * 0.9)
training_set = corpus_tagged_sentences[:training_size]
test_set = corpus_tagged_sentences[training_size:][:100]
untagged_test_set = corpus_sentences[training_size:][:100]

corpus_size = len(brown.words(categories='news'))

# Constants
START_1 = "START_1"
START_2 = "START_2"
STOP = "STOP"
TIMES = "times"
TAG_TO_UNKNOWN_WORD = "NN"
COMMON_TAGS = 20


# --------------------- Helper Function ---------------------
def get_common_tags(corpus):
    """
    :param corpus: a corpus
    :return: set with the common tags of the corpus
    """
    # Counts number of occurrences of each tag
    tags = {}
    for sen in corpus:
        for word in sen:
            if word[1] not in tags:
                tags[word[1]] = 1
            else:
                tags[word[1]] += 1

    # adds only the common tags
    common_tags = {}
    for tag in tags:
        if tags[tag] > COMMON_TAGS:
            common_tags[tag] = tags[tag]

    return common_tags


def main():
    bigram_stats, trigram_stats, tags_to_words = statistics.tags_stats(training_set)

    tags = list(get_common_tags(training_set).keys())

    # Transitions values
    transition_trigram = viterbi_algo.get_transition_values_trigram(trigram_stats, tags)
    transition_bigram = viterbi_algo.get_transition_values_bigram(bigram_stats, tags)

    # Run bigram and trigram
    viterbi_algo.run_bigram_and_trigram(training_set, test_set,
                                        untagged_test_set, transition_bigram,
                                        transition_trigram, tags_to_words, tags)

if __name__ == "__main__":
    main()
