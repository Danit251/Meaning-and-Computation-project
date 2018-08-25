import numpy as np
import main, statistics
np.set_printoptions(threshold=np.nan)

# Tags to check
verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
noun_tags = {'NN', 'NN$', 'NNS', 'NNS$', 'NP', 'NP$', 'NPS', 'NPS$'}
adj_tags = {'JJ', 'JJR', 'JJS', 'JJT'}


# --------------------- Viterbi for bigram ---------------------
def viterbi_bigram(sentence, all_tags, transition_values, emission_func, tags_to_words):
    """
    Viterbi algorithm for bigram
    :param sentence: a sentence
    :param all_tags: tags to tag the sentence
    :param transition_values: transition function for bigram
    :param emission_func: emission function
    :param tags_to_words: statistics of words with tags
    :return: a sequence of tags to the sentence
    """
    n = len(sentence)
    matrix = np.zeros((n+1, len(tags_to_words.keys())))

    # Tags for the sentence
    max_tags = []

    # Probability to "START_1"
    matrix[0, :] = 1

    for k in range(1, n+1):
        for v, tag_v in enumerate(all_tags):
            e_value = emission_func(tag_v, sentence[k-1], tags_to_words)
            val = np.zeros(len(all_tags))
            for u, tag_u in enumerate(all_tags):
                q_value = transition_values[tag_v][tag_u]
                val[u] = matrix[k - 1, u] * q_value * e_value
            matrix[k, v] = np.max(val)
        max_tags.append(all_tags[np.argmax(matrix[k])])
    return max_tags


def calc_transition_bigram(tag, tag_before, tags_dictionary):
    """
    Calculates the transition value for bigram
    :param tag: second tag
    :param tag_before: first tag
    :param tags_dictionary: dictionary with statistics of pairs tags
    :return: value of transition
    """
    trans = 0
    if tag_before in tags_dictionary:
        if tag in tags_dictionary[tag_before]:
            common = tags_dictionary[tag_before][tag]
            all_possible_tags = tags_dictionary[tag_before][main.TIMES]
            trans = common / all_possible_tags
    return trans


def get_transition_values_bigram(bigram_stats, all_tags):
    """
    Creates dictionary for bigram transition function
    :param bigram_stats: statistics of all sequences of pairs tags
    :param all_tags: all tags
    :return: dictionary that represents transition function to bigram
    """
    transition_dict = {}

    for tag1 in all_tags:
        transition_dict[tag1] = {}
        for tag2 in all_tags:
            transition_dict[tag1][tag2] = calc_transition_bigram(tag1, tag2,
                                                                 bigram_stats)

    return transition_dict


# --------------------- Viterbi for trigram ---------------------
def viterbi_trigram(sentence, all_tags, transition_values, emission_func, tags_to_words):
    """
    Viterbi algorithm for trigram
    :param sentence: a sentence
    :param all_tags: tags to tag the sentence
    :param transition_values: transition function for trigram
    :param emission_func: emission function
    :param tags_to_words: statistics of words with tags
    :return: a sequence of tags to the sentence
    """
    n = len(sentence)

    matrix = np.zeros((n+1, len(all_tags), len(all_tags)))

    # Tags for the sentence
    max_tags = []

    # Probability to "START_2"
    matrix[0, :, :] = 1

    # Dictionary for backtracking
    bp = {}

    for k in range(1, n+1):
        for v, tag_v in enumerate(all_tags):
            e_value = emission_func(tag_v, sentence[k-1], tags_to_words)
            for u, tag_u in enumerate(all_tags):
                val = np.zeros(len(all_tags))
                for s, tag_s in enumerate(all_tags):
                    q_value = transition_values[tag_v][tag_u][tag_s]
                    val[s] = matrix[k - 1, u, s] * q_value * e_value
                matrix[k, v, u] = np.max(val)
                bp[(k, tag_v, tag_u)] = all_tags[np.argmax(val)]

    max_score = float('-Inf')
    u_max, v_max = None, None
    for v, tag_v in enumerate(all_tags):
        for u, tag_u in enumerate(all_tags):
            val = matrix[n, v, u]*transition_values[main.STOP][tag_v][tag_u]
            if val > max_score:
                max_score = val
                v_max = tag_v
                u_max = tag_u
    max_tags.append(v_max)
    max_tags.append(u_max)

    for i, k in enumerate(range(n-2, 0, -1)):
        max_tags.append(bp[(k+2, max_tags[i], max_tags[i+1])])
    max_tags.reverse()

    return max_tags


def calc_transition_trigram(tag, tag_before_1, tag_before_2, tags_dictionary):
    """
    Calculates the transition value for trigram.
    :param tag: third tag
    :param tag_before_1: second tag
    :param tag_before_2: first tag
    :param tags_dictionary: dictionary with statistics of pairs tags
    :return: value of transition
    """
    trans = 0
    if tag_before_2 in tags_dictionary:
        if tag_before_1 in tags_dictionary[tag_before_2]:
            if tag in tags_dictionary[tag_before_2][tag_before_1]:
                common = tags_dictionary[tag_before_2][tag_before_1][main.TIMES]
                all_pairs = tags_dictionary[tag_before_2][tag_before_1][tag]
                trans = common / all_pairs
    return trans


# Calculates for each trio of tags his transition value
def get_transition_values_trigram(trigram_stats, all_tags):
    """
    Creates dictionary for trigram transition function
    :param trigram_stats: statistics of all sequences of three tags
    :param all_tags: all tags
    :return: dictionary that represents transition function to trigram
    """
    transition_dict = {}

    for tag1 in all_tags+["STOP"]:
        transition_dict[tag1] = {}
        for tag2 in all_tags:
            transition_dict[tag1][tag2] = {}
            for tag3 in all_tags:
                transition_dict[tag1][tag2][tag3] = \
                    calc_transition_trigram(tag1, tag2, tag3, trigram_stats)

    return transition_dict


# --------------------- Emission ---------------------
def add_one_emission(tag, word, tags_to_words):
    """
    Calculates add-one emission
    :param tag: a tag
    :param word: a word
    :param tags_to_words: statistics of words with tags
    :return: the value of the emission
    """
    delta = 1
    if tag in tags_to_words:
        common = 0
        if word in tags_to_words[tag]:
            common = tags_to_words[tag][word]
        tag_prob = tags_to_words[tag][main.TIMES]
        return (common + delta) / (tag_prob + main.corpus_size)
    return 0


# --------------------- Run Viterbi ---------------------
def run_viterbi(training, test, untagged, viterbi, tags, transition, tags_to_words):
    """
    Runs Viterbi algorithm and prints success rates
    :param training: a training set
    :param test: a test set
    :param untagged: an untagged test set
    :param viterbi: a viterbi algorithm
    :param tags: the tags
    :param transition: transition function
    :param tags_to_words: statistics of tags with words
    """
    words_to_tags_training = statistics.words_to_tags(training)

    known_success = 0
    unknown_success = 0

    known_words = 0
    unknown_words = 0

    success_tags = {'V': 0, 'N': 0, 'A': 0, 'O': 0}
    general_tags = {'V': 0, 'N': 0, 'A': 0, 'O': 0}

    for k in range(len(untagged)):
        res = viterbi(untagged[k], tags, transition, add_one_emission,
                      tags_to_words)
        for i in range(len(res)):
            right_res = test[k][i][1]
            tag_type = 'O'
            if right_res in verb_tags:
                tag_type = 'V'
            elif right_res in noun_tags:
                tag_type = 'N'
            elif right_res in adj_tags:
                tag_type = 'A'

            if res[i] == right_res:
                if test[k][i][0] in words_to_tags_training:
                    known_success += 1
                    success_tags[tag_type] +=1
                else:
                    unknown_success += 1

            if test[k][i][0] in words_to_tags_training:
                known_words += 1
                general_tags[tag_type] += 1
            else:
                unknown_words += 1

    print('Known: ', known_success / known_words)
    print('Unknown: ', unknown_success / unknown_words)
    print('Total: ',
          (known_success + unknown_success) / (known_words + unknown_words))
    print('Verb: ', success_tags['V'] / general_tags['V'])
    print('Noun: ', success_tags['N'] / general_tags['N'])
    print('Adjective: ', success_tags['A'] / general_tags['A'])


def run_bigram_and_trigram(training_set, test_set, untagged_test_set,
                           transition_bigram, transition_trigram,
                           tags_to_words, tags):

    print("Viterbi - Bigram:")
    run_viterbi(training_set, test_set, untagged_test_set, viterbi_bigram,
                tags, transition_bigram, tags_to_words)

    print("Viterbi - Trigram:")
    run_viterbi(training_set, test_set, untagged_test_set, viterbi_trigram,
                tags, transition_trigram, tags_to_words)
