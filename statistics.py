import main


def words_to_tags(corpus):
    """
    :param corpus: to take statistics from
    :return: dictionary with this structure: {"word" : {"tag" : times} }. Gives
    the number of times each word appears with each tag.
    """
    word_to_tags = {}
    for sentence in corpus:
        for word in sentence:
            the_word = word[0]
            the_tag = word[1]

            # The word is already appeared
            if the_word in word_to_tags:

                # The tag already appeared with this word
                if the_tag in word_to_tags[the_word]:
                    word_to_tags[the_word][the_tag] += 1

                # The tag didn't appear with this word yet
                else:
                    word_to_tags[the_word][the_tag] = 1

            # The word didn't appeared yet
            else:
                word_to_tags[the_word] = {the_tag: 1}
    return word_to_tags

def tags_stats(corpus):
    """
    :param corpus: corpus to give statistics from
    :return: statistics of bigram, trigram and emission function
    """
    # {tag: "times": num_of_times, tag_after: num_of_times}
    bigram_stats = {}

    # {tag: "times": num_of_times, tag_1: {tag_2: num_of_times}}
    trigram_stats = {}

    # {tag: {"times": num_of_times, word: num_of_times}}
    tags_to_words = {'NN': {main.TIMES: 0}}

    for sentence in corpus:
        trigram_stats = update_trigram_stats(trigram_stats, main.START_2, main.START_1,
                                             sentence[0][1])
        sentence = [(main.START_1, main.START_1)] + sentence
        for i in range(0, len(sentence) - 2):
            word = sentence[i][0]
            tag = sentence[i][1]
            tag_after = sentence[i + 1][1]
            tag_after_after = sentence[i + 2][1]

            bigram_stats, tags_to_words = update_bigram_stats(bigram_stats,
                                                              tag,
                                                              tag_after, word,
                                                              tags_to_words)
            trigram_stats = update_trigram_stats(trigram_stats, tag, tag_after,
                                                 tag_after_after)

        last_ind = len(sentence) - 1

        # Two last words for bigram
        bigram_stats, tags_to_words = update_bigram_stats(bigram_stats,
                                           sentence[last_ind - 1][1],
                                           sentence[last_ind][1],
                                           sentence[last_ind - 1][0],
                                           tags_to_words)
        # The last word with stop - bigram
        bigram_stats, tags_to_words = update_bigram_stats(bigram_stats,
                                                   sentence[last_ind][1],
                                                   main.STOP,
                                                   sentence[last_ind][0],
                                                   tags_to_words)
        # Two last words with "stop" for trigram
        trigram_stats = update_trigram_stats(trigram_stats,
                                             sentence[last_ind - 1][1],
                                             sentence[last_ind][1], main.STOP)

    return bigram_stats, trigram_stats, tags_to_words


def update_trigram_stats(neighbors_stats, tag, tag_1, tag_2):
    """
    Helper function to tags_stats. Updates the dictionary with the new
    three tags
    :param neighbors_stats: dictionary to update
    :param tag: first tag
    :param tag_1: second tag
    :param tag_2: third tag
    :return: updated dictionary
    """
    # Not the first time seeing the tag
    if tag in neighbors_stats:
        if tag_1 in neighbors_stats[tag]:
            if tag_2 in neighbors_stats[tag][tag_1]:
                neighbors_stats[tag][tag_1][tag_2] += 1
            else:
                neighbors_stats[tag][tag_1][tag_2] = 1
        else:
            neighbors_stats[tag][tag_1] = {main.TIMES: 0, tag_2: 1}
    else:
        neighbors_stats[tag] = {tag_1: {main.TIMES: 0, tag_2: 1}}

    # Add one to the number of times seen "tag"
    neighbors_stats[tag][tag_1][main.TIMES] += 1

    return neighbors_stats


def update_bigram_stats(neighbors_stats, tag, tag_after, word, tags_to_words):
    """
    Helper function to tags_stats. Updates the dictionary with the new
    pair tags and update the statistics of tags with words.
    :param neighbors_stats: dictionary of bigram to update
    :param tag: first tag
    :param tag_after: second tag
    :param word: a word
    :param tags_to_words: a dictionary to update
    :return: Updated dictionaries
    """
    # Not the first time seeing the tag
    if tag in neighbors_stats:

        # Not the first time seeing the tag_after after tag
        if tag_after in neighbors_stats[tag]:
            neighbors_stats[tag][tag_after] += 1
        # First time seeing the tag_after after tag
        else:
            neighbors_stats[tag][tag_after] = 1

        neighbors_stats[tag][main.TIMES] += 1

        if word in tags_to_words[tag]:
            tags_to_words[tag][word] += 1
        else:
            tags_to_words[tag][word] = 1

        tags_to_words[tag][main.TIMES] += 1
    # First time seeing the tag
    else:
        neighbors_stats[tag] = {main.TIMES: 1, tag_after: 1}
        tags_to_words[tag] = {main.TIMES: 1, word: 1}

    return neighbors_stats, tags_to_words
