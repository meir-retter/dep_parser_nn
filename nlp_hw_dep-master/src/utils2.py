from collections import defaultdict


class Vocab:
    def __init__(self, word_file, pos_file, label_file,  act_file):
        word_lines = [line.split() for line in word_file.readlines()]
        pos_lines = [line.split() for line in pos_file.readlines()]
        label_lines = [line.split() for line in label_file.readlines()]
        act_lines = [line.split() for line in act_file.readlines()]

        self.word_dict, self.pos_dict, self.label_dict, self.act_dict = {}, {}, {}, {}

        for line in word_lines:
            self.word_dict[line[0]] = int(line[1])
        for line in pos_lines:
            self.pos_dict[line[0]] = int(line[1])
        for line in label_lines:
            self.label_dict[line[0]] = int(line[1])
        for line in act_lines:
            self.act_dict[line[0]] = int(line[1])

        tags = self.pos_dict.keys()
        words = self.word_dict.keys()
        labels = self.label_dict.keys()
        actions = self.act_dict.keys()

        
        self.words = words
        self.tags = tags
        self.labels = labels
        self.actions = actions


    def tagid2tag_str(self, id):
        return self.output_tags[id]

    def tag2id(self, tag):
        return self.pos_dict[tag] if tag in self.pos_dict else self.pos_dict['<null>']

    def word2id(self, word):
        return self.word_dict[word] if word in self.word_dict else self.word_dict['<unk>']

    def label2id(self, label):
        return self.label_dict[label] if label in self.label_dict else self.label_dict['<null>']

    def act2id(self, action):
        return self.act_dict[action]

    def num_words(self):
        return len(self.words)

    # def num_tag_feats(self):
    #     return len(self.feat_tags)

    def num_tags(self):
        return len(self.tags)

    def num_labels(self):
        return len(self.labels)

    def num_actions(self):
        return len(self.actions)
