import os
from itertools import count


class Token:
    pass


class UToken(Token):
    def __init__(self, tid, form, lemma, upos, xpos, feats,
                 head, deprel, deps, misc, aspects=None):
        """
        Args:
          tid: Word index, starting at 1; may be a range for multi-word tokens;
            may be a decimal number for empty nodes.
          form: word form or punctuation symbol.
          lemma: lemma or stem of word form
          upos: universal part-of-speech tag
          xpos: language specific part-of-speech tag
          feats: morphological features
          head: head of current word (an ID or 0)
          deprel: universal dependency relation to the HEAD (root iff HEAD = 0)
          deps: enhanced dependency graph in the form of a list of head-deprel pairs
          misc: any other annotation
        """
        self.str_id = tid  # Use this for printing the conll
        self.id = int(float(tid))  # Use this for training TODO: what is this 10.1 business?
        self.form = form  # 原始词
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats
        self.head = int(head)  # head的id，为0的是root
        self.deprel = deprel  #
        self.deps = deps
        self.misc = misc
        self.aspects = aspects

    def __str__(self):
        return '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % (
            self.str_id, self.form, self.lemma, self.upos, self.xpos, self.feats,
            self.head, self.deprel, self.deps, self.misc)

    @property
    def pos(self):
        return self.upos


def read_conllu(f):

    tokens = []

    for line in f:
        line = line.strip()

        if not line:
            yield tokens
            tokens = []
            continue

        if line[0] == "#":
            continue

        parts = line.split()
        assert len(parts) == 10, "invalid conllu line"
        tokens.append(UToken(*parts))

    # possible last sentence without newline after
    if len(tokens) > 0:
        yield tokens


if __name__ == '__main__':
  pass